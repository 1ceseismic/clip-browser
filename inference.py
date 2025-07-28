import os
import json
import argparse
from pathlib import Path

import torch
import numpy as np
import faiss
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
import umap

def filter_collate_fn(batch):
    """
    A custom collate function that filters out items where the image failed to load.
    This needs to be a top-level function to be picklable by multiprocessing workers.
    """
    # Filter out samples where the image data (at index 1) is None
    batch = [b for b in batch if b[1] is not None]
    if not batch:
        # If the whole batch is bad, return None for all components
        return None, None, None
    # Otherwise, use the default collate function on the filtered batch
    return torch.utils.data.default_collate(batch)

class ImageFolderDataset(Dataset):
    def __init__(self, img_dir: Path, preprocess):
        self.paths = sorted([p for p in img_dir.rglob('*')
                              if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            with Image.open(path) as img:
                # Convert to RGBA first to handle palette transparency gracefully, then to RGB.
                # This is a robust way to handle various image modes before processing.
                img = img.convert('RGBA').convert('RGB')
            return idx, self.preprocess(img), str(path)
        except Exception as e:
            print(f"Warning: Could not load image {path}, skipping. Error: {e}")
            # Return None to indicate a failed load
            return idx, None, str(path)


class CLIPIndexer:
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device=self.device
        )
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.index = None
        self.metadata = []

    @property
    def paths(self):
        """Returns a list of paths from the metadata."""
        return [item['path'] for item in self.metadata]

    def build_index(self,img_dir: str, index_path: str, meta_path: str, batch_size: int = 32, num_workers: int = 4, n_clusters: int = 10):
        """
        Compute embeddings, cluster them, and save the index and metadata.
        """
        img_dir_path = Path(img_dir)
        dataset = ImageFolderDataset(img_dir_path, self.preprocess)
        
        # Use the top-level, picklable collate function
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, collate_fn=filter_collate_fn)

        all_embs = []
        all_meta_map = {} # Use a map to handle filtered items

        with torch.no_grad():
            for idxs, images, paths in loader:
                if idxs is None: continue # Skip empty batches from collate_fn
                images = images.to(self.device)
                feats = self.model.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_embs.append(feats.cpu().numpy())
                for i, p_str in zip(idxs.tolist(), paths):
                    p = Path(p_str)
                    stat = p.stat()
                    rel_path = p.relative_to(img_dir_path).as_posix()
                    all_meta_map[i] = {
                        'id': i,
                        'path': rel_path,
                        'mtime': stat.st_mtime
                    }

        if not all_embs:
            print("No images were successfully processed. Index not built.")
            return

        embeddings = np.vstack(all_embs).astype('float32')
        
        # --- Clustering and Dimensionality Reduction ---
        print("Performing clustering...")
        # Ensure we don't have more clusters than samples
        actual_n_clusters = min(n_clusters, len(embeddings))
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=0, n_init='auto').fit(embeddings)
        
        print("Performing UMAP reduction...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_coords = reducer.fit_transform(embeddings)

        # --- Finalize Metadata ---
        final_meta = []
        valid_ids = sorted(all_meta_map.keys())
        for new_id, old_id in enumerate(valid_ids):
            meta_item = all_meta_map[old_id]
            meta_item['id'] = new_id # Re-index to be contiguous
            meta_item['cluster_id'] = int(kmeans.labels_[new_id])
            meta_item['umap'] = umap_coords[new_id].tolist()
            final_meta.append(meta_item)

        # --- Build FAISS Index ---
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap(index)
        # Use the new contiguous IDs
        ids = np.arange(len(final_meta)).astype('int64')
        self.index.add_with_ids(embeddings, ids)

        faiss.write_index(self.index, index_path)
        with open(meta_path, 'w') as f:
            json.dump(final_meta, f, indent=2)

        print(f"Index saved to {index_path}, metadata saved to {meta_path}")

    def load_index(self, index_path: str, meta_path: str):
        """Load a saved FAISS index and its metadata."""
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)
        
        if self.index.ntotal != len(self.metadata):
            raise ValueError("Index size and metadata length do not match.")

        print(f"Loaded index ({self.index.ntotal} vectors) and {len(self.metadata)} metadata entries")

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]: 
        tokens = self.tokenizer([query]).to(self.device)
        with torch.no_grad():
            txt_emb = self.model.encode_text(tokens)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        q_np = txt_emb.cpu().numpy().astype('float32')
        D, I = self.index.search(q_np, top_k)
        results = [(self.metadata[i]['path'], float(D[0][k])) for k, i in enumerate(I[0]) if i != -1]
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Build or search a CLIP embedding index"
    )
    sub = parser.add_subparsers(dest='command', required=True)

    idx = sub.add_parser('index')
    idx.add_argument('--img-dir', required=True)
    idx.add_argument('--checkpoint', default='openai', help="Pretrained weights name or path (default: 'openai')")
    idx.add_argument('--out-index', default='image.index')
    idx.add_argument('--out-meta', default='metadata.json')
    idx.add_argument('--batch-size', type=int, default=32)
    idx.add_argument('--workers', type=int, default=4)

    srch = sub.add_parser('search')
    srch.add_argument('--query', required=True)
    srch.add_argument('--checkpoint', required=True)
    srch.add_argument('--index', default='image.index')
    srch.add_argument('--meta', default='metadata.json')
    srch.add_argument('--top-k', type=int, default=5)

    args = parser.parse_args()

    if args.command == 'index':
        indexer = CLIPIndexer('ViT-B-32', pretrained=args.checkpoint)
        indexer.build_index(
            img_dir=args.img_dir,
            index_path=args.out_index,
            meta_path=args.out_meta,
            batch_size=args.batch_size,
            num_workers=args.workers
        )
    elif args.command == 'search':
        indexer = CLIPIndexer('ViT-B-32', args.checkpoint)
        indexer.load_index(args.index, args.meta)
        results = indexer.search(args.query, top_k=args.top_k)
        for path, score in results:
            print(f"{path}    {score:.4f}")


if __name__ == '__main__':
    main()
