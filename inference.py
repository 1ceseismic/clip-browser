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


class ImageFolderDataset(Dataset):
    def __init__(self, img_dir: Path, preprocess):
        self.paths = sorted([p for p in img_dir.iterdir()
                              if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        return idx, self.preprocess(img), str(path)


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
        self.paths = []

    def build_index(self,img_dir: str, index_path: str, meta_path: str, batch_size: int = 32, num_workers: int = 4):
        """
        Compute embeddings for all images in img_dir, build a FAISS IndexIDMap,
        and save the index and metadata (paths + optional mtime)
        """
        dataset = ImageFolderDataset(Path(img_dir), self.preprocess)
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

        all_embs = []
        all_meta = []

        with torch.no_grad():
            for idxs, images, paths in loader:
                images = images.to(self.device)
                feats = self.model.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_embs.append(feats.cpu().numpy())
                for i, p in zip(idxs.tolist(), paths):
                    stat = Path(p).stat()
                    all_meta.append({
                        'id': i,
                        'path': p,
                        'mtime': stat.st_mtime
                    })

        embeddings = np.vstack(all_embs).astype('float32')
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap(index)
        ids = np.arange(len(embeddings))
        self.index.add_with_ids(embeddings, ids)

        faiss.write_index(self.index, index_path)
        with open(meta_path, 'w') as f:
            json.dump(all_meta, f, indent=2)

        print(f"Index saved to {index_path}, metadata saved to {meta_path}")

    def load_index(self, index_path: str, meta_path: str):
        #load a saved faiss index

        self.index = faiss.read_index(index_path)
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        # Sort by id to ensure consistent ordering
        meta_sorted = sorted(meta, key=lambda x: x['id'])
        self.paths = [item['path'] for item in meta_sorted]
        print(f"Loaded index ({self.index.ntotal} vectors) and {len(self.paths)} paths")

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]: 
    # Encode query into embeddings and search the faiss index; we return the scores along with img path  
        tokens = self.tokenizer([query]).to(self.device)
        with torch.no_grad():
            txt_emb = self.model.encode_text(tokens)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        q_np = txt_emb.cpu().numpy().astype('float32')
        D, I = self.index.search(q_np, top_k)
        results = [(self.paths[i], float(D[0][k])) for k, i in enumerate(I[0])]
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
