from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from inference import CLIPIndexer
import threading
import os
from pathlib import Path
import json
from PIL import Image
from typing import Optional, List, Dict, Any
from collections import defaultdict
import config

# --- Configuration ---
INDEX_FILE = "image.index"
METADATA_FILE = "metadata.json"  # Used by inference.py, contains a list of dicts with relative paths
APP_METADATA_FILE = "app_metadata.json" # Used by this app, contains a dict
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
THUMBNAIL_CACHE_DIR = Path(".thumbnails")
THUMBNAIL_MAX_SIZE = (256, 256)

# Create cache directory if it doesn't exist
THUMBNAIL_CACHE_DIR.mkdir(exist_ok=True)

app = FastAPI(title="CLIP Semantic Search API")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State ---
g = {
    "indexer": CLIPIndexer(model_name="ViT-B-32", pretrained="openai"),
    "indexed_dir": None,
    "dataset_root": None, # Will be set by the user via an API call
}
_idx_lock = threading.Lock()

class PathRequest(BaseModel):
    path: str

class SearchResponse(BaseModel):
    query: str
    results: list[dict]

class AppStatus(BaseModel):
    dataset_root: Optional[str]
    indexed_dir: Optional[str]
    index_loaded: bool
    indexed_image_count: int
    has_clusters: bool

class ClusterInfo(BaseModel):
    cluster_id: int
    count: int
    preview_paths: List[str]

class ClustersResponse(BaseModel):
    clusters: List[ClusterInfo]

class ClusterImagesResponse(BaseModel):
    cluster_id: int
    image_paths: List[str]

class AllMetadataResponse(BaseModel):
    metadata: List[Dict[str, Any]]


# --- Internal Logic ---

def _load_index_for_root(root_path: Path):
    """
    Attempts to load the index files corresponding to a given root path.
    Clears the current index if no valid index is found.
    This is the central function for managing the loaded index state.
    """
    global g
    with _idx_lock:
        # Reset current index state before attempting to load a new one
        g["indexer"].index = None
        g["indexer"].metadata = []
        g["indexed_dir"] = None
        g["dataset_root"] = root_path # Set the root, even if index loading fails

        # The app metadata file is always in the project root, not the dataset root
        if not os.path.exists(APP_METADATA_FILE):
            print(f"App metadata file not found. Cannot load index for {root_path}")
            return

        try:
            with open(APP_METADATA_FILE) as f:
                app_metadata = json.load(f)

            # Check if the index we have corresponds to the root we want to load
            if app_metadata.get("dataset_root") != str(root_path):
                print(f"Index mismatch: App metadata is for '{app_metadata.get('dataset_root')}', but trying to load '{root_path}'. Not loading index.")
                return

            # Check if the main index files exist
            if not all(os.path.exists(f) for f in [INDEX_FILE, METADATA_FILE]):
                print(f"Index files '{INDEX_FILE}' or '{METADATA_FILE}' not found. Cannot load index for {root_path}")
                return
            
            print(f"Found valid index for root: {root_path}. Loading...")
            g["indexed_dir"] = app_metadata.get("indexed_dir")
            g["indexer"].load_index(INDEX_FILE, METADATA_FILE)
            print(f"Loaded index for '{g['indexed_dir']}' with {g['indexer'].index.ntotal} embeddings.")

        except (FileNotFoundError, KeyError, json.JSONDecodeError, ValueError, AttributeError, RuntimeError) as e:
            print(f"Error during index load for {root_path}: {e}. Clearing index state.")
            g["indexer"].index = None
            g["indexer"].metadata = []
            g["indexed_dir"] = None


# --- API Endpoints ---

@app.on_event("startup")
def startup():
    """Load the last used index and metadata from disk if they exist."""
    last_root = config.get_last_used_path()
    if last_root:
        _load_index_for_root(Path(last_root))
    else:
        print("No last used dataset root found in config.")


@app.post("/set-dataset-root")
def set_dataset_root(payload: PathRequest):
    """Sets the root directory for datasets, saves it, and attempts to load its index."""
    new_root = Path(payload.path)
    if not new_root.is_dir():
        raise HTTPException(status_code=404, detail="Path is not a valid directory")
    
    # Save this path as the most recent one
    config.add_recent_path(str(new_root))
    
    # Attempt to load the index for this new root, which will update the global state
    _load_index_for_root(new_root)
    
    return {"status": "ok", "path": str(new_root)}

@app.post("/build-index")
def build_index(img_dir: str, checkpoint: str = "openai"):
    """Build or rebuild the FAISS index from a subdirectory of the dataset root."""
    if not g.get("dataset_root"):
        raise HTTPException(status_code=400, detail="Dataset root path not set")
    with _idx_lock:
        # Resolve the path to get a clean, absolute path for the indexer
        full_img_path = (g["dataset_root"] / img_dir).resolve()
        if not full_img_path.is_dir():
            raise HTTPException(status_code=404, detail=f"Directory not found: {full_img_path}")

        # --- Force a clean rebuild by deleting old files ---
        for f in [INDEX_FILE, METADATA_FILE, APP_METADATA_FILE]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError as e:
                    print(f"Error removing file {f}: {e}")
                    raise HTTPException(status_code=500, detail=f"Could not remove old index file: {f}")

        g["indexer"] = CLIPIndexer("ViT-B-32", pretrained=checkpoint)
        # build_index now correctly handles relative paths internally
        g["indexer"].build_index(str(full_img_path), INDEX_FILE, METADATA_FILE)
        
        # After building, the indexer object in memory is still empty.
        # We need to load the index and metadata we just created to sync the state.
        g["indexer"].load_index(INDEX_FILE, METADATA_FILE)

        # Save the dataset root along with the indexed directory
        with open(APP_METADATA_FILE, "w") as f:
            json.dump({
                "indexed_dir": img_dir,
                "dataset_root": str(g["dataset_root"])
            }, f)

        g["indexed_dir"] = img_dir
        print(f"Successfully built index for '{img_dir}'.")
        return {"status": "index built", "total": g["indexer"].index.ntotal}

@app.get("/search", response_model=SearchResponse)
def search(q: str, top_k: int = 5):
    if g["indexer"].index is None or g["indexed_dir"] is None:
        raise HTTPException(status_code=400, detail="Index not loaded; call /build-index first")
    if not g.get("dataset_root"):
        raise HTTPException(status_code=400, detail="Dataset root path not set. Please set it before searching.")
    
    results = g["indexer"].search(q, top_k=top_k)
    base_path = g["indexed_dir"]
    # The paths from the indexer are now relative, so we prepend the indexed directory name
    # Use os.path.join and then normalize to forward slashes for API consistency
    full_results = [{"path": os.path.join(base_path, p).replace('\\', '/'), "score": s} for p, s in results]
    
    return SearchResponse(query=q, results=full_results)

@app.get("/directories")
def get_directories():
    """
    Returns a list of subdirectories in the DATASET_ROOT.
    Also includes '.' if the root itself contains images.
    """
    if not g.get("dataset_root"):
        raise HTTPException(status_code=400, detail="Dataset root path not set")
    
    root_path = g["dataset_root"]
    dirs = [p.name for p in root_path.iterdir() if p.is_dir()]
    
    # Check if the root directory itself contains images
    has_images_in_root = any(p.suffix.lower() in IMAGE_EXTENSIONS for p in root_path.iterdir() if p.is_file())
    
    if has_images_in_root:
        # Prepend '.' to represent the root directory itself as a dataset
        dirs.insert(0, '.')
        
    return {"directories": dirs}

@app.get("/all-images")
def get_all_images():
    """Returns a list of all image paths from the current index."""
    if g["indexer"].index is None or g["indexed_dir"] is None or not g["indexer"].metadata:
        return {"images": []}
    
    base_path = g["indexed_dir"]
    full_paths = [os.path.join(base_path, item['path']).replace('\\', '/') for item in g["indexer"].metadata]
    return {"images": full_paths}

@app.get("/all-metadata", response_model=AllMetadataResponse)
def get_all_metadata():
    """Returns the complete metadata list for all indexed items, including UMAP and cluster data."""
    if not g["indexer"].metadata:
        return {"metadata": []}
    
    # The metadata is already a list of dicts, which is what we need.
    return {"metadata": g["indexer"].metadata}

@app.get("/status", response_model=AppStatus)
def get_status():
    """Returns the current status of the application."""
    with _idx_lock:
        count = 0
        has_clusters = False
        if g["indexer"].index is not None and g["indexer"].metadata:
            count = g["indexer"].index.ntotal
            if 'cluster_id' in g["indexer"].metadata[0]:
                has_clusters = True
        
        return AppStatus(
            dataset_root=str(g["dataset_root"]) if g.get("dataset_root") else None,
            indexed_dir=g.get("indexed_dir"),
            index_loaded=g["indexer"].index is not None,
            indexed_image_count=count,
            has_clusters=has_clusters
        )

@app.get("/clusters", response_model=ClustersResponse)
def get_clusters():
    """Groups images by cluster_id and returns a summary of each cluster."""
    if not g["indexer"].metadata or 'cluster_id' not in g["indexer"].metadata[0]:
        raise HTTPException(status_code=404, detail="No cluster information available. Please rebuild the index.")

    clusters = defaultdict(list)
    for item in g["indexer"].metadata:
        clusters[item['cluster_id']].append(item['path'])

    response_clusters = []
    base_path = g["indexed_dir"]
    for cluster_id, paths in sorted(clusters.items()):
        response_clusters.append(ClusterInfo(
            cluster_id=cluster_id,
            count=len(paths),
            preview_paths=[os.path.join(base_path, p).replace('\\', '/') for p in paths[:4]] # Get up to 4 previews
        ))
    
    return ClustersResponse(clusters=response_clusters)

@app.get("/cluster/{cluster_id}", response_model=ClusterImagesResponse)
def get_cluster_images(cluster_id: int):
    """Returns all image paths for a given cluster_id."""
    if not g["indexer"].metadata or 'cluster_id' not in g["indexer"].metadata[0]:
        raise HTTPException(status_code=404, detail="No cluster information available.")

    base_path = g["indexed_dir"]
    image_paths = [
        os.path.join(base_path, item['path']).replace('\\', '/')
        for item in g["indexer"].metadata
        if item['cluster_id'] == cluster_id
    ]

    if not image_paths:
        raise HTTPException(status_code=404, detail=f"Cluster ID {cluster_id} not found.")

    return ClusterImagesResponse(cluster_id=cluster_id, image_paths=image_paths)


@app.get("/health")
def health():
    return {"status": "ok", "loaded": g["indexer"].index is not None, "indexed_dir": g["indexed_dir"]}

@app.get("/thumbnail/{image_path:path}")
def get_thumbnail(image_path: str):
    """Generates and serves a thumbnail for the requested image."""
    if not g.get("dataset_root"):
        raise HTTPException(status_code=404, detail="Dataset root not set")

    try:
        full_image_path = (g["dataset_root"] / image_path).resolve()
        if g["dataset_root"].resolve() not in full_image_path.parents and full_image_path != g["dataset_root"].resolve():
             raise HTTPException(status_code=403, detail="Forbidden: Access outside of dataset root is not allowed.")
    except Exception:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid path.")

    if not full_image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    relative_path = full_image_path.relative_to(g["dataset_root"])
    # Create a consistent cache path, always using .jpg for thumbnails
    thumbnail_path = (THUMBNAIL_CACHE_DIR / relative_path).with_suffix('.jpg')
    
    if thumbnail_path.exists():
        return FileResponse(str(thumbnail_path))

    try:
        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(full_image_path) as img:
            # Convert to RGB first to handle all modes before processing
            img = img.convert("RGB")
            img.thumbnail(THUMBNAIL_MAX_SIZE)
            img.save(thumbnail_path, "JPEG", quality=85)
        return FileResponse(str(thumbnail_path))
    except Exception as e:
        print(f"Error generating thumbnail for {full_image_path}: {e}")
        raise HTTPException(status_code=500, detail="Could not generate thumbnail.")

@app.get("/image/{image_path:path}")
def get_image(image_path: str):
    """Serves a full-resolution image from the user-defined dataset root."""
    if not g.get("dataset_root"):
        raise HTTPException(status_code=404, detail="Dataset root not set")
    
    try:
        # Security: Prevent path traversal attacks
        full_path = (g["dataset_root"] / image_path).resolve()
        # Check if the resolved path is within the dataset_root
        if g["dataset_root"].resolve() not in full_path.parents and full_path != g["dataset_root"].resolve():
             raise HTTPException(status_code=403, detail="Forbidden: Access outside of dataset root is not allowed.")
    except Exception:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid path.")

    if not full_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(str(full_path))
