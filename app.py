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

# --- API Endpoints ---

@app.on_event("startup")
def startup():
    """Load the index and metadata from disk if they exist."""
    try:
        with _idx_lock:
            g["indexer"].load_index(INDEX_FILE, METADATA_FILE)

            # Consistency check to prevent IndexError
            if g["indexer"].index is not None and g["indexer"].paths is not None:
                if g["indexer"].index.ntotal != len(g["indexer"].paths):
                    print(f"Index-metadata mismatch! Index has {g['indexer'].index.ntotal} vectors, metadata has {len(g['indexer'].paths)} paths. Discarding loaded index.")
                    raise ValueError("Inconsistent index and metadata files.")

            with open(APP_METADATA_FILE) as f:
                app_metadata = json.load(f)
                g["indexed_dir"] = app_metadata.get("indexed_dir")
            if not g["indexed_dir"]:
                raise KeyError("indexed_dir not found in app metadata")
            print(f"Loaded index for '{g['indexed_dir']}' with {g['indexer'].index.ntotal} embeddings.")
    except (FileNotFoundError, KeyError, json.JSONDecodeError, ValueError, AttributeError, RuntimeError) as e:
        print(f"Index files missing or invalid — rebuild using /build-index. Error: {e}")
        g["indexer"].index = None
        g["indexer"].paths = []
        g["indexed_dir"] = None


@app.post("/set-dataset-root")
def set_dataset_root(payload: PathRequest):
    """Sets the root directory for datasets."""
    new_root = Path(payload.path)
    if not new_root.is_dir():
        raise HTTPException(status_code=404, detail="Path is not a valid directory")
    g["dataset_root"] = new_root
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

        with open(APP_METADATA_FILE, "w") as f:
            json.dump({"indexed_dir": img_dir}, f)

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
    full_results = [{"path": f"{base_path}/{p}", "score": s} for p, s in results]
    
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
    if g["indexer"].index is None or g["indexed_dir"] is None or not g["indexer"].paths:
        return {"images": []}
    
    base_path = g["indexed_dir"]
    full_paths = [f"{base_path}/{p}" for p in g["indexer"].paths]
    return {"images": full_paths}

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
    thumbnail_path = THUMBNAIL_CACHE_DIR / relative_path
    
    if thumbnail_path.exists():
        return FileResponse(str(thumbnail_path))

    try:
        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(full_image_path) as img:
            if img.mode not in ("RGB", "RGBA"):
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
