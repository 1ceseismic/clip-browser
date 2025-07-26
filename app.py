from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import CLIPIndexer
import threading

app = FastAPI(title="CLIP Semantic Search API")

# Global indexer instance
indexer = CLIPIndexer(model_name="ViT-B-32", pretrained="openai")
_idx_lock = threading.Lock()

class SearchResponse(BaseModel):
    query: str
    results: list[dict]  # {path, score}

@app.on_event("startup")
def startup():
    # load index if exists
    try:
        indexer.load_index("image.index", "metadata.json")
    except FileNotFoundError:
        print("Index files missing — rebuild using /build-index")

@app.post("/build-index")
def build_index(img_dir: str, checkpoint: str = "openai"):
    """Build or rebuild the FAISS index from a directory."""
    with _idx_lock:
        indexer = CLIPIndexer("ViT-B-32", pretrained=checkpoint)
        indexer.build_index(img_dir, "image.index", "metadata.json")
        indexer.load_index("image.index", "metadata.json")
    return {"status": "index built", "total": indexer.index.ntotal}

@app.get("/search", response_model=SearchResponse)
def search(q: str, top_k: int = 5):
    if indexer.index is None:
        raise HTTPException(status_code=400, detail="Index not loaded; call /build-index first")
    results = indexer.search(q, top_k=top_k)
    return SearchResponse(query=q, results=[{"path": p, "score": s} for p, s in results])

@app.get("/health")
def health():
    return {"status": "ok", "loaded": indexer.index is not None}
