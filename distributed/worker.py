import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from minivector.binary_engine import BinaryIndex

app = FastAPI()

SHARD_ID = int(os.getenv("SHARD_ID", "0"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data/sharded"))

index = BinaryIndex()

class SearchRequest(BaseModel):
    query_vector: List[float]
    k: int = 10

@app.on_event("startup")
async def load_shard():
    print(f"Worker {SHARD_ID}: Loading shard...")
    vectors_path = DATA_DIR / f"shard_{SHARD_ID}.npy"
    meta_path = DATA_DIR / f"shard_{SHARD_ID}_meta.json"
    
    if not vectors_path.exists():
        print(f"Worker {SHARD_ID}: Shard not found at {vectors_path}!")
        return

    index.load(str(vectors_path), str(meta_path))
    print(f"Worker {SHARD_ID}: Loaded {len(index.metadata)} vectors.")

@app.post("/search")
async def search_shard(req: SearchRequest):
    if index.vectors is None:
        raise HTTPException(status_code=503, detail="Shard not loaded")
    
    query_vec = np.array(req.query_vector, dtype=np.float32)
    results = index.search(query_vec, k=req.k)
    
    return {"shard_id": SHARD_ID, "results": results}

@app.get("/health")
async def health():
    return {"status": "ready", "shard_id": SHARD_ID, "vectors": len(index.metadata) if index.vectors is not None else 0}
