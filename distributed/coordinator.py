import os
import sys
import asyncio
import aiohttp
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
sys.path.append(str(Path(__file__).parent.parent))
from minivector.embedder import Embedder
app = FastAPI()
WORKER_URLS = os.getenv("WORKER_URLS", "http://localhost:8001,http://localhost:8002,http://localhost:8003").split(",")
embedder = Embedder()
class QueryRequest(BaseModel):
    text: str
    k: int = 10
async def query_worker(session, url, vector, k):
    try:
        async with session.post(f"{url}/search", json={"query_vector": vector, "k": k}) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                print(f"Error from {url}: {resp.status}")
                return None
    except Exception as e:
        print(f"Failed to connect to {url}: {e}")
        return None
@app.post("/search")
async def distributed_search(req: QueryRequest):
    query_vec = embedder.embed([req.text])[0].tolist()
    async with aiohttp.ClientSession() as session:
        tasks = [query_worker(session, url, query_vec, req.k) for url in WORKER_URLS]
        results = await asyncio.gather(*tasks)
    all_hits = []
    for res in results:
        if res and "results" in res:
            for hit in res["results"]:
                hit["_shard"] = res["shard_id"]
                all_hits.append(hit)
    all_hits.sort(key=lambda x: x["score"], reverse=True)
    return {
        "total_hits": len(all_hits),
        "top_k": all_hits[:req.k]
    }
@app.get("/health")
async def health():
    return {"status": "coordinator_ready", "workers": len(WORKER_URLS)}
