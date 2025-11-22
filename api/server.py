import time
import numpy as np
import sys
import json
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests


sys.path.append(str(Path(__file__).parent.parent))
from minivector.embedder import Embedder
from minivector.binary_engine import BinaryIndex 

state = {"embedder": None, "engine": None, "metadata": [], "cache": None}

class SemanticCache:
    def __init__(self, threshold=0.9):
            return self.cache[best_idx][1]
        return None

    def store(self, query_vec, response):
        self.cache.append((query_vec, response))

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 INITIALIZING SERVER...")
    state["embedder"] = Embedder()
    state["engine"] = BinaryIndex()
    state["cache"] = SemanticCache()
    
    try:
        if not Path("data/processed/vectors.npy").exists():
            print("⚠️ Data missing. Run process_data.py")
        else:
            state["engine"].load("data/processed/vectors.npy", "data/processed/metadata.json")
            state["metadata"] = state["engine"].metadata
            print(f"✅ SYSTEM READY. Loaded {len(state['metadata'])} docs.")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class SearchRequest(BaseModel):
    query: str
    k: int = 10

class ChatRequest(BaseModel):
    paper_id: str
    message: str

def optimize_context(abstract, query):
    return abstract[:1000]

@app.post("/search")
async def search(req: SearchRequest):
    if state["engine"].vectors is None: raise HTTPException(500, "Index not loaded")
    t0 = time.time()
    q_vec = state["embedder"].embed_query(req.query)
    results = state["engine"].search(q_vec, k=req.k)
    t_took = (time.time() - t0) * 1000
    return {"results": results, "took_ms": t_took, "method": "Binary Quantization"}

@app.post("/chat")
async def chat(req: ChatRequest):
    paper = next((p for p in state["metadata"] if p['id'] == req.paper_id), None)
    if not paper: raise HTTPException(404, "Not found")
    
    query_vec = state["embedder"].embed_query(req.message)

    cached_response = state["cache"].lookup(query_vec)
    if cached_response:
        def cached_stream():
            yield cached_response + " (Cached ⚡)"
        return StreamingResponse(cached_stream(), media_type="text/plain")

    abstract = paper.get('abstract') or paper.get('text') or ""
    context = optimize_context(abstract, req.message)
    
    print(f"🤖 Chat request for: {paper['title'][:50]}...")
    
    prompt = f"""You are an AI research assistant analyzing the paper titled "{paper['title']}".

Paper Abstract:
{context}

User Question: {req.message}

Provide a concise, helpful answer based on the paper's content."""

    def stream_generator():
        full_response = ""
        try:
            print("🔄 Calling Ollama (Streaming)...")
            with requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:latest",
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 200
                    }
                },
                stream=True,
                timeout=120
            ) as r:
                for line in r.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            token = chunk.get("response", "")
                            if token:
                                full_response += token
                                yield token
                        except:
                            pass
            
            if full_response:
                state["cache"].store(query_vec, full_response)
                print(f"✅ Stream complete. Cached {len(full_response)} chars.")

        except Exception as e:
            print(f"❌ Stream error: {e}")
            yield f"Error: {str(e)}"
    
    return StreamingResponse(stream_generator(), media_type="text/plain")

@app.get("/article/{doc_id}")
async def get_article(doc_id: str):
    doc = next((d for d in state["metadata"] if d['id'] == doc_id), None)
    if doc: return doc
    raise HTTPException(404, "Not found")

@app.get("/graph/{doc_id}")
async def get_graph(doc_id: str):
    path = Path("data/processed/citation_graph.json")
    if not path.exists(): return {"nodes": [], "edges": []}
    with open(path) as f: full = json.load(f)
    
    queue = [(doc_id, 0)]; visited = set(); nodes = []; edges = []
    added_ids = set()

    while queue:
        curr, depth = queue.pop(0)
        if curr in visited or depth > 1: continue
        visited.add(curr)
        

        if curr not in added_ids:
            doc = next((d for d in state["metadata"] if d['id'] == curr), None)
            if doc:
                nodes.append({"id": curr, "label": doc['title'], "isCenter": curr == doc_id})
                added_ids.add(curr)
            
        for n_id in full.get(curr, []):
             n_doc = next((d for d in state["metadata"] if d['id'] == n_id), None)
             if n_doc:
                 edges.append({"source": curr, "target": n_id})
                 if n_id not in added_ids:
                    nodes.append({"id": n_id, "label": n_doc['title'], "isCenter": False})
                    added_ids.add(n_id)
                 if n_id not in visited: queue.append((n_id, depth+1))
            
    return {"nodes": nodes, "edges": edges}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)