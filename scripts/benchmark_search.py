import time
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from minivector.embedder import Embedder
from minivector.vector_store import VectorStore

print("Initializing...")
embedder = Embedder()
store = VectorStore()
store.load_index()

queries = [
    "artificial intelligence machine learning",
    "football soccer sports championship",
    "stock market economy recession",
    "climate change global warming",
    "smartphone mobile technology innovation",
]

print("\nRunning benchmark...")
print("=" * 60)

latencies = []
for _ in range(20):
    for query in queries:
        start = time.perf_counter()
        query_vec = embedder.embed_query(query)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-12)
        results = store.search(query_vec, k=10)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

print(f"Total queries: {len(latencies)}")
print(f"Mean latency: {np.mean(latencies):.2f}ms")
print(f"Median (P50): {np.median(latencies):.2f}ms")
print(f"P95 latency: {np.percentile(latencies, 95):.2f}ms")
print(f"P99 latency: {np.percentile(latencies, 99):.2f}ms")
print(f"Min: {np.min(latencies):.2f}ms")
print(f"Max: {np.max(latencies):.2f}ms")
print("=" * 60)