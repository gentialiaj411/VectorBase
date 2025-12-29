import time
import concurrent.futures
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
test_queries = [
    "artificial intelligence",
    "football championship",
    "stock market",
    "climate change",
    "smartphone technology",
]
def run_query(query):
    query_vec = embedder.embed_query(query)
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    results = store.search(query_vec, k=10)
    return len(results)
print("\n" + "=" * 60)
print("SEQUENTIAL THROUGHPUT TEST")
print("=" * 60)
num_queries = 100
start = time.time()
for i in range(num_queries):
    query = test_queries[i % len(test_queries)]
    run_query(query)
elapsed = time.time() - start
sequential_qps = num_queries / elapsed
print(f"Total queries: {num_queries}")
print(f"Time elapsed: {elapsed:.2f}s")
print(f"Throughput: {sequential_qps:.2f} QPS")
print("\n" + "=" * 60)
print("CONCURRENT THROUGHPUT TEST (10 workers)")
print("=" * 60)
num_queries = 100
queries = [test_queries[i % len(test_queries)] for i in range(num_queries)]
start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(run_query, q) for q in queries]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]
elapsed = time.time() - start
concurrent_qps = num_queries / elapsed
print(f"Total queries: {num_queries}")
print(f"Time elapsed: {elapsed:.2f}s")
print(f"Throughput: {concurrent_qps:.2f} QPS")
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Sequential: {sequential_qps:.2f} QPS")
print(f"Concurrent (10 workers): {concurrent_qps:.2f} QPS")
print(f"Speedup: {concurrent_qps/sequential_qps:.2f}x")
print("=" * 60)
