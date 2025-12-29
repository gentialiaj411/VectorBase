import numpy as np
import time
import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from minivector.binary_engine import BinaryIndex
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
def benchmark():
    print("=" * 60)
    print("VectorBase Benchmark Suite")
    print("=" * 60)
    vectors_path = DATA_DIR / "vectors.npy"
    meta_path = DATA_DIR / "metadata.json"
    if not vectors_path.exists():
        print(f"Data not found at {DATA_DIR}")
        return
    vectors = np.load(vectors_path)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    n_vectors, dim = vectors.shape
    print(f"\nDataset: {n_vectors} vectors, {dim} dimensions")
    index = BinaryIndex(vector_dim=dim)
    print("\nBuilding index...")
    start = time.time()
    index.build_and_save(vectors, metadata, DATA_DIR / "bench_vectors.npy", DATA_DIR / "bench_meta.json")
    build_time = time.time() - start
    print(f"Index Build Time: {build_time:.3f}s")
    index.load(str(DATA_DIR / "bench_vectors.npy"), str(DATA_DIR / "bench_meta.json"))
    float_size = n_vectors * dim * 4
    binary_size = n_vectors * (dim // 8)
    compression = float_size / binary_size
    print(f"\nMemory Compression: {compression:.0f}x (float32 -> 1-bit)")
    print(f"  Original: {float_size / 1024 / 1024:.2f} MB")
    print(f"  Compressed: {binary_size / 1024 / 1024:.2f} MB")
    print("\n--- Latency Benchmark ---")
    query = np.random.rand(dim).astype(np.float32)
    latencies = []
    for _ in range(100):
        start = time.time()
        index.search(query, k=10)
        latencies.append((time.time() - start) * 1000)
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    avg = np.mean(latencies)
    print(f"  Queries: 100")
    print(f"  Avg Latency: {avg:.2f} ms")
    print(f"  P50 Latency: {p50:.2f} ms")
    print(f"  P99 Latency: {p99:.2f} ms")
    print("\n--- Throughput Benchmark ---")
    queries = [np.random.rand(dim).astype(np.float32) for _ in range(1000)]
    start = time.time()
    for q in queries:
        index.search(q, k=10)
    total_time = time.time() - start
    qps = len(queries) / total_time
    print(f"  Queries: {len(queries)}")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Throughput: {qps:.0f} queries/sec")
    print("\n" + "=" * 60)
    print("RESUME METRICS (Copy these!)")
    print("=" * 60)
    print(f"- {compression:.0f}x memory compression (float32 to 1-bit quantization)")
    print(f"- {p50:.1f}ms P50 search latency")
    print(f"- {qps:.0f} queries/second throughput")
    print(f"- {n_vectors:,} vectors indexed")
    print("=" * 60)
if __name__ == "__main__":
    benchmark()
