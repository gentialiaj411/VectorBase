import numpy as np
import time
vectors = np.load('data/processed/vectors.npy')
n_vectors, packed_dim = vectors.shape
dim = packed_dim * 8
print("="*50)
print("VECTORBASE REAL BENCHMARK")
print("="*50)
print(f"Dataset: {n_vectors:,} vectors, {dim} dimensions")
float_size = n_vectors * dim * 4
binary_size = n_vectors * packed_dim
compression = float_size / binary_size
print(f"Memory Compression: {compression:.0f}x")
print(f"Original Float32: {float_size / 1024:.1f} KB")
print(f"Binary Quantized: {binary_size / 1024:.1f} KB")
query = np.random.randint(0, 256, packed_dim, dtype=np.uint8)
latencies = []
for _ in range(200):
    start = time.perf_counter()
    xor_result = np.bitwise_xor(vectors, query)
    distances = np.unpackbits(xor_result, axis=1).sum(axis=1)
    indices = np.argsort(distances)[:10]
    end = time.perf_counter()
    latencies.append((end - start) * 1000)
print(f"\nLatency (200 queries):")
print(f"  Mean: {np.mean(latencies):.3f} ms")
print(f"  P50: {np.percentile(latencies, 50):.3f} ms")
print(f"  P99: {np.percentile(latencies, 99):.3f} ms")
queries = [np.random.randint(0, 256, packed_dim, dtype=np.uint8) for _ in range(1000)]
start = time.perf_counter()
for q in queries:
    xor_result = np.bitwise_xor(vectors, q)
    distances = np.unpackbits(xor_result, axis=1).sum(axis=1)
    np.argsort(distances)[:10]
total = time.perf_counter() - start
qps = 1000 / total
print(f"\nThroughput: {qps:.0f} queries/sec")
print("="*50)
