"""
SIMD Benchmark - Measure C++ vs NumPy Performance
==================================================
This script benchmarks the SIMD-accelerated C++ backend against
the pure NumPy implementation to verify the 10-20x speedup claim.
Usage:
    python scripts/benchmark_simd.py
    python scripts/benchmark_simd.py --vectors 100000 --queries 1000
Output:
    - Detailed timing comparison (C++ vs NumPy)
    - Speedup calculation
    - P50/P99 latency metrics
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from minivector.binary_engine import BinaryIndex, get_backend_info, _CPP_AVAILABLE
def create_synthetic_data(num_vectors: int, vector_dim: int = 384) -> tuple:
    """Create synthetic binary vectors and metadata."""
    print(f"Creating synthetic dataset: {num_vectors:,} vectors, {vector_dim} dimensions")
    bytes_per_vec = (vector_dim + 7) // 8
    vectors = np.random.randint(0, 256, size=(num_vectors, bytes_per_vec), dtype=np.uint8)
    vectors = np.ascontiguousarray(vectors)
    metadata = [{"id": i, "title": f"Document {i}"} for i in range(num_vectors)]
    return vectors, metadata, vector_dim
def benchmark_numpy(vectors: np.ndarray, queries: np.ndarray, k: int) -> dict:
    """Benchmark pure NumPy implementation."""
    times = []
    for q in queries:
        start = time.perf_counter()
        xor_result = np.bitwise_xor(vectors, q)
        distances = np.unpackbits(xor_result, axis=1).sum(axis=1)
        indices = np.argsort(distances)[:k]
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    times = np.array(times)
    return {
        "backend": "NumPy",
        "avg_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p99_ms": float(np.percentile(times, 99)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }
def benchmark_cpp(vectors: np.ndarray, queries: np.ndarray, k: int) -> dict:
    """Benchmark C++ SIMD implementation."""
    if not _CPP_AVAILABLE:
        return None
    from minivector import minivector_core as core
    times = []
    for q in queries:
        start = time.perf_counter()
        indices, distances = core.batch_search(q, vectors, k)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    times = np.array(times)
    simd_name = "AVX2" 
    try:
        if hasattr(core, 'detect_simd_id'):
            s_id = core.detect_simd_id()
            simd_name = ["Scalar", "SSE2", "AVX2", "AVX-512", "AVX-512+VPOPCNT"][min(s_id, 4)]
        elif hasattr(core, 'detect_simd'):
            simd_name = str(core.detect_simd())
    except:
        pass
    return {
        "backend": f"C++ ({simd_name})",
        "avg_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p99_ms": float(np.percentile(times, 99)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }
def print_results(numpy_stats: dict, cpp_stats: dict, num_vectors: int, num_queries: int):
    """Print benchmark results."""
    print("\n" + "=" * 70)
    print("SIMD BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Database size: {num_vectors:,} vectors")
    print(f"Queries: {num_queries:,}")
    print("-" * 70)
    print(f"\n{'NumPy (baseline)':^35}")
    print("-" * 35)
    print(f"  Average latency:  {numpy_stats['avg_ms']:>8.3f} ms")
    print(f"  P50 latency:      {numpy_stats['p50_ms']:>8.3f} ms")
    print(f"  P99 latency:      {numpy_stats['p99_ms']:>8.3f} ms")
    print(f"  Min latency:      {numpy_stats['min_ms']:>8.3f} ms")
    print(f"  Max latency:      {numpy_stats['max_ms']:>8.3f} ms")
    print(f"  Throughput:       {1000/numpy_stats['avg_ms']:>8.1f} QPS")
    if cpp_stats:
        print(f"\n{cpp_stats['backend']:^35}")
        print("-" * 35)
        print(f"  Average latency:  {cpp_stats['avg_ms']:>8.3f} ms")
        print(f"  P50 latency:      {cpp_stats['p50_ms']:>8.3f} ms")
        print(f"  P99 latency:      {cpp_stats['p99_ms']:>8.3f} ms")
        print(f"  Min latency:      {cpp_stats['min_ms']:>8.3f} ms")
        print(f"  Max latency:      {cpp_stats['max_ms']:>8.3f} ms")
        print(f"  Throughput:       {1000/cpp_stats['avg_ms']:>8.1f} QPS")
        speedup = numpy_stats['avg_ms'] / cpp_stats['avg_ms']
        speedup_p99 = numpy_stats['p99_ms'] / cpp_stats['p99_ms']
        print("\n" + "=" * 70)
        print("SPEEDUP ANALYSIS")
        print("=" * 70)
        print(f"  Average speedup:  {speedup:>8.1f}x faster")
        print(f"  P99 speedup:      {speedup_p99:>8.1f}x faster")
        if speedup >= 20:
            print(f"\n  ✓ TARGET MET: {speedup:.1f}x speedup (target: 20x)")
        elif speedup >= 10:
            print(f"\n  ~ GOOD: {speedup:.1f}x speedup (target: 20x)")
        else:
            print(f"\n  ✗ BELOW TARGET: {speedup:.1f}x speedup (target: 20x)")
    else:
        print("\n[!] C++ backend not available - install with 'pip install -e .'")
    print("=" * 70)
def main():
    parser = argparse.ArgumentParser(description="Benchmark SIMD vs NumPy performance")
    parser.add_argument("--vectors", type=int, default=50000, 
                        help="Number of database vectors")
    parser.add_argument("--queries", type=int, default=100,
                        help="Number of queries to run")
    parser.add_argument("--dim", type=int, default=384,
                        help="Vector dimension")
    parser.add_argument("--k", type=int, default=10,
                        help="Top-k results")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup queries")
    args = parser.parse_args()
    print("\n" + "=" * 70)
    print("MiniVector SIMD Performance Benchmark")
    print("=" * 70)
    info = get_backend_info()
    print(f"\nBackend: {info['backend']}")
    print(f"SIMD: {info['simd_type']}")
    print(f"C++ available: {info['cpp_available']}")
    vectors, metadata, vector_dim = create_synthetic_data(args.vectors, args.dim)
    bytes_per_vec = (args.dim + 7) // 8
    queries = np.random.randint(0, 256, size=(args.queries + args.warmup, bytes_per_vec), dtype=np.uint8)
    queries = np.ascontiguousarray(queries)
    print(f"\nRunning {args.warmup} warmup queries...")
    for q in queries[:args.warmup]:
        xor_result = np.bitwise_xor(vectors, q)
        distances = np.unpackbits(xor_result, axis=1).sum(axis=1)
        _ = np.argsort(distances)[:args.k]
    if _CPP_AVAILABLE:
        from minivector import minivector_core as core
        for q in queries[:args.warmup]:
            _, _ = core.batch_search(q, vectors, args.k)
    test_queries = queries[args.warmup:]
    print(f"Running {args.queries} benchmark queries...")
    numpy_stats = benchmark_numpy(vectors, test_queries, args.k)
    cpp_stats = benchmark_cpp(vectors, test_queries, args.k)
    print_results(numpy_stats, cpp_stats, args.vectors, args.queries)
    return {
        "numpy": numpy_stats,
        "cpp": cpp_stats,
        "speedup": numpy_stats['avg_ms'] / cpp_stats['avg_ms'] if cpp_stats else 1.0
    }
if __name__ == "__main__":
    main()
