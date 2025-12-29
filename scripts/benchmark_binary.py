import numpy as np
import time
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from minivector.binary_engine import BinaryIndex
from minivector.embedder import Embedder
def benchmark():
    print("\n" + "="*60)
    print("BINARY QUANTIZATION BENCHMARK")
    print("="*60 + "\n")
    print("Loading binary index...")
    binary_index = BinaryIndex()
    binary_index.load(
        "data/processed/vectors.npy",
        "data/processed/metadata.json",
        keep_originals=True
    )
    embedder = Embedder()
    test_queries = [
        "attention mechanisms in transformers",
        "convolutional neural networks for image classification",
        "reinforcement learning algorithms",
        "object detection with YOLO",
        "natural language processing with BERT"
    ]
    print("="*60)
    print("SPEED COMPARISON")
    print("="*60 + "\n")
    binary_times = []
    hybrid_times = []
    for query in test_queries:
        query_vec = embedder.embed_query(query)
        start = time.time()
        binary_results = binary_index.search(query_vec, k=10)
        binary_time = (time.time() - start) * 1000
        binary_times.append(binary_time)
        start = time.time()
        hybrid_results = binary_index.hybrid_search(query_vec, k=10, candidates=100)
        hybrid_time = (time.time() - start) * 1000
        hybrid_times.append(hybrid_time)
    print(f"Binary Search:")
    print(f"  Average: {np.mean(binary_times):.2f}ms")
    print(f"  Min: {np.min(binary_times):.2f}ms")
    print(f"  Max: {np.max(binary_times):.2f}ms")
    print(f"\nHybrid Search:")
    print(f"  Average: {np.mean(hybrid_times):.2f}ms")
    print(f"  Min: {np.min(hybrid_times):.2f}ms")
    print(f"  Max: {np.max(hybrid_times):.2f}ms")
    print(f"\nSpeedup: {np.mean(hybrid_times)/np.mean(binary_times):.2f}x slower (but more accurate)")
    print("\n" + "="*60)
    print("ACCURACY TEST")
    print("="*60 + "\n")
    query = test_queries[0]
    print(f"Query: '{query}'\n")
    query_vec = embedder.embed_query(query)
    binary_results = binary_index.search(query_vec, k=5)
    hybrid_results = binary_index.hybrid_search(query_vec, k=5, candidates=100)
    print("Top 5 Binary Results:")
    for i, r in enumerate(binary_results, 1):
        print(f"  {i}. {r['title'][:70]}...")
        print(f"     Score: {r['score']:.3f}\n")
    print("Top 5 Hybrid Results:")
    for i, r in enumerate(hybrid_results, 1):
        print(f"  {i}. {r['title'][:70]}...")
        print(f"     Score: {r['score']:.3f}\n")
    binary_ids = set([r['id'] for r in binary_results])
    hybrid_ids = set([r['id'] for r in hybrid_results])
    overlap = len(binary_ids & hybrid_ids)
    print(f"Result overlap: {overlap}/5 papers")
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Binary search: {np.mean(binary_times):.1f}ms (ultra-fast)")
    print(f"✓ Hybrid search: {np.mean(hybrid_times):.1f}ms (balanced)")
    print(f"✓ Memory savings: 32x compression")
    print(f"✓ Recommendation: Use hybrid for production")
    print("="*60 + "\n")
if __name__ == "__main__":
    benchmark()
