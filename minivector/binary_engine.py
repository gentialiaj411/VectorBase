"""
MiniVector Binary Engine - Hybrid Python/C++ Vector Search
===========================================================
This module implements binary (1-bit) quantized vector search with optional
SIMD acceleration via the C++ core. Falls back to NumPy when C++ is unavailable.
Architecture:
    - Python: High-level orchestration, I/O, metadata management
    - C++ (optional): SIMD-accelerated Hamming distance and top-k search
Performance:
    - C++ backend: 10-20x faster than NumPy for large databases
    - Memory: 32x compression vs float32 vectors (1-bit per dimension)
"""
import numpy as np
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import os
_CPP_AVAILABLE = False
_cpp_core = None
_simd_type = "NumPy (fallback)"
if os.name == 'nt':
    mingw_bin = r"C:\msys64\mingw64\bin"
    if os.path.exists(mingw_bin):
        try:
            os.add_dll_directory(mingw_bin)
        except AttributeError:
            os.environ['PATH'] = mingw_bin + os.pathsep + os.environ['PATH']
try:
    try:
        from . import minivector_core as _cpp_core
    except ImportError:
        import minivector_core as _cpp_core
    _CPP_AVAILABLE = True
    if hasattr(_cpp_core, 'detect_simd_id'):
        simd_id = _cpp_core.detect_simd_id()
        _simd_type = ["Scalar", "SSE2", "AVX2", "AVX-512", "AVX-512+VPOPCNT"][min(simd_id, 4)]
    else:
        _simd_type = "C++ (Generic)"
except ImportError:
    pass
def get_backend_info() -> Dict[str, Any]:
    """Get information about the current backend."""
    return {
        "cpp_available": _CPP_AVAILABLE,
        "simd_type": _simd_type,
        "backend": "C++ SIMD" if _CPP_AVAILABLE else "NumPy",
    }
class BinaryIndex:
    """
    Binary quantized vector index with SIMD-accelerated search.
    This class provides:
    - 1-bit quantization (32x memory reduction vs float32)
    - SIMD-accelerated Hamming distance search (AVX2/AVX-512)
    - Automatic fallback to NumPy when C++ is unavailable
    Example:
        >>> index = BinaryIndex(vector_dim=384)
        >>> index.load("vectors.npy", "metadata.json")
        >>> results = index.search(query_vector, k=10)
    """
    def __init__(self, vector_dim: int = 384, use_cpp: bool = True):
        """
        Initialize binary index.
        Args:
            vector_dim: Dimension of original float vectors (default: 384 for MiniLM)
            use_cpp: Whether to use C++ backend when available (default: True)
        """
        self.vectors: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []
        self.vector_dim = vector_dim
        self.use_cpp = use_cpp and _CPP_AVAILABLE
        self._search_count = 0
        self._total_search_time_ms = 0.0
    @property
    def backend(self) -> str:
        """Get current backend name."""
        if self.use_cpp:
            return f"C++ ({_simd_type})"
        return "NumPy"
    @property
    def num_vectors(self) -> int:
        """Get number of indexed vectors."""
        return len(self.metadata) if self.metadata else 0
    @property
    def bytes_per_vector(self) -> int:
        """Get bytes per packed vector."""
        return (self.vector_dim + 7) // 8  
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        avg_time = (self._total_search_time_ms / self._search_count 
                    if self._search_count > 0 else 0.0)
        return {
            "num_vectors": self.num_vectors,
            "vector_dim": self.vector_dim,
            "bytes_per_vector": self.bytes_per_vector,
            "backend": self.backend,
            "search_count": self._search_count,
            "avg_search_time_ms": avg_time,
        }
    def build_and_save(
        self,
        float_vectors: np.ndarray,
        metadata: List[Dict[str, Any]],
        save_path: Path,
        metadata_path: Path
    ) -> None:
        """
        Build binary index from float vectors and save to disk.
        Args:
            float_vectors: Float32 vectors of shape (N, dim)
            metadata: List of metadata dicts (one per vector)
            save_path: Path to save packed binary vectors
            metadata_path: Path to save metadata JSON
        """
        print("  -> Quantizing vectors to 1-bit precision...")
        norms = np.linalg.norm(float_vectors, axis=1, keepdims=True) + 1e-12
        normalized = float_vectors / norms
        bits = (normalized > 0).astype(np.uint8)
        packed = np.packbits(bits, axis=1)
        packed = np.ascontiguousarray(packed)
        np.save(save_path, packed)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
        print(f"  -> Saved index to {save_path}")
        print(f"  -> Compression: {float_vectors.nbytes / packed.nbytes:.1f}x")
    def load(
        self,
        vectors_path: str,
        metadata_path: str,
        keep_originals: bool = False
    ) -> None:
        """
        Load binary index from disk.
        Args:
            vectors_path: Path to packed binary vectors (.npy)
            metadata_path: Path to metadata JSON
            keep_originals: Ignored (for API compatibility)
        """
        self.vectors = np.load(vectors_path)
        if not self.vectors.flags['C_CONTIGUOUS']:
            self.vectors = np.ascontiguousarray(self.vectors)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        self.vector_dim = self.vectors.shape[1] * 8
    def _pack_query(self, query_vec: np.ndarray) -> np.ndarray:
        """Pack a float query vector into binary format."""
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
        q_bits = (q_norm > 0).astype(np.uint8)
        q_packed = np.packbits(q_bits)
        return q_packed
    def search(
        self,
        query_vec: np.ndarray,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for k nearest neighbors using Hamming distance.
        Uses SIMD-accelerated C++ backend when available, falls back to NumPy.
        Args:
            query_vec: Float query vector of shape (dim,)
            k: Number of results to return
        Returns:
            List of result dicts with 'score', 'text_preview', and metadata
        """
        start_time = time.perf_counter()
        q_packed = self._pack_query(query_vec)
        k = min(k, len(self.metadata))
        indices = []
        distances = []
        if self.use_cpp and _cpp_core is not None:
            try:
                indices, distances = _cpp_core.batch_search(q_packed, self.vectors, k)
                indices = indices.tolist()
                distances = distances.tolist()
            except Exception:
                indices, distances = self._numpy_search(q_packed, k)
        else:
            indices, distances = self._numpy_search(q_packed, k)
        results = []
        for i, idx in enumerate(indices):
            doc = self.metadata[idx].copy()
            doc['score'] = 1.0 - (distances[i] / self.vector_dim)
            abstract = doc.get('abstract') or doc.get('text') or ""
            doc['text_preview'] = abstract[:200] + "..." if abstract else "No preview available."
            results.append(doc)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._search_count += 1
        self._total_search_time_ms += elapsed_ms
        return results
    def _numpy_search(
        self,
        q_packed: np.ndarray,
        k: int
    ) -> Tuple[List[int], List[int]]:
        """NumPy-based Hamming distance search (fallback)."""
        xor_result = np.bitwise_xor(self.vectors, q_packed)
        distances = np.unpackbits(xor_result, axis=1).sum(axis=1)
        indices = np.argsort(distances)[:k]
        return indices.tolist(), distances[indices].tolist()
    def search_batch(
        self,
        query_vecs: np.ndarray,
        k: int = 10
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for multiple queries (batch mode).
        Args:
            query_vecs: Float query vectors of shape (Q, dim)
            k: Number of results per query
        Returns:
            List of result lists (one per query)
        """
        results = []
        for query_vec in query_vecs:
            results.append(self.search(query_vec, k))
        return results
    def hybrid_search(
        self,
        query_vec: np.ndarray,
        k: int = 10,
        candidates: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search (currently same as binary search).
        Reserved for future re-ranking with float vectors.
        Args:
            query_vec: Float query vector
            k: Number of final results
            candidates: Number of candidates for re-ranking (unused)
        Returns:
            List of result dicts
        """
        return self.search(query_vec, k)
    def benchmark(
        self,
        num_queries: int = 100,
        k: int = 10,
        warmup: int = 10
    ) -> Dict[str, float]:
        """
        Run benchmark and return timing statistics.
        Args:
            num_queries: Number of queries to run
            k: Number of results per query
            warmup: Number of warmup queries (not counted)
        Returns:
            Dict with timing statistics (avg_ms, p50_ms, p99_ms, qps)
        """
        if self.vectors is None:
            raise ValueError("Index not loaded. Call load() first.")
        np.random.seed(42)
        queries = np.random.randn(num_queries + warmup, self.vector_dim).astype(np.float32)
        for i in range(warmup):
            self.search(queries[i], k)
        times = []
        for i in range(warmup, num_queries + warmup):
            start = time.perf_counter()
            self.search(queries[i], k)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        times = np.array(times)
        return {
            "backend": self.backend,
            "num_vectors": self.num_vectors,
            "num_queries": num_queries,
            "k": k,
            "avg_ms": float(np.mean(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p99_ms": float(np.percentile(times, 99)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "qps": 1000.0 / np.mean(times),
        }
