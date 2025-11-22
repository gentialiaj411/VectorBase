import numpy as np
import json
import time
from pathlib import Path
from typing import List, Dict, Any

class BinaryIndex:
    def __init__(self, vector_dim: int = 384):
        self.vectors = None
        self.metadata = []
        self.vector_dim = vector_dim

    def build_and_save(self, float_vectors: np.ndarray, metadata: List[Dict[str, Any]], save_path: Path, metadata_path: Path):
        print("  -> Quantizing vectors to 1-bit precision...")

        norms = np.linalg.norm(float_vectors, axis=1, keepdims=True) + 1e-12
        normalized = float_vectors / norms

        bits = (normalized > 0).astype(np.uint8)

        packed = np.packbits(bits, axis=1)
        
        np.save(save_path, packed)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
        print(f"  -> Saved index to {save_path}")

    def load(self, vectors_path: str, metadata_path: str, keep_originals: bool = False):
        self.vectors = np.load(vectors_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

    def search(self, query_vec: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:

        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
        q_bits = (q_norm > 0).astype(np.uint8)
        q_packed = np.packbits(q_bits)
        

        xor_result = np.bitwise_xor(self.vectors, q_packed)

        distances = np.unpackbits(xor_result, axis=1).sum(axis=1)
        

        indices = np.argsort(distances)[:k]
        
        results = []
        for i in indices:
            doc = self.metadata[i].copy()

            doc['score'] = 1.0 - (distances[i] / self.vector_dim)

            abstract = doc.get('abstract') or doc.get('text') or ""
            doc['text_preview'] = abstract[:200] + "..." if abstract else "No preview available."
            results.append(doc)
        return results

    def hybrid_search(self, query_vec: np.ndarray, k: int = 10, candidates: int = 50):
        # For this version, we stick to the fast binary search to ensure stability
        return self.search(query_vec, k)