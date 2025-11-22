import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple

class VectorStore:   
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.metadata = []
        
    def build_index(self, vectors_path: str = "data/processed/vectors.npy",
                   metadata_path: str = "data/processed/metadata.json"):
        vectors = np.load(vectors_path).astype('float32')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(vectors)       
        
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Dict]:
     
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype('float32')
        distances, indices = self.index.search(query_vector, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(1 / (1 + dist))
                result['distance'] = float(dist)
                results.append(result)
        return results
    
    def save_index(self, index_path: str = "data/indices/faiss.index"):
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
                
    def load_index(self, index_path: str = "data/indices/faiss.index",
                   metadata_path: str = "data/processed/metadata.json"):       
        self.index = faiss.read_index(index_path)        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
       
if __name__ == "__main__":
    store = VectorStore()
    store.build_index() 
    random_query = np.random.rand(384).astype('float32')
    results = store.search(random_query, k=5)
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"Category: {result['category']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text_preview']}")
     