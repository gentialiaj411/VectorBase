# sentence-transformers is not installed, avoid importing it to prevent conflicts
HAS_TRANSFORMERS = False

import numpy as np
from typing import List

class Embedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.dim = 384 # Default dimension, common for all-MiniLM-L6-v2
        if HAS_TRANSFORMERS:
            print(f"  -> Initializing Sentence Transformer: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
        else:
            print(f"  -> Initializing Dummy Embedder (Random Vectors)")

    def embed(self, texts: List[str]) -> np.ndarray:
        if HAS_TRANSFORMERS:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.astype(np.float32)
        else:
            # Return random vectors for testing
            return np.random.rand(len(texts), self.dim).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]