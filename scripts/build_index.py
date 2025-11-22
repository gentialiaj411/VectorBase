import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from minivector.vector_store import VectorStore

def build_and_save_index():
    store = VectorStore()
    store.build_index(
        vectors_path="data/processed/vectors.npy",
        metadata_path="data/processed/metadata.json")
    store.save_index("data/indices/faiss.index")
    print("Success")

if __name__ == "__main__":
    build_and_save_index()