import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from minivector.embedder import Embedder
from minivector.vector_store import VectorStore
embedder = Embedder()
store = VectorStore()
store.load_index(
    index_path="data/indices/faiss.index",
    metadata_path="data/processed/metadata.json"
)
queries = [
    "machine learning artificial intelligence",
    "sports football basketball",
    "stock market business economy"]
for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: '{query}'")
    print('='*60)
    query_vector = embedder.embed_query(query)
    results = store.search(query_vector, k=5)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"   Category: {result['category']}")
        print(f"   Score: {result['score']:.3f}")
        print(f"   Text: {result['text_preview'][:80]}...")
