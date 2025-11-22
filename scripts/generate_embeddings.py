import json
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from minivector.embedder import Embedder
from tqdm import tqdm

def generate_embeddings(
    input_path="data/raw/texts.json",
    output_vectors="data/processed/vectors.npy",
    output_metadata="data/processed/metadata.json"):    
    Path(output_vectors).parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    embedder = Embedder()
    texts = [doc['text'] for doc in documents]
    doc_ids = [doc['id'] for doc in documents]
    
    embeddings = embedder.embed(texts, batch_size=256)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms+1e-12)
    np.save(output_vectors, embeddings)
    file_size_mb = Path(output_vectors).stat().st_size / 1024 / 1024
    5
    metadata = [{
            'id': doc['id'],
            'title': doc['title'],
            'category': doc.get('category', 'Unknown'),
            'text_preview': doc['text'][:1000]}
        for doc in documents
    ]
    with open(output_metadata, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    generate_embeddings()
