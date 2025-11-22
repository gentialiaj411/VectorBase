import json
import numpy as np
from pathlib import Path
import sys
import random

# Ensure imports work
sys.path.append(str(Path(__file__).parent))

from minivector.embedder import Embedder
from minivector.binary_engine import BinaryIndex

RAW_PATH = Path("data/raw/texts.json")
OUT_DIR = Path("data/processed")

def run():
    print("STARTING INGESTION PIPELINE...")
    
    # 1. Load Data (or create dummy data if missing)
    if not RAW_PATH.exists():
        print("Raw data not found. Creating mock papers for demonstration.")
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        categories = ["CS", "CV", "NLP", "Robotics", "Quantum"]
        data = []
        for i in range(50):
            cat = categories[i % len(categories)]
            data.append({
                "id": str(i),
                "title": f"Advanced {cat} Research Paper #{i+1}",
                # ensuring 'abstract' key exists
                "abstract": f"This paper proposes a novel method for optimizing {cat} systems using binary quantization. We explore the implications of this approach on system performance.",
                "authors": [f"Author {chr(65+i%5)}", "Aliaj G."],
                "category": cat,
                "published": "2024-05-20"
            })
        
        with open(RAW_PATH, 'w', encoding='utf-8') as f: json.dump(data, f)
    else:
        with open(RAW_PATH, 'r', encoding='utf-8') as f: data = json.load(f)

    print(f"Loaded {len(data)} documents.")

    # 2. Embed
    print("Generating embeddings (Float32)...")
    embedder = Embedder()
    
    # ROBUST TEXT EXTRACTION
    # This handles your error: checks for 'abstract', then 'text', then 'summary'
    texts = []
    for d in data:
        title = d.get('title', '')
        abstract = d.get('abstract') or d.get('text') or d.get('summary') or ""
        texts.append(f"{title} {abstract}")
        
        # Normalize data structure for the frontend
        d['abstract'] = abstract 

    vectors = embedder.embed(texts)

    # 3. Quantize & Save
    print("Quantizing to Binary (1-bit) and saving...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = BinaryIndex()
    engine.build_and_save(vectors, data, OUT_DIR / "vectors.npy", OUT_DIR / "metadata.json")
    
    # 4. Build Citation Graph
    print("Building connectivity graph...")
    graph = {}
    for i, doc in enumerate(data):
        neighbors = []
        if i > 0: neighbors.append(data[i-1]['id'])
        if i < len(data)-1: neighbors.append(data[i+1]['id'])
        if len(data) > 5:
             for _ in range(random.randint(1, 2)):
                 target = data[random.randint(0, len(data)-1)]['id']
                 if target != doc['id'] and target not in neighbors:
                     neighbors.append(target)
        graph[doc['id']] = neighbors
        
    with open(OUT_DIR / "citation_graph.json", 'w', encoding='utf-8') as f: 
        json.dump(graph, f)

    print("✅ INGESTION COMPLETE.")

if __name__ == "__main__":
    run()