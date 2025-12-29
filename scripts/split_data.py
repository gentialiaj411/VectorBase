import numpy as np
import json
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
def split_data(num_shards: int, input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Loading data from {input_path}...")
    vectors = np.load(input_path / "vectors.npy")
    with open(input_path / "metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    total_items = len(vectors)
    print(f"Total items: {total_items}")
    shard_size = (total_items + num_shards - 1) // num_shards
    for i in range(num_shards):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, total_items)
        if start_idx >= total_items:
            break
        shard_vectors = vectors[start_idx:end_idx]
        shard_metadata = metadata[start_idx:end_idx]
        np.save(output_path / f"shard_{i}.npy", shard_vectors)
        with open(output_path / f"shard_{i}_meta.json", 'w', encoding='utf-8') as f:
            json.dump(shard_metadata, f)
        print(f"Created Shard {i}: {len(shard_vectors)} items ({start_idx} to {end_idx})")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split VectorBase data into shards.")
    parser.add_argument("--shards", type=int, default=3, help="Number of shards to create")
    parser.add_argument("--input", type=str, default="data/processed", help="Input directory")
    parser.add_argument("--output", type=str, default="data/sharded", help="Output directory")
    args = parser.parse_args()
    split_data(args.shards, args.input, args.output)
