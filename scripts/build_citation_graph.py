import json
from collections import defaultdict
from pathlib import Path
def build_graph():
    """
    Build citation/similarity graph
    Strategy: Connect papers by category + shared authors
    """
    print("="*60)
    print("BUILDING CITATION GRAPH")
    print("="*60)
    print("Loading metadata...")
    with open("data/processed/metadata.json", "r", encoding="utf-8") as f:
        papers = json.load(f)
    category_map = defaultdict(list)
    author_map = defaultdict(list)
    for paper in papers:
        paper_id = paper['id']
        category = paper.get('primary_category', 'unknown')
        category_map[category].append(paper_id)
        authors = paper.get('authors', [])
        if authors:
            first_author = authors[0]
            author_map[first_author].append(paper_id)
    print(f"  Categories: {len(category_map)}")
    print(f"  Unique authors: {len(author_map)}")
    print("Building graph...")
    graph = {}
    for paper in papers:
        paper_id = paper['id']
        neighbors = set()
        category = paper.get('primary_category', 'unknown')
        same_category = [p for p in category_map[category] if p != paper_id]
        neighbors.update(same_category[:5])  
        authors = paper.get('authors', [])
        if authors:
            first_author = authors[0]
            same_author = [p for p in author_map[first_author] if p != paper_id]
            neighbors.update(same_author[:3])  
        graph[paper_id] = list(neighbors)[:8]  
    output_path = Path("data/processed/citation_graph.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
    total_edges = sum(len(neighbors) for neighbors in graph.values())
    avg_degree = total_edges / len(graph) if graph else 0
    print(f"\n✓ Graph built successfully")
    print(f"  Nodes: {len(graph)}")
    print(f"  Edges: {total_edges}")
    print(f"  Avg degree: {avg_degree:.1f}")
    print(f"  Saved to: {output_path}")
    print("="*60 + "\n")
if __name__ == "__main__":
    build_graph()
