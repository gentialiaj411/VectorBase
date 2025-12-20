import requests
import json
import time

COORDINATOR_URL = "http://localhost:8000"

def test_search():
    print(f"Testing Distributed Search at {COORDINATOR_URL}...")
    
    query = "quantum computing optimization"
    
    try:
        start_time = time.time()
        resp = requests.post(f"{COORDINATOR_URL}/search", json={"text": query, "k": 5})
        end_time = time.time()
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"\n[SUCCESS] ({end_time - start_time:.3f}s)")
            print(f"Total Hits: {data['total_hits']}")
            print("\nTop Results:")
            for i, hit in enumerate(data['top_k']):
                print(f"{i+1}. [Shard {hit.get('_shard', '?')}] {hit['title']} (Score: {hit['score']:.3f})")
        else:
            print(f"[ERROR] {resp.status_code} - {resp.text}")
            
    except Exception as e:
        print(f"[FAILED] Connection Failed: {e}")
        print("Make sure the cluster is running with: python scripts/run_local_cluster.py")

if __name__ == "__main__":
    test_search()
