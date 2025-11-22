import requests
import json

def test_graph():
    # First search to get a valid ID
    print("Searching for 'quantum'...")
    try:
        search_res = requests.post("http://localhost:8000/search", json={"query": "quantum", "k": 1})
        if search_res.status_code != 200:
            print(f"Search failed: {search_res.text}")
            return
        
        results = search_res.json().get("results", [])
        if not results:
            print("No results found.")
            return
            
        doc_id = results[0]["id"]
        print(f"Found doc_id: {doc_id}")
        
        # Now request graph
        print(f"Requesting graph for {doc_id}...")
        graph_res = requests.get(f"http://localhost:8000/graph/{doc_id}")
        
```python
import requests
import json

def test_graph():
    # First search to get a valid ID
    print("Searching for 'quantum'...")
    try:
        search_res = requests.post("http://localhost:8000/search", json={"query": "quantum", "k": 1})
        if search_res.status_code != 200:
            print(f"Search failed: {search_res.text}")
            return
        
        results = search_res.json().get("results", [])
        if not results:
            print("No results found.")
            return
            
        doc_id = results[0]["id"]
        print(f"Found doc_id: {doc_id}")
        
        # Now request graph
        print(f"Requesting graph for {doc_id}...")
        graph_res = requests.get(f"http://localhost:8000/graph/{doc_id}")
        
        if graph_res.status_code != 200:
            print(f"Graph request failed: {graph_res.text}")
            return
            
        data = graph_res.json()
        print(f"Graph response keys: {data.keys()}")
        # print(f"FULL RESPONSE: {json.dumps(data, indent=2)}")
        
        if len(data.get('nodes', [])) > 0:
            print("Sample node:", data['nodes'][0])
        else:
            print("WARNING: No nodes returned!")

        
        if 'debug' in data:
            print("\n--- SERVER DEBUG LOGS START ---")
            for log in data['debug']:
                print(log)
            print("--- SERVER DEBUG LOGS END ---")
        else:
            print("NO DEBUG LOGS IN RESPONSE")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_graph()
