import requests
import json

# Test Ollama directly
paper_id = '2511.16674v1'
with open('data/processed/metadata.json', encoding='utf-8') as f:
    metadata = json.load(f)

paper = next((p for p in metadata if p['id'] == paper_id), None)
print(f"Paper found: {paper['title'][:50]}...")

abstract = paper['abstract'][:1000]
prompt = f"""You are an AI research assistant analyzing the paper titled "{paper['title']}".

Paper Abstract:
{abstract}

User Question: What is the main contribution of this paper?

Provide a concise, helpful answer based on the paper's content."""

print("\nSending to Ollama...")
r = requests.post(
    'http://localhost:11434/api/generate',
    json={
        'model': 'llama3.2',
        'prompt': prompt,
        'stream': False,
        'options': {
            'temperature': 0.7,
            'num_predict': 200
        }
    },
    timeout=30
)

print(f"Status: {r.status_code}")
if r.status_code == 200:
    print(f"\nOllama Response:\n{r.json()['response']}")
else:
    print(f"Error: {r.text}")
