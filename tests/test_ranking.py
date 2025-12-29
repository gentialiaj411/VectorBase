import requests
import json
url = "http://localhost:8000/search"
payload = {"query": "quantum computing", "k": 5}
response = requests.post(url, json=payload)
data = response.json()
output = ["Results in order received:"]
for i, result in enumerate(data['results'], 1):
    output.append(f"{i}. Score: {result['score']:.4f} - {result['title'][:80]}")
output_text = "\n".join(output)
print(output_text)
with open("ranking_test.txt", "w", encoding="utf-8") as f:
    f.write(output_text)
