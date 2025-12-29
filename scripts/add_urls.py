import json
with open('data/processed/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)
for doc in metadata:
    doc['url'] = f'http://localhost:8000/article/{doc["id"]}'
with open('data/processed/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
