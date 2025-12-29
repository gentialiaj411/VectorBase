import json
with open('data/raw/texts.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
with open('data/processed/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)
raw_dict = {doc['id']: doc for doc in raw_data}
for doc in metadata:
    if doc['id'] in raw_dict:
        doc['text'] = raw_dict[doc['id']]['text']
with open('data/processed/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
