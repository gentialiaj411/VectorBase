import json
with open('data/processed/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)
print(f"Total documents: {len(metadata)}")
print(f"\nFirst document:")
print(json.dumps(metadata[0], indent=2))
print(f"\nLooking for doc_38852...")
found = False
for doc in metadata:
    if doc['id'] == 'doc_38852':
        print("FOUND!")
        print(json.dumps(doc, indent=2))
        found = True
        break
if not found:
    print("NOT FOUND")
    print("\nSample IDs:")
    for i in range(min(5, len(metadata))):
        print(f"  {metadata[i]['id']}")
