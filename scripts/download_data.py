import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
def download_text_data(num_samples=100000, output_path="data/raw/texts.json"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("ag_news", split=f"train[:{num_samples}]")
    documents = []
    for i, item in enumerate(tqdm(dataset)):
        text = item['text'].strip()
        if len(text) < 50:
            continue
        full_text = text
        text_preview = text[:500]
        categories = ['World', 'Sports', 'Business', 'Technology']
        category = categories[item['label']]
        documents.append({
            'id': f'doc_{i}',
            'text': full_text,
            'text_preview': text_preview,
            'title': f"{category} Article {i}",
            'category': category,
            'url': f'https://minivector-demo.com/article/{i}'
        })
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
if __name__ == "__main__":
    download_text_data(num_samples=100000)
