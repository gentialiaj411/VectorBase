import json
import arxiv 
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

def download_arxiv_papers(num_samples=10000, output_path = "data/raw/texts.json"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    categories = {
        'cs.AI': 'Artificial Intelligence',
        'cs.LG': 'Machine Learning',
        'cs.RO': 'Robotics',
        'cs.CV': 'Computer Vision',
        'cs.CL': 'Natural Language Processing',
        'eess.SP': 'Signal Processing',
        'eess.SY': 'Systems and Control',
        'stat.ML': 'Statistics - Machine Learning',
        'math.OC': 'Optimization and Control',
        'cs.DC': 'Distributed Computing'
    }
     
    print(f"Fetching {num_samples} ArXiv papers across {len(categories)} categories...")
    
    documents = []
    papers_per_category = num_samples // len(categories)
    
    for cat_code, cat_name in tqdm(categories.items(), desc="Categories"):
        # Query ArXiv for papers in this category
        search = arxiv.Search(
            query=f"cat:{cat_code}",
            max_results=papers_per_category,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        category_papers = []
        for result in search.results():
            if len(result.summary) < 100:
                continue
            
            paper = {
                'id': result.entry_id.split('/')[-1],
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'abstract': result.summary.replace('\n', ' ').strip(),
                'text': result.summary.replace('\n', ' ').strip(), 
                'text_preview': result.summary[:500].replace('\n', ' ').strip(),
                'category': cat_name,
                'primary_category': result.primary_category,
                'categories': result.categories,
                'published': result.published.strftime('%Y-%m-%d'),
                'updated': result.updated.strftime('%Y-%m-%d'),
                'pdf_url': result.pdf_url,
                'url': result.entry_id,
                'arxiv_id': result.entry_id.split('/')[-1],
                'doi': result.doi if result.doi else None,
                'journal_ref': result.journal_ref if result.journal_ref else None,
                'comment': result.comment if result.comment else None,
            }
            
            category_papers.append(paper)
            
            if len(category_papers) >= papers_per_category:
                break
        
        documents.extend(category_papers)
        print(f"  ✓ {cat_name}: {len(category_papers)} papers")
    
        if len(documents) >= num_samples:
            break
    
    for i, doc in enumerate(documents):
        doc['doc_id'] = f'doc_{i}'
    
    print(f"\nTotal papers downloaded: {len(documents)}")
    
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {output_path}")
    
    print("\nDataset Statistics:")
    print(f"  Total papers: {len(documents)}")
    print(f"  Categories: {len(categories)}")
    print(f"  Date range: {min(d['published'] for d in documents)} to {max(d['published'] for d in documents)}")
    avg_authors = sum(len(d['authors']) for d in documents) / len(documents)
    print(f"  Average authors per paper: {avg_authors:.1f}")

if __name__ == "__main__":
    download_arxiv_papers(num_samples=10000)