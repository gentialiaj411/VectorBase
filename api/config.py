from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
FAISS_INDEX_PATH = BASE_DIR / "data/indices/faiss.index"
METADATA_PATH = BASE_DIR / "data/processed/metadata.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
DEFAULT_SEARCH_K = 10
MAX_SEARCH_K = 100