from pydantic import BaseModel
from typing import List, Optional
class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 10
class SearchResult(BaseModel):
    id: str
    title: str
    category: str
    text_preview: str
    score: float
    distance: Optional[float] = None
class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    took_ms: float
    total_results: int
