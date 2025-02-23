from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class UserMetadata(BaseModel):
    author: Optional[str] = None
    department: Optional[str] = None
    project: Optional[str] = None
    doc_type: Optional[str] = None
    created_at: Optional[str] = None
    custom_tags: Optional[List[str]] = None

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    threshold: float = 0.7

class SearchResponse(BaseModel):
    id: int
    text: str
    score: float
    metadata: Dict[str, Any]
    summary: Optional[str] = None

class SearchResults(BaseModel):
    total: int
    results: List[SearchResponse]

class InsertRequest(BaseModel):
    text: str
    metadata: UserMetadata = Field(default_factory=UserMetadata)

class InsertResponse(BaseModel):
    id: int
    metadata: Dict[str, Any]
    summary: str