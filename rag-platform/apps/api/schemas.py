from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime


class HealthResponse(BaseModel):
    status: str
    service: str
    model: str
    qdrant_status: str = "unknown"
    llm_status: str = "unknown"
    indexed_vectors: int = 0


class IngestRequest(BaseModel):
    source: str
    limit: int = 100


class IngestResponse(BaseModel):
    status: str
    source: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=50)
    search_type: Literal["vector", "hybrid"] = "hybrid"
    severity_filter: Optional[List[str]] = None
    source_filter: Optional[List[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    enable_reranking: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []
    source_scores: List[float] = []
    question: str
    context_used: int
    search_type: str
    reranked: bool = False
    llm_error: Optional[str] = None

