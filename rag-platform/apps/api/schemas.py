from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class HealthResponse(BaseModel):
    status: str
    service: str


class IngestRequest(BaseModel):
    source: str
    limit: int = 100


class IngestResponse(BaseModel):
    status: str
    source: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []

