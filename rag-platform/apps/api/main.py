from fastapi import FastAPI
from contextlib import asynccontextmanager
from .config import settings
from .schemas import HealthResponse, IngestRequest, IngestResponse, QueryRequest, QueryResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f\"Loading settings: {settings}\")
    yield
    # Shutdown (if any)


app = FastAPI(lifespan=lifespan)


@app.get(\"/health\", response_model=HealthResponse)
async def health():
    return HealthResponse(status=\"ok\", service=\"rag-api\")


@app.post(\"/ingest\", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    # Stub: just return accepted
    return IngestResponse(status=\"accepted\", source=request.source)


@app.post(\"/query\", response_model=QueryResponse)
async def query(request: QueryRequest):
    # Stub: return hardcoded placeholder
    return QueryResponse(
        answer=\"This is a placeholder answer. In a real implementation, this would be generated based on the question and retrieved documents.\",
        sources=[
            {\"id\": \"doc1\", \"score\": 0.95, \"content\": \"Placeholder document content\"},
            {\"id\": \"doc2\", \"score\": 0.87, \"content\": \"Another placeholder document\"}
        ][:request.top_k]
    )

