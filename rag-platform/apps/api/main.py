from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Response
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    SparseVector,
)
from sentence_transformers import SentenceTransformer
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict
import os
import time

from .config import settings
from .schemas import HealthResponse, IngestRequest, IngestResponse, QueryRequest, QueryResponse
from .logging_config import configure_logging, get_logger, LoggingMiddleware
from .auth import AuthMiddleware
from .rate_limit import RateLimitMiddleware
from .metrics import (
    record_http_request,
    record_rag_query,
    record_rerank,
    record_qdrant_search,
    update_qdrant_stats,
    record_llm_request,
    record_embedding,
    set_active_requests,
    get_registry,
)
from .tracing import configure_tracing, instrument_fastapi, get_tracer, SpanAttributes

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingestion.pipeline import run_incremental_pipeline

_pipeline_executor = ThreadPoolExecutor(max_workers=1)

_reranker_model = None
_tracer = None


def get_reranker():
    global _reranker_model
    if _reranker_model is None and settings.ENABLE_RERANKING:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder(settings.RERANKER_MODEL)
    return _reranker_model


def rerank_chunks(chunks: List[Dict], question: str, top_k: int) -> List[Dict]:
    reranker = get_reranker()
    if reranker is None or not chunks:
        return chunks[:top_k]
    
    # Batch reranking: process all pairs at once (O(1) model calls vs O(n))
    pairs = [(question, chunk.get("text", "")) for chunk in chunks]
    scores = reranker.predict(pairs)
    
    # Sort by score descending
    sorted_pairs = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in sorted_pairs[:top_k]]


def build_filter(
    severity_filter: Optional[List[str]] = None,
    source_filter: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Optional[Filter]:
    conditions = []
    
    if severity_filter:
        conditions.append(
            FieldCondition(
                key="severity",
                match=MatchValue(value=severity_filter[0]) if len(severity_filter) == 1 
                else {"match_any": {"values": severity_filter}}
            )
        )
    
    if source_filter:
        conditions.append(
            FieldCondition(
                key="source",
                match=MatchValue(value=source_filter[0]) if len(source_filter) == 1
                else {"match_any": {"values": source_filter}}
            )
        )
    
    if date_from or date_to:
        range_conditions = {}
        if date_from:
            range_conditions["gte"] = date_from
        if date_to:
            range_conditions["lte"] = date_to
        conditions.append(FieldCondition(key="published_date", range=Range(**range_conditions)))
    
    return Filter(must=conditions) if conditions else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure structured logging
    configure_logging(log_level="INFO", json_output=True)
    logger = get_logger("api.startup")
    
    # Configure tracing
    global _tracer
    _tracer = configure_tracing(
        service_name="rag-api",
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        enable_console=True,
    )
    
    app.state.model = SentenceTransformer(settings.EMBEDDING_MODEL)
    
    # Use file-based embedded Qdrant for persistence across processes
    qdrant_path = settings.QDRANT_PATH
    os.makedirs(qdrant_path, exist_ok=True)
    
    # Prefer file-based embedded mode for local development when QDRANT_PATH is set
    # Only try server if QDRANT_PATH is empty
    if settings.QDRANT_PATH and os.path.exists(settings.QDRANT_PATH):
        logger.info("using_file_based_qdrant", path=qdrant_path)
        app.state.qdrant = QdrantClient(path=qdrant_path)
    else:
        # Try connecting to Qdrant server
        try:
            app.state.qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
            app.state.qdrant.get_collections()
            logger.info("connected_to_qdrant_server")
        except Exception as e:
            logger.warning("qdrant_server_unavailable", error=str(e))
            app.state.qdrant = QdrantClient(path=qdrant_path)
            logger.info("using_file_based_qdrant_fallback", path=qdrant_path)
    
    app.state.llm = OpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
    )
    
    try:
        collection_info = app.state.qdrant.get_collection(settings.COLLECTION_NAME)
        app.state.collection_exists = True
        logger.info("collection_exists", collection=settings.COLLECTION_NAME, points=collection_info.points_count)
        update_qdrant_stats(settings.COLLECTION_NAME, collection_info.points_count, getattr(collection_info, "indexed_vectors_count", 0))
    except Exception as e:
        logger.warning("collection_not_found", collection=settings.COLLECTION_NAME, error=str(e))
        app.state.collection_exists = False
        update_qdrant_stats(settings.COLLECTION_NAME, 0, 0)
    
    logger.info("services_ready")
    yield
    _pipeline_executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)

# Add logging middleware (ASGI middleware via Starlette)
app.add_middleware(LoggingMiddleware)

# Auth & rate limiting
app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimitMiddleware)

# Instrument with OpenTelemetry
instrument_fastapi(app)

# Prometheus metrics endpoint
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest(get_registry()), media_type="text/plain")


@app.get("/health", response_model=HealthResponse)
async def health():
    logger = get_logger("api.health")
    start_time = time.perf_counter()
    
    qdrant_status = "unknown"
    indexed_vectors = 0
    llm_status = "unknown"
    
    try:
        collection_info = app.state.qdrant.get_collection(settings.COLLECTION_NAME)
        qdrant_status = collection_info.status
        indexed_vectors = getattr(collection_info, "indexed_vectors_count", 0)
    except Exception:
        qdrant_status = "not_found"
    
    # Check LLM connectivity
    try:
        app.state.llm.models.list()
        llm_status = "connected"
    except Exception:
        llm_status = "unavailable"
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info("health_check", qdrant_status=qdrant_status, llm_status=llm_status, duration_ms=duration_ms)
    
    overall_status = "ok" if qdrant_status != "not_found" and llm_status == "connected" else "degraded"
    
    return {
        "status": overall_status,
        "service": "rag-api",
        "model": settings.OPENROUTER_MODEL,
        "qdrant_status": qdrant_status,
        "llm_status": llm_status,
        "indexed_vectors": indexed_vectors,
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    if request.source == "incremental":
        days_back = request.limit if request.limit and request.limit > 0 else 1
        background_tasks.add_task(
            lambda: asyncio.get_event_loop().run_in_executor(
                _pipeline_executor, run_incremental_pipeline, days_back
            )
        )
        return {"status": "started", "source": "incremental"}
    return {"status": "accepted", "source": request.source}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    logger = get_logger("api.query")
    tracer = get_tracer(__name__)
    start_time = time.perf_counter()
    set_active_requests(1)
    
    question = request.question
    top_k = request.top_k
    search_type = request.search_type
    enable_reranking = request.enable_reranking
    
    with tracer.start_as_current_span("rag.query") as span:
        span.set_attribute(SpanAttributes.SEARCH_TYPE, search_type)
        span.set_attribute(SpanAttributes.RERANKED, enable_reranking)
        span.set_attribute(SpanAttributes.TOP_K, top_k)
        span.set_attribute(SpanAttributes.QUESTION_LENGTH, len(question))
        span.set_attribute(SpanAttributes.QDRANT_COLLECTION, settings.COLLECTION_NAME)
        
        try:
            # Generate embedding
            embed_start = time.perf_counter()
            question_vector = app.state.model.encode(
                [question], normalize_embeddings=True
            )[0].tolist()
            record_embedding(time.perf_counter() - embed_start, 1)
            
            search_k = settings.RERANK_TOP_K if enable_reranking else top_k
            
            qdrant_filter = build_filter(
                severity_filter=request.severity_filter,
                source_filter=request.source_filter,
                date_from=request.date_from,
                date_to=request.date_to,
            )
            
            if qdrant_filter:
                span.set_attribute(SpanAttributes.QDRANT_FILTER, str(qdrant_filter))
            
            # Vector search
            qdrant_start = time.perf_counter()
            search_result = app.state.qdrant.query_points(
                collection_name=settings.COLLECTION_NAME,
                query=question_vector,
                limit=search_k,
                with_payload=True,
                query_filter=qdrant_filter,
            )
            record_qdrant_search(time.perf_counter() - qdrant_start)
            
            hits = search_result.points
            
            context_chunks = []
            source_scores = []
            for hit in hits:
                payload = hit.payload or {}
                context_chunks.append({
                    "text": payload.get("text", ""),
                    "cve_id": payload.get("cve_id", "unknown"),
                    "severity": payload.get("severity", "UNKNOWN"),
                    "source": payload.get("source", "unknown"),
                    "published_date": payload.get("published_date", ""),
                })
                source_scores.append(hit.score)
            
            span.set_attribute(SpanAttributes.CONTEXT_CHUNKS, len(context_chunks))
            
            # Reranking
            reranked = False
            if enable_reranking and len(context_chunks) > top_k:
                rerank_start = time.perf_counter()
                context_chunks = rerank_chunks(context_chunks, question, top_k)
                reranked = True
                record_rerank(time.perf_counter() - rerank_start)
            
            sources = list(dict.fromkeys(c["cve_id"] for c in context_chunks if c["cve_id"] != "unknown"))
            # Match scores to unique sources (take first occurrence score)
            seen = set()
            matched_scores = []
            for c, score in zip(context_chunks, source_scores):
                cve_id = c["cve_id"]
                if cve_id not in seen and cve_id != "unknown":
                    seen.add(cve_id)
                    matched_scores.append(score)
            context_parts = [f"ID: {c['cve_id']} | Severity: {c['severity']}\n{c['text']}" for c in context_chunks]
            context = "\n\n".join(context_parts)
            
            system_message = (
                "You are a cybersecurity threat intelligence assistant. "
                "Answer using only the provided context. "
                "Always cite CVE IDs or MITRE technique IDs. "
                "Be concise and factual."
            )
            user_message = f"Context:\n{context}\n\nQuestion: {question}"
            
            # LLM call
            llm_start = time.perf_counter()
            with tracer.start_as_current_span("llm.call") as llm_span:
                llm_span.set_attribute(SpanAttributes.LLM_MODEL, settings.OPENROUTER_MODEL)
                
                response = app.state.llm.chat.completions.create(
                    model=settings.OPENROUTER_MODEL,
                    extra_headers={
                        "HTTP-Referer": "http://localhost:8000",
                        "X-Title": "ThreatIntelRAG",
                    },
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    max_tokens=settings.LLM_MAX_TOKENS,
                    temperature=settings.LLM_TEMPERATURE,
                    timeout=settings.LLM_TIMEOUT,
                )
                
                answer = response.choices[0].message.content
                
                # Record token usage if available
                if hasattr(response, 'usage') and response.usage:
                    prompt_tokens = response.usage.prompt_tokens or 0
                    completion_tokens = response.usage.completion_tokens or 0
                    llm_span.set_attribute(SpanAttributes.LLM_TOKENS_PROMPT, prompt_tokens)
                    llm_span.set_attribute(SpanAttributes.LLM_TOKENS_COMPLETION, completion_tokens)
                    record_llm_request(settings.OPENROUTER_MODEL, "success", time.perf_counter() - llm_start, prompt_tokens, completion_tokens)
                else:
                    record_llm_request(settings.OPENROUTER_MODEL, "success", time.perf_counter() - llm_start)
            
            total_duration = time.perf_counter() - start_time
            record_rag_query(search_type, reranked, total_duration, len(context_chunks))
            
            logger.info(
                "query_completed",
                question=question[:100],
                search_type=search_type,
                reranked=reranked,
                context_chunks=len(context_chunks),
                sources=len(sources),
                duration_ms=round(total_duration * 1000, 2),
            )
            
            return {
                "answer": answer,
                "sources": sources,
                "source_scores": matched_scores,
                "question": question,
                "context_used": len(context_chunks),
                "search_type": search_type,
                "reranked": reranked,
                "llm_error": None,
            }
        
        except Exception as e:
            total_duration = time.perf_counter() - start_time
            record_rag_query(search_type, False, total_duration, 0)
            record_llm_request(settings.OPENROUTER_MODEL, "error", 0)
            
            logger.error(
                "query_failed",
                question=question[:100],
                error=str(e),
                duration_ms=round(total_duration * 1000, 2),
            )
            
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "source_scores": [],
                "question": question,
                "context_used": 0,
                "search_type": search_type,
                "reranked": False,
                "llm_error": str(e),
            }
        finally:
            set_active_requests(0)