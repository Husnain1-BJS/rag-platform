"""Prometheus metrics for the RAG API."""
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from typing import Optional


# Create a custom registry
registry = CollectorRegistry()

# Request metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
    registry=registry,
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=registry,
)

# Query-specific metrics
rag_queries_total = Counter(
    "rag_queries_total",
    "Total RAG queries",
    ["search_type", "reranked"],
    registry=registry,
)

rag_query_duration_seconds = Histogram(
    "rag_query_duration_seconds",
    "RAG query duration in seconds",
    ["search_type"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=registry,
)

rag_context_chunks = Histogram(
    "rag_context_chunks",
    "Number of context chunks retrieved",
    buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
    registry=registry,
)

rag_rerank_duration_seconds = Histogram(
    "rag_rerank_duration_seconds",
    "Reranking duration in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=registry,
)

# Qdrant metrics
qdrant_search_duration_seconds = Histogram(
    "qdrant_search_duration_seconds",
    "Qdrant search duration in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=registry,
)

qdrant_points_total = Gauge(
    "qdrant_points_total",
    "Total points in Qdrant collection",
    ["collection"],
    registry=registry,
)

qdrant_indexed_vectors = Gauge(
    "qdrant_indexed_vectors",
    "Indexed vectors in Qdrant collection",
    ["collection"],
    registry=registry,
)

# LLM metrics
llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["model", "status"],
    registry=registry,
)

llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "LLM request duration in seconds",
    ["model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
    registry=registry,
)

llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total LLM tokens used",
    ["model", "type"],  # type: prompt, completion
    registry=registry,
)

# Ingestion metrics
ingestion_runs_total = Counter(
    "ingestion_runs_total",
    "Total ingestion pipeline runs",
    ["source", "status"],
    registry=registry,
)

ingestion_duration_seconds = Histogram(
    "ingestion_duration_seconds",
    "Ingestion pipeline duration in seconds",
    ["source"],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0],
    registry=registry,
)

ingestion_records_processed = Histogram(
    "ingestion_records_processed",
    "Number of records processed in ingestion",
    ["source"],
    buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000],
    registry=registry,
)

# Embedding metrics
embedding_duration_seconds = Histogram(
    "embedding_duration_seconds",
    "Embedding generation duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=registry,
)

embedding_batch_size = Histogram(
    "embedding_batch_size",
    "Embedding batch size",
    buckets=[1, 2, 4, 8, 16, 32, 64, 128],
    registry=registry,
)

# System metrics
active_requests = Gauge(
    "active_requests",
    "Number of active requests",
    registry=registry,
)

collection_exists = Gauge(
    "collection_exists",
    "Whether the collection exists (1) or not (0)",
    ["collection"],
    registry=registry,
)


def record_http_request(method: str, endpoint: str, status: int, duration: float):
    """Record HTTP request metrics."""
    http_requests_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)


def record_rag_query(search_type: str, reranked: bool, duration: float, context_chunks: int):
    """Record RAG query metrics."""
    rag_queries_total.labels(search_type=search_type, reranked=str(reranked).lower()).inc()
    rag_query_duration_seconds.labels(search_type=search_type).observe(duration)
    rag_context_chunks.observe(context_chunks)


def record_rerank(duration: float):
    """Record reranking duration."""
    rag_rerank_duration_seconds.observe(duration)


def record_qdrant_search(duration: float):
    """Record Qdrant search duration."""
    qdrant_search_duration_seconds.observe(duration)


def update_qdrant_stats(collection: str, points: int, indexed: int):
    """Update Qdrant collection stats."""
    qdrant_points_total.labels(collection=collection).set(points)
    qdrant_indexed_vectors.labels(collection=collection).set(indexed)
    collection_exists.labels(collection=collection).set(1 if points > 0 else 0)


def record_llm_request(model: str, status: str, duration: float, prompt_tokens: int = 0, completion_tokens: int = 0):
    """Record LLM request metrics."""
    llm_requests_total.labels(model=model, status=status).inc()
    llm_request_duration_seconds.labels(model=model).observe(duration)
    if prompt_tokens:
        llm_tokens_total.labels(model=model, type="prompt").inc(prompt_tokens)
    if completion_tokens:
        llm_tokens_total.labels(model=model, type="completion").inc(completion_tokens)


def record_ingestion(source: str, status: str, duration: float, records: int):
    """Record ingestion pipeline metrics."""
    ingestion_runs_total.labels(source=source, status=status).inc()
    ingestion_duration_seconds.labels(source=source).observe(duration)
    ingestion_records_processed.labels(source=source).observe(records)


def record_embedding(duration: float, batch_size: int):
    """Record embedding generation metrics."""
    embedding_duration_seconds.observe(duration)
    embedding_batch_size.observe(batch_size)


def set_active_requests(count: int):
    """Set active requests gauge."""
    active_requests.set(count)


def get_registry() -> CollectorRegistry:
    """Get the Prometheus registry."""
    return registry