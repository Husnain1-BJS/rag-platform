"""OpenTelemetry tracing configuration."""
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from typing import Optional


def configure_tracing(
    service_name: str = "rag-api",
    otlp_endpoint: Optional[str] = None,
    enable_console: bool = True,
) -> trace.Tracer:
    """Configure OpenTelemetry tracing."""
    
    # Create resource
    resource = Resource.create({SERVICE_NAME: service_name})
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Add console exporter for debugging
    if enable_console:
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(console_exporter))
    
    # Add OTLP exporter if endpoint provided
    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    
    # Set global tracer provider
    trace.set_tracer_provider(provider)
    
    # Get tracer
    tracer = trace.get_tracer(__name__)
    
    return tracer


def instrument_fastapi(app):
    """Instrument FastAPI app with OpenTelemetry."""
    FastAPIInstrumentor.instrument_app(app)


def instrument_httpx():
    """Instrument httpx client with OpenTelemetry."""
    HTTPXClientInstrumentor().instrument()


def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer instance."""
    return trace.get_tracer(name)


# Custom span attributes for RAG
class SpanAttributes:
    """Custom span attributes for RAG operations."""
    SEARCH_TYPE = "rag.search_type"
    RERANKED = "rag.reranked"
    CONTEXT_CHUNKS = "rag.context_chunks"
    TOP_K = "rag.top_k"
    QUESTION_LENGTH = "rag.question_length"
    QDRANT_COLLECTION = "qdrant.collection"
    QDRANT_FILTER = "qdrant.filter"
    LLM_MODEL = "llm.model"
    LLM_TOKENS_PROMPT = "llm.tokens.prompt"
    LLM_TOKENS_COMPLETION = "llm.tokens.completion"
    INGESTION_SOURCE = "ingestion.source"
    INGESTION_RECORDS = "ingestion.records"
    EMBEDDING_MODEL = "embedding.model"
    EMBEDDING_BATCH_SIZE = "embedding.batch_size"