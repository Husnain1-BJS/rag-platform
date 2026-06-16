"""Unit tests for logging and metrics."""
import pytest
from apps.api.logging_config import configure_logging, LoggingMiddleware
from apps.api.metrics import (
    rag_queries_total,
    http_requests_total,
)


class TestLogging:
    def test_configure_logging_runs(self):
        configure_logging()


class TestMetrics:
    def test_rag_query_counter(self):
        before = rag_queries_total.labels(search_type="hybrid", reranked="false")._value.get()
        rag_queries_total.labels(search_type="hybrid", reranked="false").inc()
        assert rag_queries_total.labels(search_type="hybrid", reranked="false")._value.get() == before + 1
