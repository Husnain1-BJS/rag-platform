"""Integration tests for the RAG API endpoints.

Patches heavy dependencies (Qdrant, SentenceTransformer, OpenAI) at module level
to avoid file-locking and model-loading issues in CI/test environments.
"""
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_qdrant():
    """Create a mock QdrantClient with in-memory data."""
    client = MagicMock()

    collection_info = MagicMock()
    collection_info.status = "green"
    collection_info.points_count = 25
    collection_info.indexed_vectors_count = 25
    client.get_collection.return_value = collection_info
    client.get_collections.return_value = MagicMock(collections=[MagicMock(name="threat_intel")])

    search_result = MagicMock()
    search_result.points = [
        MagicMock(
            id="test-point-1",
            payload={
                "text": "CVE-2024-1234 buffer overflow vulnerability",
                "cve_id": "CVE-2024-1234",
                "severity": "HIGH",
                "source": "nvd",
                "published_date": "2024-01-15",
            },
        )
    ]
    client.query_points.return_value = search_result
    return client


@pytest.fixture
def mock_model():
    """Create a mock SentenceTransformer."""
    import numpy as np
    model = MagicMock()
    model.encode.return_value = np.random.randn(1, 768).astype(np.float32)
    return model


@pytest.fixture
def mock_llm():
    """Create a mock OpenAI client."""
    client = MagicMock()
    client.models.list.return_value = []
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="Test answer about CVE-2024-1234"))]
    response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
    client.chat.completions.create.return_value = response
    return client


@pytest.fixture
def client(mock_qdrant, mock_model, mock_llm):
    """Create TestClient with mocked dependencies."""

    with patch("qdrant_client.QdrantClient", return_value=mock_qdrant), \
         patch("sentence_transformers.SentenceTransformer", return_value=mock_model), \
         patch("openai.OpenAI", return_value=mock_llm):
        from apps.api.main import app
        from fastapi.testclient import TestClient
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "rag-api"
        assert data["qdrant_status"] == "green"
        assert data["indexed_vectors"] == 25


class TestMetricsEndpoint:
    def test_metrics_returns_prometheus(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200


class TestIngestEndpoint:
    def test_ingest_nvd(self, client):
        response = client.post("/ingest", json={"source": "nvd", "limit": 1})
        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "nvd"

    def test_ingest_incremental(self, client):
        response = client.post("/ingest", json={"source": "incremental", "limit": 1})
        assert response.status_code == 200


class TestQueryEndpoint:
    def test_query_returns_response(self, client):
        response = client.post("/query", json={
            "question": "What is CVE-2024-1234?",
            "search_type": "hybrid",
            "top_k": 3,
        })
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "search_type" in data
        assert data["search_type"] == "hybrid"
        assert data["context_used"] >= 1

    def test_query_validation_error(self, client):
        response = client.post("/query", json={})
        assert response.status_code == 422


@pytest.fixture
def client_with_auth(mock_qdrant, mock_model, mock_llm):
    with patch("qdrant_client.QdrantClient", return_value=mock_qdrant), \
         patch("sentence_transformers.SentenceTransformer", return_value=mock_model), \
         patch("openai.OpenAI", return_value=mock_llm), \
         patch("apps.api.config.settings.API_KEY", "test-key-123"):
        from apps.api.main import app
        from fastapi.testclient import TestClient
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestAuth:
    def test_missing_key_returns_401(self, client_with_auth):
        response = client_with_auth.post("/query", json={"question": "test"})
        assert response.status_code == 401
        assert "API key" in response.text

    def test_wrong_key_returns_401(self, client_with_auth):
        response = client_with_auth.post("/query", json={"question": "test"}, headers={"x-api-key": "wrong"})
        assert response.status_code == 401

    def test_valid_key_succeeds(self, client_with_auth):
        response = client_with_auth.post("/query", json={
            "question": "test",
            "search_type": "hybrid",
            "top_k": 1,
        }, headers={"x-api-key": "test-key-123"})
        assert response.status_code == 200

    def test_health_bypasses_auth(self, client_with_auth):
        response = client_with_auth.get("/health")
        assert response.status_code == 200