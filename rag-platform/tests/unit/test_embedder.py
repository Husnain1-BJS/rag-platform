"""Unit tests for embedder."""
import pytest
from apps.ingestion.embedder import get_model, embed_chunks


class TestGetModel:
    def test_returns_model(self):
        model = get_model()
        assert model is not None
        assert hasattr(model, "encode")


class TestEmbedChunks:
    def test_embeds_chunks(self):
        chunks = [
            {"text": "test vulnerability", "cve_id": "CVE-2024-1"},
            {"text": "another exploit", "cve_id": "CVE-2024-2"},
        ]
        result = embed_chunks(chunks, batch_size=2)
        assert len(result) == 2
        assert "vector" in result[0]
        assert isinstance(result[0]["vector"], list)
        assert len(result[0]["vector"]) > 0

    def test_empty_input(self):
        assert embed_chunks([]) == []

    def test_embedding_dimension(self):
        chunks = [{"text": "test", "cve_id": "CVE-2024-1"}]
        result = embed_chunks(chunks)
        assert len(result[0]["vector"]) == 768
