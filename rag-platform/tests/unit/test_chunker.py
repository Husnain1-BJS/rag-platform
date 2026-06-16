"""Unit tests for chunker."""
import pytest
from apps.ingestion.chunker import chunk_documents


class TestChunkDocuments:
    def test_chunks_single_document(self):
        docs = [{
            "cve_id": "CVE-2024-1",
            "description": "A " * 500,
            "severity": "HIGH",
            "published_date": "2024-01-01",
            "source": "nvd",
        }]
        chunks = chunk_documents(docs)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "text" in chunk
            assert "cve_id" in chunk

    def test_preserves_metadata(self):
        docs = [{
            "cve_id": "CVE-2024-1",
            "severity": "HIGH",
            "published_date": "2024-01-01",
            "source": "nvd",
            "description": "Vuln " * 200,
        }]
        chunks = chunk_documents(docs)
        assert all(c["cve_id"] == "CVE-2024-1" for c in chunks)
        assert all(c["severity"] == "HIGH" for c in chunks)

    def test_handles_empty_input(self):
        assert chunk_documents([]) == []

    def test_short_text_single_chunk(self):
        docs = [{
            "cve_id": "CVE-2024-1",
            "description": "Short",
            "severity": "MEDIUM",
            "published_date": "2024-01-01",
            "source": "nvd",
        }]
        chunks = chunk_documents(docs)
        assert len(chunks) == 1
