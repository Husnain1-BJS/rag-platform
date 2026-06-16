"""Unit tests for indexer."""
import pytest
import tempfile
import shutil
from qdrant_client import QdrantClient
from apps.ingestion.indexer import get_client, upsert_chunks, get_existing_ids


@pytest.fixture
def tmp_qdrant():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def client(tmp_qdrant):
    return get_client(path=tmp_qdrant)


def _make_chunks(n=3):
    return [
        {"text": f"text {i}", "cve_id": f"CVE-2024-{i}", "vector": [0.1] * 768, "severity": "HIGH"}
        for i in range(n)
    ]


class TestGetClient:
    def test_returns_client(self, tmp_qdrant):
        client = get_client(path=tmp_qdrant)
        assert isinstance(client, QdrantClient)


class TestUpsertChunks:
    def test_upserts_new_chunks(self, client):
        chunks = _make_chunks(2)
        count = upsert_chunks(chunks, "test_col", client=client, deduplicate=False)
        assert count == 2

    def test_deduplication_skips_existing(self, client):
        chunks = _make_chunks(2)
        upsert_chunks(chunks, "test_col", client=client, deduplicate=False)
        count = upsert_chunks(chunks, "test_col", client=client, deduplicate=True)
        assert count == 0


class TestGetExistingIds:
    def test_returns_empty_for_new_collection(self, client):
        ids = get_existing_ids(client, "test_col")
        assert ids == set()

    def test_returns_point_ids_after_upsert(self, client):
        chunks = _make_chunks(2)
        upsert_chunks(chunks, "test_col", client=client, deduplicate=False)
        ids = get_existing_ids(client, "test_col")
        assert len(ids) == 2
