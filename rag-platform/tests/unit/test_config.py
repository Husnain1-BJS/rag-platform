"""Unit tests for API config."""
import pytest
from apps.api.config import Settings


class TestSettings:
    def test_default_values(self):
        s = Settings()
        assert s.EMBEDDING_MODEL == "BAAI/bge-base-en-v1.5"
        assert s.QDRANT_PATH == "./qdrant_data"
        assert s.COLLECTION_NAME == "threat_intel"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("QDRANT_PATH", "/tmp/test_qdrant")
        s = Settings()
        assert s.QDRANT_PATH == "/tmp/test_qdrant"
