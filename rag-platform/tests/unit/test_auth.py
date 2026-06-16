"""Unit tests for auth middleware."""
import pytest
from unittest.mock import MagicMock, patch
from fastapi import Request
from apps.api.auth import get_api_key, AuthMiddleware, SKIP_AUTH_PATHS


class TestGetApiKey:
    def test_returns_key_from_header(self):
        request = MagicMock(spec=Request)
        request.headers = {"x-api-key": "my-secret"}
        assert get_api_key(request) == "my-secret"

    def test_returns_empty_when_missing(self):
        request = MagicMock(spec=Request)
        request.headers = {}
        assert get_api_key(request) == ""


class TestSkipAuthPaths:
    def test_health_is_skipped(self):
        assert "/health" in SKIP_AUTH_PATHS

    def test_metrics_is_skipped(self):
        assert "/metrics" in SKIP_AUTH_PATHS