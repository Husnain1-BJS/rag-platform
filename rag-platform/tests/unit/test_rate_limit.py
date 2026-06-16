"""Unit tests for rate limiting."""
import time
import pytest
from unittest.mock import MagicMock, patch
from apps.api.rate_limit import RateLimitMiddleware, SKIP_RATE_LIMIT_PATHS


class TestSkipRateLimitPaths:
    def test_health_is_skipped(self):
        assert "/health" in SKIP_RATE_LIMIT_PATHS

    def test_metrics_is_skipped(self):
        assert "/metrics" in SKIP_RATE_LIMIT_PATHS


class TestRateLimitMiddleware:
    def test_allows_requests_under_limit(self):
        middleware = RateLimitMiddleware(MagicMock())
        assert len(middleware._buckets) == 0