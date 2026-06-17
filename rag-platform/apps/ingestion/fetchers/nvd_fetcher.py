"""Fetch CVEs from NVD REST API v2 with streaming and rate limiting."""
import os
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Iterator
import requests

from ..parsers.base import extract_cve_data


# NVD API v2 rate limits:
# Without API key: 5 requests per 30 seconds (1 req/6s)
# With API key: 50 requests per 30 seconds (~1.7 req/s)
RATE_LIMIT_WITHOUT_KEY = 6.0  # seconds between requests
RATE_LIMIT_WITH_KEY = 0.6     # seconds between requests


def _get_rate_limit_delay(has_api_key: bool) -> float:
    """Get appropriate delay between API requests based on key availability."""
    return RATE_LIMIT_WITH_KEY if has_api_key else RATE_LIMIT_WITHOUT_KEY


def fetch_recent_cves(days_back: int = 1, api_key: str = None) -> List[Dict]:
    """Fetch recent CVEs from NVD REST API v2.

    Args:
        days_back: Number of days back to fetch (uses pubStartDate/pubEndDate).
        api_key: Optional NVD API key for higher rate limits.

    Returns:
        List of CVE dictionaries with keys: cve_id, description, severity,
        published_date, references, source.
    """
    if api_key is None:
        api_key = os.getenv("NVD_API_KEY")

    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    pub_start = start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    pub_end = end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    params = {
        "pubStartDate": pub_start,
        "pubEndDate": pub_end,
        "resultsPerPage": 2000,
        "startIndex": 0,
    }

    headers = {}
    if api_key:
        headers["apiKey"] = api_key

    all_results = []
    total_results = None
    delay = _get_rate_limit_delay(bool(api_key))

    while True:
        response = requests.get(base_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        if total_results is None:
            total_results = data.get("totalResults", 0)
            if total_results == 0:
                break

        vulnerabilities = data.get("vulnerabilities", [])
        for item in vulnerabilities:
            cve = item.get("cve", {})
            extracted = extract_cve_data(cve)
            if extracted:
                all_results.append(extracted)

        if len(all_results) >= total_results:
            break

        params["startIndex"] += params["resultsPerPage"]
        time.sleep(delay)

    return all_results


def fetch_cves_streaming(
    days_back: int = 1,
    api_key: str = None,
    results_per_page: int = 2000,
) -> Iterator[Dict]:
    """Stream CVEs from NVD API one at a time (memory-efficient).

    Args:
        days_back: Number of days back to fetch.
        api_key: Optional NVD API key.
        results_per_page: Results per API page (max 2000).

    Yields:
        Individual parsed CVE dicts.
    """
    if api_key is None:
        api_key = os.getenv("NVD_API_KEY")

    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    pub_start = start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    pub_end = end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    params = {
        "pubStartDate": pub_start,
        "pubEndDate": pub_end,
        "resultsPerPage": results_per_page,
        "startIndex": 0,
    }

    headers = {}
    if api_key:
        headers["apiKey"] = api_key

    total_results = None
    delay = _get_rate_limit_delay(bool(api_key))

    while True:
        response = requests.get(base_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        if total_results is None:
            total_results = data.get("totalResults", 0)
            if total_results == 0:
                return

        vulnerabilities = data.get("vulnerabilities", [])
        for item in vulnerabilities:
            cve = item.get("cve", {})
            extracted = extract_cve_data(cve)
            if extracted:
                yield extracted

        fetched = params["startIndex"] + len(vulnerabilities)
        if fetched >= total_results:
            return

        params["startIndex"] += results_per_page
        time.sleep(delay)


def fetch_cves_streaming_batched(
    days_back: int = 1,
    api_key: str = None,
    batch_size: int = 500,
    results_per_page: int = 2000,
) -> Iterator[List[Dict]]:
    """Stream CVEs from NVD API in batches (memory-efficient).

    Args:
        days_back: Number of days back to fetch.
        api_key: Optional NVD API key.
        batch_size: Number of CVEs per batch.
        results_per_page: Results per API page.

    Yields:
        Lists of parsed CVE dicts, each up to batch_size long.
    """
    batch = []
    for cve in fetch_cves_streaming(days_back, api_key, results_per_page):
        batch.append(cve)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


if __name__ == "__main__":
    cves = fetch_recent_cves(days_back=1)
    print(f"Fetched {len(cves)} CVEs")
    if cves:
        import json
        print(json.dumps(cves[0], indent=2))
