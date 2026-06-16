"""Fetch CVEs from NVD REST API v2."""
import os
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import requests

from ..parsers.base import extract_cve_data


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
        time.sleep(0.6)

    return all_results


if __name__ == "__main__":
    cves = fetch_recent_cves(days_back=1)
    print(f"Fetched {len(cves)} CVEs")
    if cves:
        import json
        print(json.dumps(cves[0], indent=2))