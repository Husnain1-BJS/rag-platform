"""Parse NVD CVE JSON feed (version 2.0)."""
import gzip
import json
from pathlib import Path
from typing import List, Dict, Optional

from .base import extract_cve_data

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_cve_file(filepath: Path, limit: Optional[int] = None) -> List[Dict]:
    """Parse NVD CVE JSON file (v2.0 schema). Loads all into memory.

    Args:
        filepath: Path to the gzipped JSON file.
        limit: Maximum number of CVE items to parse.

    Returns:
        List of dictionaries with keys: cve_id, description, severity,
        published_date, references, source.
    """
    results = []

    with gzip.open(filepath, "rb") as f:
        data = json.load(f)

    vulnerabilities = data.get("vulnerabilities", [])

    for item in vulnerabilities:
        if limit is not None and len(results) >= limit:
            break

        cve = item.get("cve", {})
        extracted = extract_cve_data(cve)
        if extracted:
            results.append(extracted)

    return results


if __name__ == "__main__":
    filepath = PROJECT_ROOT / "data/raw/nvdcve-2.0-recent.json.gz"
    parsed = parse_cve_file(filepath, limit=10)
    if parsed:
        print(json.dumps(parsed[0], indent=2))
    else:
        print("No CVE items found")
