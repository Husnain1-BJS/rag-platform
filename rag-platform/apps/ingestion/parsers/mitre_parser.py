"""Parse MITRE ATT&CK STIX 2.0 JSON."""
import json
from pathlib import Path
from typing import List, Dict, Optional

from .base import extract_mitre_technique

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_mitre_file(filepath: Path, limit: Optional[int] = None) -> List[Dict]:
    """Parse MITRE ATT&CK STIX 2.0 JSON and extract technique information.

    Args:
        filepath: Path to the enterprise-attack.json file.
        limit: Maximum number of techniques to parse.

    Returns:
        List of dictionaries with keys: cve_id, name, description,
        tactic, platforms, source, severity, published_date, references.
    """
    results = []

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    objects = data.get("objects", [])

    for obj in objects:
        if limit is not None and len(results) >= limit:
            break

        extracted = extract_mitre_technique(obj)
        if extracted:
            results.append(extracted)

    return results


if __name__ == "__main__":
    filepath = PROJECT_ROOT / "data/raw/enterprise-attack.json"
    techniques = parse_mitre_file(filepath)
    print(f"Found {len(techniques)} techniques")
    if techniques:
        print(json.dumps(techniques[0], indent=2))