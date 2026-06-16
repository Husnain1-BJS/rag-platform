"""Shared extraction logic for CVE and MITRE parsers."""
from typing import Dict, Optional, List


def extract_cve_data(cve: Dict) -> Optional[Dict]:
    """Extract CVE data from NVD API v2 schema.
    
    Args:
        cve: CVE object from NVD API (under "vulnerabilities" -> "cve")
        
    Returns:
        Dict with keys: cve_id, description, severity, published_date, references, source
    """
    cve_id = cve.get("id")
    if not cve_id:
        return None

    description = ""
    descriptions = cve.get("descriptions", [])
    for desc in descriptions:
        if desc.get("lang") == "en":
            description = desc.get("value", "")
            break

    if not description or description.strip() == "** RESERVED **":
        return None

    severity = "UNKNOWN"
    metrics = cve.get("metrics", {})

    cvss_v31 = metrics.get("cvssMetricV31", [])
    if cvss_v31:
        severity = cvss_v31[0].get("cvssData", {}).get("baseSeverity", "UNKNOWN")
    else:
        cvss_v2 = metrics.get("cvssMetricV2", [])
        if cvss_v2:
            severity = cvss_v2[0].get("baseSeverity", "UNKNOWN")

    published_date = cve.get("published", "")

    references = []
    refs = cve.get("references", [])
    for ref in refs:
        url = ref.get("url")
        if url:
            references.append(url)

    return {
        "cve_id": cve_id,
        "description": description,
        "severity": severity,
        "published_date": published_date,
        "references": references,
        "source": "nvd",
    }


def extract_mitre_technique(obj: Dict) -> Optional[Dict]:
    """Extract MITRE ATT&CK technique from STIX object.
    
    Args:
        obj: STIX attack-pattern object
        
    Returns:
        Dict with keys: cve_id (technique_id), name, description, tactic, platforms, source
    """
    if obj.get("type") != "attack-pattern":
        return None
    if obj.get("x_mitre_deprecated") is True:
        return None
    if obj.get("revoked") is True:
        return None

    technique_id = None
    external_refs = obj.get("external_references", [])
    for ref in external_refs:
        if ref.get("source_name") == "mitre-attack":
            technique_id = ref.get("external_id")
            break

    if not technique_id:
        return None

    name = obj.get("name", "")
    description = obj.get("description", "")
    import re
    description = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", description)

    tactics = []
    kill_chain_phases = obj.get("kill_chain_phases", [])
    for phase in kill_chain_phases:
        if phase.get("kill_chain_name") == "mitre-attack":
            tactics.append(phase.get("phase_name", ""))

    platforms = obj.get("x_mitre_platforms", [])

    return {
        "cve_id": technique_id,
        "name": name,
        "description": description,
        "tactic": tactics,
        "platforms": platforms,
        "source": "mitre",
        "severity": "INFO",
        "published_date": "2024-01-01",
        "references": [],
    }


def normalize_to_chunk(doc: Dict, chunk_text: str, chunk_index: int) -> Dict:
    """Normalize any parsed document to chunk schema.
    
    Args:
        doc: Parsed document with cve_id, description, etc.
        chunk_text: Text content of this chunk
        chunk_index: Index of chunk within document
        
    Returns:
        Dict with keys: text, cve_id, severity, published_date, source, chunk_index
    """
    return {
        "text": chunk_text,
        "cve_id": doc.get("cve_id", "unknown"),
        "severity": doc.get("severity", "UNKNOWN"),
        "published_date": doc.get("published_date", ""),
        "source": doc.get("source", "unknown"),
        "chunk_index": chunk_index,
    }