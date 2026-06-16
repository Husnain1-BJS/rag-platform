"""Run the threat intelligence processing pipeline."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps"))
from api.config import settings

from ingestion.parsers.cve_parser import parse_cve_file
from ingestion.parsers.mitre_parser import parse_mitre_file
from ingestion.fetchers.nvd_fetcher import fetch_recent_cves
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_chunks
from ingestion.indexer import upsert_chunks, get_client


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CVE_RECENT = PROJECT_ROOT / "data/raw/nvdcve-2.0-recent.json.gz"
CVE_FULL = PROJECT_ROOT / "data/raw/nvdcve-2.0-2025.json.gz"
MITRE_FILE = PROJECT_ROOT / "data/raw/enterprise-attack.json"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def get_qdrant_client():
    """Get Qdrant client using connection pool."""
    return get_client(path=settings.QDRANT_PATH)


def _run_pipeline(
    source: str,
    parsed_data: list,
    deduplicate: bool = True,
    use_checkpoint: bool = False,
) -> dict:
    """Run the common pipeline stages (chunk -> embed -> upsert)."""
    chunked_docs = chunk_documents(parsed_data)
    embedded_chunks = embed_chunks(chunked_docs, batch_size=32)
    
    checkpoint_path = None
    if use_checkpoint:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = str(CHECKPOINT_DIR / f"{source}_checkpoint.json")
    
    client = get_qdrant_client()
    upsert_count = upsert_chunks(
        embedded_chunks,
        collection=settings.COLLECTION_NAME,
        client=client,
        deduplicate=deduplicate,
        checkpoint_path=checkpoint_path,
    )
    
    return {
        "parsed": len(parsed_data),
        "chunks": len(chunked_docs),
        "embedded": len(embedded_chunks),
        "upserted": upsert_count,
    }


def run_nvd_pipeline(limit: int, deduplicate: bool = True) -> dict:
    """Run the NVD CVE processing pipeline from local gz files."""
    if limit <= 200:
        filepath = CVE_RECENT
    else:
        filepath = CVE_FULL
    
    parsed_cves = parse_cve_file(filepath, limit=limit)
    result = _run_pipeline("nvd", parsed_cves, deduplicate=deduplicate)
    print(f"NVD Pipeline Summary: {result}")
    return result


def run_nvd_api_pipeline(limit: int, days_back: int = 1) -> dict:
    """Run the NVD CVE processing pipeline from live NVD REST API."""
    api_key = settings.NVD_API_KEY if settings.NVD_API_KEY else None
    parsed_cves = fetch_recent_cves(days_back=days_back, api_key=api_key)
    
    if limit > 0:
        parsed_cves = parsed_cves[:limit]
    
    result = _run_pipeline("nvd_api", parsed_cves, deduplicate=True)
    print(f"NVD API Pipeline Summary: {result}")
    return result


def run_incremental_pipeline(days_back: int = 1) -> dict:
    """Run incremental NVD CVE pipeline from live NVD REST API."""
    api_key = os.getenv("NVD_API_KEY") or settings.NVD_API_KEY
    parsed_cves = fetch_recent_cves(days_back=days_back, api_key=api_key)
    
    result = _run_pipeline("incremental", parsed_cves, deduplicate=True, use_checkpoint=True)
    print(f"Incremental Pipeline Summary: {result}")
    return result


def run_mitre_pipeline(limit: int, deduplicate: bool = True) -> dict:
    """Run the MITRE ATT&CK processing pipeline."""
    parsed_techniques = parse_mitre_file(MITRE_FILE, limit=limit)
    result = _run_pipeline("mitre", parsed_techniques, deduplicate=deduplicate)
    print(f"MITRE Pipeline Summary: {result}")
    return result


def run_pipeline(source: str, limit: int, days_back: int = 1, deduplicate: bool = True) -> dict:
    """Run the processing pipeline based on source."""
    if source.lower() == "nvd":
        return run_nvd_pipeline(limit=limit, deduplicate=deduplicate)
    elif source.lower() == "mitre":
        return run_mitre_pipeline(limit=limit, deduplicate=deduplicate)
    elif source.lower() == "nvd-api":
        return run_nvd_api_pipeline(limit=limit, days_back=days_back)
    elif source.lower() == "all":
        nvd_result = run_nvd_pipeline(limit=limit, deduplicate=deduplicate)
        mitre_result = run_mitre_pipeline(limit=limit, deduplicate=deduplicate)
        return {
            "parsed": nvd_result.get("parsed", 0) + mitre_result.get("parsed", 0),
            "chunks": nvd_result.get("chunks", 0) + mitre_result.get("chunks", 0),
            "embedded": nvd_result.get("embedded", 0) + mitre_result.get("embedded", 0),
            "upserted": nvd_result.get("upserted", 0) + mitre_result.get("upserted", 0),
            "nvd_pipeline": nvd_result,
            "mitre_pipeline": mitre_result,
        }
    else:
        print(f"Error: Unknown source '{source}'. Must be 'nvd', 'mitre', 'nvd-api', or 'all'.")
        return {"error": f"Unknown source '{source}'"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the threat intelligence processing pipeline")
    parser.add_argument("--source", type=str, default="all", choices=["nvd", "mitre", "nvd-api", "all"])
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--days-back", type=int, default=1)
    parser.add_argument("--no-deduplicate", action="store_true", help="Skip deduplication")
    args = parser.parse_args()
    
    result = run_pipeline(args.source, args.limit, args.days_back, deduplicate=not args.no_deduplicate)
    print(f"Final result: {result}")
