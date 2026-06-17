"""Run the threat intelligence processing pipeline with batched processing.

Supports processing large datasets (44k+ CVEs) without OOM by loading data
once and processing in configurable batches with checkpointing.
"""
import argparse
import gzip
import math
import os
import time
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps"))
from api.config import settings

from ingestion.parsers.cve_parser import parse_cve_file
from ingestion.parsers.base import extract_cve_data
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
    """Run the common pipeline stages (chunk -> embed -> upsert).
    Legacy all-in-memory version for small datasets.
    """
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


def _load_checkpoint(source: str) -> dict:
    path = CHECKPOINT_DIR / f"{source}_batch_checkpoint.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {"batches_completed": 0, "total_upserted": 0}


def _save_checkpoint(source: str, data: dict):
    path = CHECKPOINT_DIR / f"{source}_batch_checkpoint.json"
    with open(path, "w") as f:
        json.dump(data, f)


def _clear_checkpoint(source: str):
    path = CHECKPOINT_DIR / f"{source}_batch_checkpoint.json"
    if path.exists():
        os.remove(path)


def _load_and_parse_cves(filepath: Path, limit: int = None) -> list:
    """Load gzipped NVD JSON once and parse all CVEs into memory.

    Decompresses the gz file ONCE. For 44k CVEs this uses ~200-400MB RAM.
    """
    print(f"  Loading {filepath.name}...")
    load_start = time.time()

    with gzip.open(filepath, "rb") as f:
        data = json.load(f)

    vulnerabilities = data.get("vulnerabilities", [])
    del data  # Release top-level JSON structure
    print(f"  Raw CVEs in file: {len(vulnerabilities)}")

    parsed = []
    for item in vulnerabilities:
        if limit is not None and len(parsed) >= limit:
            break
        cve = item.get("cve", {})
        extracted = extract_cve_data(cve)
        if extracted:
            parsed.append(extracted)

    del vulnerabilities  # Release raw list
    load_time = time.time() - load_start
    print(f"  Parsed {len(parsed)} valid CVEs in {load_time:.1f}s")
    return parsed


def run_batched_nvd_pipeline(
    filepath: Path,
    source_name: str = "nvd",
    batch_size: int = None,
    limit: int = None,
    deduplicate: bool = True,
    resume: bool = True,
) -> dict:
    """Process NVD CVE file in batches. Loads data once, processes in chunks."""
    batch_size = batch_size or settings.INGEST_BATCH_SIZE
    client = get_qdrant_client()

    # Ensure checkpoint directory exists once
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    start_batch = 0
    total_upserted = 0
    if resume:
        checkpoint = _load_checkpoint(source_name)
        start_batch = checkpoint.get("batches_completed", 0)
        total_upserted = checkpoint.get("total_upserted", 0)
        if start_batch > 0:
            print(f"  Resuming from batch {start_batch} ({total_upserted} already upserted)")

    # Load all data ONCE
    all_cves = _load_and_parse_cves(filepath, limit=limit)

    # Process in batches
    total_batches = math.ceil(len(all_cves) / batch_size)
    total_parsed = 0
    total_chunks = 0
    total_embedded = 0
    failed_batches = []

    for batch_idx in range(total_batches):
        batch_num = batch_idx + 1
        start = batch_idx * batch_size
        end = min(start + batch_size, len(all_cves))
        batch = all_cves[start:end]

        if batch_num <= start_batch:
            total_parsed += len(batch)
            continue

        batch_start = time.time()

        try:
            chunked = chunk_documents(batch)
            embedded = embed_chunks(chunked, batch_size=32)
            upserted = upsert_chunks(
                embedded,
                collection=settings.COLLECTION_NAME,
                client=client,
                deduplicate=deduplicate,
            )

            total_parsed += len(batch)
            total_chunks += len(chunked)
            total_embedded += len(embedded)
            total_upserted += upserted

            batch_elapsed = time.time() - batch_start
            _save_checkpoint(source_name, {
                "batches_completed": batch_num,
                "total_upserted": total_upserted,
            })

            pct = (batch_num / total_batches) * 100
            print(f"  [{batch_num}/{total_batches}] {pct:.0f}% "
                  f"parsed={len(batch)} chunks={len(chunked)} upserted={upserted} "
                  f"time={batch_elapsed:.0f}s cumulative_upserted={total_upserted}")

        except Exception as e:
            batch_elapsed = time.time() - batch_start
            failed_batches.append({"batch": batch_num, "error": str(e)})
            print(f"  [{batch_num}/{total_batches}] FAILED after {batch_elapsed:.0f}s: {e}")
            # Save checkpoint so resume skips this failed batch
            _save_checkpoint(source_name, {
                "batches_completed": batch_num,
                "total_upserted": total_upserted,
            })
            continue

    _clear_checkpoint(source_name)

    result = {
        "parsed": total_parsed,
        "chunks": total_chunks,
        "embedded": total_embedded,
        "upserted": total_upserted,
    }
    if failed_batches:
        result["failed_batches"] = failed_batches
        print(f"  WARNING: {len(failed_batches)} batches failed: "
              f"{[b['batch'] for b in failed_batches]}")

    return result


def run_nvd_pipeline(limit: int, deduplicate: bool = True) -> dict:
    """Run the NVD CVE processing pipeline from local gz files."""
    filepath = CVE_RECENT if limit <= 200 else CVE_FULL
    parsed_cves = parse_cve_file(filepath, limit=limit)
    result = _run_pipeline("nvd", parsed_cves, deduplicate=deduplicate)
    print(f"NVD Pipeline Summary: {result}")
    return result


def run_nvd_pipeline_batched(
    limit: int = None,
    batch_size: int = None,
    deduplicate: bool = True,
    resume: bool = True,
) -> dict:
    """Run NVD pipeline with batched processing for large datasets."""
    filepath = CVE_FULL if (limit is None or limit > 200) else CVE_RECENT
    result = run_batched_nvd_pipeline(
        filepath=filepath,
        source_name="nvd_batched",
        batch_size=batch_size,
        limit=limit,
        deduplicate=deduplicate,
        resume=resume,
    )
    print(f"NVD Batched Pipeline Summary: {result}")
    return result


def run_nvd_api_pipeline(limit: int, days_back: int = 1) -> dict:
    api_key = settings.NVD_API_KEY if settings.NVD_API_KEY else None
    parsed_cves = fetch_recent_cves(days_back=days_back, api_key=api_key)
    if limit > 0:
        parsed_cves = parsed_cves[:limit]
    result = _run_pipeline("nvd_api", parsed_cves, deduplicate=True)
    print(f"NVD API Pipeline Summary: {result}")
    return result


def run_incremental_pipeline(days_back: int = 1) -> dict:
    api_key = os.getenv("NVD_API_KEY") or settings.NVD_API_KEY
    parsed_cves = fetch_recent_cves(days_back=days_back, api_key=api_key)
    result = _run_pipeline("incremental", parsed_cves, deduplicate=True, use_checkpoint=True)
    print(f"Incremental Pipeline Summary: {result}")
    return result


def run_mitre_pipeline(limit: int, deduplicate: bool = True) -> dict:
    parsed_techniques = parse_mitre_file(MITRE_FILE, limit=limit)
    result = _run_pipeline("mitre", parsed_techniques, deduplicate=deduplicate)
    print(f"MITRE Pipeline Summary: {result}")
    return result


def run_pipeline(
    source: str,
    limit: int,
    days_back: int = 1,
    deduplicate: bool = True,
    batch_size: int = None,
    resume: bool = True,
) -> dict:
    if source.lower() == "nvd":
        return run_nvd_pipeline(limit=limit, deduplicate=deduplicate)
    elif source.lower() == "nvd-batched":
        return run_nvd_pipeline_batched(
            limit=limit, batch_size=batch_size, deduplicate=deduplicate, resume=resume,
        )
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
    elif source.lower() == "all-batched":
        nvd_result = run_nvd_pipeline_batched(
            limit=limit, batch_size=batch_size, deduplicate=deduplicate, resume=resume,
        )
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
        print(f"Error: Unknown source '{source}'.")
        return {"error": f"Unknown source '{source}'"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the threat intelligence processing pipeline")
    parser.add_argument("--source", type=str, default="all",
                        choices=["nvd", "nvd-batched", "mitre", "nvd-api", "all", "all-batched"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--days-back", type=int, default=1)
    parser.add_argument("--no-deduplicate", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    result = run_pipeline(
        args.source,
        limit=args.limit,
        days_back=args.days_back,
        deduplicate=not args.no_deduplicate,
        batch_size=args.batch_size,
        resume=not args.no_resume,
    )
    print(f"Final result: {result}")
