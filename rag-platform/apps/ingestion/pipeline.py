import argparse
import sys
import os

# Add the apps directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ingestion.parsers.cve_parser import parse_nvd_cve
from ingestion.parsers.mitre_parser import parse_mitre_attack
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_chunks
from ingestion.indexer import upsert_chunks


def run_cve_pipeline(limit: int = 2000) -> dict:
    """
    Run the CVE processing pipeline.
    
    Args:
        limit: Maximum number of CVE records to parse (default: 2000)
        
    Returns:
        dict: Summary of the pipeline run with counts
    """
    print("Starting CVE processing pipeline...")
    print(f"  Limit: {limit}")
    
    # Step 1: Parse CVEs
    print("\n1. Parsing CVE data...")
    # Parse from the NVD JSON file
    file_path = os.path.join('datasets', 'raw', 'nvdcve-2024.json.gz')
    # Fallback to recent file if 2024 doesn't exist
    if not os.path.exists(file_path):
        file_path = os.path.join('datasets', 'raw', 'nvdcve-2.0-recent.json.gz')
        print(f"  Note: nvdcve-2024.json.gz not found, using {file_path}")
    
    try:
        parsed_cves = parse_nvd_cve(file_path, limit=limit)
        print(f"  Parsed {len(parsed_cves)} CVE records")
    except Exception as e:
        print(f"  Error parsing CVE data: {e}")
        return {"parsed": 0, "chunks": 0, "upserted": 0, "error": str(e)}
    
    # Step 2: Chunk the parsed CVEs
    print("\n2. Chunking CVE descriptions...")
    try:
        chunked_docs = chunk_documents(parsed_cves)
        print(f"  Created {len(chunked_docs)} chunks")
    except Exception as e:
        print(f"  Error chunking documents: {e}")
        return {"parsed": len(parsed_cves), "chunks": 0, "upserted": 0, "error": str(e)}
    
    # Step 3: Embed the chunks
    print("\n3. Embedding chunks...")
    try:
        embedded_chunks = embed_chunks(chunked_docs, batch_size=32)
        print(f"  Embedded {len(embedded_chunks)} chunks")
    except Exception as e:
        print(f"  Error embedding chunks: {e}")
        return {"parsed": len(parsed_cves), "chunks": len(chunked_docs), "upserted": 0, "error": str(e)}
    
    # Step 4: Upsert to Qdrant
    print("\n4. Upserting to Qdrant collection 'threat_intel'...")
    try:
        upsert_count = upsert_chunks(embedded_chunks, collection="threat_intel")
        print(f"  Upserted {upsert_count} points to Qdrant")
    except Exception as e:
        print(f"  Error upserting to Qdrant: {e}")
        # Note: This might fail if Qdrant isn't running, which is expected in some environments
        return {"parsed": len(parsed_cves), "chunks": len(chunked_docs), "upserted": 0, "embedded": len(embedded_chunks), "error": str(e)}
    
    # Summary
    print("\n" + "="*50)
    print("CVE PIPELINE SUMMARY:")
    print(f"  Total parsed: {len(parsed_cves)}")
    print(f"  Total chunks: {len(chunked_docs)}")
    print(f"  Total embedded: {len(embedded_chunks)}")
    print(f"  Total upserted: {upsert_count}")
    print("="*50)
    
    return {
        "parsed": len(parsed_cves),
        "chunks": len(chunked_docs),
        "embedded": len(embedded_chunks),
        "upserted": upsert_count
    }


def run_mitre_pipeline() -> dict:
    """
    Run the MITRE ATT&CK processing pipeline.
    
    Returns:
        dict: Summary of the pipeline run with counts
    """
    print("Starting MITRE ATT&CK processing pipeline...")
    
    # Step 1: Parse MITRE data
    print("\n1. Parsing MITRE ATT&CK data...")
    file_path = os.path.join('datasets', 'raw', 'enterprise-attack.json')
    
    try:
        parsed_techniques = parse_mitre_attack(file_path)
        print(f"  Parsed {len(parsed_techniques)} MITRE techniques")
    except Exception as e:
        print(f"  Error parsing MITRE data: {e}")
        return {"parsed": 0, "chunks": 0, "upserted": 0, "error": str(e)}
    
    # Step 2: Chunk the parsed techniques
    print("\n2. Chunking MITRE descriptions...")
    try:
        chunked_docs = chunk_documents(parsed_techniques)
        print(f"  Created {len(chunked_docs)} chunks")
    except Exception as e:
        print(f"  Error chunking documents: {e}")
        return {"parsed": len(parsed_techniques), "chunks": 0, "upserted": 0, "error": str(e)}
    
    # Step 3: Embed the chunks
    print("\n3. Embedding chunks...")
    try:
        embedded_chunks = embed_chunks(chunked_docs, batch_size=32)
        print(f"  Embedded {len(embedded_chunks)} chunks")
    except Exception as e:
        print(f"  Error embedding chunks: {e}")
        return {"parsed": len(parsed_techniques), "chunks": len(chunked_docs), "upserted": 0, "error": str(e)}
    
    # Step 4: Upsert to Qdrant
    print("\n4. Upserting to Qdrant collection 'threat_intel'...")
    try:
        upsert_count = upsert_chunks(embedded_chunks, collection="threat_intel")
        print(f"  Upserted {upsert_count} points to Qdrant")
    except Exception as e:
        print(f"  Error upserting to Qdrant: {e}")
        # Note: This might fail if Qdrant isn't running, which is expected in some environments
        return {"parsed": len(parsed_techniques), "chunks": len(chunked_docs), "upserted": 0, "embedded": len(embedded_chunks), "error": str(e)}
    
    # Summary
    print("\n" + "="*50)
    print("MITRE PIPELINE SUMMARY:")
    print(f"  Total parsed: {len(parsed_techniques)}")
    print(f"  Total chunks: {len(chunked_docs)}")
    print(f"  Total embedded: {len(embedded_chunks)}")
    print(f"  Total upserted: {upsert_count}")
    print("="*50)
    
    return {
        "parsed": len(parsed_techniques),
        "chunks": len(chunked_docs),
        "embedded": len(embedded_chunks),
        "upserted": upsert_count
    }


def run_pipeline(limit: int = 2000, source: str = "all") -> dict:
    """
    Run the processing pipeline based on source.
    
    Args:
        limit: Maximum number of records to parse for CVE pipeline (default: 2000)
        source: Source identifier ("nvd", "mitre", or "all") (default: "all")
        
    Returns:
        dict: Summary of the pipeline run with counts
    """
    if source.lower() == "nvd":
        return run_cve_pipeline(limit=limit)
    elif source.lower() == "mitre":
        return run_mitre_pipeline()
    elif source.lower() == "all":
        print("Running both CVE and MITRE pipelines...")
        print("="*60)
        
        # Run CVE pipeline first
        cve_result = run_cve_pipeline(limit=limit)
        
        print("\n" + "="*60)
        print("Switching to MITRE pipeline...")
        
        # Run MITRE pipeline
        mitre_result = run_mitre_pipeline()
        
        # Combined summary
        print("\n" + "="*60)
        print("COMBINED PIPELINE SUMMARY:")
        print(f"  Total parsed: {cve_result.get('parsed', 0) + mitre_result.get('parsed', 0)}")
        print(f"  Total chunks: {cve_result.get('chunks', 0) + mitre_result.get('chunks', 0)}")
        print(f"  Total embedded: {cve_result.get('embedded', 0) + mitre_result.get('embedded', 0)}")
        print(f"  Total upserted: {cve_result.get('upserted', 0) + mitre_result.get('upserted', 0)}")
        print("="*60)
        
        return {
            "parsed": cve_result.get('parsed', 0) + mitre_result.get('parsed', 0),
            "chunks": cve_result.get('chunks', 0) + mitre_result.get('chunks', 0),
            "embedded": cve_result.get('embedded', 0) + mitre_result.get('embedded', 0),
            "upserted": cve_result.get('upserted', 0) + mitre_result.get('upserted', 0),
            "cve_pipeline": cve_result,
            "mitre_pipeline": mitre_result
        }
    else:
        print(f"Error: Unknown source '{source}'. Must be 'nvd', 'mitre', or 'all'.")
        return {"error": f"Unknown source '{source}'"}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the threat intelligence processing pipeline')
    parser.add_argument('--limit', type=int, default=2000,
                       help='Maximum number of CVE records to parse (default: 2000)')
    parser.add_argument('--source', type=str, default='all',
                       help='Source to process: "nvd", "mitre", or "all" (default: all)')
    
    args = parser.parse_args()
    
    result = run_pipeline(limit=args.limit, source=args.source)
    
    # Exit with error code if there was an error
    if 'error' in result and result['error']:
        sys.exit(1)
    else:
        sys.exit(0)