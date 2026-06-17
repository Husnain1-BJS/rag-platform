"""Upsert chunk vectors to Qdrant with connection pooling and deduplication."""
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, SparseVectorParams, OptimizersConfigDiff, SparseVector
import uuid
from typing import List, Dict, Optional, Set
import os
import json
import re
from pathlib import Path
import math
from collections import Counter


_clients: Dict[str, QdrantClient] = {}


def bm25_tokenize(text: str) -> List[str]:
    """Simple BM25 tokenizer: lowercase, split on non-alphanumeric, filter short tokens."""
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [t for t in tokens if len(t) > 2]


def compute_bm25_sparse(text: str, idf_cache: Dict[str, float] = None) -> SparseVector:
    """Compute BM25 sparse vector from text.
    
    Returns a SparseVector with token indices and TF-IDF scores.
    Uses a simple TF scoring since IDF requires corpus statistics.
    """
    tokens = bm25_tokenize(text)
    if not tokens:
        return SparseVector(indices=[], values=[])
    
    # Term frequency with logarithmic scaling
    tf = Counter(tokens)
    max_tf = max(tf.values()) if tf else 1
    
    seen_indices = set()
    indices = []
    values = []
    
    for token, count in tf.items():
        # Normalized TF score
        score = 0.5 + 0.5 * (count / max_tf)
        
        # Apply IDF if available
        if idf_cache and token in idf_cache:
            score *= idf_cache[token]
        
        # Use hash of token as index, handle collisions by incrementing
        token_idx = hash(token) % 30000
        while token_idx in seen_indices:
            token_idx = (token_idx + 1) % 30000
        seen_indices.add(token_idx)
        
        indices.append(token_idx)
        values.append(float(score))
    
    return SparseVector(indices=indices, values=values)


def get_client(
    path: str = "./qdrant_data",
    host: str = "localhost",
    port: int = 6333,
) -> QdrantClient:
    """Get or create a Qdrant client with connection pooling."""
    client_key = path if path else f"{host}:{port}"
    
    if client_key not in _clients:
        try:
            _clients[client_key] = QdrantClient(path=path)
        except Exception:
            _clients[client_key] = QdrantClient(host=host, port=port)
    
    return _clients[client_key]


def create_collection_if_not_exists(
    client: QdrantClient,
    collection: str,
    vector_size: int = 768,
) -> None:
    """Create collection with dense and sparse (BM25) vectors for hybrid search."""
    try:
        client.get_collection(collection_name=collection)
        print(f"Collection '{collection}' already exists")
    except Exception:
        client.create_collection(
            collection_name=collection,
            vectors_config={
                "dense": VectorParams(
                    size=vector_size,
                    distance="Cosine",
                ),
            },
            sparse_vectors_config={
                "text": SparseVectorParams(),
            },
            optimizers_config=OptimizersConfigDiff(indexing_threshold=1),
        )
        print(f"Created collection '{collection}' with hybrid search support")


def get_existing_ids(client: QdrantClient, collection: str) -> Set[str]:
    """Get all existing point IDs in the collection for deduplication."""
    existing_ids = set()
    try:
        offset = None
        while True:
            result = client.scroll(
                collection_name=collection,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            points, next_offset = result
            for point in points:
                existing_ids.add(str(point.id))
            if next_offset is None:
                break
            offset = next_offset
    except Exception:
        pass
    return existing_ids


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load ingestion checkpoint for resume capability."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return {"last_index": 0, "upserted": 0}


def save_checkpoint(checkpoint_path: str, data: dict):
    """Save ingestion checkpoint for resume capability."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump(data, f)


def upsert_chunks(
    chunks: List[Dict],
    collection: str = "threat_intel",
    client: Optional[QdrantClient] = None,
    deduplicate: bool = True,
    checkpoint_path: Optional[str] = None,
) -> int:
    """Upsert chunk vectors to Qdrant collection.
    
    Args:
        chunks: List of chunk dictionaries, each containing a 'vector' key.
        collection: Name of the Qdrant collection.
        client: Optional QdrantClient instance. If not provided, creates one.
        deduplicate: Whether to skip chunks that already exist in the collection.
        checkpoint_path: Optional path for checkpoint file to support resume.
        
    Returns:
        Total number of points upserted.
    """
    if not chunks:
        print("No chunks to upsert")
        return 0

    if client is None:
        client = get_client()
    create_collection_if_not_exists(client, collection)
    
    existing_ids = set()
    if deduplicate:
        existing_ids = get_existing_ids(client, collection)
        print(f"Found {len(existing_ids)} existing points for deduplication")
    
    last_index = 0
    total_upserted = 0
    
    if checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path)
        last_index = checkpoint.get("last_index", 0)
        total_upserted = checkpoint.get("upserted", 0)
        print(f"Resuming from index {last_index} (already upserted: {total_upserted})")
    
    batch_size = 100
    skipped = 0
    total_batches = (len(chunks) - last_index + batch_size - 1) // batch_size

    for i in range(last_index, len(chunks), batch_size):
        batch_num = ((i - last_index) // batch_size) + 1
        batch_chunks = chunks[i : i + batch_size]

        points = []
        for chunk in batch_chunks:
            vector = chunk.get("vector")
            if vector is None:
                continue

            payload = {
                k: v
                for k, v in chunk.items()
                if k != "vector"
                and k in ("cve_id", "text", "severity", "published_date", "source", "chunk_index")
            }

            seed_string = f"{chunk.get('cve_id', 'unknown')}_{chunk.get('chunk_index', 0)}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, seed_string))
            
            if deduplicate and point_id in existing_ids:
                skipped += 1
                continue

            # Generate BM25 sparse vector from text
            sparse_vector = compute_bm25_sparse(chunk.get("text", ""))
            
            point = PointStruct(
                id=point_id,
                vector={"dense": vector, "text": sparse_vector},
                payload=payload,
            )
            points.append(point)

        if points:
            client.upsert(collection_name=collection, points=points)
            total_upserted += len(points)
            print(f"Batch {batch_num}/{total_batches} - upserted {len(points)} points, total: {total_upserted}")
        
        if checkpoint_path:
            save_checkpoint(checkpoint_path, {
                "last_index": i + batch_size,
                "upserted": total_upserted,
            })

    if skipped > 0:
        print(f"Skipped {skipped} duplicate chunks")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return total_upserted


if __name__ == "__main__":
    print("Indexer module ready.")
    print("To test, run:")
    print("  1. Start Qdrant: docker run -d -p 6333:6333 qdrant/qdrant:latest")
    print("  2. Then run integration tests that call upsert_chunks()")
