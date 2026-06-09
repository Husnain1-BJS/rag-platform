from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from typing import List, Dict, Any
import uuid


# Simple dict config for Qdrant connection
QDRANT_CONFIG = {
    "host": "localhost",
    "port": 6333
}


def get_qdrant_client() -> QdrantClient:
    """
    Create and return a Qdrant client instance.
    
    Returns:
        QdrantClient: Connected Qdrant client
    """
    return QdrantClient(
        host=QDRANT_CONFIG["host"],
        port=QDRANT_CONFIG["port"]
    )


def upsert_chunks(chunks: List[Dict[str, Any]], collection: str = "threat_intel") -> int:
    """
    Upsert chunk vectors to Qdrant collection.
    
    Args:
        chunks: List of chunk dictionaries, each containing a 'vector' key
        collection: Name of the Qdrant collection (default: "threat_intel")
        
    Returns:
        int: Total number of points upserted
    """
    if not chunks:
        print("No chunks to upsert")
        return 0
    
    # Initialize Qdrant client
    client = get_qdrant_client()
    
    # Ensure collection exists
    try:
        client.get_collection(collection_name=collection)
        print(f"Collection '{collection}' already exists")
    except Exception:
        # Collection doesn't exist, create it
        # Get vector size from first chunk
        vector_size = len(chunks[0].get('vector', []))
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Created collection '{collection}' with vector size {vector_size}")
    
    # Process chunks in batches of 100
    batch_size = 100
    total_upserted = 0
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch_num = (i // batch_size) + 1
        batch_chunks = chunks[i:i + batch_size]
        
        # Prepare points for this batch
        points = []
        for chunk in batch_chunks:
            # Extract vector
            vector = chunk.get('vector')
            if vector is None:
                print(f"Warning: Chunk {chunk.get('cve_id', 'unknown')} missing vector, skipping")
                continue
            
            # Create payload (all keys except 'vector')
            payload = {k: v for k, v in chunk.items() if k != 'vector'}
            
            # Generate deterministic UUID using UUID5 namespace
            # Using cve_id + chunk_index as the seed for deterministic UUID
            seed_string = f"{chunk.get('cve_id', 'unknown')}_{chunk.get('chunk_index', 0)}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, seed_string))
            
            # Create PointStruct
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
            points.append(point)
        
        # Upsert batch to Qdrant
        if points:
            client.upsert(
                collection_name=collection,
                points=points
            )
            
            total_upserted += len(points)
            print(f"Upserted batch {batch_num}/{total_batches} — total points so far: {total_upserted}")
    
    return total_upserted


if __name__ == '__main__':
    # This section would be used for testing, but requires a running Qdrant instance
    print("Indexer module ready.")
    print("To test, run:")
    print("  1. Start Qdrant: docker-compose -f infra/docker/docker-compose.yml up -d qdrant")
    print("  2. Then run integration tests that call upsert_chunks()")