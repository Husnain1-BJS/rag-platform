"""Create Qdrant collection for threat intelligence vectors with hybrid search support."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps"))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    OptimizersConfigDiff,
)

from api.config import settings


def main():
    """Connect to Qdrant and create the threat_intel collection with hybrid search."""
    # Use file-based or server client based on config
    if settings.QDRANT_PATH:
        client = QdrantClient(path=settings.QDRANT_PATH)
    else:
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

    collection_name = settings.COLLECTION_NAME
    vector_size = 768  # BAAI/bge-base-en-v1.5 dimension

    try:
        client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' already exists")
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=vector_size, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "text": SparseVectorParams(),
            },
            optimizers_config=OptimizersConfigDiff(indexing_threshold=1),
        )
        print(f"Created collection '{collection_name}' with hybrid search (dense + BM25)")


if __name__ == "__main__":
    main()
