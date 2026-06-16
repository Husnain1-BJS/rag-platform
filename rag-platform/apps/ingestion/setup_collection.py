"""Create Qdrant collection for threat intelligence vectors."""
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


def main():
    """Connect to Qdrant and create the threat_intel collection."""
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "threat_intel"
    vector_size = 768
    distance = Distance.COSINE

    try:
        client.get_collection(collection_name=collection_name)
        print("collection already exists")
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
        print("collection threat_intel ready")


if __name__ == "__main__":
    main()