from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

def main():
    # Connect to Qdrant at localhost:6333
    client = QdrantClient(host="localhost", port=6333)
    
    collection_name = "threat_intel"
    vector_size = 1024
    distance = Distance.COSINE
    
    try:
        # Check if collection already exists
        client.get_collection(collection_name=collection_name)
        print("collection already exists")
    except Exception:
        # Collection does not exist, create it
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
        print("collection threat_intel ready")

if __name__ == "__main__":
    main()