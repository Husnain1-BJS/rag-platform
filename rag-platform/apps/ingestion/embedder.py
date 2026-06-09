import sys
import os
# Fix for local datasets directory conflicting with huggingface datasets library
# Add the site-packages directory to the beginning of sys.path to prioritize it
site_packages_path = os.path.join(os.path.dirname(os.__file__), '..', 'Lib', 'site-packages')
if os.path.exists(site_packages_path):
    sys.path.insert(0, site_packages_path)
# Also add the apps directory so our modules can still be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np


# Module-level variable for caching the model
_model = None


def get_model() -> SentenceTransformer:
    """
    Load and cache the SentenceTransformer model.
    
    Returns:
        SentenceTransformer: The loaded model instance
    """
    global _model
    if _model is None:
        print("Loading BAAI/bge-base-en-v1.5 model...")
        _model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        print("Model loaded successfully!")
    return _model


def embed_chunks(chunks: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, Any]]:
    """
    Embed a list of chunk dictionaries using the BAAI/bge-base-en-v1.5 model.
    
    Args:
        chunks: List of chunk dictionaries, each containing a 'text' field
        batch_size: Number of chunks to process in each batch (default: 32)
        
    Returns:
        List of chunk dictionaries with added 'vector' field containing embeddings
    """
    if not chunks:
        return chunks
    
    model = get_model()
    
    # Extract texts from chunks
    texts = [chunk.get('text', '') for chunk in chunks]
    
    # Process in batches
    total_batches = (len(texts) + batch_size - 1) // batch_size
    embedded_chunks = []
    
    for i in range(0, len(texts), batch_size):
        batch_num = (i // batch_size) + 1
        batch_texts = texts[i:i + batch_size]
        
        # Generate embeddings for the batch
        batch_embeddings = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Add embeddings to chunks
        for j, embedding in enumerate(batch_embeddings):
            chunk_index = i + j
            if chunk_index < len(chunks):
                # Create a copy of the chunk to avoid modifying the original
                embedded_chunk = chunks[chunk_index].copy()
                embedded_chunk['vector'] = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                embedded_chunks.append(embedded_chunk)
        
        # Print progress
        print(f"Embedded batch {batch_num}/{total_batches}")
    
    return embedded_chunks


if __name__ == '__main__':
    # Create 3 fake chunk dicts with text field
    fake_chunks = [
        {
            'text': 'This is the first test chunk for embedding.',
            'cve_id': 'CVE-2026-TEST1',
            'severity': 'LOW',
            'published_date': '2026-01-01T00:00:00.000Z',
            'source': 'nvd',
            'chunk_index': 0
        },
        {
            'text': 'This is the second test chunk with a bit more content to make it slightly longer.',
            'cve_id': 'CVE-2026-TEST2',
            'severity': 'HIGH',
            'published_date': '2026-01-02T00:00:00.000Z',
            'source': 'nvd',
            'chunk_index': 0
        },
        {
            'text': 'This is the third test chunk. It has enough text to be meaningful for embedding testing purposes.',
            'cve_id': 'CVE-2026-TEST3',
            'severity': 'MEDIUM',
            'published_date': '2026-01-03T00:00:00.000Z',
            'source': 'nvd',
            'chunk_index': 0
        }
    ]
    
    print("Testing embedder with fake chunks...")
    print(f"Number of chunks to embed: {len(fake_chunks)}")
    
    # Embed the chunks
    embedded_chunks = embed_chunks(fake_chunks, batch_size=2)
    
    # Print results
    if embedded_chunks:
        first_vector = embedded_chunks[0].get('vector')
        if first_vector:
            print(f"\nVector dimension: {len(first_vector)}")
            print(f"First 5 values of vector[0]: {first_vector[:5]}")
            print(f"Vector type: {type(first_vector)}")
            print(f"All chunks now have vectors: {all('vector' in chunk for chunk in embedded_chunks)}")
        else:
            print("ERROR: No vector found in first chunk")
    else:
        print("ERROR: No chunks returned from embedding")