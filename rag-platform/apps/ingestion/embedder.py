"""Embed text chunks using SentenceTransformer."""
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import partial


_model = None
_model_name = None
_executor = None


def get_executor(max_workers: int = 2) -> ThreadPoolExecutor:
    """Get or create the thread pool executor for async embedding."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=max_workers)
    return _executor


def get_model(model_name: str = None) -> SentenceTransformer:
    """Load and cache the SentenceTransformer model."""
    global _model, _model_name
    name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    if _model is None or _model_name != name:
        _model = SentenceTransformer(name)
        _model_name = name
    return _model


def _encode_batch(model: SentenceTransformer, texts: List[str], normalize: bool = True) -> List[List[float]]:
    """Encode a batch of texts (runs in thread pool)."""
    embeddings = model.encode(texts, normalize_embeddings=normalize, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]


def embed_chunks(chunks: List[Dict], batch_size: int = 32, model_name: str = None) -> List[Dict]:
    """Embed a list of chunk dictionaries (synchronous version).
    
    Args:
        chunks: List of chunk dictionaries, each containing a 'text' field.
        batch_size: Number of chunks to process in each batch.
        model_name: Optional model name override.
        
    Returns:
        List of chunk dictionaries with added 'vector' field.
    """
    if not chunks:
        return chunks

    model = get_model(model_name)
    texts = [chunk.get("text", "") for chunk in chunks]

    total_batches = (len(texts) + batch_size - 1) // batch_size
    embedded_chunks = []

    for i in range(0, len(texts), batch_size):
        batch_num = (i // batch_size) + 1
        batch_texts = texts[i : i + batch_size]

        batch_embeddings = model.encode(
            batch_texts, normalize_embeddings=True, show_progress_bar=False
        )

        for j, embedding in enumerate(batch_embeddings):
            chunk_index = i + j
            if chunk_index < len(chunks):
                embedded_chunk = chunks[chunk_index].copy()
                embedded_chunk["vector"] = embedding.tolist()
                embedded_chunks.append(embedded_chunk)

        print(f"Embedded batch {batch_num}/{total_batches}")

    return embedded_chunks


async def embed_chunks_async(
    chunks: List[Dict], 
    batch_size: int = 32, 
    model_name: str = None,
    max_workers: int = 2,
) -> List[Dict]:
    """Embed a list of chunk dictionaries asynchronously using thread pool.
    
    Args:
        chunks: List of chunk dictionaries, each containing a 'text' field.
        batch_size: Number of chunks to process in each batch.
        model_name: Optional model name override.
        max_workers: Number of worker threads for parallel batch encoding.
        
    Returns:
        List of chunk dictionaries with added 'vector' field.
    """
    if not chunks:
        return chunks

    model = get_model(model_name)
    texts = [chunk.get("text", "") for chunk in chunks]

    total_batches = (len(texts) + batch_size - 1) // batch_size
    executor = get_executor(max_workers)
    loop = asyncio.get_event_loop()

    # Create batch tasks
    batch_tasks = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        task = loop.run_in_executor(
            executor, 
            partial(_encode_batch, model, batch_texts, True)
        )
        batch_tasks.append((i, task))

    # Execute all batches in parallel
    embedded_chunks = [None] * len(chunks)
    
    for batch_idx, (start_idx, task) in enumerate(batch_tasks, 1):
        batch_embeddings = await task
        
        for j, embedding in enumerate(batch_embeddings):
            chunk_index = start_idx + j
            if chunk_index < len(chunks):
                embedded_chunk = chunks[chunk_index].copy()
                embedded_chunk["vector"] = embedding
                embedded_chunks[chunk_index] = embedded_chunk

        print(f"Embedded batch {batch_idx}/{total_batches}")

    # Filter out any None values (shouldn't happen but safety)
    return [c for c in embedded_chunks if c is not None]


def shutdown_executor():
    """Shutdown the thread pool executor."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None


if __name__ == "__main__":
    fake_chunks = [
        {"text": "This is the first test chunk for embedding.", "cve_id": "CVE-2026-TEST1"},
        {"text": "This is the second test chunk with more content.", "cve_id": "CVE-2026-TEST2"},
        {"text": "This is the third test chunk for testing purposes.", "cve_id": "CVE-2026-TEST3"},
    ]

    print("Testing embedder with fake chunks...")
    print(f"Number of chunks to embed: {len(fake_chunks)}")

    embedded_chunks = embed_chunks(fake_chunks, batch_size=2)

    if embedded_chunks:
        first_vector = embedded_chunks[0].get("vector")
        if first_vector:
            print(f"\nVector dimension: {len(first_vector)}")
            print(f"First 5 values: {first_vector[:5]}")
            print(f"All chunks have vectors: {all('vector' in c for c in embedded_chunks)}")
        else:
            print("ERROR: No vector found in first chunk")
    else:
        print("ERROR: No chunks returned from embedding")