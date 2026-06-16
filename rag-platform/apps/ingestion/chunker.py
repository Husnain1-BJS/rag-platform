"""Chunk documents into smaller pieces for embedding."""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict


def chunk_document(doc: Dict) -> List[Dict]:
    """Chunk a single document based on description length.

    Args:
        doc: Dictionary containing at least 'description', 'cve_id', 'severity',
             'published_date', 'source'.

    Returns:
        List of chunk dictionaries with text and metadata.
    """
    description = doc.get("description", "")
    word_count = len(description.split())

    if word_count < 400:
        return [
            {
                "text": description,
                "cve_id": doc["cve_id"],
                "severity": doc["severity"],
                "published_date": doc["published_date"],
                "source": doc["source"],
                "chunk_index": 0,
            }
        ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_text(description)
    chunked_docs = []
    for i, chunk_text in enumerate(chunks):
        chunked_docs.append(
            {
                "text": chunk_text,
                "cve_id": doc["cve_id"],
                "severity": doc["severity"],
                "published_date": doc["published_date"],
                "source": doc["source"],
                "chunk_index": i,
            }
        )

    return chunked_docs


def chunk_documents(docs: List[Dict]) -> List[Dict]:
    """Chunk a list of documents.

    Args:
        docs: List of document dictionaries.

    Returns:
        List of all chunked documents.
    """
    chunked_docs = []
    for doc in docs:
        chunked_docs.extend(chunk_document(doc))
    return chunked_docs


if __name__ == "__main__":
    short_doc = {
        "cve_id": "CVE-2026-TEST1",
        "description": "This is a short description. It has fewer than 400 words.",
        "severity": "LOW",
        "published_date": "2026-01-01T00:00:00.000Z",
        "source": "nvd",
    }

    long_description = "This is a test sentence for chunking. " * 100

    long_doc = {
        "cve_id": "CVE-2026-TEST2",
        "description": long_description,
        "severity": "HIGH",
        "published_date": "2026-01-02T00:00:00.000Z",
        "source": "nvd",
    }

    short_chunks = chunk_document(short_doc)
    long_chunks = chunk_document(long_doc)

    print(f"Short doc chunks: {len(short_chunks)}")
    print(f"Long doc chunks: {len(long_chunks)}")

    if long_chunks:
        first_chunk = long_chunks[0]
        print("\nFirst chunk of long doc:")
        print(f"  cve_id: {first_chunk.get('cve_id')}")
        print(f"  chunk_index: {first_chunk.get('chunk_index')}")
        print(f"  text length: {len(first_chunk.get('text', ''))}")