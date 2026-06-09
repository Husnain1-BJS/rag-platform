from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any


def chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chunk a single document based on description length.
    
    Args:
        doc: Dictionary containing at least 'description', 'cve_id', 'severity', 
             'published_date', 'source' and other metadata fields.
             
    Returns:
        List of chunk dictionaries, each containing the chunk text and metadata.
    """
    description = doc.get('description', '')
    # Approximate word count by splitting on whitespace
    word_count = len(description.split())
    
    # If description is under 400 words, return as single chunk
    if word_count < 400:
        chunk = doc.copy()
        chunk['text'] = description
        chunk['chunk_index'] = 0
        return [chunk]
    
    # Otherwise, split the description
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Split the description text
    chunks = text_splitter.split_text(description)
    
    # Build chunk documents
    chunked_docs = []
    for i, chunk_text in enumerate(chunks):
        chunk_doc = doc.copy()
        chunk_doc['text'] = chunk_text
        chunk_doc['chunk_index'] = i
        chunked_docs.append(chunk_doc)
    
    return chunked_docs


def chunk_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chunk a list of documents.
    
    Args:
        docs: List of document dictionaries.
        
    Returns:
        List of all chunked documents.
    """
    chunked_docs = []
    for doc in docs:
        chunked_docs.extend(chunk_document(doc))
    return chunked_docs


if __name__ == '__main__':
    # Create two fake docs for testing
    short_doc = {
        'cve_id': 'CVE-2026-TEST1',
        'description': 'This is a short description. It has fewer than 400 words.',
        'severity': 'LOW',
        'published_date': '2026-01-01T00:00:00.000Z',
        'source': 'nvd',
        'extra_field': 'extra_value'
    }
    
    # Create a long description (over 400 words)
    long_description = ('This is a test sentence for chunking. ' * 100)  # ~400 words
    
    long_doc = {
        'cve_id': 'CVE-2026-TEST2',
        'description': long_description,
        'severity': 'HIGH',
        'published_date': '2026-01-02T00:00:00.000Z',
        'source': 'nvd',
        'extra_field': 'extra_value'
    }
    
    # Test chunking
    short_chunks = chunk_document(short_doc)
    long_chunks = chunk_document(long_doc)
    
    print(f'Short doc chunks: {len(short_chunks)}')
    print(f'Long doc chunks: {len(long_chunks)}')
    
    # Print details of first chunk of long doc for verification
    if long_chunks:
        first_chunk = long_chunks[0]
        print('\nFirst chunk of long doc:')
        print(f"  cve_id: {first_chunk.get('cve_id')}")
        print(f"  chunk_index: {first_chunk.get('chunk_index')}")
        print(f"  text length: {len(first_chunk.get('text', ''))}")
        print(f"  text preview: {first_chunk.get('text', '')[:100]}...")