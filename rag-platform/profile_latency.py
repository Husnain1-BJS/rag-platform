"""Profile embedding vs LLM latency."""
import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load from env or use defaults
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

print("=" * 60)
print("PROFILING: Embedding vs LLM Latency")
print("=" * 60)

# 1. Profile Embedding Model
print(f"\n1. Loading embedding model: {EMBEDDING_MODEL}")
start = time.perf_counter()
model = SentenceTransformer(EMBEDDING_MODEL)
load_time = time.perf_counter() - start
print(f"   Model load time: {load_time:.2f}s")

# Test embedding
test_texts = [
    "What is CVE-2024-1234?",
    "Buffer overflow vulnerability in Linux kernel",
    "MITRE ATT&CK technique T1566 Phishing",
] * 10  # 30 texts

print(f"\n2. Embedding {len(test_texts)} texts...")
start = time.perf_counter()
embeddings = model.encode(test_texts, normalize_embeddings=True, show_progress_bar=False)
embed_time = time.perf_counter() - start
print(f"   Total time: {embed_time:.2f}s")
print(f"   Per text: {embed_time/len(test_texts)*1000:.1f}ms")
print(f"   Embedding dim: {embeddings.shape[1]}")

# 2. Profile LLM (NVIDIA Build API)
if NVIDIA_API_KEY:
    print(f"\n3. Testing LLM: {NVIDIA_MODEL}")
    client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)
    
    messages = [
        {"role": "system", "content": "You are a cybersecurity assistant."},
        {"role": "user", "content": "What is CVE-2024-1234? Be concise."},
    ]
    
    print("   First request (cold)...")
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=messages,
        max_tokens=512,
    )
    llm_time = time.perf_counter() - start
    print(f"   Time: {llm_time:.2f}s")
    print(f"   Response: {response.choices[0].message.content[:100]}...")
    if hasattr(response, 'usage') and response.usage:
        print(f"   Tokens: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")
    
    print("   Second request (warm)...")
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=messages,
        max_tokens=512,
    )
    llm_time2 = time.perf_counter() - start
    print(f"   Time: {llm_time2:.2f}s")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Embedding (30 texts): {embed_time:.2f}s total, {embed_time/len(test_texts)*1000:.1f}ms/text")
    print(f"LLM first request:    {llm_time:.2f}s")
    print(f"LLM subsequent:       {llm_time2:.2f}s")
    print(f"\nLLM is ~{llm_time/embed_time:.0f}x SLOWER than batch embedding")
    print(f"LLM is the BOTTLENECK for query latency")
else:
    print("\n3. Skipping LLM test (no NVIDIA_API_KEY)")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Embedding (30 texts): {embed_time:.2f}s total, {embed_time/len(test_texts)*1000:.1f}ms/text")
    print("Set NVIDIA_API_KEY to profile LLM")

print("\nRecommendation: LLM is the bottleneck. Consider:")
print("  - Reduce max_tokens")
print("  - Enable streaming for perceived latency")
print("  - Cache frequent queries")
