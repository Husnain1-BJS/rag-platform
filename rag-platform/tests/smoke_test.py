"""Smoke test for the RAG API."""
import sys
import httpx

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)


BASE_URL = "http://localhost:8000"
TIMEOUT = 120.0

QUERIES = [
    "Explain the Log4Shell vulnerability",
    "What ATT&CK techniques are used for credential dumping?",
    "High severity Apache vulnerabilities in 2024",
    "How does phishing relate to MITRE ATT&CK?",
    "What is a critical severity CVE from 2025?",
]


def main():
    """Run smoke tests against the API."""
    client = httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)

    # Test health
    print("Testing /health...")
    resp = client.get("/health")
    print(f"  Status: {resp.status_code}")
    print(f"  Response: {resp.json()}")
    print("-" * 80)

    # Test queries
    for i, question in enumerate(QUERIES, 1):
        print(f"Query {i}: {question}")
        resp = client.post("/query", json={"question": question, "top_k": 5})

        if resp.status_code == 200:
            data = resp.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])
            question_resp = data.get("question", "")
            context_used = data.get("context_used", 0)
            llm_error = data.get("llm_error", "")

            print(f"  Answer (first 300 chars): {answer[:300]}")
            print(f"  Sources: {sources}")
            print(f"  Question echoed: {question_resp}")
            print(f"  Context used: {context_used}")
            if llm_error:
                print(f"  LLM Error: {llm_error}")
        else:
            print(f"  Error: {resp.status_code} - {resp.text}")

        print("-" * 80)

    client.close()


if __name__ == "__main__":
    main()