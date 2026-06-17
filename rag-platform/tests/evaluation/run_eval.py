"""Run RAGAS evaluation on the RAG API."""
import os
import sys
import json
import httpx
from pathlib import Path
import types

# Patch missing langchain_community.chat_models.vertexai import for ragas
mock_vertexai = types.ModuleType('langchain_community.chat_models.vertexai')
mock_vertexai.ChatVertexAI = type('ChatVertexAI', (), {})
sys.modules['langchain_community.chat_models.vertexai'] = mock_vertexai

import langchain_community.chat_models as chat_models
chat_models.vertexai = mock_vertexai

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.evaluation.eval_dataset import EVAL_EXAMPLES

BASE_URL = "http://localhost:8000"
TIMEOUT = 600.0


def fetch_contexts_from_qdrant(question: str, top_k: int = 5) -> list[str]:
    """Fetch raw context chunks from Qdrant for a given question."""
    try:
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
        from apps.api.config import settings

        model = SentenceTransformer(settings.EMBEDDING_MODEL)
        qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

        vector = model.encode([question], normalize_embeddings=True)[0].tolist()
        hits = qdrant.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=vector,
            limit=top_k,
        ).points

        contexts = []
        for hit in hits:
            payload = hit.payload
            cve_id = payload.get("cve_id", "unknown")
            severity = payload.get("severity", "UNKNOWN")
            text = payload.get("text", "")
            contexts.append(f"ID: {cve_id} | Severity: {severity}\n{text}")
        return contexts
    except Exception as e:
        print(f"  Warning: Failed to fetch contexts from Qdrant: {e}")
        return []


def main():
    """Run RAGAS evaluation against the RAG API."""
    print("=" * 60)
    print("RAGAS Evaluation - Starting")
    print("=" * 60)

    client = httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)

    # Collect evaluation data
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, example in enumerate(EVAL_EXAMPLES, 1):
        question = example["question"]
        ground_truth = example["ground_truth"]

        print(f"\n[{i}/{len(EVAL_EXAMPLES)}] {question}")

        # Query the API
        resp = client.post("/query", json={"question": question, "top_k": 5})
        if resp.status_code != 200:
            print(f"  Error: {resp.status_code} - {resp.text}")
            answer = "ERROR"
            sources = []
        else:
            data = resp.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])
            print(f"  Answer: {answer[:100]}...")
            print(f"  Sources: {sources}")

        # Fetch raw contexts from Qdrant
        ctxs = fetch_contexts_from_qdrant(question, top_k=5)
        print(f"  Contexts fetched: {len(ctxs)}")

        questions.append(question)
        answers.append(answer)
        contexts.append(ctxs)
        ground_truths.append(ground_truth)

    client.close()

    print("\n" + "=" * 60)
    print("Building RAGAS dataset...")
    print("=" * 60)

    # Build ragas-compatible dataset
    from datasets import Dataset

    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(eval_data)

    print(f"Dataset size: {len(dataset)}")
    print(dataset)

    # Configure RAGAS to use NVIDIA Build API
    print("\n" + "=" * 60)
    print("Configuring RAGAS with NVIDIA Build API...")
    print("=" * 60)

    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision

    # Get NVIDIA settings from environment
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    nvidia_base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    nvidia_model = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")

    if not nvidia_api_key:
        print("ERROR: NVIDIA_API_KEY not set in environment")
        sys.exit(1)

    llm = ChatOpenAI(
        model=nvidia_model,
        openai_api_key=nvidia_api_key,
        openai_api_base=nvidia_base_url,
        temperature=0,
        max_tokens=1024,
    )
    ragas_llm = LangchainLLMWrapper(llm)

    # Run evaluation
    print("\nRunning RAGAS evaluation...")
    print("Metrics: faithfulness, answer_relevancy")

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
    )

    print("\n" + "=" * 60)
    print("RAGAS Evaluation Results")
    print("=" * 60)
    print(result)

    # Convert to pandas DataFrame for better display
    df = result.to_pandas()
    print("\nScore DataFrame:")
    print(df.to_string())

    # Save results
    output_dir = Path(__file__).parent
    output_file = output_dir / "results.json"
    df.to_json(output_file, orient="records", indent=2)
    print(f"\nResults saved to: {output_file}")

    # Also save raw result dict
    raw_output_file = output_dir / "results_raw.json"
    with open(raw_output_file, "w") as f:
        json.dump(result._scores_dict, f, indent=2)
    print(f"Raw scores saved to: {raw_output_file}")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()