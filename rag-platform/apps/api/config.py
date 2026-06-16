from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_PATH: str = "./qdrant_data"
    COLLECTION_NAME: str = "threat_intel"
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    RAGAS_EVAL_MODEL: str = "meta-llama/llama-3.1-8b-instruct:free"
    NVD_API_KEY: str = ""

    ENABLE_HYBRID_SEARCH: bool = True
    SPARSE_MODEL: str = "Qdrant/bm25"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ENABLE_RERANKING: bool = True
    RERANK_TOP_K: int = 20
    FINAL_TOP_K: int = 5

    API_KEY: str = ""
    RATE_LIMIT_PER_MINUTE: int = 30

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()