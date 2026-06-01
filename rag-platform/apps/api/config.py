from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
