"""
config.py — central configuration loaded from .env
All modules import from here; never read os.environ directly elsewhere.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.1))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", 1024))

    # Embeddings
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    HUGGINGFACE_MODEL: str = os.getenv(
        "HUGGINGFACE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector store
    VECTOR_STORE: str = os.getenv("VECTOR_STORE", "faiss")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "data/processed/faiss_index")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "astrorag")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))

    # Retrieval
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", 5))

    # NASA
    NASA_API_KEY: str = os.getenv("NASA_API_KEY", "DEMO_KEY")

    # App
    APP_ENV: str = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
