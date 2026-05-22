"""
embedder.py — create embedding models for vector indexing.

Supports:
  - HuggingFace sentence-transformers (free, local, good for dev)
  - OpenAI text-embedding-3-small (fast, high quality, costs money)

Switch via EMBEDDING_PROVIDER in .env
"""
from loguru import logger
from langchain.embeddings.base import Embeddings

from config import settings


class EmbeddingManager:
    """Factory that returns a LangChain-compatible embedding object."""

    def __init__(self, provider: str = settings.EMBEDDING_PROVIDER):
        self.provider = provider
        self._embeddings: Embeddings | None = None

    def get_embeddings(self) -> Embeddings:
        """Return (and cache) the configured embedding model."""
        if self._embeddings is not None:
            return self._embeddings

        if self.provider == "openai":
            self._embeddings = self._load_openai()
        elif self.provider == "huggingface":
            self._embeddings = self._load_huggingface()
        else:
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER='{self.provider}'. "
                "Choose 'openai' or 'huggingface'."
            )
        return self._embeddings

    # ------------------------------------------------------------------ #
    #  Backends                                                             #
    # ------------------------------------------------------------------ #

    def _load_openai(self) -> Embeddings:
        from langchain_openai import OpenAIEmbeddings

        if not settings.OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY is not set in .env")

        logger.info("Loading OpenAI embeddings (text-embedding-3-small)")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.OPENAI_API_KEY,
        )

    def _load_huggingface(self) -> Embeddings:
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = settings.HUGGINGFACE_MODEL
        logger.info(f"Loading HuggingFace embeddings: {model_name}")

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # ------------------------------------------------------------------ #
    #  Utility                                                              #
    # ------------------------------------------------------------------ #

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (helper for testing)."""
        return self.get_embeddings().embed_query(text)

    @property
    def model_name(self) -> str:
        if self.provider == "openai":
            return "text-embedding-3-small"
        return settings.HUGGINGFACE_MODEL
