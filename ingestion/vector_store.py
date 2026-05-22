"""
vector_store.py — manage FAISS (local) and Pinecone (cloud) vector stores.

FAISS  → fast local dev, no API key needed, index saved to disk.
Pinecone → cloud, persists between restarts, production-ready.

Switch via VECTOR_STORE in .env
"""
import os
from pathlib import Path
from typing import List, Tuple
from loguru import logger

from langchain.schema import Document
from langchain.vectorstores.base import VectorStore

from config import settings
from ingestion.embedder import EmbeddingManager


class VectorStoreManager:
    """Create, persist, and query the vector index."""

    def __init__(self, embedding_manager: EmbeddingManager = None):
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self._store: VectorStore | None = None

    # ------------------------------------------------------------------ #
    #  Build / load                                                         #
    # ------------------------------------------------------------------ #

    def build_from_documents(self, documents: List[Document]) -> VectorStore:
        """Embed and index a list of chunked documents."""
        if not documents:
            raise ValueError("Cannot build vector store from empty document list")

        embeddings = self.embedding_manager.get_embeddings()
        provider = settings.VECTOR_STORE.lower()

        logger.info(
            f"Building {provider.upper()} index from {len(documents)} chunks…"
        )

        if provider == "faiss":
            self._store = self._build_faiss(documents, embeddings)
        elif provider == "pinecone":
            self._store = self._build_pinecone(documents, embeddings)
        else:
            raise ValueError(f"Unknown VECTOR_STORE='{provider}'")

        logger.success("Vector store built successfully")
        return self._store

    def load_existing(self) -> VectorStore:
        """Load a previously saved vector store from disk (FAISS only)."""
        provider = settings.VECTOR_STORE.lower()

        if provider == "faiss":
            self._store = self._load_faiss()
        elif provider == "pinecone":
            self._store = self._connect_pinecone()
        else:
            raise ValueError(f"Unknown VECTOR_STORE='{provider}'")

        return self._store

    def get_store(self) -> VectorStore:
        """Return the active store, loading from disk if needed."""
        if self._store is None:
            self._store = self.load_existing()
        return self._store

    # ------------------------------------------------------------------ #
    #  Retrieval                                                            #
    # ------------------------------------------------------------------ #

    def similarity_search(
        self, query: str, k: int = settings.TOP_K_RESULTS
    ) -> List[Document]:
        """Return top-k most relevant chunks for a query."""
        return self.get_store().similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int = settings.TOP_K_RESULTS
    ) -> List[Tuple[Document, float]]:
        """Return top-k chunks with relevance scores."""
        return self.get_store().similarity_search_with_score(query, k=k)

    def as_retriever(self, k: int = settings.TOP_K_RESULTS):
        """Return a LangChain Retriever (used by chains)."""
        return self.get_store().as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    # ------------------------------------------------------------------ #
    #  FAISS internals                                                      #
    # ------------------------------------------------------------------ #

    def _build_faiss(self, documents, embeddings):
        from langchain_community.vectorstores import FAISS

        store = FAISS.from_documents(documents, embeddings)
        self._save_faiss(store)
        return store

    def _save_faiss(self, store):
        index_path = settings.FAISS_INDEX_PATH
        Path(index_path).mkdir(parents=True, exist_ok=True)
        store.save_local(index_path)
        logger.info(f"FAISS index saved to {index_path}")

    def _load_faiss(self):
        from langchain_community.vectorstores import FAISS

        index_path = settings.FAISS_INDEX_PATH
        if not Path(index_path).exists():
            raise FileNotFoundError(
                f"No FAISS index at '{index_path}'. Run ingestion first."
            )
        embeddings = self.embedding_manager.get_embeddings()
        store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"FAISS index loaded from {index_path}")
        return store

    # ------------------------------------------------------------------ #
    #  Pinecone internals                                                   #
    # ------------------------------------------------------------------ #

    def _build_pinecone(self, documents, embeddings):
        from pinecone import Pinecone, ServerlessSpec
        from langchain_community.vectorstores import Pinecone as LCPinecone

        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index_name = settings.PINECONE_INDEX_NAME

        # Create index if it doesn't exist
        existing = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing:
            pc.create_index(
                name=index_name,
                dimension=384,  # all-MiniLM-L6-v2 output dim
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info(f"Created Pinecone index: {index_name}")

        store = LCPinecone.from_documents(
            documents,
            embeddings,
            index_name=index_name,
        )
        return store

    def _connect_pinecone(self):
        from pinecone import Pinecone
        from langchain_community.vectorstores import Pinecone as LCPinecone

        embeddings = self.embedding_manager.get_embeddings()
        return LCPinecone.from_existing_index(
            index_name=settings.PINECONE_INDEX_NAME,
            embedding=embeddings,
        )
