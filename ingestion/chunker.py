"""
chunker.py — split documents into overlapping chunks for embedding.

Strategy:
  - RecursiveCharacterTextSplitter for most documents (respects sentence/para boundaries)
  - Smaller chunks for dense technical PDFs
  - Metadata is preserved and augmented on every chunk
"""
from typing import List
from loguru import logger

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import settings


class DocumentChunker:
    """Split LangChain Documents into retrieval-ready chunks."""

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Try to split at paragraph, sentence, then word boundaries
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk a list of documents and enrich metadata on each chunk."""
        if not documents:
            logger.warning("chunk_documents received an empty list")
            return []

        logger.info(
            f"Chunking {len(documents)} documents "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        chunks = self.splitter.split_documents(documents)

        # Enrich each chunk with positional metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_total"] = len(chunks)
            # Derive a short display title for the UI
            chunk.metadata["display_title"] = self._derive_title(chunk)

        logger.success(f"Produced {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    def chunk_text(self, text: str, base_metadata: dict = None) -> List[Document]:
        """Chunk a raw string directly (useful for API-sourced text)."""
        base_metadata = base_metadata or {}
        docs = [Document(page_content=text, metadata=base_metadata)]
        return self.chunk_documents(docs)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _derive_title(self, chunk: Document) -> str:
        """Pull a short human-readable title from chunk metadata."""
        meta = chunk.metadata
        if "title" in meta and meta["title"]:
            return meta["title"][:80]
        if "file_name" in meta:
            return meta["file_name"]
        if "source_label" in meta:
            return meta["source_label"][:80]
        # Fall back to first 60 chars of content
        return chunk.page_content[:60].replace("\n", " ") + "…"

    def get_stats(self, chunks: List[Document]) -> dict:
        """Return chunk length statistics (useful for debugging/evaluation)."""
        if not chunks:
            return {}
        lengths = [len(c.page_content) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_length": round(sum(lengths) / len(lengths), 1),
            "source_types": list(
                {c.metadata.get("source_type", "unknown") for c in chunks}
            ),
        }
