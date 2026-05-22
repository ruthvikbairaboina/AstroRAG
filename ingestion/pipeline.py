"""
pipeline.py — orchestrates the full ingestion flow:
  load → chunk → embed → store

Run directly:  python -m ingestion.pipeline
Or import:     IngestionPipeline().run_from_directory("data/raw")
"""
import time
from typing import List, Optional
from loguru import logger

from langchain.schema import Document

from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.embedder import EmbeddingManager
from ingestion.vector_store import VectorStoreManager


class IngestionPipeline:
    """End-to-end document ingestion: load → chunk → embed → index."""

    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker()
        self.embedding_manager = EmbeddingManager()
        self.vector_store_manager = VectorStoreManager(self.embedding_manager)

    # ------------------------------------------------------------------ #
    #  Primary entry points                                                 #
    # ------------------------------------------------------------------ #

    def run_from_directory(self, directory: str = "data/raw") -> dict:
        """Ingest all PDFs from a local directory."""
        logger.info(f"=== Starting ingestion from directory: {directory} ===")
        start = time.time()

        raw_docs = self.loader.load_pdf_directory(directory)
        return self._process_and_index(raw_docs, start, source=f"dir:{directory}")

    def run_from_nasa_api(self, apod_count: int = 50) -> dict:
        """Ingest NASA APOD entries + Earth events."""
        logger.info("=== Starting ingestion from NASA APIs ===")
        start = time.time()

        raw_docs: List[Document] = []
        raw_docs.extend(self.loader.load_nasa_apod(count=apod_count))
        raw_docs.extend(self.loader.load_nasa_earth_events())
        return self._process_and_index(raw_docs, start, source="nasa_api")

    def run_from_arxiv(
        self, query: str = "space mission NASA ESA", max_results: int = 20
    ) -> dict:
        """Ingest arXiv space papers."""
        logger.info(f"=== Starting ingestion from arXiv: '{query}' ===")
        start = time.time()

        raw_docs = self.loader.load_arxiv_papers(query=query, max_results=max_results)
        return self._process_and_index(raw_docs, start, source="arxiv")

    def run_full(
        self,
        pdf_dir: Optional[str] = "data/raw",
        nasa_apod_count: int = 50,
        arxiv_query: str = "NASA ESA space mission exploration",
        arxiv_max: int = 20,
    ) -> dict:
        """
        Master ingestion: pulls from ALL sources and builds one unified index.
        This is what you'd call once before serving the app.
        """
        logger.info("=== Full multi-source ingestion starting ===")
        start = time.time()
        all_docs: List[Document] = []

        # 1. PDFs (if directory exists and has files)
        import os
        if os.path.isdir(pdf_dir):
            all_docs.extend(self.loader.load_pdf_directory(pdf_dir))

        # 2. NASA APIs
        try:
            all_docs.extend(self.loader.load_nasa_apod(count=nasa_apod_count))
            all_docs.extend(self.loader.load_nasa_earth_events())
        except Exception as e:
            logger.warning(f"NASA API ingestion failed (continuing): {e}")

        # 3. arXiv
        try:
            all_docs.extend(
                self.loader.load_arxiv_papers(query=arxiv_query, max_results=arxiv_max)
            )
        except Exception as e:
            logger.warning(f"arXiv ingestion failed (continuing): {e}")

        return self._process_and_index(all_docs, start, source="full")

    # ------------------------------------------------------------------ #
    #  Internal                                                             #
    # ------------------------------------------------------------------ #

    def _process_and_index(
        self, raw_docs: List[Document], start: float, source: str
    ) -> dict:
        if not raw_docs:
            logger.error("No documents loaded — aborting ingestion")
            return {"success": False, "error": "No documents loaded"}

        # Step 1: Chunk
        chunks = self.chunker.chunk_documents(raw_docs)
        stats = self.chunker.get_stats(chunks)

        # Step 2: Embed + store
        self.vector_store_manager.build_from_documents(chunks)

        elapsed = round(time.time() - start, 2)
        result = {
            "success": True,
            "source": source,
            "raw_documents": len(raw_docs),
            "chunks_created": len(chunks),
            "elapsed_seconds": elapsed,
            **stats,
        }
        logger.success(f"Ingestion complete in {elapsed}s: {result}")
        return result


# ------------------------------------------------------------------ #
#  CLI entrypoint                                                       #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AstroRAG ingestion pipeline")
    parser.add_argument(
        "--source",
        choices=["directory", "nasa", "arxiv", "full"],
        default="full",
        help="Data source to ingest",
    )
    parser.add_argument("--pdf-dir", default="data/raw")
    parser.add_argument("--arxiv-query", default="NASA ESA space mission")
    args = parser.parse_args()

    pipeline = IngestionPipeline()

    if args.source == "directory":
        pipeline.run_from_directory(args.pdf_dir)
    elif args.source == "nasa":
        pipeline.run_from_nasa_api()
    elif args.source == "arxiv":
        pipeline.run_from_arxiv(args.arxiv_query)
    else:
        pipeline.run_full(pdf_dir=args.pdf_dir, arxiv_query=args.arxiv_query)
