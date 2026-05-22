"""
main.py — AstroRAG FastAPI application.

Endpoints:
  GET  /health              — liveness + config check
  POST /query               — RAG question answering with memory
  DELETE /query/{session_id} — clear conversation memory
  POST /ingest/nasa         — ingest from NASA APIs
  POST /ingest/arxiv        — ingest from arXiv
  POST /ingest/url          — ingest a web URL
  POST /ingest/text         — ingest raw text
  POST /ingest/file         — upload and ingest a PDF
  POST /summarize           — summarize document text into timeline
  POST /summarize/file      — upload PDF and summarize
  POST /classify            — classify document text into mission type
"""
import uuid
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config import settings
from ingestion.pipeline import IngestionPipeline
from ingestion.vector_store import VectorStoreManager
from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.embedder import EmbeddingManager
from retrieval.qa_chain import QAChain
from retrieval.summarizer import MissionSummarizer
from retrieval.classifier import MissionClassifier
from api.models import (
    QueryRequest, QueryResponse, SourceDocument,
    IngestURLRequest, IngestTextRequest, IngestResponse,
    SummarizeTextRequest, SummarizeResponse,
    ClassifyRequest, ClassifyResponse,
    HealthResponse,
)

# ── Session store (in-memory; use Redis in production) ──────────────── #
sessions: dict[str, QAChain] = {}

# ── Shared singletons ────────────────────────────────────────────────── #
embedding_manager = EmbeddingManager()
vector_store_manager = VectorStoreManager(embedding_manager)
ingestion_pipeline = IngestionPipeline()
summarizer = MissionSummarizer()
classifier = MissionClassifier()
_index_loaded = False


# ── Lifespan: try loading existing index on startup ──────────────────── #
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _index_loaded
    try:
        vector_store_manager.load_existing()
        _index_loaded = True
        logger.success("Existing vector index loaded on startup")
    except FileNotFoundError:
        logger.warning("No existing index found. Run /ingest/* first.")
    yield
    logger.info("AstroRAG shutting down")


# ── App init ────────────────────────────────────────────────────────── #
app = FastAPI(
    title="AstroRAG",
    description="Space Mission Intelligence Assistant — RAG-powered Q&A over NASA, ESA, and arXiv documents",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ─────────────────────────────────────────────────────────── #
def get_or_create_chain(session_id: str) -> QAChain:
    if session_id not in sessions:
        sessions[session_id] = QAChain(vector_store_manager)
        logger.info(f"New session: {session_id}")
    return sessions[session_id]


def require_index():
    if not _index_loaded and not _has_index():
        raise HTTPException(
            status_code=503,
            detail="No vector index loaded. POST to /ingest/nasa, /ingest/arxiv, or /ingest/file first.",
        )


def _has_index() -> bool:
    try:
        vector_store_manager.get_store()
        return True
    except Exception:
        return False


# ════════════════════════════════════════════════════════════════════ #
#  HEALTH                                                               #
# ════════════════════════════════════════════════════════════════════ #

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check — confirms the app is running and shows config."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        vector_store=settings.VECTOR_STORE,
        embedding_provider=settings.EMBEDDING_PROVIDER,
        llm_model=settings.LLM_MODEL,
        index_loaded=_has_index(),
    )


# ════════════════════════════════════════════════════════════════════ #
#  QUERY — core RAG endpoint                                            #
# ════════════════════════════════════════════════════════════════════ #

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Ask a natural-language question over the indexed documents.

    - Maintains conversation memory per session_id.
    - Returns the answer plus cited source documents.
    - Omit session_id for a stateless single-turn query.
    """
    require_index()

    session_id = request.session_id or str(uuid.uuid4())
    chain = get_or_create_chain(session_id)

    try:
        result = chain.ask(request.question)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    sources = [SourceDocument(**s) for s in result["sources"]]
    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        session_id=session_id,
        chat_history_length=result["chat_history_length"],
    )


@app.delete("/query/{session_id}", tags=["Query"])
async def clear_session(session_id: str):
    """Clear conversation memory for a given session."""
    if session_id in sessions:
        sessions[session_id].clear_memory()
        del sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    raise HTTPException(status_code=404, detail="Session not found")


# ════════════════════════════════════════════════════════════════════ #
#  INGEST                                                               #
# ════════════════════════════════════════════════════════════════════ #

@app.post("/ingest/nasa", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_nasa(apod_count: int = Query(default=50, ge=1, le=200)):
    """Ingest NASA APOD entries and Earth event data."""
    global _index_loaded
    try:
        result = ingestion_pipeline.run_from_nasa_api(apod_count=apod_count)
        _index_loaded = True
        return IngestResponse(**result, message="NASA data ingested successfully")
    except Exception as e:
        logger.error(f"NASA ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/arxiv", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_arxiv(
    query: str = Query(default="NASA ESA space mission", min_length=3),
    max_results: int = Query(default=20, ge=1, le=100),
):
    """Ingest arXiv space research papers by keyword query."""
    global _index_loaded
    try:
        result = ingestion_pipeline.run_from_arxiv(query=query, max_results=max_results)
        _index_loaded = True
        return IngestResponse(**result, message=f"arXiv papers for '{query}' ingested")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/url", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_url(request: IngestURLRequest):
    """Fetch and ingest a web page (NASA mission page, Wikipedia, etc.)."""
    global _index_loaded
    try:
        loader = DocumentLoader()
        chunker = DocumentChunker()
        docs = loader.load_url(request.url, source_label=request.source_label)
        chunks = chunker.chunk_documents(docs)
        vector_store_manager.build_from_documents(chunks)
        _index_loaded = True
        return IngestResponse(
            success=True,
            source=request.url,
            raw_documents=len(docs),
            chunks_created=len(chunks),
            elapsed_seconds=0.0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/text", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_text(request: IngestTextRequest):
    """Ingest raw text directly (useful for testing or pasting content)."""
    global _index_loaded
    start = time.time()
    chunker = DocumentChunker()
    chunks = chunker.chunk_text(request.text, base_metadata=request.metadata or {})
    vector_store_manager.build_from_documents(chunks)
    _index_loaded = True
    return IngestResponse(
        success=True,
        source="raw_text",
        raw_documents=1,
        chunks_created=len(chunks),
        elapsed_seconds=round(time.time() - start, 2),
    )


@app.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(...)):
    """Upload and ingest a PDF document."""
    global _index_loaded
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    import tempfile, shutil
    start = time.time()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        loader = DocumentLoader()
        chunker = DocumentChunker()
        docs = loader.load_pdf(tmp_path)
        # Override file_name metadata with the original filename
        for doc in docs:
            doc.metadata["file_name"] = file.filename
        chunks = chunker.chunk_documents(docs)
        vector_store_manager.build_from_documents(chunks)
        _index_loaded = True

        return IngestResponse(
            success=True,
            source=file.filename,
            raw_documents=len(docs),
            chunks_created=len(chunks),
            elapsed_seconds=round(time.time() - start, 2),
        )
    finally:
        import os
        os.unlink(tmp_path)


# ════════════════════════════════════════════════════════════════════ #
#  SUMMARIZE                                                            #
# ════════════════════════════════════════════════════════════════════ #

@app.post("/summarize", response_model=SummarizeResponse, tags=["Analysis"])
async def summarize_text(request: SummarizeTextRequest):
    """Summarize mission document text into a structured timeline + key facts."""
    try:
        result = summarizer.summarize_text(request.text)
        return SummarizeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize/file", response_model=SummarizeResponse, tags=["Analysis"])
async def summarize_file(file: UploadFile = File(...)):
    """Upload a PDF and receive a structured mission summary."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    import tempfile, shutil
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        loader = DocumentLoader()
        docs = loader.load_pdf(tmp_path)
        result = summarizer.summarize_documents(docs)
        return SummarizeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        import os
        os.unlink(tmp_path)


# ════════════════════════════════════════════════════════════════════ #
#  CLASSIFY                                                             #
# ════════════════════════════════════════════════════════════════════ #

@app.post("/classify", response_model=ClassifyResponse, tags=["Analysis"])
async def classify(request: ClassifyRequest):
    """Classify document text into a space mission category."""
    try:
        result = classifier.classify(request.text)
        return ClassifyResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Dev server ──────────────────────────────────────────────────────── #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
