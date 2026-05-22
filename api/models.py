"""
models.py — Pydantic request/response schemas for all API endpoints.
"""
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── /query ─────────────────────────────────────────────────────────── #

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, example="What fuel did Artemis I use?")
    session_id: Optional[str] = Field(None, description="Conversation session ID for memory")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Override default top-k retrieval")


class SourceDocument(BaseModel):
    title: str
    source_type: str
    snippet: str
    url: Optional[str] = None
    page: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    session_id: str
    chat_history_length: int


# ── /ingest ────────────────────────────────────────────────────────── #

class IngestURLRequest(BaseModel):
    url: str = Field(..., example="https://www.nasa.gov/missions/artemis")
    source_label: Optional[str] = None


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=50)
    metadata: Optional[dict] = None


class IngestResponse(BaseModel):
    success: bool
    source: str
    raw_documents: int
    chunks_created: int
    elapsed_seconds: float
    message: Optional[str] = None


# ── /summarize ─────────────────────────────────────────────────────── #

class SummarizeTextRequest(BaseModel):
    text: str = Field(..., min_length=100)


class TimelineEvent(BaseModel):
    date: str
    event: str


class KeyFacts(BaseModel):
    crew_size: Optional[str] = None
    launch_vehicle: Optional[str] = None
    destination: Optional[str] = None
    mission_duration: Optional[str] = None


class SummarizeResponse(BaseModel):
    mission_name: Optional[str] = None
    agency: Optional[str] = None
    mission_type: Optional[str] = None
    objective: Optional[str] = None
    timeline: Optional[list[TimelineEvent]] = None
    key_facts: Optional[KeyFacts] = None
    summary: Optional[str] = None
    raw_summary: Optional[str] = None  # fallback if JSON parse fails


# ── /classify ──────────────────────────────────────────────────────── #

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=20)


class LabelScore(BaseModel):
    label: str
    score: float


class ClassifyResponse(BaseModel):
    top_label: str
    top_score: float
    all_labels: list[LabelScore]


# ── /health ────────────────────────────────────────────────────────── #

class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store: str
    embedding_provider: str
    llm_model: str
    index_loaded: bool
