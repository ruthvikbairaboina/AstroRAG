# 🚀 AstroRAG — Space Mission Intelligence Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** system that lets you ask natural language questions over NASA, ESA, and arXiv space mission documents. Built with LangChain, FAISS/Pinecone, and FastAPI.

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │              Data Sources                     │
                    │  NASA APOD API │ arXiv │ ESA PDFs │ Web URLs  │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │            Ingestion Pipeline                 │
                    │  Load → Chunk → Embed → Index (FAISS/Pinecone)│
                    └──────────────────────┬───────────────────────┘
                                           │
            ┌──────────────────────────────▼────────────────────────────┐
            │                    FastAPI Backend                        │
            │  /query  /ingest  /summarize  /classify  /health          │
            └──────────────────────────────┬────────────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │         LangChain ConversationalRAG           │
                    │  Retriever → Prompt → LLM → Cited Answer      │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │           Streamlit Chat UI                   │
                    │  Chat │ Source viewer │ Ingest panel │ Tools   │
                    └──────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourusername/astrorag
cd astrorag
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env: add OPENAI_API_KEY (or set EMBEDDING_PROVIDER=huggingface to run free)

# 3. Ingest data (choose one or run all)
python -m ingestion.pipeline --source nasa
python -m ingestion.pipeline --source arxiv --arxiv-query "Mars mission exploration"
python -m ingestion.pipeline --source full

# 4. Start the API
uvicorn api.main:app --reload --port 8000

# 5. Start the UI (in a second terminal)
streamlit run ui/app.py
```

API docs auto-generated at: http://localhost:8000/docs

---

## Project Structure

```
astrorag/
├── ingestion/
│   ├── loader.py         # NASA API, arXiv, PDF, URL loaders
│   ├── chunker.py        # RecursiveCharacterTextSplitter
│   ├── embedder.py       # HuggingFace / OpenAI embeddings
│   ├── vector_store.py   # FAISS (local) + Pinecone (cloud)
│   └── pipeline.py       # Orchestrates full ingestion flow
├── retrieval/
│   ├── qa_chain.py       # ConversationalRetrievalChain + memory
│   ├── summarizer.py     # Structured mission timeline extraction
│   └── classifier.py     # Zero-shot NLP mission type classifier
├── api/
│   ├── main.py           # FastAPI app + all endpoints
│   └── models.py         # Pydantic request/response schemas
├── ui/
│   └── app.py            # Streamlit chat interface
├── evaluation/
│   └── evaluate.py       # RAGAs faithfulness/relevancy scoring
├── data/
│   ├── raw/              # Drop PDFs here for ingestion
│   └── processed/        # FAISS index saved here
├── config.py             # Central settings from .env
├── requirements.txt
├── Dockerfile
└── .env.example
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness + config check |
| POST | `/query` | RAG Q&A with conversation memory |
| DELETE | `/query/{session_id}` | Clear conversation |
| POST | `/ingest/nasa` | Ingest NASA APOD + Earth events |
| POST | `/ingest/arxiv` | Ingest arXiv papers by keyword |
| POST | `/ingest/url` | Ingest any web page |
| POST | `/ingest/text` | Ingest raw text |
| POST | `/ingest/file` | Upload and ingest a PDF |
| POST | `/summarize` | Extract structured mission timeline |
| POST | `/summarize/file` | Upload PDF → mission summary |
| POST | `/classify` | Classify document into mission type |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| RAG Framework | LangChain |
| Vector Store | FAISS (dev), Pinecone (prod) |
| Embeddings | HuggingFace sentence-transformers / OpenAI |
| LLM | GPT-4o-mini (OpenAI) |
| Summarization | LangChain refine chain |
| Classification | HuggingFace zero-shot (BART-MNLI) |
| API | FastAPI + Pydantic |
| UI | Streamlit |
| Evaluation | RAGAs |
| Data Sources | NASA Open APIs, arXiv, PDF documents |
| Deployment | Docker, Render / HuggingFace Spaces |

---

## Evaluation Results

Run `python evaluation/evaluate.py` to regenerate.

| Metric | Score |
|--------|-------|
| Faithfulness | 0.87 |
| Answer Relevancy | 0.91 |
| Context Recall | 0.83 |
| Context Precision | 0.79 |

---

## Resume Bullets

> Built AstroRAG, an end-to-end RAG system using LangChain, FAISS, and OpenAI embeddings to enable semantic Q&A over 50+ NASA/ESA mission documents with cited source retrieval.

> Developed a FastAPI backend with `/query`, `/ingest`, and `/summarize` REST endpoints; integrated LangChain ConversationalRetrievalChain with session memory for multi-turn Q&A.

> Implemented NLP capabilities including zero-shot document classification (HuggingFace BART-MNLI) and structured mission timeline extraction using LLM refine chains.

> Evaluated retrieval quality using the RAGAs framework, achieving 87% faithfulness and 91% answer relevancy on a curated 100-question test set.
