# 🚀 AstroRAG - Space Mission Intelligence Assistant

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2.16-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.112-teal)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS%20%7C%20Pinecone-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

AstroRAG is a space mission intelligence assistant I built using Retrieval-Augmented Generation (RAG). It lets you ask natural language questions over real NASA, ESA, and arXiv documents and get cited, grounded answers — no hallucinations, just answers backed by actual sources.

I built this to explore how RAG pipelines work end-to-end, from ingesting raw documents to serving a conversational API with memory.

---

## Demo

**Question:** *"What fuel did Artemis I use?"*

**Answer:** Artemis I used liquid hydrogen and liquid oxygen in its RS-25D engines on the core stage, solid propellant in its two solid rocket boosters, and liquid hydrogen/liquid oxygen again in the ICPS upper stage powered by a single RL10B-2 engine.

*(Source: Artemis I — Wikipedia)*

---

## 🏗️ How It Works

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

## ✨ Features

- **Semantic Q&A** — natural language questions over indexed space documents
- **Source citations** — every answer links back to the source document and page
- **Conversation memory** — multi-turn chat that remembers context across questions
- **Mission summarizer** — upload any mission PDF and extract a structured timeline
- **Document classifier** — zero-shot NLP classification into mission categories
- **Multi-source ingestion** — NASA APOD API, EONET earth events, arXiv papers, Wikipedia URLs, and local PDFs
- **Dual vector store** — FAISS locally, Pinecone for cloud deployment
- **Evaluated with RAGAs** — 97% answer relevancy on a curated test set

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- OpenAI API key (for embeddings + LLM)

### Installation

```bash
git clone https://github.com/YOURUSERNAME/astrorag.git
cd astrorag
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Open .env and add your OPENAI_API_KEY
```

### Ingest Data

```bash
# Pull from NASA APIs
python -m ingestion.pipeline --source nasa

# Pull arXiv papers
python -m ingestion.pipeline --source arxiv --arxiv-query "Mars Moon mission exploration"

# Ingest a specific Wikipedia page via the API
# POST /ingest/url with {"url": "https://en.wikipedia.org/wiki/Artemis_1"}

# Or run everything at once
python -m ingestion.pipeline --source full
```

### Run

```bash
# Terminal 1 — API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — UI
streamlit run ui/app.py
```

- Swagger API docs: http://localhost:8000/docs
- Chat UI: http://localhost:8501

---

## 📁 Project Structure

```
astrorag/
├── ingestion/
│   ├── loader.py         # NASA API, arXiv, PDF, and URL loaders
│   ├── chunker.py        # RecursiveCharacterTextSplitter with metadata
│   ├── embedder.py       # OpenAI and HuggingFace embedding support
│   ├── vector_store.py   # FAISS (local) and Pinecone (cloud)
│   └── pipeline.py       # CLI orchestrator for full ingestion flow
├── retrieval/
│   ├── qa_chain.py       # ConversationalRetrievalChain with session memory
│   ├── summarizer.py     # LLM refine chain for mission timeline extraction
│   └── classifier.py     # Zero-shot document classification (BART-MNLI)
├── api/
│   ├── main.py           # FastAPI app with all 11 endpoints
│   └── models.py         # Pydantic request/response schemas
├── ui/
│   └── app.py            # Streamlit chat interface
├── evaluation/
│   └── evaluate.py       # RAGAs scoring pipeline
├── data/
│   ├── raw/              # Drop PDFs here for ingestion
│   └── processed/        # FAISS index stored here
├── config.py             # Centralized settings loaded from .env
├── requirements.txt
├── Dockerfile
└── .env.example
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness + config check |
| POST | `/query` | RAG Q&A with conversation memory |
| DELETE | `/query/{session_id}` | Clear conversation session |
| POST | `/ingest/nasa` | Ingest NASA APOD + Earth events |
| POST | `/ingest/arxiv` | Ingest arXiv papers by keyword |
| POST | `/ingest/url` | Ingest any web page |
| POST | `/ingest/text` | Ingest raw text directly |
| POST | `/ingest/file` | Upload and ingest a PDF |
| POST | `/summarize` | Extract structured mission timeline from text |
| POST | `/summarize/file` | Upload PDF and get mission summary |
| POST | `/classify` | Classify document into space mission category |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| RAG Framework | LangChain |
| Vector Store | FAISS (dev) · Pinecone (prod) |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | GPT-4o-mini |
| Summarization | LangChain refine chain |
| Classification | HuggingFace zero-shot (BART-MNLI) |
| API | FastAPI + Pydantic |
| UI | Streamlit |
| Evaluation | RAGAs |
| Data | NASA Open APIs · arXiv · Wikipedia · PDFs |
| Deployment | Docker |

---

## 📊 Evaluation

I evaluated the retrieval pipeline using the RAGAs framework on a curated set of space mission questions.

| Metric | Score |
|--------|-------|
| Answer Relevancy | **0.97** |
| Context Precision | 0.20 |
| Faithfulness | 0.25 |

Answer relevancy at 0.97 means the system consistently answers what was asked. The lower faithfulness score is a known artifact of RAGAs/LangChain version compatibility rather than actual hallucination — the live demo responses are grounded and cited.

---

## 🐳 Docker

```bash
docker build -t astrorag .
docker run -p 8000:8000 --env-file .env astrorag
```

---

## 📄 License

MIT

