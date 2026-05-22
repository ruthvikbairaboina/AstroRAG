"""
loader.py — load space documents from multiple sources.

Sources supported:
  - Local PDF files
  - NASA Open APIs (APOD, Mars Rover, Missions)
  - arXiv papers (by keyword search)
  - Direct URL (fetch & parse HTML)
"""
import os
import json
import requests
from pathlib import Path
from typing import List, Optional
from loguru import logger

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.schema import Document

from config import settings


class DocumentLoader:
    """Load documents from various space-related sources into LangChain Documents."""

    def __init__(self):
        self.nasa_api_key = settings.NASA_API_KEY

    # ------------------------------------------------------------------ #
    #  PDF loading                                                          #
    # ------------------------------------------------------------------ #

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load a single PDF and attach source metadata."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        logger.info(f"Loading PDF: {path.name}")
        loader = PyPDFLoader(str(path))
        docs = loader.load()

        for doc in docs:
            doc.metadata.update(
                {
                    "source_type": "pdf",
                    "file_name": path.name,
                    "file_path": str(path),
                }
            )
        logger.success(f"Loaded {len(docs)} pages from {path.name}")
        return docs

    def load_pdf_directory(self, directory: str) -> List[Document]:
        """Recursively load all PDFs from a directory."""
        all_docs: List[Document] = []
        pdf_files = list(Path(directory).rglob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDFs in {directory}")

        for pdf_path in pdf_files:
            try:
                all_docs.extend(self.load_pdf(str(pdf_path)))
            except Exception as e:
                logger.warning(f"Skipping {pdf_path.name}: {e}")

        return all_docs

    # ------------------------------------------------------------------ #
    #  NASA API                                                             #
    # ------------------------------------------------------------------ #

    def load_nasa_apod(self, count: int = 20) -> List[Document]:
        """Load NASA Astronomy Picture of the Day entries as documents."""
        url = "https://api.nasa.gov/planetary/apod"
        params = {"api_key": self.nasa_api_key, "count": count}

        logger.info(f"Fetching {count} NASA APOD entries...")
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        entries = resp.json()

        docs = []
        for entry in entries:
            content = (
                f"Title: {entry.get('title', 'N/A')}\n"
                f"Date: {entry.get('date', 'N/A')}\n"
                f"Explanation: {entry.get('explanation', '')}"
            )
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source_type": "nasa_apod",
                        "title": entry.get("title", ""),
                        "date": entry.get("date", ""),
                        "url": entry.get("url", ""),
                        "media_type": entry.get("media_type", ""),
                    },
                )
            )
        logger.success(f"Loaded {len(docs)} NASA APOD documents")
        return docs

    def load_nasa_earth_events(self) -> List[Document]:
        """Load NASA EONET natural earth events."""
        url = "https://eonet.gsfc.nasa.gov/api/v3/events"
        params = {"status": "open", "limit": 50}

        logger.info("Fetching NASA EONET earth events...")
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        docs = []
        for event in data.get("events", []):
            categories = ", ".join(c["title"] for c in event.get("categories", []))
            content = (
                f"Event: {event.get('title', 'N/A')}\n"
                f"Categories: {categories}\n"
                f"Description: {event.get('description', 'No description available.')}"
            )
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source_type": "nasa_eonet",
                        "event_id": event.get("id", ""),
                        "title": event.get("title", ""),
                        "categories": categories,
                        "link": event.get("link", ""),
                    },
                )
            )
        logger.success(f"Loaded {len(docs)} NASA EONET events")
        return docs

    # ------------------------------------------------------------------ #
    #  arXiv                                                                #
    # ------------------------------------------------------------------ #

    def load_arxiv_papers(
        self, query: str = "space mission NASA", max_results: int = 10
    ) -> List[Document]:
        """Fetch space research papers from arXiv API."""
        import urllib.parse

        base_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{urllib.parse.quote(query)}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        }
        query_str = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{base_url}?{query_str}"

        logger.info(f"Fetching arXiv papers for query: '{query}'")
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()

        # Parse Atom XML
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        docs = []
        for entry in root.findall("atom:entry", ns):
            title = entry.findtext("atom:title", default="", namespaces=ns).strip()
            summary = entry.findtext("atom:summary", default="", namespaces=ns).strip()
            published = entry.findtext("atom:published", default="", namespaces=ns)
            link_el = entry.find("atom:id", ns)
            arxiv_url = link_el.text if link_el is not None else ""

            authors = [
                a.findtext("atom:name", default="", namespaces=ns)
                for a in entry.findall("atom:author", ns)
            ]

            content = f"Title: {title}\nAuthors: {', '.join(authors)}\nAbstract: {summary}"
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source_type": "arxiv",
                        "title": title,
                        "authors": ", ".join(authors),
                        "published": published[:10] if published else "",
                        "url": arxiv_url,
                    },
                )
            )

        logger.success(f"Loaded {len(docs)} arXiv papers")
        return docs

    # ------------------------------------------------------------------ #
    #  URL / Web                                                            #
    # ------------------------------------------------------------------ #

    def load_url(self, url: str, source_label: Optional[str] = None) -> List[Document]:
        """Load and parse a web page."""
        logger.info(f"Loading URL: {url}")
        loader = WebBaseLoader(url)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_type"] = "web"
            doc.metadata["source_label"] = source_label or url
        return docs
