"""
qa_chain.py — the core RAG chain.

Uses:
  - LangChain ConversationalRetrievalChain for multi-turn memory
  - Custom prompt engineering for space domain + citation format
  - Returns answer + source documents for the UI to display
"""
from typing import Any
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from config import settings
from ingestion.vector_store import VectorStoreManager


# ── Prompt templates ────────────────────────────────────────────────── #

SYSTEM_TEMPLATE = """You are AstroRAG, an expert space mission intelligence assistant.
You have access to documents from NASA, ESA, arXiv, and other space agencies.

Use the context provided below to answer the question as helpfully as possible.
If the context contains relevant information, use it to give a detailed answer.
If the context is only partially relevant, still use what you have and be clear about it.
Only say you don't have information if the context is completely unrelated to the question.

Always cite the source document title in your answer.

Context:
{context}
"""

HUMAN_TEMPLATE = "{question}"

CONDENSE_QUESTION_TEMPLATE = """Given the conversation history and a follow-up question,
rephrase the follow-up into a standalone question that contains all necessary context.

Chat History:
{chat_history}

Follow-up: {question}
Standalone question:"""


class QAChain:
    """Conversational RAG chain with memory and source attribution."""

    def __init__(self, vector_store_manager: VectorStoreManager = None):
        self.vsm = vector_store_manager or VectorStoreManager()
        self._chain: ConversationalRetrievalChain | None = None
        self.memory = ConversationBufferWindowMemory(
            k=5,  # remember last 5 exchanges
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

    def get_chain(self) -> ConversationalRetrievalChain:
        """Build (and cache) the retrieval chain."""
        if self._chain is not None:
            return self._chain

        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            openai_api_key=settings.OPENAI_API_KEY,
        )

        # Prompt for final answer generation
        qa_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
            HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
        ])

        # Prompt for condensing follow-up questions
        condense_prompt = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)

        retriever = self.vsm.as_retriever(k=settings.TOP_K_RESULTS)

        self._chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory,
            condense_question_prompt=condense_prompt,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
            output_key="answer",
            verbose=settings.APP_ENV == "development",
        )
        logger.info("QA chain initialized")
        return self._chain

    def ask(self, question: str) -> dict:
        """
        Ask a question and return answer + sources.

        Returns:
            {
                "answer": str,
                "sources": [{"title": str, "source_type": str, "snippet": str, "url": str}],
                "chat_history_length": int
            }
        """
        logger.info(f"Query: {question!r}")
        chain = self.get_chain()

        result = chain.invoke({"question": question})

        sources = self._format_sources(result.get("source_documents", []))
        response = {
            "answer": result["answer"],
            "sources": sources,
            "chat_history_length": len(self.memory.chat_memory.messages),
        }
        logger.info(f"Answer generated, {len(sources)} sources cited")
        return response

    def clear_memory(self):
        """Reset conversation history."""
        self.memory.clear()
        logger.info("Conversation memory cleared")

    # ------------------------------------------------------------------ #
    #  Internal                                                             #
    # ------------------------------------------------------------------ #

    def _format_sources(self, source_docs: list) -> list[dict]:
        seen = set()
        sources = []
        for doc in source_docs:
            meta = doc.metadata
            key = meta.get("title") or meta.get("file_name") or meta.get("source_label", "")
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "title": meta.get("display_title") or key or "Unknown source",
                    "source_type": meta.get("source_type", "unknown"),
                    "snippet": doc.page_content[:200].replace("\n", " ") + "…",
                    "url": meta.get("url") or meta.get("link") or "",
                    "page": meta.get("page"),
                }
            )
        return sources
