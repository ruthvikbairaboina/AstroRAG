"""
summarizer.py — summarize a space mission document into a structured timeline.

Demonstrates: LLM summarization, structured output, prompt engineering.
"""
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document

from config import settings


TIMELINE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are an expert space mission analyst. Read the following document
and extract a structured mission timeline and summary.

Document:
{text}

Respond in this exact JSON format:
{{
  "mission_name": "Name of the mission",
  "agency": "NASA / ESA / ISRO / etc.",
  "mission_type": "Crewed | Robotic | Telescope | Launch Vehicle | Other",
  "objective": "1-2 sentence mission objective",
  "timeline": [
    {{"date": "YYYY-MM-DD or approx", "event": "event description"}},
    ...
  ],
  "key_facts": {{
    "crew_size": "number or N/A",
    "launch_vehicle": "rocket name",
    "destination": "Moon / Mars / LEO / etc.",
    "mission_duration": "X days or ongoing"
  }},
  "summary": "3-4 sentence narrative summary of the mission"
}}

Return only the JSON with no extra commentary.""",
)

REFINE_PROMPT = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template="""You are refining a space mission summary. Here is the current summary:

{existing_answer}

Refine it using additional context below. Keep the same JSON structure.
Only update fields if the new context provides better information.

Additional context:
{text}

Return only the updated JSON.""",
)


class MissionSummarizer:
    """Extract structured mission summaries from document chunks."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.0,  # deterministic for structured extraction
            openai_api_key=settings.OPENAI_API_KEY,
        )

    def summarize_documents(self, documents: list[Document]) -> dict:
        """
        Summarize a list of documents (e.g. all chunks of one PDF) into
        a structured mission timeline.
        """
        if not documents:
            return {"error": "No documents provided"}

        logger.info(f"Summarizing {len(documents)} document chunks")

        # Use refine chain — processes docs sequentially, refining the answer
        chain = load_summarize_chain(
            llm=self.llm,
            chain_type="refine",
            question_prompt=TIMELINE_PROMPT,
            refine_prompt=REFINE_PROMPT,
            return_intermediate_steps=False,
        )

        result = chain.invoke({"input_documents": documents})
        raw_output = result.get("output_text", "{}")

        # Parse JSON safely
        import json
        try:
            # Strip markdown fences if LLM wrapped the output
            cleaned = raw_output.strip().removeprefix("```json").removesuffix("```").strip()
            structured = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Structured JSON parse failed, returning raw text")
            structured = {"raw_summary": raw_output}

        logger.success("Mission summary generated")
        return structured

    def summarize_text(self, text: str) -> dict:
        """Convenience wrapper for summarizing raw text."""
        doc = Document(page_content=text, metadata={"source_type": "raw_text"})
        return self.summarize_documents([doc])
