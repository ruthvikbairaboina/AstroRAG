"""
evaluate.py — measure RAG quality using the RAGAs framework.

Metrics:
  - faithfulness:      does the answer stick to the retrieved context?
  - answer_relevancy:  is the answer on-topic with the question?
  - context_recall:    did retrieval find the right chunks?
  - context_precision: are the retrieved chunks actually useful?

Run: python -m evaluation.evaluate
"""
import json
from pathlib import Path
from loguru import logger

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import settings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from retrieval.qa_chain import QAChain
from ingestion.vector_store import VectorStoreManager


# ── Ground-truth test set ────────────────────────────────────────────── #
TEST_SET = [
    {
        "question": "What fuel did Artemis I use?",
        "ground_truth": "Artemis I used liquid hydrogen and liquid oxygen in its RS-25D engines and RL10B-2 upper stage engine, plus solid propellant in its two solid rocket boosters.",
    },
    {
        "question": "What was the main objective of Artemis I?",
        "ground_truth": "The main objective of Artemis I was to test the Orion spacecraft and its heat shield in preparation for subsequent crewed Artemis missions to the Moon.",
    },
    {
        "question": "How far did Artemis I travel from Earth?",
        "ground_truth": "Artemis I achieved a maximum distance of 432,210 km from Earth, breaking the record previously held by Apollo 13.",
    },
    {
        "question": "What is the Space Launch System?",
        "ground_truth": "The Space Launch System is NASA's heavy-lift rocket used for the Artemis program, consisting of a core stage with four RS-25D engines and two solid rocket boosters.",
    },
    {
        "question": "When did Artemis I launch and land?",
        "ground_truth": "Artemis I launched on November 16, 2022 and landed on December 11, 2022, lasting 25 days in total.",
    },
]


def run_evaluation(output_path: str = "evaluation/results.json") -> dict:
    """Run RAGAs evaluation against the test set and save results."""
    logger.info(f"Running RAGAs evaluation on {len(TEST_SET)} questions…")

    vsm = VectorStoreManager()
    chain = QAChain(vsm)

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in TEST_SET:
        q = item["question"]
        logger.info(f"  Evaluating: {q!r}")

        result = chain.ask(q)
        chain.clear_memory()

        answers.append(result["answer"])
        questions.append(q)
        ground_truths.append(item["ground_truth"])
        contexts.append([s["snippet"] for s in result["sources"]])

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    # Explicitly pass LLM and embeddings to avoid version compatibility issues
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=settings.OPENAI_API_KEY,
    )
    embeddings = OpenAIEmbeddings(
        openai_api_key=settings.OPENAI_API_KEY,
    )

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
        embeddings=embeddings,
    )

    # Safely extract scores — some metrics may return None if they fail
    def safe_round(val):
        try:
            return round(float(val), 4)
        except (TypeError, ValueError):
            return None

    results = {
        "faithfulness": safe_round(scores["faithfulness"]),
        "answer_relevancy": safe_round(scores["answer_relevancy"]),
        "context_recall": safe_round(scores["context_recall"]),
        "context_precision": safe_round(scores["context_precision"]),
        "num_questions": len(TEST_SET),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.success(f"Evaluation complete. Results saved to {output_path}")
    logger.info(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_evaluation()
    