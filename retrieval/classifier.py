"""
classifier.py — classify uploaded documents into mission categories.

Demonstrates: NLP text classification using zero-shot HuggingFace pipeline.
No fine-tuning required — works out of the box.
"""
from loguru import logger


MISSION_CATEGORIES = [
    "Crewed spaceflight",
    "Robotic planetary exploration",
    "Space telescope / observatory",
    "Launch vehicle / rocket",
    "Earth observation satellite",
    "Deep space / interstellar mission",
    "Space station / habitat",
]


class MissionClassifier:
    """Zero-shot classify space documents using HuggingFace transformers."""

    def __init__(self):
        self._classifier = None

    def _load(self):
        if self._classifier is None:
            from transformers import pipeline

            logger.info("Loading zero-shot classification pipeline...")
            self._classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1,  # CPU
            )
            logger.success("Classifier loaded")
        return self._classifier

    def classify(self, text: str) -> dict:
        """
        Classify a document excerpt.

        Returns:
            {
                "top_label": str,
                "top_score": float,
                "all_labels": [{"label": str, "score": float}, ...]
            }
        """
        # Use first 512 chars to keep inference fast
        snippet = text[:512]
        logger.info(f"Classifying text snippet ({len(snippet)} chars)")

        clf = self._load()
        result = clf(snippet, MISSION_CATEGORIES, multi_label=False)

        labels_scores = [
            {"label": label, "score": round(score, 4)}
            for label, score in zip(result["labels"], result["scores"])
        ]
        labels_scores.sort(key=lambda x: x["score"], reverse=True)

        response = {
            "top_label": labels_scores[0]["label"],
            "top_score": labels_scores[0]["score"],
            "all_labels": labels_scores,
        }
        logger.info(f"Classification: {response['top_label']} ({response['top_score']:.1%})")
        return response
