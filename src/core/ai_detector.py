"""
src/core/ai_detector.py
-----------------------
AI content detection module using transformer models.
"""

import os
from typing import Any, Dict, List

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None

_DEFAULT_MODEL = "roberta-base-openai-detector"


def _get_model_name() -> str:
    """Return the configured AI detection model name."""
    return os.getenv("AI_DETECTION_MODEL", _DEFAULT_MODEL)


def _get_model_and_tokenizer():
    """Load and cache the transformer model and tokenizer."""
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        model_name = _get_model_name()
        logger.info(f"[ai_detector] Loading model: {model_name} …")

        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            )

            logger.info("[ai_detector] Model loaded successfully.")

        except Exception as err:
            logger.warning(
                "[ai_detector] Warning: Could not load transformer model "
                f"({err}). Using fallback mode."
            )

            _model = "fallback"
            _tokenizer = "fallback"

    return _model, _tokenizer


def detect_ai_probability_batch(texts: List[str], batch_size: int = 8) -> List[float]:
    """Detects AI generated text probabilities for a batch of strings."""
    if not texts:
        return []

    # Remove empty or whitespace-only texts while keeping track
    # of their original positions.
    valid_texts = []
    valid_indices = []

    for index, text in enumerate(texts):
        if text and text.strip():
            valid_texts.append(text)
            valid_indices.append(index)

    # If all texts are empty, return zero scores
    if not valid_texts:
        return [0.0] * len(texts)

    try:
        model, tokenizer = _get_model_and_tokenizer()

        # Use fallback mode if the transformer model could not load
        if model == "fallback":
            return [0.0] * len(texts)

    except Exception:
        return [0.0] * len(texts)

    # Initialize results for all original inputs
    probabilities = [0.0] * len(texts)

    # Process valid texts in batches
    for i in range(0, len(valid_texts), batch_size):
        batch_texts = valid_texts[i : i + batch_size]
        batch_indices = valid_indices[i : i + batch_size]

        try:
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            # Move tensors to GPU if available
            if torch.cuda.is_available():
                inputs = {
                    key: value.to("cuda")
                    for key, value in inputs.items()
                }

            # Run model inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

                if isinstance(logits, torch.Tensor):
                    probs = torch.softmax(logits, dim=-1)

                    # Class 1 corresponds to Fake/AI
                    if probs.shape[1] > 1:
                        batch_probs = probs[:, 1].tolist()
                    else:
                        batch_probs = probs[:, 0].tolist()
                else:
                    batch_probs = [0.5] * len(batch_texts)

            # Map predictions back to their original input positions
            for index, probability in zip(
                batch_indices,
                batch_probs,
            ):
                probabilities[index] = float(probability)

        except Exception as err:
            logger.warning(
                f"[ai_detector] Warning: Failed to process batch "
                f"starting at index {i}: {err}"
            )

            # Keep zero probability for failed batch items
            for index in batch_indices:
                probabilities[index] = 0.0

    return probabilities


def detect_ai_probability(text: str) -> float:
    """
    Detect the probability that a given text was AI-generated.

    Args:
        text: Input text string to analyze.

    Returns:
        Probability score between 0.0 (human-written)
        and 1.0 (AI-generated).
    """

    if not text or not text.strip():
        return 0.0

    results = detect_ai_probability_batch([text])

    return results[0] if results else 0.0


def detect_document_ai_probability(chunks: List[str]) -> Dict[str, Any]:
    """Calculates AI generated text statistics for a single document's chunks."""
    if not chunks:
        return {
            "overall": 0.0,
            "max": 0.0,
            "chunk_scores": [],
        }

    chunk_scores = detect_ai_probability_batch(chunks)

    return {
        "overall": (
            float(np.mean(chunk_scores))
            if chunk_scores
            else 0.0
        ),
        "max": (
            float(np.max(chunk_scores))
            if chunk_scores
            else 0.0
        ),
        "chunk_scores": chunk_scores,
    }


def detect_documents_ai_probability(
    chunked_docs: Dict[str, List[str]]
) -> Dict[str, Dict[str, Any]]:
    """Calculates AI generated probabilities across multiple documents."""
    results = {}

    for doc_name, chunks in chunked_docs.items():
        results[doc_name] = detect_document_ai_probability(
            chunks
        )

    return results