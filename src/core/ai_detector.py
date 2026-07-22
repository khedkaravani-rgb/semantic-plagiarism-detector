"""
src/core/ai_detector.py
-----------------------
AI content detection module using transformer models.
"""

import numpy as np

_model = None
_tokenizer = None
_DEFAULT_MODEL = "roberta-base-openai-detector"


def _get_model_name() -> str:
    return _DEFAULT_MODEL


def _get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        model_name = _get_model_name()
        print(f"[ai_detector] Loading model: {model_name} …")
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModelForSequenceClassification.from_pretrained(model_name)
            print("[ai_detector] Model loaded successfully.")
        except Exception as err:
            print(f"[ai_detector] Warning: Could not load transformer model ({err}). Using fallback mode.")
            _model = "fallback"
            _tokenizer = "fallback"
    return _model, _tokenizer


def detect_ai_probability_batch(texts: list[str]) -> list[float]:
    """Detects AI generated text probabilities for a batch of strings."""
    if not texts:
        return []

    try:
        model, tokenizer = _get_model_and_tokenizer()
        if model == "fallback":
            return [0.0] * len(texts)
    except Exception:
        return [0.0] * len(texts)

    probabilities = []
    batch_size = 8

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        try:
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            import torch

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                # Class 1 corresponds to Fake/AI
                ai_probs = probs[:, 1].tolist()
                probabilities.extend(ai_probs)
        except Exception:
            probabilities.extend([0.0] * len(batch_texts))

    return probabilities


def detect_ai_probability(text: str) -> float:
    """Detects AI generated text probability for a single string input."""
    if not text or not text.strip():
        return 0.0
    results = detect_ai_probability_batch([text])
    return results[0] if results else 0.0


def detect_document_ai_probability(chunks: list[str]) -> dict:
    """Calculates AI generated text statistics for a single document's chunks."""
    if not chunks:
        return {"overall": 0.0, "max": 0.0, "chunk_scores": []}

    chunk_scores = detect_ai_probability_batch(chunks)

    return {
        "overall": float(np.mean(chunk_scores)) if chunk_scores else 0.0,
        "max": float(np.max(chunk_scores)) if chunk_scores else 0.0,
        "chunk_scores": chunk_scores,
    }


def detect_documents_ai_probability(chunked_docs: dict[str, list[str]]) -> dict[str, dict]:
    """Calculates AI generated probabilities across multiple documents."""
    results = {}

    for doc_name, chunks in chunked_docs.items():
        results[doc_name] = detect_document_ai_probability(chunks)

    return results
