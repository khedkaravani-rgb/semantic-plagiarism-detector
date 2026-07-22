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

# ── Public API ─────────────────────────────────────────────────────────────────


def detect_ai_probability(text: str) -> float:
    """
    Detect the probability that a given text was AI-generated.

    Args:
        text: Input text string to analyze.

    Returns:
        Probability score between 0.0 (human-written) and 1.0 (AI-generated).
    """
    if not text or not text.strip():
        return 0.0

    model, tokenizer = _get_model_and_tokenizer()

    # Tokenize input
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    )

    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Apply softmax to get probabilities
        if isinstance(logits, torch.Tensor):
            probs = torch.softmax(logits, dim=-1)
            ai_prob = probs[0, 1].item() if probs.shape[1] > 1 else probs[0, 0].item()
        else:
            ai_prob = 0.5

    return float(ai_prob)


def detect_ai_probability_batch(texts: List[str], batch_size: int = 8) -> List[float]:
    """
    Detect AI probability for multiple texts in batch for efficiency.

    Args:
        texts: List of text strings to analyze.
        batch_size: Number of texts to process per batch.

    Returns:
        List of probability scores (0.0 to 1.0) corresponding to input texts.
    """

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


            continue

        # Tokenize batch
        inputs = tokenizer(
            valid_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batch_probs = []
            if isinstance(logits, torch.Tensor):
                probs = torch.softmax(logits, dim=-1)
                for j in range(probs.shape[0]):
                    ai_prob = (
                        probs[j, 1].item() if probs.shape[1] > 1 else probs[j, 0].item()
                    )
                    batch_probs.append(float(ai_prob))
            else:
                batch_probs = [0.5] * len(valid_texts)

        # Map back to original batch order
        batch_result = [0.0] * len(batch_texts)
        for idx, prob in zip(valid_indices, batch_probs):
            batch_result[idx] = prob

        probabilities.extend(batch_result)


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
