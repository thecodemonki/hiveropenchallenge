"""Retrieve similar past email/reply pairs for a new inbound email.

Uses TF-IDF + cosine similarity over the dataset to surface
few-shot examples that inform reply generation.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def incoming_text(item: dict) -> str:
    """Subject + body of an incoming email (excludes reply)."""
    return f"{item.get('subject', '')}\n\n{item.get('body', '')}".strip()


def top_k_similar(query_text: str, corpus: list[dict], k: int = 3) -> list[dict]:
    """Return the top-k corpus items most similar to query_text.

    Similarity is TF-IDF cosine over incoming email text (subject + body) only.
    Each result includes the original item fields plus ``id`` and ``score``.
    """
    if not corpus or k <= 0:
        return []

    docs = [incoming_text(item) for item in corpus]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(docs + [query_text])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()

    ranked = sorted(range(len(corpus)), key=lambda i: float(scores[i]), reverse=True)
    results = []
    for i in ranked[: min(k, len(corpus))]:
        item = dict(corpus[i])
        item.setdefault("id", i)
        item["score"] = float(scores[i])
        results.append(item)
    return results
