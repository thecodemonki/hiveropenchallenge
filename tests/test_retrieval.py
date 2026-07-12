"""Smoke tests for TF-IDF retrieval."""

from __future__ import annotations

from retrieval import top_k_similar


CORPUS = [
    {
        "id": "a",
        "subject": "Refund for double charge",
        "body": "I was charged twice on invoice 100. Please refund the duplicate.",
        "reply": "Refund issued.",
        "category": "billing_refund",
    },
    {
        "id": "b",
        "subject": "Export CSV fails",
        "body": "CSV export returns a 500 error in Chrome.",
        "reply": "Fix deploying today.",
        "category": "bug_report",
    },
    {
        "id": "c",
        "subject": "Cancel subscription",
        "body": "Please cancel our Pro plan at period end.",
        "reply": "Cancellation scheduled.",
        "category": "cancellation_churn",
    },
    {
        "id": "d",
        "subject": "Slack notifications",
        "body": "Can new tickets post to Slack?",
        "reply": "Slack beta available.",
        "category": "feature_request",
    },
]


def test_top_k_respects_k():
    results = top_k_similar("I need a refund for a double charge", CORPUS, k=2)
    assert len(results) <= 2
    assert len(results) == 2


def test_top_k_includes_id_and_score():
    results = top_k_similar("CSV export 500 error", CORPUS, k=3)
    assert results
    for item in results:
        assert "id" in item
        assert "score" in item
        assert isinstance(item["score"], float)


def test_top_k_empty_corpus():
    assert top_k_similar("anything", [], k=3) == []


def test_top_k_prefers_related_example():
    results = top_k_similar(
        "Please refund the duplicate charge on my invoice",
        CORPUS,
        k=1,
    )
    assert results[0]["id"] == "a"
