"""Smoke tests for evaluation helpers (no LLM calls)."""

from __future__ import annotations

from evaluate import (
    DIMENSIONS,
    accuracy_from_dims,
    derange_ids,
    rouge_l_f1,
    score_item,
)


def test_rouge_l_identical_is_one():
    text = "refund issued within three business days"
    assert rouge_l_f1(text, text) == 1.0


def test_rouge_l_unrelated_is_low():
    score = rouge_l_f1(
        "please enable slack alerts for new tickets",
        "we processed a billing refund for invoice nine",
    )
    assert 0.0 <= score < 0.5


def test_accuracy_from_dims_bounds():
    assert accuracy_from_dims({d: 5 for d in DIMENSIONS}) == 100.0
    assert accuracy_from_dims({d: 1 for d in DIMENSIONS}) == 20.0
    assert accuracy_from_dims({d: 3 for d in DIMENSIONS}) == 60.0


def test_derange_ids_has_no_fixed_points():
    ids = ["t0", "t1", "t2", "t3"]
    mapping = derange_ids(ids, seed=42)
    assert set(mapping.keys()) == set(ids)
    assert set(mapping.values()) == set(ids)
    assert all(mapping[i] != i for i in ids)


def test_dry_run_score_item_structure():
    item = {
        "id": "test_0",
        "subject": "Refund request",
        "body": "Please refund the duplicate charge.",
        "generated_reply": "[dry-run] placeholder reply",
    }
    scores = score_item(None, item, "We issued the refund today.", dry_run=True)
    for key in DIMENSIONS:
        assert key in scores
        assert 1 <= scores[key] <= 5
    assert "accuracy_score" in scores
    assert "rouge_l_f1" in scores
    assert scores["accuracy_score"] == 60.0  # dry-run constant dims = 3 each
