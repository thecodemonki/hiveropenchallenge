"""Smoke test: dry-run pipeline produces expected data files."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data"

EXPECTED_FILES = (
    "dataset.json",
    "train.json",
    "test.json",
    "generated.json",
    "evaluations.json",
    "validation.json",
)


def test_dry_run_pipeline_writes_outputs():
    cmd = [sys.executable, str(SRC / "pipeline.py"), "--n", "5", "--dry-run"]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout

    for name in EXPECTED_FILES:
        path = DATA / name
        assert path.exists(), f"missing {path}"
        payload = json.loads(path.read_text())
        assert payload, f"{name} is empty"

    generated = json.loads((DATA / "generated.json").read_text())
    assert isinstance(generated, list)
    row = generated[0]
    for key in ("id", "generated_reply", "reference_reply", "retrieved_ids"):
        assert key in row

    evaluations = json.loads((DATA / "evaluations.json").read_text())
    ev = evaluations[0]
    for key in (
        "id",
        "accuracy_score",
        "rouge_l_f1",
        "intent_coverage",
        "tone_fidelity",
        "correctness",
        "conciseness",
    ):
        assert key in ev

    validation = json.loads((DATA / "validation.json").read_text())
    for key in (
        "avg_accuracy_correct",
        "avg_accuracy_mismatched",
        "accuracy_gap",
        "interpretation",
        "dry_run",
    ):
        assert key in validation
    assert validation["dry_run"] is True
