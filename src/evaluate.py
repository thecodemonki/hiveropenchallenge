"""Evaluate suggested replies against ground-truth or quality criteria.

Scores generated replies with an LLM judge (primary accuracy) and
ROUGE-L overlap (secondary signal), then validates that the judge
discriminates correct vs mismatched references.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

from anthropic import Anthropic

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL = "claude-sonnet-4-5"
DIMENSIONS = ("intent_coverage", "tone_fidelity", "correctness", "conciseness")

# Dry-run uses a constant judge so correct vs mismatched averages match —
# validation gap will be ~0. That is expected; it does not mean the metric works.
DRY_RUN_DIM_SCORE = 3

FALLBACK_WRONG_REPLY = (
    "Hi,\n\nWe've processed a refund for invoice #99999 and cancelled your "
    "enterprise SSO add-on. Expect the credit in 3–5 days.\n\n— Accounts"
)


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def load_json(path: Path) -> list | dict:
    return json.loads(path.read_text())


def with_ids(items: list[dict], prefix: str) -> list[dict]:
    out = []
    for i, item in enumerate(items):
        row = dict(item)
        row["id"] = row.get("id", f"{prefix}{i}")
        out.append(row)
    return out


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence (token-level)."""
    if not a or not b:
        return 0
    # rolling two rows — O(min(n,m)) space
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 via token LCS. Secondary signal only — not the accuracy number."""
    hyp = tokenize(hypothesis)
    ref = tokenize(reference)
    if not hyp or not ref:
        return 0.0
    lcs = lcs_length(hyp, ref)
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def accuracy_from_dims(dims: dict[str, int]) -> float:
    """Map four 1–5 scores onto a 0–100 accuracy_score."""
    total = sum(int(dims[d]) for d in DIMENSIONS)
    return total / 20 * 100


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group())


def judge_pair(
    client: Anthropic | None,
    generated_reply: str,
    reference_reply: str,
    incoming: dict,
    dry_run: bool,
) -> dict[str, int]:
    """Score generated vs reference on the four dimensions (1–5 each)."""
    if dry_run:
        return {d: DRY_RUN_DIM_SCORE for d in DIMENSIONS}

    assert client is not None
    prompt = f"""You judge a suggested support-email reply against a reference reply that an agent actually sent.

This is NOT exact-match grading. A good reply may use totally different words than the reference
if it covers the same intent, matches tone, is factually careful, and is concise.

Score each dimension from 1 (poor) to 5 (excellent):
- intent_coverage: does the generated reply address what the customer needed?
- tone_fidelity: warm, direct, action-oriented in the same spirit as the reference?
- correctness: avoids inventing policies/facts the reference wouldn't support?
- conciseness: appropriately tight without omitting needed next steps?

Incoming email subject: {incoming.get('subject', '')}
Incoming email body:
{incoming.get('body', '')}

Reference reply (what was actually sent):
{reference_reply}

Generated reply (to score):
{generated_reply}

Return JSON only, no markdown:
{{"intent_coverage": n, "tone_fidelity": n, "correctness": n, "conciseness": n}}
"""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = _parse_json_object(msg.content[0].text)
    dims = {}
    for d in DIMENSIONS:
        val = int(raw[d])
        dims[d] = max(1, min(5, val))
    return dims


def derange_ids(ids: list[str], seed: int) -> dict[str, str]:
    """Map each id to a different id (derangement). Requires len(ids) >= 2."""
    if len(ids) < 2:
        raise ValueError("derangement needs at least 2 ids")
    rng = random.Random(seed)
    targets = list(ids)
    # random cyclic shift guarantees no fixed points for n >= 2
    shift = rng.randrange(1, len(ids))
    rotated = targets[shift:] + targets[:shift]
    return dict(zip(ids, rotated))


def score_item(
    client: Anthropic | None,
    item: dict,
    reference_reply: str,
    dry_run: bool,
) -> dict:
    dims = judge_pair(
        client,
        item["generated_reply"],
        reference_reply,
        item,
        dry_run,
    )
    return {
        **dims,
        "accuracy_score": accuracy_from_dims(dims),
        "rouge_l_f1": rouge_l_f1(item["generated_reply"], reference_reply),
    }


def interpret_gap(gap: float, dry_run: bool, n: int) -> str:
    if dry_run:
        return (
            "Dry-run used a constant placeholder judge score for every pair, so the "
            "correct-vs-mismatched gap is ~0 by construction. This does not validate "
            "the metric — re-run without --dry-run (and with ≥2 test emails) to check "
            "that the LLM judge discriminates."
        )
    if n < 2:
        return (
            "Only one test email was available, so mismatched pairs used a canned "
            "unrelated reference instead of a shuffled test ID. Treat the gap as a "
            "weak sanity check; regenerate a larger test set for a real derangement."
        )
    if gap >= 15:
        return (
            f"Average accuracy on correct pairs exceeds mismatched pairs by {gap:.1f} points. "
            "That gap indicates the judge tracks reply–reference fit rather than emitting "
            "a plausible constant score."
        )
    if gap >= 5:
        return (
            f"Modest gap of {gap:.1f} points between correct and mismatched averages. "
            "The metric shows some discrimination; a larger test set would make this clearer."
        )
    return (
        f"Gap of only {gap:.1f} points between correct and mismatched averages. "
        "The judge may not be discriminating well — inspect per-item scores before trusting accuracy."
    )


def run(dry_run: bool = False, seed: int = 42) -> tuple[list[dict], dict]:
    generated = load_json(DATA_DIR / "generated.json")
    test = with_ids(load_json(DATA_DIR / "test.json"), "test_")
    test_by_id = {t["id"]: t for t in test}

    client = None
    if not dry_run:
        _load_dotenv()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY not set. Copy .env.example to .env or export the key.")
        client = Anthropic()

    # --- correct-reference scoring ---
    evaluations = []
    for i, item in enumerate(generated, 1):
        ref_item = test_by_id.get(item["id"])
        reference = ref_item["reply"] if ref_item else item.get("reference_reply", "")
        print(f"[{i}/{len(generated)}] judging correct pair: {item['id']}")
        scores = score_item(client, item, reference, dry_run)
        evaluations.append(
            {
                "id": item["id"],
                "category": item.get("category"),
                "subject": item.get("subject"),
                "reference_reply": reference,
                "generated_reply": item["generated_reply"],
                "intent_coverage": scores["intent_coverage"],
                "tone_fidelity": scores["tone_fidelity"],
                "correctness": scores["correctness"],
                "conciseness": scores["conciseness"],
                "accuracy_score": scores["accuracy_score"],
                "rouge_l_f1": scores["rouge_l_f1"],
                "pair_type": "correct",
            }
        )
        print(
            f"[{i}/{len(generated)}] accuracy={scores['accuracy_score']:.0f} "
            f"rouge_l={scores['rouge_l_f1']:.3f}"
        )

    # --- mismatched-reference validation ---
    ids = [e["id"] for e in evaluations]
    mismatched_rows = []
    used_fallback = False

    if len(ids) >= 2:
        mapping = derange_ids(ids, seed)
        print(f"validation derangement: {mapping}")
    else:
        mapping = {ids[0]: None} if ids else {}
        used_fallback = True
        print("validation: only 1 test item — using canned unrelated reference")

    for i, item in enumerate(generated, 1):
        if mapping.get(item["id"]) is None:
            wrong_ref = FALLBACK_WRONG_REPLY
            wrong_id = "fallback_unrelated"
        else:
            wrong_id = mapping[item["id"]]
            wrong_src = test_by_id.get(wrong_id) or next(
                e for e in evaluations if e["id"] == wrong_id
            )
            wrong_ref = (
                wrong_src["reply"]
                if "reply" in wrong_src
                else wrong_src["reference_reply"]
            )

        print(f"[{i}/{len(generated)}] judging mismatched pair: {item['id']} vs {wrong_id}")
        scores = score_item(client, item, wrong_ref, dry_run)
        mismatched_rows.append(
            {
                "id": item["id"],
                "mismatched_reference_id": wrong_id,
                "accuracy_score": scores["accuracy_score"],
                "rouge_l_f1": scores["rouge_l_f1"],
                "intent_coverage": scores["intent_coverage"],
                "tone_fidelity": scores["tone_fidelity"],
                "correctness": scores["correctness"],
                "conciseness": scores["conciseness"],
            }
        )
        print(f"[{i}/{len(generated)}] mismatched accuracy={scores['accuracy_score']:.0f}")

    avg_correct = sum(e["accuracy_score"] for e in evaluations) / len(evaluations)
    avg_mismatch = sum(r["accuracy_score"] for r in mismatched_rows) / len(mismatched_rows)
    avg_rouge_correct = sum(e["rouge_l_f1"] for e in evaluations) / len(evaluations)
    avg_rouge_mismatch = sum(r["rouge_l_f1"] for r in mismatched_rows) / len(mismatched_rows)
    gap = avg_correct - avg_mismatch

    validation = {
        "n": len(evaluations),
        "dry_run": dry_run,
        "used_fallback_wrong_reference": used_fallback,
        "avg_accuracy_correct": avg_correct,
        "avg_accuracy_mismatched": avg_mismatch,
        "accuracy_gap": gap,
        "avg_rouge_l_correct": avg_rouge_correct,
        "avg_rouge_l_mismatched": avg_rouge_mismatch,
        "note": (
            "accuracy_score (LLM judge, 0–100) is the primary metric. "
            "rouge_l_f1 is a coarse lexical secondary signal only."
        ),
        "interpretation": interpret_gap(gap, dry_run, len(evaluations)),
        "mismatched_pairs": mismatched_rows,
    }

    eval_path = DATA_DIR / "evaluations.json"
    val_path = DATA_DIR / "validation.json"
    eval_path.write_text(json.dumps(evaluations, indent=2) + "\n")
    val_path.write_text(json.dumps(validation, indent=2) + "\n")
    print(f"wrote {eval_path}")
    print(f"wrote {val_path}")
    print(
        f"avg accuracy correct={avg_correct:.1f} mismatched={avg_mismatch:.1f} "
        f"gap={gap:.1f}"
    )
    print(validation["interpretation"])
    return evaluations, validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="placeholder judge scores (no API); validation gap will be ~0",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed for ID derangement")
    args = parser.parse_args()
    run(dry_run=args.dry_run, seed=args.seed)


if __name__ == "__main__":
    main()
