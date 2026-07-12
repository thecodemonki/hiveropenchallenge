"""Generate an AI suggested reply for an inbound email.

Takes the inbound message, retrieved similar examples, and brand voice
guidelines, then calls the Anthropic API to produce a draft reply.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from anthropic import Anthropic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import incoming_text, top_k_similar  # noqa: E402
DATA_DIR = ROOT / "data"
MODEL = "claude-sonnet-4-5"


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


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def with_ids(items: list[dict], prefix: str) -> list[dict]:
    out = []
    for i, item in enumerate(items):
        row = dict(item)
        row["id"] = row.get("id", f"{prefix}{i}")
        out.append(row)
    return out


def load_brand_voice() -> str:
    return (DATA_DIR / "brand_voice.md").read_text().strip()


def build_system_prompt(brand_voice: str) -> str:
    return f"""You draft support email replies for a shared inbox team.

Be warm, direct, concise, and action-oriented.
Do not invent policies or facts not supported by the incoming email or the past examples.
Output only the final reply text — no markdown fences, no explanation.

Brand voice guide:
{brand_voice}
"""


def build_user_prompt(email: dict, retrieved: list[dict]) -> str:
    examples = []
    for i, ex in enumerate(retrieved, 1):
        examples.append(
            f"### Example {i} (id={ex['id']}, score={ex['score']:.3f})\n"
            f"Subject: {ex['subject']}\n"
            f"Incoming:\n{ex['body']}\n\n"
            f"Reply that was sent:\n{ex['reply']}"
        )
    examples_block = "\n\n".join(examples) if examples else "(no examples)"

    return f"""Past similar email/reply pairs:
{examples_block}

---
Incoming email to reply to:
Subject: {email['subject']}
From: {email.get('customer_name', 'Customer')}
Tier: {email.get('customer_tier', 'unknown')}

{email['body']}

Draft the reply now.
"""


def placeholder_reply(email: dict, retrieved: list[dict]) -> str:
    ids = ", ".join(str(r["id"]) for r in retrieved) or "none"
    return (
        f"[dry-run] Suggested reply for: {email['subject']}\n"
        f"Retrieved examples: {ids}"
    )


def generate_reply(
    client: Anthropic | None,
    email: dict,
    retrieved: list[dict],
    brand_voice: str,
    dry_run: bool,
) -> str:
    if dry_run:
        return placeholder_reply(email, retrieved)

    assert client is not None
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=build_system_prompt(brand_voice),
        messages=[{"role": "user", "content": build_user_prompt(email, retrieved)}],
    )
    return msg.content[0].text.strip()


def run(k: int = 3, dry_run: bool = False) -> list[dict]:
    train = with_ids(load_json(DATA_DIR / "train.json"), "train_")
    test = with_ids(load_json(DATA_DIR / "test.json"), "test_")
    brand_voice = load_brand_voice()

    client = None
    if not dry_run:
        _load_dotenv()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY not set. Copy .env.example to .env or export the key.")
        client = Anthropic()

    results = []
    for i, email in enumerate(test, 1):
        print(f"[{i}/{len(test)}] retrieving for: {email['subject']}")
        retrieved = top_k_similar(incoming_text(email), train, k=k)
        retrieved_ids = [r["id"] for r in retrieved]

        print(f"[{i}/{len(test)}] generating reply (retrieved={retrieved_ids})...")
        generated = generate_reply(client, email, retrieved, brand_voice, dry_run)

        results.append(
            {
                "id": email["id"],
                "category": email.get("category"),
                "subject": email["subject"],
                "body": email["body"],
                "reference_reply": email["reply"],
                "generated_reply": generated,
                "retrieved_ids": retrieved_ids,
                "retrieved_examples": [
                    {
                        "id": r["id"],
                        "score": r["score"],
                        "subject": r["subject"],
                        "category": r.get("category"),
                    }
                    for r in retrieved
                ],
            }
        )
        print(f"[{i}/{len(test)}] ok")

    out_path = DATA_DIR / "generated.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out_path} ({len(results)} replies)")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3, help="number of retrieved examples")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run retrieval normally; skip API and use placeholder replies",
    )
    args = parser.parse_args()
    run(k=args.k, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
