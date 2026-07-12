"""Generate a synthetic email/reply dataset for training and evaluation.

Creates paired examples of inbound emails and brand-aligned replies,
used as the corpus for retrieval and as ground truth for evaluation.
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

CATEGORIES = [
    "billing_refund",
    "bug_report",
    "cancellation_churn",
    "feature_request",
    "sales_inquiry",
]
TIERS = ["standard", "pro", "enterprise"]
SENTIMENTS = ["frustrated", "neutral", "positive", "confused", "urgent"]

FIXTURES = [
    {
        "customer_name": "Priya Shah",
        "subject": "Refund for double charge on invoice #4821",
        "body": (
            "Hi team,\n\nI was charged twice for our March invoice (#4821). "
            "One charge went through on the 1st and another on the 3rd for the same amount. "
            "Can you refund the duplicate and confirm once it's done?\n\nThanks,\nPriya"
        ),
        "reply": (
            "Hi Priya,\n\nYou're right — we see two charges for invoice #4821. "
            "I've issued a full refund for the duplicate ($249), which should land "
            "in 3–5 business days. I'll email you the refund confirmation once it posts.\n\n"
            "Sorry for the hassle.\n\nBest,\nAlex"
        ),
        "category": "billing_refund",
        "customer_tier": "pro",
        "sentiment": "frustrated",
    },
    {
        "customer_name": "Marcus Chen",
        "subject": "Export CSV keeps failing with 500 error",
        "body": (
            "Hello,\n\nWhenever I try to export contacts as CSV from Settings → Data, "
            "I get a 500 error after about 10 seconds. Happening on Chrome and Firefox. "
            "Account: acme-ops. Started yesterday afternoon.\n\nMarcus"
        ),
        "reply": (
            "Hi Marcus,\n\nThanks for the details — we've reproduced the 500 on CSV export "
            "for larger contact lists. Engineering has a fix queued for deploy today. "
            "In the meantime, try exporting in batches under 5k rows, or reply and I'll "
            "pull a CSV for you manually.\n\nI'll update you when the fix is live.\n\n— Jordan"
        ),
        "category": "bug_report",
        "customer_tier": "enterprise",
        "sentiment": "frustrated",
    },
    {
        "customer_name": "Elena Rossi",
        "subject": "Cancel our subscription",
        "body": (
            "Hi,\n\nWe need to cancel our Pro plan effective end of this billing cycle. "
            "We're consolidating tools and won't need the seats. Please confirm cancellation "
            "and whether we keep access until the period ends.\n\nElena"
        ),
        "reply": (
            "Hi Elena,\n\nI've scheduled your Pro plan to cancel at the end of your current "
            "billing period (April 30). You'll keep full access until then, and you won't "
            "be charged again. If you want an export of your data before then, say the word "
            "and I'll walk you through it.\n\nSorry to see you go — we're here if plans change.\n\n— Sam"
        ),
        "category": "cancellation_churn",
        "customer_tier": "pro",
        "sentiment": "neutral",
    },
    {
        "customer_name": "Devon Blake",
        "subject": "Can we get Slack alerts for new tickets?",
        "body": (
            "Hey,\n\nLove the product so far. One ask: can we push new ticket notifications "
            "into a Slack channel? We miss emails when we're heads-down. Is this on the roadmap "
            "or possible via API?\n\nDevon"
        ),
        "reply": (
            "Hi Devon,\n\nGreat timing — Slack ticket alerts are in beta. I can enable it on "
            "your workspace today: Settings → Integrations → Slack, then pick a channel. "
            "If you'd rather use the API, here's the webhook docs: docs.example.com/slack. "
            "Want me to turn on beta access for your account?\n\n— Riley"
        ),
        "category": "feature_request",
        "customer_tier": "standard",
        "sentiment": "positive",
    },
    {
        "customer_name": "Aisha Rahman",
        "subject": "Pricing for a 40-person team",
        "body": (
            "Hello,\n\nWe're evaluating your platform for a ~40-person customer support team. "
            "Could you share pricing for Pro vs Enterprise, and whether annual billing includes "
            "a discount? We'd also like a short demo next week if possible.\n\nAisha"
        ),
        "reply": (
            "Hi Aisha,\n\nHappy to help. For ~40 seats, Pro is usually the fit; Enterprise "
            "adds SSO, custom roles, and a dedicated CSM. Annual billing is 20% off monthly. "
            "I've held a 30-min demo slot Tuesday 2pm ET — reply with a better time if needed, "
            "and I'll send a calendar invite plus a one-pager comparing the plans.\n\n— Casey"
        ),
        "category": "sales_inquiry",
        "customer_tier": "standard",
        "sentiment": "positive",
    },
]


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


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group())


def _generate_pair(client: Anthropic, category: str, tier: str, sentiment: str) -> dict:
    prompt = f"""Generate one realistic shared-inbox support email pair as JSON only (no markdown).

Schema:
{{
  "customer_name": string,
  "subject": string,
  "body": string,   // the incoming customer email
  "reply": string,  // the reply an agent actually sent — warm, direct, specific next steps
  "category": "{category}",
  "customer_tier": "{tier}",
  "sentiment": "{sentiment}"
}}

Constraints:
- Product is a fictional B2B SaaS helpdesk called "Hiver Desk"
- category must be exactly "{category}"
- customer_tier must be exactly "{tier}"
- sentiment must be exactly "{sentiment}"
- body and reply should be realistic multi-sentence emails
- reply must include a concrete next step (refund timeline, fix ETA, cancel confirmation, etc.)
"""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    pair = _parse_json_object(msg.content[0].text)
    pair["category"] = category
    pair["customer_tier"] = tier
    pair["sentiment"] = sentiment
    return pair


def generate_pairs(n: int, seed: int, dry_run: bool) -> list[dict]:
    rng = random.Random(seed)
    if dry_run:
        pairs = [dict(p) for p in FIXTURES]
        print(f"[dry-run] using {len(pairs)} hardcoded fixture pairs")
        return pairs

    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set. Copy .env.example to .env or export the key.")

    client = Anthropic()
    pairs: list[dict] = []
    for i in range(n):
        category = CATEGORIES[i % len(CATEGORIES)]
        tier = rng.choice(TIERS)
        sentiment = rng.choice(SENTIMENTS)
        print(f"[{i + 1}/{n}] generating {category} ({tier}, {sentiment})...")
        pair = _generate_pair(client, category, tier, sentiment)
        pairs.append(pair)
        print(f"[{i + 1}/{n}] ok — {pair.get('subject', '(no subject)')}")
    return pairs


def split_and_write(pairs: list[dict], test_frac: float, seed: int) -> None:
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    n_test = max(1, int(round(len(shuffled) * test_frac))) if len(shuffled) > 1 else 0
    if len(shuffled) == 1:
        n_test = 0
    test = shuffled[:n_test]
    train = shuffled[n_test:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in [
        ("dataset.json", shuffled),
        ("train.json", train),
        ("test.json", test),
    ]:
        path = DATA_DIR / name
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"wrote {path} ({len(data)} pairs)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=40, help="number of pairs to generate")
    parser.add_argument("--test-frac", type=float, default=0.25, help="fraction held out as test")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="use 5 hardcoded fixtures (no API calls)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for shuffle/tiers")
    args = parser.parse_args()

    pairs = generate_pairs(args.n, args.seed, args.dry_run)
    split_and_write(pairs, args.test_frac, args.seed)


if __name__ == "__main__":
    main()
