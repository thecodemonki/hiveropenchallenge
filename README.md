# TrustDraft

TrustDraft is a shared-inbox suggested-reply prototype. It retrieves similar past support tickets, drafts a reply with Claude, and evaluates that draft against the reply that was actually sent.

The project is built around a practical product question: not just whether an AI can draft a reply, but whether the reply is actually useful and accurate.

## What it does

TrustDraft runs a simple pipeline:

1. Generate a synthetic support-email dataset.
2. Retrieve similar past tickets with TF-IDF.
3. Draft a reply with Claude using few-shot examples.
4. Score the draft against the held-out reference reply.
5. Show the results in a Streamlit dashboard.

## Dataset

The dataset is synthetic, but it is designed to resemble a support inbox rather than a generic chat dataset. Each record includes:

- customer name
- subject
- body
- reply
- category
- customer tier
- sentiment

The categories cover common shared-inbox traffic:

- billing/refund
- bug report
- cancellation/churn
- feature request
- sales inquiry

That mix keeps the system grounded in realistic support workflows: money issues, product bugs, churn risk, feature asks, and sales conversations.

## Generation approach

For each incoming ticket, the system:

- finds the most similar historical tickets using TF-IDF over subject + body,
- passes those examples into Claude with a brand voice guide,
- asks for a warm, direct, concise reply.

I chose retrieval + few-shot prompting instead of fine-tuning because it is easier to inspect, faster to update, and much lighter to build in a short project window. If the tone or policy changes, you can update the prompt or the retrieved examples without retraining anything.

## Evaluation

The main challenge in this project is measuring quality.

Exact match is not a good metric for support replies, because a correct reply can be phrased many different ways. Instead, TrustDraft uses an LLM judge that scores each reply on:

- intent coverage
- tone fidelity
- correctness
- conciseness

Those four dimensions are combined into a 0–100 accuracy score.

ROUGE-L is also logged as a secondary lexical signal, but it is not the main metric.

## Validation

To check that the evaluator is meaningful, the system re-scores replies against deliberately wrong reference replies. If the metric is working, correct pairs should score higher than mismatched pairs.

In a real local run, the correct-pair average was 71.7, the mismatched average was 38.3, and the gap was 33.3. That suggests the judge is tracking reply quality rather than giving a constant score.

**Note:** `--dry-run` uses placeholder judge scores, so the gap is ~0 by construction and does not validate the metric. Checked-in `data/` files may reflect a dry-run unless you re-run the live pipeline.

## Testing

The repo includes lightweight smoke tests for the core pieces of the system. They verify:

- retrieval behavior,
- evaluation helpers,
- dry-run pipeline outputs.

`pytest tests/ -v` passes.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add ANTHROPIC_API_KEY to .env
```

Dry-run:

```bash
python src/pipeline.py --n 40 --dry-run
streamlit run app.py
```

Live run:

```bash
python src/pipeline.py --n 40
streamlit run app.py
```

## Repository layout

| Path | Purpose |
|---|---|
| `src/generate_dataset.py` | Builds the synthetic email dataset |
| `src/retrieval.py` | Finds similar past tickets |
| `src/generate_reply.py` | Drafts replies with Claude |
| `src/evaluate.py` | Scores replies and runs validation |
| `src/pipeline.py` | Runs the full workflow |
| `app.py` | Streamlit dashboard |
| `tests/` | Smoke tests |
| `data/brand_voice.md` | Tone guide |

## Why this is solid

The project is intentionally small, but it shows the right product thinking:

- retrieval instead of blind generation,
- a judge that measures actual reply quality,
- validation against mismatched references,
- tests for core behavior,
- a dashboard for review.

It is built to be understandable, testable, and easy to iterate on.

## AI usage

AI tools were used to speed up implementation and iteration. The design choices, evaluation strategy, and final framing were intentional and reviewed carefully.
