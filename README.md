# TrustDraft

Suggested-reply system for a shared support inbox. Given an incoming email, it retrieves similar past tickets, drafts a reply with Claude grounded in those examples and a brand-voice guide, then scores the draft against the real reference reply with an LLM judge (plus a secondary lexical overlap signal).

## 1. Dataset

The corpus is a synthetic shared-inbox archive of `(incoming_email, reply_that_was_sent)` pairs. Each pair includes `customer_name`, `subject`, `body`, `reply`, `category`, `customer_tier` (`standard` / `pro` / `enterprise`), and `sentiment`.

Categories mirror common support work:

- `billing_refund`
- `bug_report`
- `cancellation_churn`
- `feature_request`
- `sales_inquiry`

That mix is representative of a real helpdesk queue: money issues, product defects, churn risk, product asks, and inbound sales — with tier and sentiment variation so retrieval and replies are not one-dimensional. Live generation uses Claude (`claude-sonnet-4-5`); `--dry-run` swaps in five hardcoded fixtures so the rest of the pipeline can be tested without API spend. Pairs are shuffled (seeded) into `data/train.json` (retrieval corpus), `data/test.json` (held out), and `data/dataset.json` (all pairs).

## 2. Why RAG / few-shot instead of fine-tuning

TrustDraft retrieves the top-*k* train examples by TF-IDF cosine similarity over **incoming** subject+body only, then passes those past email/reply pairs into the prompt as few-shot context (plus `data/brand_voice.md`).

**Why this over fine-tuning**

- Works with a small archive; no training loop or GPU job.
- Updating behavior means editing brand voice or adding tickets — not re-training.
- Each draft stays inspectable: you can see which past replies were retrieved.

**Trade-off**

- Quality depends on retrieval and prompt context limits; rare ticket types may not find good neighbors.
- Fine-tuning could absorb house style more tightly at scale, but costs more to train, evaluate, and refresh when policies change. For a support inbox that changes weekly, retrieval + prompting is the cheaper control surface.

## 3. Accuracy is not exact match

A good agent reply can use different words than the historical reference and still be correct. Exact string match (or treating ROUGE as the headline metric) would punish valid paraphrases.

Primary metric: an **LLM judge** scores each generated reply against that ticket’s **real** reference reply on four 1–5 dimensions:

| Dimension | What it measures |
|-----------|------------------|
| `intent_coverage` | Did the reply address what the customer needed? |
| `tone_fidelity` | Warm, direct, action-oriented in the same spirit as the reference? |
| `correctness` | Avoid inventing policies/facts the reference wouldn’t support? |
| `conciseness` | Tight without dropping needed next steps? |

Those four scores sum to a **0–100 `accuracy_score`** (`sum / 20 × 100`).

**ROUGE-L** (token LCS F1, implemented in-repo with no extra dependency) is logged only as a coarse lexical secondary signal. It is **not** the accuracy number.

## 4. Validation / corruption test

After scoring every generated reply against the correct reference, the evaluator **re-scores** each reply against a **wrong** reference: test IDs are deranged so no ID maps to itself (with a single test item, a canned unrelated reply is used instead).

Compare:

- average `accuracy_score` on **correct** pairs
- average `accuracy_score` on **mismatched** pairs

A **large positive gap** means the judge tracks reply–reference fit rather than emitting a plausible constant. A near-zero gap is a warning that the metric may not discriminate.

**Dry-run note:** `--dry-run` uses a constant placeholder judge score, so the validation gap is **~0 by construction**. That is expected and documented in `data/validation.json` — it does **not** prove the metric works. Re-run without `--dry-run` (and with enough test emails for a real derangement) to validate discrimination.

## 5. How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set ANTHROPIC_API_KEY for live runs
```

**Dry-run (no API calls)** — fixtures + placeholder replies/scores:

```bash
python src/pipeline.py --n 40 --dry-run
streamlit run app.py
```

**Live run** (uses Anthropic; costs tokens):

```bash
python src/pipeline.py --n 40
streamlit run app.py
```

Or step by step:

```bash
python src/generate_dataset.py --n 40          # or --dry-run
python src/generate_reply.py                  # or --dry-run --k 3
python src/evaluate.py                        # or --dry-run
```

Outputs land in `data/`: `train.json`, `test.json`, `dataset.json`, `generated.json`, `evaluations.json`, `validation.json`.

Dashboard (`app.py`): sidebar controls dataset size and dry-run, runs `pipeline.py`, then shows summary metrics, charts, validation interpretation, a filterable table, and a per-ticket scorecard.

## 6. AI tools used

This project was built with Cursor (AI-assisted coding): scaffolding the package layout, implementing retrieval / generation / evaluation / pipeline / Streamlit UI from staged prompts, and iterating on CLI flags and dry-run behavior. Design choices (TF-IDF few-shot over fine-tuning, LLM-judge accuracy with a mismatched-reference check, ROUGE-L as secondary only) were specified in the challenge prompts and implemented accordingly. Always re-run the pipeline yourself before treating numbers in `data/` as live evaluation results — checked-in JSON may reflect a prior `--dry-run`.

## Project layout

| Path | Purpose |
|------|---------|
| `src/generate_dataset.py` | Synthetic email/reply dataset |
| `src/retrieval.py` | TF-IDF top-*k* over incoming email text |
| `src/generate_reply.py` | Few-shot Claude suggested replies |
| `src/evaluate.py` | LLM-judge accuracy + ROUGE-L + validation |
| `src/pipeline.py` | Runs the three stages in order |
| `app.py` | Streamlit dashboard |
| `data/brand_voice.md` | Brand tone/style guidelines |
| `requirements.txt` | `anthropic`, `streamlit`, `pandas`, `scikit-learn` |
