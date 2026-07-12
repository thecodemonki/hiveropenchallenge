# TrustDraft

Shared-inbox suggested-reply prototype: retrieve similar past tickets, draft a reply with Claude, then score that draft against the held-out reference reply.

Pipeline: `generate_dataset` → `generate_reply` (TF-IDF retrieval + few-shot) → `evaluate` (LLM judge + lexical secondary + mismatched-reference check). `app.py` is a Streamlit dashboard over the same outputs.

## Dataset

Synthetic `(incoming_email, reply_that_was_sent)` pairs meant to look like a support archive, not a generic chat corpus. Fields: `customer_name`, `subject`, `body`, `reply`, `category`, `customer_tier` (`standard` / `pro` / `enterprise`), `sentiment`.

Categories (balanced across generation):

| Category | Typical ticket |
|----------|----------------|
| `billing_refund` | Charges, invoices, refunds |
| `bug_report` | Product failures with repro detail |
| `cancellation_churn` | Cancel / downgrade requests |
| `feature_request` | Product asks / roadmap questions |
| `sales_inquiry` | Pricing, seats, demos |

That mix is representative of a real queue: money, defects, churn, product, and sales, with tier and sentiment so retrieval is not dominated by one template. Live generation uses Claude (`claude-sonnet-4-5`). `--dry-run` uses five fixtures and ignores `--n` for content (still useful for wiring tests). Seeded shuffle → `data/train.json` (retrieval), `data/test.json` (held out), `data/dataset.json` (all).

**Limitation:** synthetic data understates real messiness (escalations, partial context, policy edge cases). It is enough to exercise retrieval, prompting, and the judge loop—not a claim of production coverage.

## Generation approach

For each test email:

1. Build TF-IDF over train **incoming** text only (`subject` + `body`), not replies.
2. Take top-*k* by cosine similarity (`src/retrieval.py`).
3. Prompt Claude with brand voice (`data/brand_voice.md`), the retrieved email/reply pairs, and the new incoming email (`src/generate_reply.py`). One API call per test email.

The model is instructed to stay warm, direct, and action-oriented, and not to invent policies unsupported by the email or examples.

### Why retrieval + few-shot instead of fine-tuning

| | Retrieval + few-shot | Fine-tuning |
|--|----------------------|-------------|
| Data need | Small labeled archive | Larger, cleaner training set |
| Update path | Add tickets / edit brand voice | Retrain / redeploy |
| Inspectability | Retrieved neighbors are visible | Behavior is baked into weights |
| Cost | Inference (+ retrieval) | Train + eval + ongoing refresh |

**Trade-off:** quality tracks retrieval. Thin categories or odd wording get weak neighbors; context window caps how many examples you can show. Fine-tuning can lock in house style more tightly at scale, but is slower to change when refund rules or tone guidelines shift. For an inbox that changes often, prompting over retrieved tickets is the lower-ops control surface.

## Evaluation

### Why not exact match

Agent replies that are correct often paraphrase the reference. Exact string match (and treating ROUGE as the headline number) rewards copying and punishes valid wording. Exact match is the wrong primary metric here.

### LLM judge (primary)

Each generated reply is scored against **that ticket’s real reference** on four 1–5 dimensions:

| Dimension | Measures |
|-----------|----------|
| `intent_coverage` | Addresses the customer’s ask |
| `tone_fidelity` | Warm / direct / action-oriented like the reference |
| `correctness` | Does not invent unsupported policy or facts |
| `conciseness` | Tight without dropping needed next steps |

`accuracy_score` = sum of the four / 20 × 100 (range 0–100).

**Limitation:** the judge is itself an LLM. It can be noisy or biased; the corruption test below is there to catch “always scores ~high” failure modes.

### Lexical overlap (secondary only)

ROUGE-L (token LCS F1, implemented in-repo) is logged as a coarse overlap signal. It is **not** the accuracy number.

### Validation / corruption test

After correct-reference scoring, every generated reply is re-scored against a **wrong** reference: test IDs are deranged so no ID maps to itself (≥2 test items). With a single test item, a canned unrelated reply is used instead.

Report:

- avg `accuracy_score` on correct pairs  
- avg `accuracy_score` on mismatched pairs  
- **gap** = correct − mismatched  

A large positive gap means the judge tracks reply–reference fit. A near-zero gap means it may be emitting a plausible constant regardless of input—do not trust the accuracy number until that gap is real.

## Dry-run limitations

`--dry-run` skips Anthropic calls:

- dataset → 5 fixtures  
- replies → deterministic placeholders  
- judge → constant dimension scores  

So validation gap is **~0 by construction**. That confirms the pipeline wiring; it does **not** validate the metric or reply quality. Checked-in `data/*.json` may be from dry-run—treat numbers there as fixtures unless you re-ran without `--dry-run`.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set ANTHROPIC_API_KEY for live runs
```

Dry-run (no API):

```bash
python src/pipeline.py --n 40 --dry-run
streamlit run app.py
```

Live (uses API credits):

```bash
python src/pipeline.py --n 40
streamlit run app.py
```

Steps individually: `src/generate_dataset.py`, `src/generate_reply.py` (`--k`), `src/evaluate.py`.

Outputs: `data/train.json`, `test.json`, `dataset.json`, `generated.json`, `evaluations.json`, `validation.json`.

## AI tool usage

Built in Cursor with AI-assisted implementation of the modules above from staged requirements. Method choices (TF-IDF few-shot vs fine-tune, LLM judge + corruption check, ROUGE as secondary) follow the challenge spec. Review the code and re-run the pipeline before treating any `data/` scores as live results.

## Layout

| Path | Role |
|------|------|
| `src/generate_dataset.py` | Synthetic archive |
| `src/retrieval.py` | TF-IDF top-*k* |
| `src/generate_reply.py` | Few-shot Claude drafts |
| `src/evaluate.py` | Judge + ROUGE-L + corruption test |
| `src/pipeline.py` | Runs the three stages |
| `app.py` | Dashboard |
| `data/brand_voice.md` | Tone guide |
| `requirements.txt` | anthropic, streamlit, pandas, scikit-learn |
