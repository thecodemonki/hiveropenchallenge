# TrustDraft

AI email suggested-reply system. Retrieves similar past replies, applies brand voice guidelines, and drafts a suggested response via the Anthropic API.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your ANTHROPIC_API_KEY
```

## Project layout

| Path | Purpose |
|------|---------|
| `src/generate_dataset.py` | Build synthetic email/reply dataset |
| `src/retrieval.py` | Find similar past email/reply pairs |
| `src/generate_reply.py` | Call Anthropic to draft a reply |
| `src/evaluate.py` | Score reply quality |
| `src/pipeline.py` | End-to-end orchestration |
| `app.py` | Streamlit demo UI |
| `data/brand_voice.md` | Brand tone/style guidelines |
