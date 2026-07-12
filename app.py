"""Streamlit product demo for the TrustDraft suggested-reply pipeline.

Run generate → reply → evaluate, then review validation proof and ticket quality.
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PIPELINE = ROOT / "src" / "pipeline.py"

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');

:root {
    --td-ink: #0f172a;
    --td-muted: #64748b;
    --td-line: #e2e8f0;
    --td-soft: #f8fafc;
    --td-accent: #0f766e;
    --td-accent-soft: #ccfbf1;
    --td-accent-ink: #115e59;
}

html, body, [class*="css"] {
    font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
}

header[data-testid="stHeader"] { background: transparent; }
div[data-testid="stDecoration"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1480px;
}

section[data-testid="stSidebar"] {
    background: #f1f5f9;
    border-right: 1px solid var(--td-line);
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
}
.sidebar-brand {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--td-accent);
    margin: 0 0 0.3rem 0;
}
.sidebar-title {
    font-size: 1.12rem;
    font-weight: 700;
    color: var(--td-ink);
    letter-spacing: -0.02em;
    margin: 0 0 0.2rem 0;
}
.sidebar-help {
    color: var(--td-muted);
    font-size: 0.84rem;
    line-height: 1.4;
    margin: 0 0 1rem 0;
}
.stButton > button[kind="primary"] {
    background: var(--td-accent) !important;
    border-color: var(--td-accent) !important;
    color: #fff !important;
    font-weight: 600;
    border-radius: 8px;
}
.stButton > button[kind="primary"]:hover {
    background: var(--td-accent-ink) !important;
    border-color: var(--td-accent-ink) !important;
}

.hero {
    padding: 0.1rem 0 1rem 0;
    border-bottom: 1px solid var(--td-line);
    margin-bottom: 1.1rem;
}
.hero-top {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.75rem 1rem;
    margin-bottom: 0.35rem;
}
.mode-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    border: 1px solid var(--td-line);
    white-space: nowrap;
}
.mode-badge.live {
    background: var(--td-accent-soft);
    border-color: #99f6e4;
    color: var(--td-accent-ink);
}
.mode-badge.dry {
    background: #f5f5f4;
    border-color: #d6d3d1;
    color: #57534e;
}
.mode-badge .dot {
    width: 0.45rem;
    height: 0.45rem;
    border-radius: 999px;
    background: currentColor;
}
.reviewer-guide {
    background: #fff;
    border: 1px solid var(--td-line);
    border-left: 4px solid var(--td-accent);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin: 0.95rem 0 0 0;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
}
.reviewer-guide .rg-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--td-accent-ink);
    margin: 0 0 0.35rem 0;
}
.reviewer-guide p {
    margin: 0;
    color: #334155;
    font-size: 0.92rem;
    line-height: 1.5;
}
.reviewer-guide ol {
    margin: 0.45rem 0 0 1.1rem;
    padding: 0;
    color: #475569;
    font-size: 0.88rem;
    line-height: 1.45;
}
.reviewer-guide li { margin: 0.2rem 0; }
.story-step {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--td-accent);
    margin: 0 0 0.15rem 0;
}
.hero-eyebrow {
    color: var(--td-accent);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 0 0 0.45rem 0;
}
.hero-title {
    font-size: 2.35rem;
    font-weight: 700;
    letter-spacing: -0.035em;
    color: var(--td-ink);
    margin: 0 0 0.45rem 0;
    line-height: 1.12;
}
.hero-subtitle {
    color: #475569;
    font-size: 1.05rem;
    margin: 0 0 1.05rem 0;
    line-height: 1.55;
    max-width: 40rem;
}
.process-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.45rem;
}
.process-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #fff;
    border: 1px solid var(--td-line);
    color: var(--td-ink);
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.38rem 0.7rem;
    border-radius: 999px;
}
.process-pill span.num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.15rem;
    height: 1.15rem;
    border-radius: 999px;
    background: var(--td-accent);
    color: #fff;
    font-size: 0.66rem;
    font-weight: 700;
}
.process-arrow { color: #94a3b8; font-size: 0.85rem; }

.section-head {
    margin: 2.2rem 0 0.95rem 0;
    padding-top: 0.85rem;
    border-top: 1px solid var(--td-line);
}
.section-label {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--td-ink);
    margin: 0 0 0.28rem 0;
}
.section-label::before {
    content: "";
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 2px;
    background: var(--td-accent);
    flex: 0 0 auto;
}
.section-lead {
    color: var(--td-muted);
    font-size: 0.9rem;
    margin: 0;
    line-height: 1.45;
    max-width: 42rem;
}

div[data-testid="stMetric"] {
    background: #fff;
    border: 1px solid var(--td-line);
    border-radius: 12px;
    padding: 1rem 1.05rem 0.9rem 1.05rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
div[data-testid="stMetric"] label {
    color: var(--td-muted) !important;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--td-ink) !important;
    font-weight: 700 !important;
}

.proof-card {
    background: #0b1220;
    color: #e2e8f0;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.4rem 1.45rem 1.25rem 1.45rem;
    margin: 0.1rem 0 0.85rem 0;
    box-shadow: 0 14px 32px rgba(15, 23, 42, 0.16);
}
.proof-card.proof-success {
    background: #042f2e;
    border-color: #0f766e;
    box-shadow: 0 14px 32px rgba(15, 118, 110, 0.18);
}
.proof-card.proof-dry {
    background: #111827;
    border-color: #374151;
    box-shadow: none;
}
.proof-kicker {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 0 0 0.35rem 0;
}
.proof-card.proof-success .proof-kicker { color: #5eead4; }
.proof-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0 0 0.35rem 0;
    letter-spacing: -0.02em;
}
.proof-lead {
    color: #94a3b8;
    font-size: 0.9rem;
    margin: 0 0 1.15rem 0;
    line-height: 1.5;
    max-width: 42rem;
}
.compare-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1.15fr;
    gap: 0.75rem;
}
@media (max-width: 900px) {
    .compare-grid { grid-template-columns: 1fr; }
}
.compare-cell {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 11px;
    padding: 0.95rem 1rem;
}
.compare-cell.gap-cell {
    background: rgba(15, 118, 110, 0.16);
    border-color: rgba(45, 212, 191, 0.35);
}
.compare-label {
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 0 0 0.35rem 0;
    font-weight: 650;
}
.compare-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.03em;
    margin: 0;
    line-height: 1.1;
}
.compare-cell.gap-cell .compare-value {
    font-size: 2.2rem;
    color: #99f6e4;
}
.compare-sub {
    font-size: 0.76rem;
    color: #94a3b8;
    margin: 0.4rem 0 0 0;
    line-height: 1.35;
}
.status-chip {
    display: inline-block;
    margin-top: 0.9rem;
    font-size: 0.76rem;
    font-weight: 600;
    padding: 0.32rem 0.7rem;
    border-radius: 999px;
    border: 1px solid transparent;
}
.status-chip.ok {
    color: #99f6e4;
    background: rgba(15, 118, 110, 0.28);
    border-color: rgba(45, 212, 191, 0.4);
}
.status-chip.warn {
    color: #e7e5e4;
    background: rgba(120, 113, 108, 0.22);
    border-color: rgba(168, 162, 158, 0.35);
}
.status-chip.neutral {
    color: #cbd5e1;
    background: rgba(148, 163, 184, 0.12);
    border-color: rgba(148, 163, 184, 0.3);
}
.dry-run-inline {
    margin-top: 0.85rem;
    font-size: 0.8rem;
    color: #a8a29e;
    background: rgba(68, 64, 60, 0.35);
    border: 1px solid rgba(168, 162, 158, 0.25);
    border-radius: 8px;
    padding: 0.55rem 0.75rem;
    line-height: 1.4;
}
.interpretation {
    background: #fff;
    border: 1px solid var(--td-line);
    border-left: 4px solid var(--td-accent);
    border-radius: 10px;
    padding: 0.95rem 1.05rem;
    color: #334155;
    font-size: 0.92rem;
    line-height: 1.55;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.interpretation.interp-success {
    background: #f0fdfa;
    border-color: #99f6e4;
}
.interpretation .interp-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--td-muted);
    margin-bottom: 0.35rem;
}
.interpretation.interp-success .interp-label { color: var(--td-accent-ink); }
.interpretation .note {
    color: var(--td-muted);
    font-size: 0.8rem;
    margin-top: 0.5rem;
}

.panel-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--td-ink);
    margin: 0;
}
.panel-meta {
    font-size: 0.76rem;
    color: var(--td-muted);
    margin: 0 0 0.7rem 0;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #fff;
    border-radius: 12px !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.score-strip {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.6rem;
    margin: 0.15rem 0 1.05rem 0;
}
@media (max-width: 900px) {
    .score-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
.score-chip {
    background: #fff;
    border: 1px solid var(--td-line);
    border-radius: 11px;
    padding: 0.7rem 0.8rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
}
.score-chip.total {
    background: var(--td-accent);
    border-color: var(--td-accent);
}
.score-chip .slabel {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--td-muted);
    margin: 0 0 0.2rem 0;
}
.score-chip.total .slabel { color: rgba(255,255,255,0.78); }
.score-chip .svalue {
    font-size: 1.28rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--td-ink);
    margin: 0;
    line-height: 1.15;
}
.score-chip.total .svalue { color: #fff; }
.score-chip .sunit {
    font-size: 0.74rem;
    color: #94a3b8;
    font-weight: 500;
}
.mail-card-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    margin-bottom: 0.3rem;
}
.mail-badge {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 0.18rem 0.5rem;
    border-radius: 999px;
    border: 1px solid var(--td-line);
    color: #475569;
    background: var(--td-soft);
}
.mail-badge.gen {
    background: var(--td-accent-soft);
    border-color: #99f6e4;
    color: var(--td-accent-ink);
}
.mail-badge.ref {
    background: #ecfdf5;
    border-color: #a7f3d0;
    color: #047857;
}
.retrieved-panel {
    background: #fff;
    border: 1px solid var(--td-line);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin-top: 1rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
}
.retrieved-panel .rp-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--td-muted);
    margin: 0 0 0.55rem 0;
}
.retrieved-row {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.35rem 0.75rem;
    padding: 0.42rem 0;
    border-top: 1px solid #eef2f7;
    font-size: 0.86rem;
    color: #334155;
}
.retrieved-row:first-of-type { border-top: none; padding-top: 0; }
.retrieved-row .rid {
    font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace;
    font-size: 0.76rem;
    color: var(--td-muted);
}
.retrieved-row .rscore {
    font-size: 0.76rem;
    color: var(--td-accent-ink);
    font-weight: 600;
    margin-left: auto;
}
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--td-muted) !important;
}
</style>

"""

def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def run_pipeline(n: int, dry_run: bool) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(PIPELINE), "--n", str(n)]
    if dry_run:
        cmd.append("--dry-run")
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def section(label: str, lead: str = "") -> None:
    lead_html = f'<p class="section-lead">{lead}</p>' if lead else ""
    st.markdown(
        f'<div class="section-head"><div class="section-label">{label}</div>{lead_html}</div>',
        unsafe_allow_html=True,
    )


def category_chart(df: pd.DataFrame):
    by_cat = (
        df.groupby("category", as_index=False)["accuracy_score"]
        .mean()
        .rename(columns={"accuracy_score": "avg_accuracy"})
    )
    chart = (
        alt.Chart(by_cat)
        .mark_bar(color="#0f766e", cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "category:N",
                title="Category",
                sort="-y",
                axis=alt.Axis(labelAngle=-25, labelLimit=140),
            ),
            y=alt.Y(
                "avg_accuracy:Q",
                title="Average accuracy (0–100)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("avg_accuracy:Q", title="Avg accuracy", format=".1f"),
            ],
        )
        .properties(height=280)
        .configure_axis(labelFontSize=12, titleFontSize=12, grid=True)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, width="stretch")


def distribution_chart(df: pd.DataFrame):
    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0–20", "21–40", "41–60", "61–80", "81–100"]
    bucketed = pd.cut(
        df["accuracy_score"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    hist = (
        bucketed.value_counts()
        .reindex(labels, fill_value=0)
        .rename_axis("bucket")
        .reset_index(name="count")
    )
    chart = (
        alt.Chart(hist)
        .mark_bar(color="#115e59", cornerRadiusEnd=3)
        .encode(
            x=alt.X("bucket:N", title="Accuracy score range", sort=labels),
            y=alt.Y("count:Q", title="Number of tickets"),
            tooltip=[
                alt.Tooltip("bucket:N", title="Range"),
                alt.Tooltip("count:Q", title="Tickets"),
            ],
        )
        .properties(height=280)
        .configure_axis(labelFontSize=12, titleFontSize=12)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, width="stretch")


def render_hero(dry_run: bool | None = None) -> None:
    if dry_run is True:
        mode = (
            '<span class="mode-badge dry"><span class="dot"></span>'
            "Dry-run results</span>"
        )
    elif dry_run is False:
        mode = (
            '<span class="mode-badge live"><span class="dot"></span>'
            "Live API results</span>"
        )
    else:
        mode = ""

    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-top">
            <div>
              <p class="hero-eyebrow">Hiring-manager demo</p>
              <h1 class="hero-title">TrustDraft</h1>
            </div>
            {mode}
          </div>
          <p class="hero-subtitle">
            Suggested support replies with a measurable quality check —
            not just generation.
          </p>
          <div class="process-row">
            <div class="process-pill"><span class="num">1</span>Retrieve</div>
            <span class="process-arrow">→</span>
            <div class="process-pill"><span class="num">2</span>Draft</div>
            <span class="process-arrow">→</span>
            <div class="process-pill"><span class="num">3</span>Validate</div>
          </div>
          <div class="reviewer-guide">
            <p class="rg-label">What to look at</p>
            <p>
              This app shows an AI drafting replies from similar past tickets,
              then <strong>stress-tests the score</strong> by comparing correct vs
              deliberately wrong references. Start with the validation gap below;
              then open a ticket to see the draft side-by-side with the real reply.
            </p>
            <ol>
              <li><strong>Validation gap</strong> — does accuracy fall on mismatched references?</li>
              <li><strong>Ticket review</strong> — is the draft useful vs. what was sent?</li>
              <li><strong>Mode badge</strong> — dry-run uses placeholders; live uses Claude.</li>
            </ol>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_validation_proof(validation: dict) -> None:
    gap = float(validation["accuracy_gap"])
    correct = float(validation["avg_accuracy_correct"])
    mismatched = float(validation["avg_accuracy_mismatched"])
    interpretation = validation.get("interpretation", "")
    dry_run = bool(validation.get("dry_run"))
    meaningful = (not dry_run) and gap >= 15

    if dry_run:
        card_class = "proof-card proof-dry"
        status = (
            '<span class="status-chip warn">'
            "Dry-run — placeholder judge; gap ~0 by design"
            "</span>"
        )
        dry_banner = (
            '<div class="dry-run-inline">'
            "<strong>Dry-run vs live:</strong> scores here are constants, so they cannot "
            "prove discrimination. Uncheck Dry-run in the sidebar and re-run for a real gap."
            "</div>"
        )
    elif meaningful:
        card_class = "proof-card proof-success"
        status = (
            '<span class="status-chip ok">'
            "Live run — gap shows the judge discriminates"
            "</span>"
        )
        dry_banner = ""
    else:
        card_class = "proof-card"
        status = (
            '<span class="status-chip neutral">'
            "Live run — gap is small; inspect before trusting"
            "</span>"
        )
        dry_banner = ""

    interp_class = "interpretation interp-success" if meaningful else "interpretation"

    st.markdown(
        f"""
        <div class="{card_class}">
          <p class="proof-kicker">01 · Product insight</p>
          <p class="proof-title">Correct vs mismatched references</p>
          <p class="proof-lead">
            Same drafts, wrong references on purpose. A real metric should score
            correct pairs higher.
          </p>
          <div class="compare-grid">
            <div class="compare-cell">
              <p class="compare-label">Correct-pair average</p>
              <p class="compare-value">{correct:.1f}</p>
              <p class="compare-sub">True reference</p>
            </div>
            <div class="compare-cell">
              <p class="compare-label">Mismatched-pair average</p>
              <p class="compare-value">{mismatched:.1f}</p>
              <p class="compare-sub">Wrong reference</p>
            </div>
            <div class="compare-cell gap-cell">
              <p class="compare-label">Gap</p>
              <p class="compare-value">{gap:.1f}</p>
              <p class="compare-sub">What to believe</p>
            </div>
          </div>
          {status}
          {dry_banner}
        </div>
        <div class="{interp_class}">
          <div class="interp-label">Plain English</div>
          <div>{interpretation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ticket_review(df: pd.DataFrame, gen_by_id: dict) -> None:
    st.markdown('<p class="story-step">02 · See a draft</p>', unsafe_allow_html=True)
    section(
        "Ticket review",
        "Incoming · model draft · what the agent sent.",
    )

    id_to_label = {}
    for _, r in df.iterrows():
        subj = str(r.get("subject") or "").strip()
        tid = str(r["id"])
        id_to_label[tid] = f"{tid} — {subj}" if subj else tid

    ticket_id = st.selectbox(
        "Ticket",
        list(id_to_label.keys()),
        format_func=lambda i: id_to_label[i],
        label_visibility="collapsed",
    )
    row = df[df["id"] == ticket_id].iloc[0]
    gen = gen_by_id.get(ticket_id, {})

    incoming_body = gen.get("body") or ""
    incoming_subject = row.get("subject") or gen.get("subject", "")
    category = row.get("category") or gen.get("category") or ""

    # Compact scorecard first — glanceable before reading prose
    st.markdown(
        f"""
        <div class="score-strip">
          <div class="score-chip">
            <p class="slabel">Intent</p>
            <p class="svalue">{int(row['intent_coverage'])}<span class="sunit"> / 5</span></p>
          </div>
          <div class="score-chip">
            <p class="slabel">Tone</p>
            <p class="svalue">{int(row['tone_fidelity'])}<span class="sunit"> / 5</span></p>
          </div>
          <div class="score-chip">
            <p class="slabel">Correctness</p>
            <p class="svalue">{int(row['correctness'])}<span class="sunit"> / 5</span></p>
          </div>
          <div class="score-chip">
            <p class="slabel">Conciseness</p>
            <p class="svalue">{int(row['conciseness'])}<span class="sunit"> / 5</span></p>
          </div>
          <div class="score-chip total">
            <p class="slabel">Accuracy</p>
            <p class="svalue">{row['accuracy_score']:.0f}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    d1, d2, d3 = st.columns(3, gap="medium")
    with d1:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="mail-card-head">
                  <div class="panel-title">Incoming email</div>
                  <span class="mail-badge">Customer</span>
                </div>
                <div class="panel-meta">{html.escape(str(category))} · {html.escape(str(ticket_id))}</div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(f"**Subject:** {incoming_subject}")
            st.text_area(
                "Incoming body",
                incoming_body or "(body not in generated.json)",
                height=270,
                disabled=True,
                label_visibility="collapsed",
            )
    with d2:
        with st.container(border=True):
            st.markdown(
                """
                <div class="mail-card-head">
                  <div class="panel-title">Generated reply</div>
                  <span class="mail-badge gen">Model</span>
                </div>
                <div class="panel-meta">Suggested draft under evaluation</div>
                """,
                unsafe_allow_html=True,
            )
            st.text_area(
                "Generated reply",
                row["generated_reply"],
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )
    with d3:
        with st.container(border=True):
            st.markdown(
                """
                <div class="mail-card-head">
                  <div class="panel-title">Reference reply</div>
                  <span class="mail-badge ref">Sent</span>
                </div>
                <div class="panel-meta">Ground-truth agent reply</div>
                """,
                unsafe_allow_html=True,
            )
            st.text_area(
                "Reference reply",
                row["reference_reply"],
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )

    retrieved = gen.get("retrieved_examples") or []
    if retrieved:
        rows_html = []
        for ex in retrieved:
            score = ex.get("score")
            score_txt = f"sim {score:.3f}" if isinstance(score, (int, float)) else ""
            rows_html.append(
                "<div class='retrieved-row'>"
                f"<span class='rid'>{html.escape(str(ex.get('id', '')))}</span>"
                f"<span>{html.escape(str(ex.get('subject', '')))}</span>"
                f"<span class='rid'>{html.escape(str(ex.get('category', '')))}</span>"
                f"<span class='rscore'>{html.escape(score_txt)}</span>"
                "</div>"
            )
        st.markdown(
            "<div class='retrieved-panel'>"
            "<p class='rp-title'>Retrieved few-shot examples</p>"
            + "".join(rows_html)
            + "</div>",
            unsafe_allow_html=True,
        )
    elif gen.get("retrieved_ids"):
        ids = ", ".join(str(i) for i in gen["retrieved_ids"])
        st.caption(f"Retrieved example ids: {ids}")


def main() -> None:
    st.set_page_config(
        page_title="TrustDraft",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            """
            <p class="sidebar-brand">TrustDraft</p>
            <p class="sidebar-title">Run pipeline</p>
            <p class="sidebar-help">Dry-run is safe for demos. Live needs an API key.</p>
            """,
            unsafe_allow_html=True,
        )
        n = st.slider("Dataset size", min_value=5, max_value=80, value=40, step=5)
        dry_run = st.checkbox("Dry-run (no API calls)", value=True)
        if dry_run:
            st.info("Dry-run: fixtures + placeholder judge. Gap ≈ 0.")
        else:
            st.warning("Live: calls Claude. Requires ANTHROPIC_API_KEY.")
        run_clicked = st.button("Run full pipeline", type="primary", width="stretch")

        if run_clicked:
            with st.spinner("Running pipeline…"):
                result = run_pipeline(n=n, dry_run=dry_run)
            if result.returncode != 0:
                st.error("Pipeline failed")
                st.code(result.stderr or result.stdout or "(no output)")
            else:
                st.success("Done")
                with st.expander("Log"):
                    st.code(result.stdout or "(no stdout)")
                st.rerun()

    evaluations = load_json(DATA / "evaluations.json")
    validation = load_json(DATA / "validation.json")
    generated = load_json(DATA / "generated.json") or []

    dry_flag = None if not validation else bool(validation.get("dry_run"))
    render_hero(dry_run=dry_flag)

    if not evaluations or not validation:
        st.info("No results yet. Run the pipeline from the sidebar (dry-run is fine to start).")
        return

    gen_by_id = {g["id"]: g for g in generated}
    df = pd.DataFrame(evaluations)

    # 1. Validation first — hiring-manager insight in the first viewport
    st.markdown('<p class="story-step">01 · Believe the score?</p>', unsafe_allow_html=True)
    section(
        "Validation proof",
        "The gap is the punchline. Charts come later.",
    )
    render_validation_proof(validation)

    # 2. Ticket review
    render_ticket_review(df, gen_by_id)

    # 3. Supporting detail — collapsed by default to cut clutter
    st.markdown('<p class="story-step">03 · Supporting detail</p>', unsafe_allow_html=True)
    with st.expander("Charts & full results table", expanded=False):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**Accuracy by category**")
            if "category" in df.columns and df["category"].notna().any():
                category_chart(df)
            else:
                st.write("No category data.")
        with c2:
            st.markdown("**Score distribution**")
            distribution_chart(df)

        st.markdown("**All results**")
        categories = sorted({c for c in df["category"].dropna().unique()})
        f1, f2 = st.columns([2, 1], gap="medium")
        with f1:
            selected_cats = st.multiselect("Category", categories, default=categories)
        with f2:
            min_acc = st.slider("Minimum accuracy", 0, 100, 0)

        filtered = df.copy()
        if selected_cats:
            filtered = filtered[filtered["category"].isin(selected_cats)]
        filtered = filtered[filtered["accuracy_score"] >= min_acc]

        show_cols = [
            c
            for c in [
                "id",
                "category",
                "subject",
                "accuracy_score",
                "rouge_l_f1",
                "intent_coverage",
                "tone_fidelity",
                "correctness",
                "conciseness",
            ]
            if c in filtered.columns
        ]
        display = filtered[show_cols].rename(
            columns={
                "accuracy_score": "accuracy",
                "rouge_l_f1": "rouge-l",
                "intent_coverage": "intent",
                "tone_fidelity": "tone",
            }
        )
        st.dataframe(display, width="stretch", hide_index=True, height=300)


if __name__ == "__main__":
    main()
