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
@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --td-primary: #0F766E;
    --td-accent: #14B8A6;
    --td-navy: #0F172A;
    --td-bg: #F8FAFC;
    --td-surface: #FFFFFF;
    --td-border: #E2E8F0;
    --td-text: #0F172A;
    --td-muted: #475569;
    --td-radius: 12px;
    --td-radius-sm: 10px;
    --td-pad: 1.05rem 1.15rem;
    --td-space: 2rem;
    --td-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}

html, body, [class*="css"], .stApp {
    font-family: Inter, "Segoe UI", system-ui, sans-serif !important;
    color: var(--td-text);
}
.stApp {
    background: var(--td-bg);
}
h1, h2, h3, .hero-title, .proof-title, .panel-title, .section-label,
.sidebar-mark .copy .name, .score-row .acc, .compare-value {
    font-family: "Inter Tight", Inter, system-ui, sans-serif !important;
}

header[data-testid="stHeader"] { background: transparent; }
div[data-testid="stDecoration"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    padding-left: 1.75rem !important;
    padding-right: 1.75rem !important;
    max-width: 1120px;
}

/* —— Sidebar —— */
section[data-testid="stSidebar"] {
    background: var(--td-navy);
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] > div:first-child { background: transparent; }
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 2rem !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #94a3b8 !important;
}
.sidebar-mark {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0 0 1.35rem 0;
    padding-bottom: 1.1rem;
    border-bottom: 1px solid rgba(226, 232, 240, 0.12);
}
.sidebar-mark .logo {
    width: 2.2rem;
    height: 2.2rem;
    border-radius: var(--td-radius-sm);
    background: linear-gradient(145deg, var(--td-accent) 0%, var(--td-primary) 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ecfeff;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.sidebar-mark .copy .name {
    font-size: 0.98rem;
    font-weight: 700;
    color: #F8FAFC !important;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.2;
}
.sidebar-mark .copy .tag {
    font-size: 0.72rem;
    color: #94a3b8 !important;
    margin: 0.18rem 0 0 0;
}
.sidebar-panel-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--td-accent) !important;
    margin: 0 0 0.55rem 0;
}
.sidebar-mode-note {
    font-size: 0.74rem;
    line-height: 1.35;
    border-radius: var(--td-radius-sm);
    padding: 0.5rem 0.65rem;
    margin: 0.55rem 0 0.85rem 0;
    border: 1px solid transparent;
}
.sidebar-mode-note.dry {
    background: rgba(71, 85, 105, 0.35);
    border-color: rgba(226, 232, 240, 0.12);
    color: #e2e8f0 !important;
}
.sidebar-mode-note.live {
    background: rgba(15, 118, 110, 0.28);
    border-color: rgba(20, 184, 166, 0.35);
    color: #99f6e4 !important;
}
.stButton > button[kind="primary"] {
    background: var(--td-primary) !important;
    border-color: var(--td-primary) !important;
    color: #fff !important;
    font-weight: 600;
    border-radius: var(--td-radius-sm);
    min-height: 2.65rem;
}
.stButton > button[kind="primary"]:hover {
    background: var(--td-accent) !important;
    border-color: var(--td-accent) !important;
}
section[data-testid="stSidebar"] div[data-baseweb="slider"] div[role="slider"] {
    background-color: var(--td-accent) !important;
}

/* —— Hero —— */
.hero {
    padding: 0 0 0.75rem 0;
    border-bottom: 1px solid var(--td-border);
    margin-bottom: 1.15rem;
}
.hero-top {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem 1rem;
    margin-bottom: 0.25rem;
}
.hero-eyebrow {
    color: var(--td-primary);
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 0 0 0.25rem 0;
}
.hero-title {
    font-size: 1.95rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    color: var(--td-text);
    margin: 0;
    line-height: 1.05;
}
.hero-subtitle {
    color: var(--td-muted);
    font-size: 0.9rem;
    margin: 0.35rem 0 0.55rem 0;
    line-height: 1.35;
    max-width: 26rem;
}
.hero-subtitle strong {
    color: var(--td-primary);
    font-weight: 700;
}
.mode-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.32rem 0.65rem;
    border-radius: var(--td-radius-sm);
    border: 1px solid var(--td-border);
    white-space: nowrap;
    background: var(--td-surface);
}
.mode-badge.live {
    background: #F0FDFA;
    border-color: #99F6E4;
    color: var(--td-primary);
}
.mode-badge.dry {
    background: var(--td-bg);
    border-color: var(--td-border);
    color: var(--td-muted);
}
.mode-badge .dot {
    width: 0.36rem;
    height: 0.36rem;
    border-radius: 999px;
    background: currentColor;
    box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.12);
}
.mode-badge.dry .dot {
    box-shadow: 0 0 0 3px rgba(71, 85, 105, 0.12);
}
.process-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.3rem;
}
.process-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: var(--td-surface);
    border: 1px solid var(--td-border);
    color: var(--td-text);
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.26rem 0.5rem;
    border-radius: var(--td-radius-sm);
}
.process-pill span.num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 0.95rem;
    height: 0.95rem;
    border-radius: 4px;
    background: var(--td-navy);
    color: #fff;
    font-size: 0.58rem;
    font-weight: 700;
}
.process-arrow { color: #94a3b8; font-size: 0.75rem; }

/* —— Sections —— */
.section-head {
    margin: 1.25rem 0 0.55rem 0;
    padding-top: 0.65rem;
    border-top: 1px solid var(--td-border);
}
.section-head.proof-focus {
    margin-top: 0.85rem;
}
.section-label {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--td-text);
    margin: 0;
}
.section-label::before {
    content: "";
    width: 0.4rem;
    height: 0.4rem;
    border-radius: 2px;
    background: var(--td-primary);
    flex: 0 0 auto;
}
.section-lead {
    color: var(--td-muted);
    font-size: 0.8rem;
    margin: 0.2rem 0 0 0;
    line-height: 1.35;
    max-width: 32rem;
}

div[data-testid="stMetric"] {
    background: var(--td-surface);
    border: 1px solid var(--td-border);
    border-radius: var(--td-radius);
    padding: var(--td-pad);
    box-shadow: var(--td-shadow);
}
div[data-testid="stMetric"] label {
    color: var(--td-muted) !important;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--td-text) !important;
    font-weight: 700 !important;
}

/* —— Proof —— */
.proof-card {
    background: var(--td-navy);
    color: #e2e8f0;
    border: 1px solid #1e293b;
    border-radius: var(--td-radius);
    padding: 1.05rem 1.15rem 1rem 1.15rem;
    margin: 0 0 0.5rem 0;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.18);
}
.proof-card.proof-success {
    background: linear-gradient(165deg, #042f2e 0%, var(--td-navy) 52%);
    border-color: var(--td-primary);
    box-shadow: 0 18px 40px rgba(15, 118, 110, 0.28);
    padding: 1.25rem 1.35rem 1.15rem 1.35rem;
}
.proof-card.proof-dry {
    background: #111827;
    border-color: #334155;
    box-shadow: none;
}
.proof-header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.45rem 0.85rem;
    margin-bottom: 0.85rem;
}
.proof-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.proof-card.proof-success .proof-title {
    font-size: 1.12rem;
}
.compare-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.55rem;
    margin: 0;
}
.compare-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.55rem;
}
@media (max-width: 640px) {
    .compare-row { grid-template-columns: 1fr; }
}
.compare-cell {
    background: rgba(248, 250, 252, 0.04);
    border: 1px solid rgba(226, 232, 240, 0.12);
    border-radius: var(--td-radius-sm);
    padding: 0.7rem 0.85rem;
}
.compare-cell.gap-cell {
    background: linear-gradient(160deg, rgba(20, 184, 166, 0.2) 0%, rgba(15, 118, 110, 0.1) 100%);
    border: 1px solid rgba(20, 184, 166, 0.35);
    padding: 0.95rem 1.05rem;
    text-align: left;
}
.proof-card.proof-success .compare-cell.gap-cell {
    background: linear-gradient(155deg, rgba(20, 184, 166, 0.32) 0%, rgba(15, 118, 110, 0.14) 100%);
    border: 1px solid rgba(45, 212, 191, 0.55);
    box-shadow: inset 0 1px 0 rgba(204, 251, 241, 0.12);
    padding: 1.15rem 1.2rem 1.05rem 1.2rem;
}
.compare-label {
    font-size: 0.64rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 0 0 0.25rem 0;
    font-weight: 700;
}
.compare-cell.gap-cell .compare-label { color: #99F6E4; }
.compare-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: -0.03em;
    margin: 0;
    line-height: 1.05;
}
.compare-cell.gap-cell .compare-value {
    font-size: 2.75rem;
    color: #CCFBF1;
    letter-spacing: -0.045em;
}
.proof-card.proof-success .compare-cell.gap-cell .compare-value {
    font-size: 3.35rem;
    color: #F0FDFA;
}
.compare-hint {
    margin: 0.35rem 0 0 0;
    font-size: 0.78rem;
    color: #99F6E4;
    font-weight: 500;
    line-height: 1.3;
}
.status-chip {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 650;
    padding: 0.28rem 0.55rem;
    border-radius: var(--td-radius-sm);
    border: 1px solid transparent;
}
.status-chip.ok {
    color: #CCFBF1;
    background: rgba(15, 118, 110, 0.38);
    border-color: rgba(20, 184, 166, 0.45);
}
.status-chip.warn {
    color: #e2e8f0;
    background: rgba(71, 85, 105, 0.35);
    border-color: rgba(226, 232, 240, 0.12);
}
.status-chip.neutral {
    color: #e2e8f0;
    background: rgba(148, 163, 184, 0.14);
    border-color: rgba(226, 232, 240, 0.14);
}
.dry-run-inline {
    margin-top: 0.55rem;
    font-size: 0.76rem;
    color: #cbd5e1;
    background: rgba(71, 85, 105, 0.35);
    border: 1px solid rgba(226, 232, 240, 0.12);
    border-radius: var(--td-radius-sm);
    padding: 0.5rem 0.7rem;
    line-height: 1.35;
}
.interpretation {
    background: var(--td-surface);
    border: 1px solid var(--td-border);
    border-left: 3px solid var(--td-primary);
    border-radius: var(--td-radius);
    padding: 0.65rem 0.8rem;
    color: var(--td-muted);
    font-size: 0.8rem;
    line-height: 1.4;
    margin-top: 0.35rem;
}
.interpretation.interp-success {
    background: #F0FDFA;
    border-color: #99F6E4;
}

/* —— Ticket review —— */
.ticket-shell {
    background: var(--td-surface);
    border: 1px solid var(--td-border);
    border-radius: var(--td-radius);
    box-shadow: var(--td-shadow);
    padding: 0.85rem 1rem;
    margin: 0.25rem 0 0.65rem 0;
}
.ticket-shell .ticket-head {
    margin-bottom: 0.65rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid var(--td-border);
}
.ticket-shell .ticket-head .th-subject {
    font-size: 0.95rem;
    font-weight: 650;
    color: var(--td-text);
    margin: 0;
    letter-spacing: -0.015em;
}
.ticket-shell .ticket-head .th-meta {
    font-size: 0.76rem;
    color: var(--td-muted);
    margin: 0.2rem 0 0 0;
}
.score-row {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.3rem;
}
.score-row .acc {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--td-primary);
    letter-spacing: -0.03em;
    margin: 0;
    line-height: 1;
}
.score-row .acc span {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--td-muted);
    margin-right: 0.4rem;
}
.score-row .dims {
    font-size: 0.76rem;
    color: var(--td-muted);
    margin: 0;
    line-height: 1.35;
}
.score-row .dims b {
    color: var(--td-text);
    font-weight: 650;
}
.panel-title {
    font-size: 0.86rem;
    font-weight: 700;
    color: var(--td-text);
    margin: 0 0 0.55rem 0;
    letter-spacing: -0.01em;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--td-bg) !important;
    border: 1px solid var(--td-border) !important;
    border-radius: var(--td-radius) !important;
    box-shadow: none;
    padding: 0.1rem;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 0.7rem 0.8rem 0.85rem 0.8rem !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] textarea {
    background: var(--td-surface) !important;
    border: 1px solid var(--td-border) !important;
    border-radius: var(--td-radius-sm) !important;
    color: var(--td-text) !important;
    font-size: 0.88rem !important;
    line-height: 1.5 !important;
    padding: 0.7rem 0.8rem !important;
}
.retrieved-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.55rem;
    margin-top: 0.25rem;
}
@media (max-width: 900px) {
    .retrieved-grid { grid-template-columns: 1fr; }
}
.evidence-card {
    background: var(--td-bg);
    border: 1px solid var(--td-border);
    border-radius: var(--td-radius-sm);
    padding: 0.65rem 0.75rem;
}
.evidence-card .ev-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.4rem;
    margin-bottom: 0.25rem;
}
.evidence-card .ev-id {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.7rem;
    color: var(--td-muted);
}
.evidence-card .ev-score {
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--td-primary);
}
.evidence-card .ev-subject {
    font-size: 0.82rem;
    font-weight: 650;
    color: var(--td-text);
    margin: 0 0 0.15rem 0;
    line-height: 1.3;
}
.evidence-card .ev-cat {
    font-size: 0.72rem;
    color: var(--td-muted);
    margin: 0;
}

.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--td-muted) !important;
}

/* —— Supporting detail —— */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0.2rem;
    background: var(--td-bg);
    border: 1px solid var(--td-border);
    border-radius: var(--td-radius-sm);
    padding: 0.2rem;
    margin-bottom: 0.55rem;
}
div[data-testid="stTabs"] button[data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 600 !important;
    color: var(--td-muted) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    background: var(--td-surface) !important;
    color: var(--td-text) !important;
    box-shadow: var(--td-shadow);
}
div[data-testid="stDataFrame"] {
    border: 1px solid var(--td-border);
    border-radius: var(--td-radius);
    overflow: hidden;
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


def section(label: str, lead: str = "", *, focus: bool = False) -> None:
    lead_html = f'<p class="section-lead">{lead}</p>' if lead else ""
    cls = "section-head proof-focus" if focus else "section-head"
    st.markdown(
        f'<div class="{cls}"><div class="section-label">{label}</div>{lead_html}</div>',
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
        .mark_bar(color="#0F766E", cornerRadiusEnd=3)
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
        .properties(height=220)
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
        .mark_bar(color="#14B8A6", cornerRadiusEnd=3)
        .encode(
            x=alt.X("bucket:N", title="Accuracy score range", sort=labels),
            y=alt.Y("count:Q", title="Number of tickets"),
            tooltip=[
                alt.Tooltip("bucket:N", title="Range"),
                alt.Tooltip("count:Q", title="Tickets"),
            ],
        )
        .properties(height=220)
        .configure_axis(labelFontSize=12, titleFontSize=12)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, width="stretch")


def render_hero(
    dry_run: bool | None = None,
    gap: float | None = None,
) -> None:
    if dry_run is True:
        mode = (
            '<span class="mode-badge dry"><span class="dot"></span>'
            "Dry-run</span>"
        )
        subtitle = "Fixtures only. Live run proves the gap."
    elif dry_run is False and gap is not None:
        mode = (
            '<span class="mode-badge live"><span class="dot"></span>'
            "Live</span>"
        )
        gap_sign = "+" if gap > 0 else ""
        subtitle = (
            f'Look at the <strong>gap ({gap_sign}{gap:.1f})</strong>, '
            "then one ticket."
        )
    elif dry_run is False:
        mode = (
            '<span class="mode-badge live"><span class="dot"></span>'
            "Live</span>"
        )
        subtitle = "Look at the gap, then one ticket."
    else:
        mode = ""
        subtitle = "Retrieve · draft · validate."

    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-top">
            <div>
              <p class="hero-eyebrow">Shared-inbox demo</p>
              <h1 class="hero-title">TrustDraft</h1>
            </div>
            {mode}
          </div>
          <p class="hero-subtitle">{subtitle}</p>
          <div class="process-row">
            <div class="process-pill"><span class="num">1</span>Retrieve</div>
            <span class="process-arrow">→</span>
            <div class="process-pill"><span class="num">2</span>Draft</div>
            <span class="process-arrow">→</span>
            <div class="process-pill"><span class="num">3</span>Validate</div>
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
        status = '<span class="status-chip warn">Dry-run</span>'
        title = "Validation gap"
        dry_banner = (
            '<div class="dry-run-inline">'
            "Placeholder scores · gap ≈ 0. Re-run in Live."
            "</div>"
        )
        gap_hint = ""
    elif meaningful:
        card_class = "proof-card proof-success"
        status = '<span class="status-chip ok">Discriminates</span>'
        title = "Validation gap"
        dry_banner = ""
        gap_hint = (
            f'<p class="compare-hint">'
            f"Correct {correct:.1f} − mismatched {mismatched:.1f}"
            f"</p>"
        )
    else:
        card_class = "proof-card"
        status = '<span class="status-chip neutral">Weak gap</span>'
        title = "Validation gap"
        dry_banner = ""
        gap_hint = ""

    gap_sign = "+" if gap > 0 else ""
    interp_class = "interpretation interp-success" if meaningful else "interpretation"

    st.markdown(
        f"""
        <div class="{card_class}">
          <div class="proof-header">
            <p class="proof-title">{title}</p>
            {status}
          </div>
          <div class="compare-grid">
            <div class="compare-cell gap-cell">
              <p class="compare-label">Accuracy gap</p>
              <p class="compare-value">{gap_sign}{gap:.1f}</p>
              {gap_hint}
            </div>
            <div class="compare-row">
              <div class="compare-cell">
                <p class="compare-label">Correct</p>
                <p class="compare-value">{correct:.1f}</p>
              </div>
              <div class="compare-cell">
                <p class="compare-label">Mismatched</p>
                <p class="compare-value">{mismatched:.1f}</p>
              </div>
            </div>
          </div>
          {dry_banner}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if interpretation:
        with st.expander("Notes", expanded=False):
            st.markdown(
                f'<div class="{interp_class}">{html.escape(str(interpretation))}</div>',
                unsafe_allow_html=True,
            )


def render_ticket_review(df: pd.DataFrame, gen_by_id: dict) -> None:
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

    st.markdown(
        f"""
        <div class="ticket-shell">
          <div class="ticket-head">
            <p class="th-subject">{html.escape(str(incoming_subject))}</p>
            <p class="th-meta">{html.escape(str(category))} · {html.escape(str(ticket_id))}</p>
          </div>
          <div class="score-row">
            <p class="acc"><span>Accuracy</span>{row['accuracy_score']:.0f}</p>
            <p class="dims">
              Intent <b>{int(row['intent_coverage'])}</b>
              · Tone <b>{int(row['tone_fidelity'])}</b>
              · Correct <b>{int(row['correctness'])}</b>
              · Concise <b>{int(row['conciseness'])}</b>
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            '<p class="panel-title">Customer email</p>',
            unsafe_allow_html=True,
        )
        st.text_area(
            "Customer email",
            incoming_body or "(body not in generated.json)",
            height=150,
            disabled=True,
            label_visibility="collapsed",
        )

    c_draft, c_ref = st.columns(2, gap="small")
    with c_draft:
        with st.container(border=True):
            st.markdown(
                '<p class="panel-title">Model draft</p>',
                unsafe_allow_html=True,
            )
            st.text_area(
                "Model draft",
                row["generated_reply"],
                height=220,
                disabled=True,
                label_visibility="collapsed",
            )
    with c_ref:
        with st.container(border=True):
            st.markdown(
                '<p class="panel-title">Agent-sent reply</p>',
                unsafe_allow_html=True,
            )
            st.text_area(
                "Agent-sent reply",
                row["reference_reply"],
                height=220,
                disabled=True,
                label_visibility="collapsed",
            )

    retrieved = gen.get("retrieved_examples") or []
    if retrieved:
        n_ex = len(retrieved)
        with st.expander(f"Retrieval · {n_ex}", expanded=False):
            cards = []
            for i, ex in enumerate(retrieved, 1):
                score = ex.get("score")
                score_html = (
                    f'<span class="ev-score">{score:.2f}</span>'
                    if isinstance(score, (int, float))
                    else ""
                )
                cards.append(
                    "<div class='evidence-card'>"
                    "<div class='ev-top'>"
                    f"<span class='ev-id'>#{i} · {html.escape(str(ex.get('id', '')))}</span>"
                    f"{score_html}"
                    "</div>"
                    f"<p class='ev-subject'>{html.escape(str(ex.get('subject', '')))}</p>"
                    f"<p class='ev-cat'>{html.escape(str(ex.get('category', '') or 'uncategorized'))}</p>"
                    "</div>"
                )
            st.markdown(
                "<div class='retrieved-grid'>" + "".join(cards) + "</div>",
                unsafe_allow_html=True,
            )
    elif gen.get("retrieved_ids"):
        ids = ", ".join(str(i) for i in gen["retrieved_ids"])
        with st.expander("Retrieval", expanded=False):
            st.caption(ids)



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
            <div class="sidebar-mark">
              <div class="logo">TD</div>
              <div class="copy">
                <p class="name">TrustDraft</p>
                <p class="tag">Reply quality</p>
              </div>
            </div>
            <p class="sidebar-panel-label">Run</p>
            """,
            unsafe_allow_html=True,
        )
        n = st.slider("Tickets", min_value=5, max_value=80, value=40, step=5)
        dry_run = st.checkbox("Dry-run", value=True)
        if dry_run:
            st.markdown(
                '<p class="sidebar-mode-note dry">Fixtures · gap ≈ 0</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="sidebar-mode-note live">Claude · needs API key</p>',
                unsafe_allow_html=True,
            )
        run_clicked = st.button("Run pipeline", type="primary", width="stretch")

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
    gap_val = (
        None
        if not validation
        else float(validation.get("accuracy_gap", 0))
    )
    render_hero(dry_run=dry_flag, gap=gap_val)

    if not evaluations or not validation:
        st.info("No results yet. Run from the sidebar.")
        return

    gen_by_id = {g["id"]: g for g in generated}
    df = pd.DataFrame(evaluations)

    is_dry = bool(validation.get("dry_run"))
    live_proof = (not is_dry) and gap_val is not None and gap_val >= 15
    section("Proof", focus=live_proof)
    render_validation_proof(validation)

    section("Ticket")
    render_ticket_review(df, gen_by_id)

    support_label = "More" + (" · dry-run" if is_dry else "")
    with st.expander(support_label, expanded=False):
        tab_results, tab_charts = st.tabs(["Table", "Charts"])

        with tab_results:
            categories = sorted({c for c in df["category"].dropna().unique()})
            f1, f2 = st.columns([2, 1], gap="small")
            with f1:
                selected_cats = st.multiselect(
                    "Category",
                    categories,
                    default=categories,
                    label_visibility="collapsed",
                    placeholder="Category",
                )
            with f2:
                min_acc = st.slider("Min accuracy", 0, 100, 0)

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
                    "intent_coverage": "intent",
                    "tone_fidelity": "tone",
                }
            )
            st.dataframe(
                display,
                width="stretch",
                hide_index=True,
                height=min(200, 42 + 35 * max(len(display), 1)),
            )
            st.caption(f"{len(display)} shown")

        with tab_charts:
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.caption("By category")
                if "category" in df.columns and df["category"].notna().any():
                    category_chart(df)
                else:
                    st.write("No category data.")
            with c2:
                st.caption("Score bands")
                distribution_chart(df)


if __name__ == "__main__":
    main()
