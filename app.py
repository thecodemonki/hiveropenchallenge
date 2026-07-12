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
.sidebar-mark .copy .name, .score-chip .svalue, .compare-value {
    font-family: "Inter Tight", Inter, system-ui, sans-serif !important;
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
    margin: 0 0 0.4rem 0;
}
.sidebar-help {
    color: #94a3b8 !important;
    font-size: 0.82rem;
    line-height: 1.45;
    margin: 0 0 1.15rem 0;
}
.sidebar-mode-note {
    font-size: 0.78rem;
    line-height: 1.4;
    border-radius: var(--td-radius-sm);
    padding: 0.6rem 0.75rem;
    margin: 0.7rem 0 1rem 0;
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
    padding: 0.2rem 0 1.35rem 0;
    border-bottom: 1px solid var(--td-border);
    margin-bottom: var(--td-space);
}
.hero-top {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.85rem 1.25rem;
    margin-bottom: 0.55rem;
}
.hero-eyebrow {
    color: var(--td-primary);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0 0 0.55rem 0;
}
.hero-title {
    font-size: 2.55rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    color: var(--td-text);
    margin: 0;
    line-height: 1.06;
}
.hero-subtitle {
    color: var(--td-muted);
    font-size: 1.05rem;
    margin: 0.7rem 0 1rem 0;
    line-height: 1.55;
    max-width: 34rem;
}
.mode-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.42rem 0.78rem;
    border-radius: var(--td-radius-sm);
    border: 1px solid var(--td-border);
    white-space: nowrap;
    margin-top: 0.35rem;
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
    width: 0.42rem;
    height: 0.42rem;
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
    gap: 0.4rem;
}
.process-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--td-surface);
    border: 1px solid var(--td-border);
    color: var(--td-text);
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.36rem 0.68rem;
    border-radius: var(--td-radius-sm);
}
.process-pill span.num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.1rem;
    height: 1.1rem;
    border-radius: 6px;
    background: var(--td-navy);
    color: #fff;
    font-size: 0.64rem;
    font-weight: 700;
}
.process-arrow { color: #94a3b8; font-size: 0.85rem; }

.reviewer-guide {
    background: var(--td-surface);
    border: 1px solid var(--td-border);
    border-radius: var(--td-radius);
    padding: var(--td-pad);
    margin: 1.15rem 0 0 0;
    box-shadow: var(--td-shadow);
}
.reviewer-guide .rg-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--td-primary);
    margin: 0 0 0.75rem 0;
}
.reviewer-guide .rg-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.75rem;
}
@media (max-width: 900px) {
    .reviewer-guide .rg-grid { grid-template-columns: 1fr; }
}
.reviewer-guide .rg-item {
    background: var(--td-bg);
    border: 1px solid var(--td-border);
    border-radius: var(--td-radius-sm);
    padding: 0.75rem 0.85rem;
}
.reviewer-guide .rg-item .n {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--td-primary);
    margin: 0 0 0.25rem 0;
}
.reviewer-guide .rg-item .t {
    font-size: 0.9rem;
    font-weight: 650;
    color: var(--td-text);
    margin: 0 0 0.2rem 0;
    letter-spacing: -0.01em;
}
.reviewer-guide .rg-item .d {
    font-size: 0.8rem;
    color: var(--td-muted);
    margin: 0;
    line-height: 1.4;
}

.story-step {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--td-primary);
    margin: 0 0 0.15rem 0;
}

/* —— Sections —— */
.section-head {
    margin: var(--td-space) 0 1rem 0;
    padding-top: 0.9rem;
    border-top: 1px solid var(--td-border);
}
.section-label {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--td-text);
    margin: 0 0 0.28rem 0;
}
.section-label::before {
    content: "";
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 2px;
    background: var(--td-primary);
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
    padding: 1.5rem 1.55rem 1.35rem 1.55rem;
    margin: 0.15rem 0 0.85rem 0;
    box-shadow: 0 16px 36px rgba(15, 23, 42, 0.22);
}
.proof-card.proof-success {
    background: linear-gradient(165deg, #042f2e 0%, var(--td-navy) 58%);
    border-color: var(--td-primary);
}
.proof-card.proof-dry {
    background: #111827;
    border-color: #334155;
}
.proof-header {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.75rem 1.25rem;
    margin-bottom: 1.15rem;
}
.proof-kicker {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 0 0 0.4rem 0;
}
.proof-card.proof-success .proof-kicker { color: var(--td-accent); }
.proof-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.025em;
    line-height: 1.2;
}
.proof-lead {
    color: #94a3b8;
    font-size: 0.92rem;
    margin: 0;
    line-height: 1.5;
    max-width: 36rem;
}
.compare-grid {
    display: grid;
    grid-template-columns: 1.45fr 1fr 1fr;
    gap: 0.85rem;
    margin: 0 0 1rem 0;
    align-items: stretch;
}
@media (max-width: 900px) {
    .compare-grid { grid-template-columns: 1fr; }
}
.compare-cell {
    background: rgba(248, 250, 252, 0.04);
    border: 1px solid rgba(226, 232, 240, 0.14);
    border-radius: var(--td-radius-sm);
    padding: var(--td-pad);
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.compare-cell.gap-cell {
    background: linear-gradient(160deg, rgba(20, 184, 166, 0.22) 0%, rgba(15, 118, 110, 0.12) 100%);
    border: 1px solid rgba(20, 184, 166, 0.4);
    min-height: 8.2rem;
}
.compare-label {
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 0 0 0.45rem 0;
    font-weight: 700;
}
.compare-cell.gap-cell .compare-label { color: #99F6E4; }
.compare-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: -0.04em;
    margin: 0;
    line-height: 1.05;
}
.compare-cell.gap-cell .compare-value {
    font-size: 3rem;
    color: #CCFBF1;
}
.compare-sub {
    font-size: 0.8rem;
    color: #94a3b8;
    margin: 0.5rem 0 0 0;
    line-height: 1.4;
}
.compare-cell.gap-cell .compare-sub {
    color: #99F6E4;
    font-weight: 500;
}
.proof-takeaway {
    display: flex;
    gap: 0.65rem;
    align-items: flex-start;
    background: rgba(248, 250, 252, 0.05);
    border: 1px solid rgba(226, 232, 240, 0.12);
    border-radius: var(--td-radius-sm);
    padding: 0.8rem 0.95rem;
    margin: 0 0 0.85rem 0;
}
.proof-takeaway .mark {
    flex: 0 0 auto;
    width: 0.4rem;
    height: 0.4rem;
    border-radius: 999px;
    background: var(--td-accent);
    margin-top: 0.4rem;
    box-shadow: 0 0 0 4px rgba(20, 184, 166, 0.15);
}
.proof-takeaway p {
    margin: 0;
    color: #e2e8f0;
    font-size: 0.9rem;
    line-height: 1.45;
}
.proof-takeaway strong {
    color: #F8FAFC;
    font-weight: 650;
}
.status-chip {
    display: inline-block;
    font-size: 0.74rem;
    font-weight: 650;
    padding: 0.34rem 0.72rem;
    border-radius: var(--td-radius-sm);
    border: 1px solid transparent;
}
.status-chip.ok {
    color: #CCFBF1;
    background: rgba(15, 118, 110, 0.32);
    border-color: rgba(20, 184, 166, 0.4);
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
    margin-top: 0.15rem;
    font-size: 0.8rem;
    color: #cbd5e1;
    background: rgba(71, 85, 105, 0.35);
    border: 1px solid rgba(226, 232, 240, 0.12);
    border-radius: var(--td-radius-sm);
    padding: 0.65rem 0.85rem;
    line-height: 1.45;
}
.interpretation {
    background: var(--td-surface);
    border: 1px solid var(--td-border);
    border-left: 4px solid var(--td-primary);
    border-radius: var(--td-radius);
    padding: 0.9rem 1.05rem;
    color: var(--td-muted);
    font-size: 0.9rem;
    line-height: 1.5;
    box-shadow: var(--td-shadow);
}
.interpretation.interp-success {
    background: #F0FDFA;
    border-color: #99F6E4;
}
.interpretation .interp-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--td-muted);
    margin-bottom: 0.3rem;
}
.interpretation.interp-success .interp-label { color: var(--td-primary); }

/* —— Ticket review —— */
.ticket-shell {
    background: var(--td-surface);
    border: 1px solid var(--td-border);
    border-radius: var(--td-radius);
    box-shadow: var(--td-shadow);
    padding: 1rem 1.1rem 1.15rem 1.1rem;
    margin: 0.35rem 0 1rem 0;
}
.ticket-shell .ticket-head {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.35rem 1rem;
    margin-bottom: 0.85rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--td-border);
}
.ticket-shell .ticket-head .th-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--td-primary);
    margin: 0 0 0.2rem 0;
}
.ticket-shell .ticket-head .th-meta {
    font-size: 0.8rem;
    color: var(--td-muted);
    margin: 0;
}
.ticket-shell .ticket-head .th-subject {
    font-size: 1rem;
    font-weight: 650;
    color: var(--td-text);
    margin: 0.15rem 0 0 0;
    letter-spacing: -0.015em;
    max-width: 42rem;
}
.panel-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--td-text);
    margin: 0;
    letter-spacing: -0.01em;
}
.panel-meta {
    font-size: 0.76rem;
    color: var(--td-muted);
    margin: 0 0 0.85rem 0;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--td-bg) !important;
    border: 1px solid var(--td-border) !important;
    border-radius: var(--td-radius) !important;
    box-shadow: none;
    padding: 0.15rem;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 0.85rem 0.95rem 1rem 0.95rem !important;
}
.score-strip {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.55rem;
    margin: 0;
}
@media (max-width: 900px) {
    .score-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
.score-chip {
    background: var(--td-bg);
    border: 1px solid var(--td-border);
    border-radius: var(--td-radius-sm);
    padding: 0.7rem 0.8rem;
}
.score-chip.total {
    background: var(--td-primary);
    border-color: var(--td-primary);
}
.score-chip .slabel {
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--td-muted);
    margin: 0 0 0.2rem 0;
}
.score-chip.total .slabel { color: rgba(255,255,255,0.78); }
.score-chip .svalue {
    font-size: 1.22rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--td-text);
    margin: 0;
    line-height: 1.15;
}
.score-chip.total .svalue { color: #fff; }
.score-chip .sunit {
    font-size: 0.72rem;
    color: #94a3b8;
    font-weight: 500;
}
.mail-card-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    margin-bottom: 0.25rem;
}
.mail-badge {
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 0.2rem 0.5rem;
    border-radius: 999px;
    border: 1px solid var(--td-border);
    color: var(--td-muted);
    background: var(--td-surface);
}
.mail-badge.gen {
    background: #F0FDFA;
    border-color: #99F6E4;
    color: var(--td-primary);
}
.mail-badge.ref {
    background: #ECFDF5;
    border-color: #A7F3D0;
    color: #047857;
}
.mail-subject {
    font-size: 0.84rem;
    color: var(--td-text);
    margin: 0 0 0.65rem 0;
    line-height: 1.4;
}
.mail-subject strong {
    color: var(--td-muted);
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    display: block;
    margin-bottom: 0.15rem;
}
/* Text areas in compare panels */
div[data-testid="stVerticalBlockBorderWrapper"] textarea {
    background: var(--td-surface) !important;
    border: 1px solid var(--td-border) !important;
    border-radius: var(--td-radius-sm) !important;
    color: var(--td-text) !important;
    font-size: 0.9rem !important;
    line-height: 1.55 !important;
    padding: 0.85rem 0.95rem !important;
    min-height: 16rem;
}
.retrieved-panel {
    background: var(--td-surface);
    border: 1px solid var(--td-border);
    border-radius: var(--td-radius);
    padding: 1rem 1.1rem;
    margin-top: 0.85rem;
    box-shadow: var(--td-shadow);
}
.retrieved-panel .rp-title {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--td-muted);
    margin: 0 0 0.75rem 0;
}
.retrieved-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.65rem;
}
@media (max-width: 900px) {
    .retrieved-grid { grid-template-columns: 1fr; }
}
.evidence-card {
    background: var(--td-bg);
    border: 1px solid var(--td-border);
    border-radius: var(--td-radius-sm);
    padding: 0.75rem 0.85rem;
}
.evidence-card .ev-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    margin-bottom: 0.35rem;
}
.evidence-card .ev-id {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.72rem;
    color: var(--td-muted);
}
.evidence-card .ev-score {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--td-primary);
    background: #F0FDFA;
    border: 1px solid #99F6E4;
    border-radius: 999px;
    padding: 0.12rem 0.45rem;
}
.evidence-card .ev-subject {
    font-size: 0.86rem;
    font-weight: 650;
    color: var(--td-text);
    margin: 0 0 0.25rem 0;
    line-height: 1.35;
    letter-spacing: -0.01em;
}
.evidence-card .ev-cat {
    font-size: 0.74rem;
    color: var(--td-muted);
    margin: 0;
}

.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--td-muted) !important;
}

/* —— Supporting detail —— */
.support-note {
    color: var(--td-muted);
    font-size: 0.86rem;
    line-height: 1.45;
    margin: 0 0 0.65rem 0;
    max-width: 42rem;
}
.support-note strong { color: var(--td-text); font-weight: 650; }
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0.25rem;
    background: var(--td-bg);
    border: 1px solid var(--td-border);
    border-radius: var(--td-radius-sm);
    padding: 0.25rem;
    margin-bottom: 0.65rem;
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
.chart-muted {
    background: var(--td-bg);
    border: 1px solid var(--td-border);
    border-radius: var(--td-radius-sm);
    padding: 0.7rem 0.85rem;
    color: var(--td-muted);
    font-size: 0.84rem;
    line-height: 1.45;
    margin: 0 0 0.65rem 0;
}
.tab-sub {
    color: var(--td-muted);
    font-size: 0.82rem;
    margin: 0 0 0.75rem 0;
    line-height: 1.4;
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


def render_hero(dry_run: bool | None = None) -> None:
    if dry_run is True:
        mode = (
            '<span class="mode-badge dry"><span class="dot"></span>'
            "Dry-run</span>"
        )
    elif dry_run is False:
        mode = (
            '<span class="mode-badge live"><span class="dot"></span>'
            "Live</span>"
        )
    else:
        mode = ""

    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-top">
            <div>
              <p class="hero-eyebrow">Shared-inbox quality demo</p>
              <h1 class="hero-title">TrustDraft</h1>
            </div>
            {mode}
          </div>
          <p class="hero-subtitle">
            Draft support replies from similar tickets — then prove the score
            actually measures quality.
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
            <div class="rg-grid">
              <div class="rg-item">
                <p class="n">01</p>
                <p class="t">Validation gap</p>
                <p class="d">Does accuracy fall when the reference is wrong?</p>
              </div>
              <div class="rg-item">
                <p class="n">02</p>
                <p class="t">Ticket review</p>
                <p class="d">Compare the draft to the reply that was sent.</p>
              </div>
              <div class="rg-item">
                <p class="n">03</p>
                <p class="t">Mode</p>
                <p class="d">Dry-run = placeholders. Live = Claude scores.</p>
              </div>
            </div>
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
            "Dry-run · placeholder scores</span>"
        )
        dry_banner = (
            '<div class="dry-run-inline">'
            "Dry-run cannot prove discrimination — the judge returns a constant, "
            "so the gap stays near zero. Switch to Live and re-run for a real test."
            "</div>"
        )
    elif meaningful:
        card_class = "proof-card proof-success"
        status = (
            '<span class="status-chip ok">'
            "Live · metric discriminates</span>"
        )
        dry_banner = ""
    else:
        card_class = "proof-card"
        status = (
            '<span class="status-chip neutral">'
            "Live · gap is small</span>"
        )
        dry_banner = ""

    interp_class = "interpretation interp-success" if meaningful else "interpretation"
    gap_sign = "+" if gap > 0 else ""

    st.markdown(
        f"""
        <div class="{card_class}">
          <div class="proof-header">
            <div>
              <p class="proof-kicker">Core proof</p>
              <p class="proof-title">Does the score discriminate?</p>
              <p class="proof-lead">
                Same drafts, scored twice: once against the real reply, once against
                a wrong one. Correct should win.
              </p>
            </div>
            {status}
          </div>
          <div class="compare-grid">
            <div class="compare-cell gap-cell">
              <p class="compare-label">Accuracy gap</p>
              <p class="compare-value">{gap_sign}{gap:.1f}</p>
              <p class="compare-sub">Correct average − mismatched average</p>
            </div>
            <div class="compare-cell">
              <p class="compare-label">Correct references</p>
              <p class="compare-value">{correct:.1f}</p>
              <p class="compare-sub">Avg score vs. the true reply</p>
            </div>
            <div class="compare-cell">
              <p class="compare-label">Mismatched references</p>
              <p class="compare-value">{mismatched:.1f}</p>
              <p class="compare-sub">Avg score vs. a wrong reply</p>
            </div>
          </div>
          <div class="proof-takeaway">
            <span class="mark"></span>
            <p>
              <strong>Positive gap means the metric is discriminating</strong>
              between correct and wrong references — not outputting a constant.
            </p>
          </div>
          {dry_banner}
        </div>
        <div class="{interp_class}">
          <div class="interp-label">Run notes</div>
          <div>{interpretation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ticket_review(df: pd.DataFrame, gen_by_id: dict) -> None:
    st.markdown('<p class="story-step">02 · See a draft</p>', unsafe_allow_html=True)
    section(
        "Ticket review",
        "Customer email · model draft · agent-sent reply.",
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

    st.markdown(
        f"""
        <div class="ticket-shell">
          <div class="ticket-head">
            <div>
              <p class="th-label">Selected ticket</p>
              <p class="th-subject">{html.escape(str(incoming_subject))}</p>
              <p class="th-meta">{html.escape(str(category))} · {html.escape(str(ticket_id))}</p>
            </div>
          </div>
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
        </div>
        """,
        unsafe_allow_html=True,
    )

    d1, d2, d3 = st.columns(3, gap="medium")
    with d1:
        with st.container(border=True):
            st.markdown(
                """
                <div class="mail-card-head">
                  <div class="panel-title">Customer email</div>
                  <span class="mail-badge">Inbound</span>
                </div>
                <div class="panel-meta">What the customer wrote</div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p class="mail-subject"><strong>Subject</strong>'
                f"{html.escape(str(incoming_subject))}</p>",
                unsafe_allow_html=True,
            )
            st.text_area(
                "Customer email",
                incoming_body or "(body not in generated.json)",
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )
    with d2:
        with st.container(border=True):
            st.markdown(
                """
                <div class="mail-card-head">
                  <div class="panel-title">Model draft</div>
                  <span class="mail-badge gen">Suggested</span>
                </div>
                <div class="panel-meta">AI reply under evaluation</div>
                """,
                unsafe_allow_html=True,
            )
            st.text_area(
                "Model draft",
                row["generated_reply"],
                height=340,
                disabled=True,
                label_visibility="collapsed",
            )
    with d3:
        with st.container(border=True):
            st.markdown(
                """
                <div class="mail-card-head">
                  <div class="panel-title">Agent-sent reply</div>
                  <span class="mail-badge ref">Ground truth</span>
                </div>
                <div class="panel-meta">What the agent actually sent</div>
                """,
                unsafe_allow_html=True,
            )
            st.text_area(
                "Agent-sent reply",
                row["reference_reply"],
                height=340,
                disabled=True,
                label_visibility="collapsed",
            )

    retrieved = gen.get("retrieved_examples") or []
    if retrieved:
        cards = []
        for i, ex in enumerate(retrieved, 1):
            score = ex.get("score")
            score_html = (
                f'<span class="ev-score">sim {score:.3f}</span>'
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
            "<div class='retrieved-panel'>"
            "<p class='rp-title'>Retrieval evidence</p>"
            "<div class='retrieved-grid'>"
            + "".join(cards)
            + "</div></div>",
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
            <div class="sidebar-mark">
              <div class="logo">TD</div>
              <div class="copy">
                <p class="name">TrustDraft</p>
                <p class="tag">Suggested-reply evaluation</p>
              </div>
            </div>
            <p class="sidebar-panel-label">Controls</p>
            <p class="sidebar-help">Rebuild the corpus, draft replies, and score them.</p>
            """,
            unsafe_allow_html=True,
        )
        n = st.slider("Dataset size", min_value=5, max_value=80, value=40, step=5)
        dry_run = st.checkbox("Dry-run (no API calls)", value=True)
        if dry_run:
            st.markdown(
                '<p class="sidebar-mode-note dry">'
                "Dry-run uses fixtures and a placeholder judge. Gap ≈ 0."
                "</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="sidebar-mode-note live">'
                "Live mode calls Claude. Requires ANTHROPIC_API_KEY."
                "</p>",
                unsafe_allow_html=True,
            )
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
        "Scan the gap first. That is the product claim.",
    )
    render_validation_proof(validation)

    # 2. Ticket review
    render_ticket_review(df, gen_by_id)

    # 3. Supporting detail — accessible, not dominant
    is_dry = bool(validation.get("dry_run"))
    st.markdown('<p class="story-step">03 · Supporting detail</p>', unsafe_allow_html=True)
    section("Supporting detail")
    st.markdown(
        '<p class="support-note">'
        "<strong>Optional.</strong> Use this for category patterns and a full ticket table. "
        "The proof and ticket review above are the hiring-manager path."
        "</p>",
        unsafe_allow_html=True,
    )

    tab_results, tab_charts = st.tabs(["Results table", "Charts"])

    with tab_results:
        st.markdown(
            '<p class="tab-sub">Filter evaluated tickets. Compact view — open a row above for the full scorecard.</p>',
            unsafe_allow_html=True,
        )
        categories = sorted({c for c in df["category"].dropna().unique()})
        f1, f2 = st.columns([2, 1], gap="small")
        with f1:
            selected_cats = st.multiselect(
                "Category",
                categories,
                default=categories,
                label_visibility="collapsed",
                placeholder="Filter by category",
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
            height=min(260, 42 + 35 * max(len(display), 1)),
        )
        st.caption(f"{len(display)} ticket(s) shown")

    with tab_charts:
        if is_dry:
            st.markdown(
                '<div class="chart-muted">'
                "<strong>Dry-run:</strong> judge scores are constant, so category and "
                "distribution charts carry little signal. Expand only if you want to "
                "confirm the wiring."
                "</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Show charts anyway", expanded=False):
                st.markdown(
                    '<p class="tab-sub">Mean accuracy by category and score bands.</p>',
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns(2, gap="medium")
                with c1:
                    st.markdown("**By category**")
                    if "category" in df.columns and df["category"].notna().any():
                        category_chart(df)
                    else:
                        st.write("No category data.")
                with c2:
                    st.markdown("**Score bands**")
                    distribution_chart(df)
        else:
            st.markdown(
                '<p class="tab-sub">Secondary patterns after a live run — not the primary proof.</p>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.markdown("**By category**")
                st.caption("Mean LLM-judge accuracy (0–100).")
                if "category" in df.columns and df["category"].notna().any():
                    category_chart(df)
                else:
                    st.write("No category data.")
            with c2:
                st.markdown("**Score bands**")
                st.caption("How tickets fall across accuracy ranges.")
                distribution_chart(df)


if __name__ == "__main__":
    main()
