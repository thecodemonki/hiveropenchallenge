"""Streamlit dashboard for the TrustDraft suggested-reply pipeline.

Run the full generate → reply → evaluate pipeline and inspect accuracy,
validation gap, and per-ticket scorecards.
"""

from __future__ import annotations

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
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    h1.app-title {
        font-size: 2rem;
        font-weight: 650;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
        line-height: 1.2;
    }
    p.app-subtitle {
        color: #5c6570;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 0.35rem;
        line-height: 1.45;
    }
    p.app-eyebrow {
        color: #6b7280;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .section-label {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #6b7280;
        margin: 1.75rem 0 0.65rem 0;
    }
    div[data-testid="stMetric"] {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.85rem 1rem;
    }
    div[data-testid="stMetric"] label {
        color: #6b7280 !important;
    }
    .panel-title {
        font-size: 0.85rem;
        font-weight: 650;
        color: #111827;
        margin-bottom: 0.35rem;
    }
    .panel-meta {
        font-size: 0.8rem;
        color: #6b7280;
        margin-bottom: 0.75rem;
    }
    .dry-run-note {
        display: inline-block;
        font-size: 0.8rem;
        color: #78716c;
        background: #f5f5f4;
        border: 1px solid #e7e5e4;
        border-radius: 6px;
        padding: 0.35rem 0.65rem;
        margin: 0.5rem 0 0.25rem 0;
    }
    .validation-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #334155;
        border-radius: 8px;
        padding: 1rem 1.15rem 0.85rem 1.15rem;
        margin-top: 0.35rem;
    }
    .validation-box p {
        margin: 0.4rem 0 0 0;
        color: #334155;
        line-height: 1.5;
        font-size: 0.95rem;
    }
    .validation-box .note {
        color: #64748b;
        font-size: 0.82rem;
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


def section(label: str) -> None:
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)


def category_chart(df: pd.DataFrame):
    by_cat = (
        df.groupby("category", as_index=False)["accuracy_score"]
        .mean()
        .rename(columns={"accuracy_score": "avg_accuracy"})
    )
    chart = (
        alt.Chart(by_cat)
        .mark_bar(color="#334155", cornerRadiusEnd=3)
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
        .properties(height=300)
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
        .mark_bar(color="#475569", cornerRadiusEnd=3)
        .encode(
            x=alt.X("bucket:N", title="Accuracy score range", sort=labels),
            y=alt.Y("count:Q", title="Number of tickets"),
            tooltip=[
                alt.Tooltip("bucket:N", title="Range"),
                alt.Tooltip("count:Q", title="Tickets"),
            ],
        )
        .properties(height=300)
        .configure_axis(labelFontSize=12, titleFontSize=12)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, width="stretch")


def main() -> None:
    st.set_page_config(
        page_title="TrustDraft",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Controls")
        st.caption("Generate data, draft replies, and score them end-to-end.")
        n = st.slider("Dataset size", min_value=5, max_value=80, value=40, step=5)
        dry_run = st.checkbox("Dry-run (no API calls)", value=True)
        st.caption("Dry-run uses fixtures and placeholder judge scores.")
        run_clicked = st.button("Run pipeline", type="primary", width="stretch")

        if run_clicked:
            with st.spinner("Running pipeline…"):
                result = run_pipeline(n=n, dry_run=dry_run)
            if result.returncode != 0:
                st.error("Pipeline failed")
                st.code(result.stderr or result.stdout or "(no output)")
            else:
                st.success("Pipeline finished")
                with st.expander("Pipeline log"):
                    st.code(result.stdout or "(no stdout)")
                st.rerun()

    st.markdown('<p class="app-eyebrow">Shared inbox evaluation</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="app-title">TrustDraft</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">'
        "Measure suggested-reply quality against real reference replies — "
        "with retrieval context, an LLM judge, and a mismatched-reference check."
        "</p>",
        unsafe_allow_html=True,
    )

    evaluations = load_json(DATA / "evaluations.json")
    validation = load_json(DATA / "validation.json")
    generated = load_json(DATA / "generated.json") or []

    if not evaluations or not validation:
        st.info("No results yet. Run the pipeline from the sidebar to populate this dashboard.")
        return

    if validation.get("dry_run"):
        st.markdown(
            '<div class="dry-run-note">'
            "Showing dry-run results — judge scores are placeholders; validation gap will be ~0."
            "</div>",
            unsafe_allow_html=True,
        )

    gen_by_id = {g["id"]: g for g in generated}
    df = pd.DataFrame(evaluations)

    # --- summary metrics ---
    section("Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test set size", int(validation.get("n", len(df))))
    m2.metric("Avg accuracy", f"{validation['avg_accuracy_correct']:.1f}")
    m3.metric("Avg lexical overlap", f"{validation['avg_rouge_l_correct']:.3f}")
    m4.metric("Validation gap", f"{validation['accuracy_gap']:.1f}")

    # --- charts ---
    section("Overview charts")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Accuracy by category**")
        st.caption("Mean LLM-judge accuracy (0–100) per ticket category.")
        if "category" in df.columns and df["category"].notna().any():
            category_chart(df)
        else:
            st.write("No category data.")
    with c2:
        st.markdown("**Score distribution**")
        st.caption("How many tickets fall into each accuracy band.")
        distribution_chart(df)

    # --- validation panel ---
    section("Metric validation")
    with st.container(border=True):
        st.markdown("**Correct vs mismatched references**")
        st.caption(
            "Primary metric is LLM-judge accuracy. Lexical overlap is secondary only."
        )
        v1, v2, v3 = st.columns(3)
        v1.metric("Correct pairs", f"{validation['avg_accuracy_correct']:.1f}")
        v2.metric("Mismatched pairs", f"{validation['avg_accuracy_mismatched']:.1f}")
        v3.metric("Gap", f"{validation['accuracy_gap']:.1f}")

        interpretation = validation.get("interpretation", "")
        note = validation.get("note", "")
        st.markdown(
            f'<div class="validation-box">'
            f"<strong>Interpretation</strong>"
            f"<p>{interpretation}</p>"
            f'{f"<p class=note>{note}</p>" if note else ""}'
            f"</div>",
            unsafe_allow_html=True,
        )

    # --- filterable table ---
    section("Results")
    st.caption("Filter the evaluation table, then open a ticket below for the full scorecard.")
    categories = sorted({c for c in df["category"].dropna().unique()})
    f1, f2 = st.columns([2, 1], gap="medium")
    with f1:
        selected_cats = st.multiselect(
            "Category",
            categories,
            default=categories,
            help="Show only selected categories.",
        )
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
    st.dataframe(display, width="stretch", hide_index=True, height=280)

    # --- detail view ---
    section("Ticket detail")
    ids = filtered["id"].tolist() if len(filtered) else df["id"].tolist()
    if not ids:
        st.write("No tickets match the current filters.")
        return

    ticket_id = st.selectbox("Ticket", ids, help="Compare incoming, reference, and generated text.")
    row = df[df["id"] == ticket_id].iloc[0]
    gen = gen_by_id.get(ticket_id, {})
    incoming_body = gen.get("body") or ""
    incoming_subject = row.get("subject") or gen.get("subject", "")

    d1, d2, d3 = st.columns(3, gap="medium")
    with d1:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Incoming email</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="panel-meta">{row.get("category", "")} · {ticket_id}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Subject:** {incoming_subject}")
            st.text_area(
                "Incoming body",
                incoming_body or "(body not in generated.json)",
                height=260,
                disabled=True,
                label_visibility="collapsed",
            )
    with d2:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Reference reply</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="panel-meta">What the agent actually sent</div>',
                unsafe_allow_html=True,
            )
            st.text_area(
                "Reference reply",
                row["reference_reply"],
                height=280,
                disabled=True,
                label_visibility="collapsed",
            )
    with d3:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Generated reply</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="panel-meta">Model suggestion under evaluation</div>',
                unsafe_allow_html=True,
            )
            st.text_area(
                "Generated reply",
                row["generated_reply"],
                height=280,
                disabled=True,
                label_visibility="collapsed",
            )

    st.markdown("**Scorecard** *(1–5 dimensions → 0–100 accuracy)*")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Intent", f"{int(row['intent_coverage'])} / 5")
    s2.metric("Tone", f"{int(row['tone_fidelity'])} / 5")
    s3.metric("Correctness", f"{int(row['correctness'])} / 5")
    s4.metric("Conciseness", f"{int(row['conciseness'])} / 5")
    s5.metric("Accuracy", f"{row['accuracy_score']:.0f}")


if __name__ == "__main__":
    main()
