"""Streamlit dashboard for the TrustDraft suggested-reply pipeline.

Run the full generate → reply → evaluate pipeline and inspect accuracy,
validation gap, and per-ticket scorecards.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PIPELINE = ROOT / "src" / "pipeline.py"


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def run_pipeline(n: int, dry_run: bool) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(PIPELINE), "--n", str(n)]
    if dry_run:
        cmd.append("--dry-run")
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def main() -> None:
    st.set_page_config(page_title="TrustDraft", layout="wide")
    st.title("TrustDraft")
    st.caption("AI suggested-reply accuracy dashboard")

    with st.sidebar:
        st.header("Pipeline")
        n = st.slider("Dataset size (--n)", min_value=5, max_value=80, value=40, step=5)
        dry_run = st.checkbox("Dry-run (no API calls)", value=True)
        run_clicked = st.button("Run pipeline", type="primary", use_container_width=True)

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

    evaluations = load_json(DATA / "evaluations.json")
    validation = load_json(DATA / "validation.json")
    generated = load_json(DATA / "generated.json") or []

    if not evaluations or not validation:
        st.info("No results yet. Run the pipeline from the sidebar.")
        return

    gen_by_id = {g["id"]: g for g in generated}
    df = pd.DataFrame(evaluations)

    # --- summary metrics ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test set size", int(validation.get("n", len(df))))
    m2.metric("Avg accuracy", f"{validation['avg_accuracy_correct']:.1f}")
    m3.metric("Avg lexical overlap", f"{validation['avg_rouge_l_correct']:.3f}")
    m4.metric("Validation gap", f"{validation['accuracy_gap']:.1f}")

    # --- charts ---
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Accuracy by category")
        if "category" in df.columns and df["category"].notna().any():
            by_cat = (
                df.groupby("category", as_index=False)["accuracy_score"]
                .mean()
                .rename(columns={"accuracy_score": "avg_accuracy"})
            )
            st.bar_chart(by_cat.set_index("category"))
        else:
            st.write("No category data.")
    with c2:
        st.subheader("Score distribution")
        hist = (
            pd.cut(df["accuracy_score"], bins=[0, 20, 40, 60, 80, 100], include_lowest=True)
            .value_counts()
            .sort_index()
        )
        hist_df = pd.DataFrame({"count": hist.astype(int).values}, index=hist.index.astype(str))
        st.bar_chart(hist_df)

    # --- validation panel ---
    st.subheader("Validation")
    v1, v2, v3 = st.columns(3)
    v1.metric("Correct pairs (avg)", f"{validation['avg_accuracy_correct']:.1f}")
    v2.metric("Mismatched pairs (avg)", f"{validation['avg_accuracy_mismatched']:.1f}")
    v3.metric("Gap (correct − mismatched)", f"{validation['accuracy_gap']:.1f}")
    st.info(validation.get("interpretation", ""))
    if validation.get("note"):
        st.caption(validation["note"])
    if validation.get("dry_run"):
        st.warning("These results include dry-run placeholder judge scores.")

    # --- filterable table ---
    st.subheader("Results")
    categories = sorted({c for c in df["category"].dropna().unique()})
    f1, f2 = st.columns(2)
    with f1:
        selected_cats = st.multiselect("Filter category", categories, default=categories)
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
            "rouge_l_f1",
            "intent_coverage",
            "tone_fidelity",
            "correctness",
            "conciseness",
        ]
        if c in filtered.columns
    ]
    st.dataframe(filtered[show_cols], use_container_width=True, hide_index=True)

    # --- detail view ---
    st.subheader("Ticket detail")
    ids = filtered["id"].tolist() if len(filtered) else df["id"].tolist()
    if not ids:
        st.write("No tickets match the current filters.")
        return

    ticket_id = st.selectbox("Select ticket", ids)
    row = df[df["id"] == ticket_id].iloc[0]
    gen = gen_by_id.get(ticket_id, {})
    incoming_body = gen.get("body") or ""
    incoming_subject = row.get("subject") or gen.get("subject", "")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**Incoming email**")
        st.markdown(f"**Subject:** {incoming_subject}")
        st.text(incoming_body or "(body not in generated.json)")
    with d2:
        st.markdown("**Reference reply**")
        st.text(row["reference_reply"])
    with d3:
        st.markdown("**Generated reply**")
        st.text(row["generated_reply"])

    st.markdown("**Scorecard**")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Intent", int(row["intent_coverage"]))
    s2.metric("Tone", int(row["tone_fidelity"]))
    s3.metric("Correctness", int(row["correctness"]))
    s4.metric("Conciseness", int(row["conciseness"]))
    s5.metric("Accuracy", f"{row['accuracy_score']:.0f}")


if __name__ == "__main__":
    main()
