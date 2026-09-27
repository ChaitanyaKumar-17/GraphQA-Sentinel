"""
dashboard/app.py

Streamlit dashboard (M6) that reads eval/regression_log.json and plots
RAGAS metric trends across recorded eval runs, so a regression (or an
improvement, like the M5 agentic upgrade) is visible at a glance.

Run with: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Allow running `streamlit run dashboard/app.py` directly from the repo
# root without needing the project installed as a package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.store import load_log

st.set_page_config(page_title="GraphQA-Sentinel Eval Dashboard", layout="wide")
st.title("GraphQA-Sentinel — Eval Regression Dashboard")

log = load_log()

if not log:
    st.warning(
        "No eval runs recorded yet. Run `python -m eval.run_eval --label baseline` "
        "(or `--label agentic`) at least once to populate eval/regression_log.json."
    )
    st.stop()

df = pd.DataFrame(log)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df["run"] = df.index + 1  # simple run counter, cleaner on the x-axis than raw timestamps

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

st.sidebar.header("Filters")
labels = sorted(df["label"].unique())
selected_labels = st.sidebar.multiselect("Pipeline label", labels, default=labels)
selected_metrics = st.sidebar.multiselect("Metrics", METRICS, default=METRICS)

filtered = df[df["label"].isin(selected_labels)]

if filtered.empty:
    st.info("No runs match the current filters.")
    st.stop()

st.subheader("Metric trends across runs")

for metric in selected_metrics:
    fig = px.line(
        filtered,
        x="run",
        y=metric,
        color="label",
        markers=True,
        hover_data=["timestamp", "commit", "n_questions"],
        title=metric.replace("_", " ").title(),
    )
    fig.update_layout(yaxis_range=[0, 1], xaxis_title="Run #", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Raw run log")
display_cols = ["run", "timestamp", "label", "commit", "n_questions"] + METRICS
st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

if len(selected_labels) >= 2:
    st.subheader("Latest run per label")
    latest_per_label = filtered.sort_values("timestamp").groupby("label").tail(1)
    st.dataframe(latest_per_label[display_cols], use_container_width=True, hide_index=True)