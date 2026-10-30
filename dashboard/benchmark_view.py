"""Streamlit P2 visual analytics for completed benchmark runs."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

def render_benchmarks(root="storage/benchmarks"):
    st.subheader("P2 · Multi-seed benchmark comparison")
    folders = sorted((p for p in Path(root).glob("benchmark-*") if (p/"summary.json").exists()), reverse=True)
    if not folders:
        st.info("No completed benchmarks. Run: python scripts/benchmark.py --epochs 3")
        return
    selected = st.selectbox("Benchmark run", folders, format_func=lambda p:p.name)
    manifest = json.loads((selected/"manifest.json").read_text())
    st.caption(f"Dataset SHA-256: {manifest['dataset_sha256']} | seeds: {manifest['seeds']} | epochs: {manifest['epochs']}")
    summary = pd.read_json(selected/"summary.json")
    st.dataframe(summary, use_container_width=True)
    st.bar_chart(summary.set_index("operator")[["macro_f1_mean", "accuracy_mean"]])
    st.bar_chart(summary.set_index("operator")[["elapsed_mean_seconds"]])
    st.bar_chart(summary.set_index("operator")[["parameter_count"]])
    st.caption("The leaderboard is descriptive. Equal epochs do not imply equal compute; model selection on the test set causes bias.")
    runs_path = selected/"runs.json"
    if runs_path.exists():
        runs = json.loads(runs_path.read_text())
        selected_operator = st.selectbox("Learning curves", sorted({r["operator"] for r in runs}))
        for r in runs:
            if r["operator"] == selected_operator:
                with st.expander(f"Seed {r['seed']}"):
                    df = pd.DataFrame(r["history"]).set_index("epoch")
                    st.line_chart(df[["train_loss", "val_loss"]])
                    st.line_chart(df[["val_accuracy", "val_macro_f1"]])
