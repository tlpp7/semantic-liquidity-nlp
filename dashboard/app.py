"""Run: streamlit run dashboard/app.py"""
import json
import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st
from semantic_liquidity import OPERATOR_NAMES

st.set_page_config(page_title="Semantic Liquidity Lab", layout="wide")
st.title("Semantic Liquidity Lab")
st.caption("P1 · Dataset registry, supervised training and reproducible experiment metrics")
datasets, training, metrics, benchmarks, continual = st.tabs(["Datasets", "Training", "Metrics", "Benchmarks", "Continual Learning"])
with datasets:
    st.subheader("Import CSV")
    upload = st.file_uploader("CSV file", type="csv")
    text_col = st.text_input("Text column", "text")
    label_col = st.text_input("Label column", "target")
    if upload and st.button("Import dataset"):
        from semantic_liquidity.data import ingest_csv
        target = Path("storage/uploads") / Path(upload.name).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(upload.getvalue())
        try:
            st.success(ingest_csv(target, text_col, label_col))
        except ValueError as e:
            st.error(str(e))
with training:
    operator = st.selectbox("Operator", ["baseline", *OPERATOR_NAMES])
    epochs = st.slider("Epochs", 1, 30, 5)
    seed = st.number_input("Seed", min_value=0, max_value=100000, value=42)
    dataset = st.text_input("Normalized JSONL dataset", "storage/dataset.jsonl")
    if st.button("Start training", type="primary"):
        from semantic_liquidity.train import run_training
        try:
            with st.spinner("Training model..."):
                result = run_training(dataset, operator, epochs, int(seed))
            st.success(f"Saved checkpoint: {result['checkpoint']}")
            st.metric("Test accuracy", f"{result['test_accuracy']:.1%}")
            st.metric("Test macro-F1", f"{result['test_macro_f1']:.3f}")
            st.line_chart(pd.DataFrame(result["history"]).set_index("epoch")[["train_loss", "val_loss"]])
        except (ValueError, FileNotFoundError) as exc:
            st.error(str(exc))
with metrics:
    registry = Path("storage/registry.sqlite")
    if registry.exists():
        with sqlite3.connect(registry) as con:
            st.subheader("Imported datasets")
            st.dataframe(pd.read_sql_query("SELECT * FROM datasets ORDER BY id DESC", con))
    experiments = Path("storage/runs/experiments.sqlite")
    if experiments.exists():
        with sqlite3.connect(experiments) as con:
            st.subheader("Experiment comparison")
            st.dataframe(pd.read_sql_query("SELECT * FROM runs ORDER BY run_id DESC", con))
        for path in sorted(Path("storage/runs").glob("*.json"), reverse=True)[:10]:
            item = json.loads(path.read_text(encoding="utf-8"))
            with st.expander(item["run_id"]):
                df = pd.DataFrame(item["history"]).set_index("epoch")
                st.line_chart(df[["train_loss", "val_loss"]])
                st.line_chart(df[["val_accuracy", "val_macro_f1"]])
    else:
        st.info("No experiments yet. Import a dataset and run training.")

with benchmarks:
    from benchmark_view import render_benchmarks
    render_benchmarks()

with continual:
    st.subheader("P3 · Continual learning and semantic stability")
    task_count = st.slider("Sequential tasks", 2, 8, 3)
    task_epochs = st.slider("Epochs per task", 1, 10, 2)
    ewc_strength = st.number_input("EWC strength", min_value=0.0, max_value=100.0, value=1.0)
    continual_operator = st.selectbox("Continual operator", ["baseline", *OPERATOR_NAMES])
    if st.button("Run continual learning"):
        from semantic_liquidity.train import load_records
        from semantic_liquidity.continual import run_continual
        try:
            with st.spinner("Training sequential tasks..."):
                result = run_continual(load_records("storage/dataset.jsonl"), operator=continual_operator,
                                       n_tasks=task_count, epochs=task_epochs, ewc_lambda=ewc_strength)
            st.metric("Mean forgetting (training-set proxy)", f"{result['mean_forgetting']:.3f}")
            matrix = pd.DataFrame(result["accuracy_matrix"],
                                  columns=[f"Task {i+1}" for i in range(task_count)])
            matrix.index = [f"After task {i+1}" for i in range(task_count)]
            st.dataframe(matrix)
            st.line_chart(matrix)
            st.warning("Exploratory proxy: evaluation uses task training records, not independent held-out samples.")
        except (ValueError, FileNotFoundError) as exc:
            st.error(str(exc))
    st.subheader("Saved continual experiments")
    for path in sorted(Path("storage/continual").glob("*.json"), reverse=True)[:15]:
        item = json.loads(path.read_text(encoding="utf-8"))
        with st.expander(path.name):
            st.json(item)
