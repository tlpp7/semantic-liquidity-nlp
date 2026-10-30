"""Run: streamlit run dashboard/app.py"""
import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st
from semantic_liquidity import OPERATOR_NAMES

st.set_page_config(page_title="Semantic Liquidity Lab", layout="wide")
st.title("Semantic Liquidity Lab")
st.caption("P0 · Experimental NLP laboratory · metrics are displayed only when available")
datasets, operators, metrics = st.tabs(["Datasets", "Operators", "Metrics"])
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
with operators:
    selected = st.selectbox("Neural operator", OPERATOR_NAMES)
    st.code(f"SemanticLiquidityLayer(dim=64, operator='{selected}')")
    st.info("P0 validates differentiable operators; full benchmark training is a later milestone.")
with metrics:
    db = Path("storage/registry.sqlite")
    if db.exists():
        with sqlite3.connect(db) as con:
            st.dataframe(pd.read_sql_query("SELECT * FROM datasets ORDER BY id DESC", con))
    else:
        st.info("No dataset imported yet.")
