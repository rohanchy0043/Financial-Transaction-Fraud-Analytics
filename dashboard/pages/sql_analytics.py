import streamlit as st
import pandas as pd

from dashboard.components.ui import require_dataset
from dashboard.utils.database import create_database_engine, replace_transactions, run_sql
from dashboard.utils.ml_models import dataset_fingerprint


def render() -> None:
    st.title("SQL Analytics")
    frame = require_dataset()
    if frame is None:
        return
    if not {"Amount", "Class"}.issubset(frame.columns):
        st.warning("SQL KPI examples need mapped Amount and Class columns.")
        return
    amounts = pd.to_numeric(frame["Amount"], errors="coerce")
    invalid_amounts = int((amounts.isna() & frame["Amount"].notna()).sum())
    if invalid_amounts:
        st.warning(f"{invalid_amounts:,} non-numeric amount values are present; SQL amount summaries may not match numeric Pandas results.")
    fingerprint = dataset_fingerprint(frame)
    if st.session_state.get("sql_fingerprint") != fingerprint:
        engine = create_database_engine()
        replace_transactions(engine, frame)
        st.session_state["sql_engine"] = engine
        st.session_state["sql_fingerprint"] = fingerprint
    query = st.selectbox("Prepared SQL analysis", [
        "Summary KPIs", "Amount by fraud status", "Amount deciles", "Time ranges"
    ])
    if query == "Time ranges" and "Time" not in frame:
        st.warning("Time ranges require the original Time field.")
        return
    result = run_sql(st.session_state["sql_engine"], query)
    st.dataframe(result, width="stretch", hide_index=True)
    st.caption("Queries execute against the current in-memory SQLite transaction table via SQLAlchemy.")
