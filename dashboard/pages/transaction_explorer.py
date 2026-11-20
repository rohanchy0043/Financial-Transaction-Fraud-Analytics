import pandas as pd
import streamlit as st

from dashboard.components.ui import require_dataset
from dashboard.utils.analytics import filter_transactions


def render() -> None:
    st.title("Transaction Explorer")
    frame = require_dataset()
    if frame is None:
        return
    selected = frame
    if "Amount" in frame:
        amount = pd.to_numeric(frame["Amount"], errors="coerce").dropna()
        if not amount.empty and amount.min() < amount.max():
            bounds = st.slider("Transaction amount", float(amount.min()), float(amount.max()),
                               (float(amount.min()), float(amount.max())), key="explorer_amount")
            selected = filter_transactions(selected, bounds[0], bounds[1])
    if "Class" in frame:
        status = st.selectbox("Fraud status", ["All", "Legitimate", "Fraud"])
        selected = filter_transactions(selected, fraud_status=status)
    search = st.text_input("Search visible transaction values")
    if search:
        mask = selected.astype(str).apply(
            lambda column: column.str.contains(search, case=False, regex=False, na=False)
        ).any(axis=1)
        selected = selected.loc[mask]

    page_size = st.selectbox("Rows per page", [25, 50, 100, 250], index=0)
    page_count = max(1, (len(selected) + page_size - 1) // page_size)
    page_number = st.number_input("Page", min_value=1, max_value=page_count, value=1)
    start = (int(page_number) - 1) * page_size
    visible = selected.iloc[start:start + page_size]
    st.caption(f"{len(selected):,} filtered records | page {page_number} of {page_count}")
    st.dataframe(visible, width="stretch", hide_index=True, height=440)
    if not visible.empty:
        detail_position = st.selectbox(
            "Inspect visible record", range(len(visible)),
            format_func=lambda position: f"Record {start + position + 1}",
        )
        st.subheader("Transaction details")
        st.dataframe(visible.iloc[detail_position].to_frame("Value"),
                     width="stretch")
    st.download_button("Download filtered CSV", selected.to_csv(index=False).encode("utf-8"),
                       "filtered_transactions.csv", "text/csv")
