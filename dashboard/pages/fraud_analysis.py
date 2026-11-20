import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.ui import require_dataset


def render() -> None:
    st.title("Fraud Analysis")
    frame = require_dataset()
    if frame is None:
        return
    if "Class" not in frame:
        st.warning("This dataset has no fraud-label column. Map one on Upload Dataset to enable fraud analysis.")
        return
    labels = pd.to_numeric(frame["Class"], errors="coerce")
    labeled = frame.loc[labels.isin([0, 1])].copy()
    labeled["Status"] = labels.loc[labeled.index].map({0: "Legitimate", 1: "Fraud"})
    fraud = labeled.loc[labeled["Status"] == "Fraud"]
    legitimate = labeled.loc[labeled["Status"] == "Legitimate"]
    count = len(labeled)
    metrics = st.columns(3)
    metrics[0].metric("Fraudulent transactions", f"{len(fraud):,}")
    metrics[1].metric("Fraud percentage", f"{(100 * len(fraud) / count) if count else 0:.4f}%")
    metrics[2].metric("Labeled records", f"{count:,}")
    if fraud.empty:
        st.info("No fraud-labeled records are present in the selected dataset.")
        return
    if "Amount" in labeled:
        amount = pd.to_numeric(labeled["Amount"], errors="coerce")
        fraud_amount = amount.loc[fraud.index].dropna()
        legit_amount = amount.loc[legitimate.index].dropna()
        c1, c2, c3 = st.columns(3)
        c1.metric("Fraud median amount", f"${fraud_amount.median():,.2f}" if not fraud_amount.empty else "N/A")
        c2.metric("Fraud mean amount", f"${fraud_amount.mean():,.2f}" if not fraud_amount.empty else "N/A")
        c3.metric("Fraud maximum amount", f"${fraud_amount.max():,.2f}" if not fraud_amount.empty else "N/A")
        st.plotly_chart(px.box(labeled, x="Status", y="Amount", color="Status",
                               title="Transaction amounts by label", points=False),
                        width="stretch")
    available = [column for column in labeled.select_dtypes(include="number").columns
                 if column not in ("Class",)]
    if available:
        feature = st.selectbox("Compare a numeric feature", available, key="fraud_feature")
        st.plotly_chart(px.histogram(labeled, x=feature, color="Status", barmode="overlay",
                                     opacity=0.7, nbins=50,
                                     title=f"{feature} distribution by observed label"),
                        width="stretch")
    st.caption("These are descriptive comparisons of the supplied labels; correlation or distribution differences do not establish cause.")
