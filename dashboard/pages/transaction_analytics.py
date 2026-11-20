import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.ui import require_dataset


def render() -> None:
    st.title("Transaction Analytics")
    frame = require_dataset()
    if frame is None:
        return
    if "Amount" not in frame:
        st.warning("No amount column is mapped. Map an amount field on Upload Dataset.")
        return

    amount = pd.to_numeric(frame["Amount"], errors="coerce").dropna()
    if amount.empty:
        st.warning("The amount column contains no usable numeric values.")
        return
    stats = st.columns(4)
    stats[0].metric("Mean", f"${amount.mean():,.2f}")
    stats[1].metric("Median", f"${amount.median():,.2f}")
    stats[2].metric("Minimum", f"${amount.min():,.2f}")
    stats[3].metric("Maximum", f"${amount.max():,.2f}")

    min_amount, max_amount = float(amount.min()), float(amount.max())
    low, high = st.slider(
        "Amount range", min_value=min_amount, max_value=max_amount,
        value=(min_amount, max_amount), key="analytics_amount_range",
    )
    selected = frame.loc[
        pd.to_numeric(frame["Amount"], errors="coerce").between(low, high)
    ].copy()
    st.caption(f"{len(selected):,} of {len(frame):,} records in the selected range.")

    left, right = st.columns(2)
    left.plotly_chart(px.histogram(selected, x="Amount", nbins=50, title="Amount histogram"),
                      width="stretch")
    if "Class" in selected:
        selected["Fraud status"] = selected["Class"].map({0: "Legitimate", 1: "Fraud"})
        right.plotly_chart(px.box(selected, x="Fraud status", y="Amount",
                                  title="Amount by fraud status", points=False),
                           width="stretch")
    else:
        right.info("Status comparison requires a mapped fraud-label column.")

    if "Time" in selected:
        selected["Elapsed hour"] = (
            pd.to_numeric(selected["Time"], errors="coerce") // 3600
        )
        hourly = selected.dropna(subset=["Elapsed hour"]).groupby("Elapsed hour").size().reset_index(name="Transactions")
        st.plotly_chart(px.line(hourly, x="Elapsed hour", y="Transactions",
                                title="Transaction frequency by elapsed hour"),
                        width="stretch")
    q1, q3 = amount.quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = amount[(amount < q1 - 1.5 * iqr) | (amount > q3 + 1.5 * iqr)]
    st.metric("Amount outliers by 1.5×IQR rule", f"{len(outliers):,}")
    st.caption("The IQR rule flags statistical outliers; it does not determine fraud.")
