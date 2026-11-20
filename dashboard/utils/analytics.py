from __future__ import annotations

import numpy as np
import pandas as pd

from dashboard.utils.data_loader import TARGET_COLUMN


def calculate_kpis(frame: pd.DataFrame) -> dict:
    amount = pd.to_numeric(frame.get("Amount", pd.Series(dtype=float)), errors="coerce")
    amount = amount.where(np.isfinite(amount))
    total = len(frame)
    has_amounts = amount.notna().any()
    total_amount = (
        float(amount.sum())
        if has_amounts
        else 0.0 if frame.empty and "Amount" in frame.columns
        else None
    )
    average_amount = float(amount.mean()) if has_amounts else None
    if TARGET_COLUMN in frame:
        labels = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        fraud = int(labels.eq(1).sum())
        legitimate = int(labels.eq(0).sum())
        valid_labels = fraud + legitimate
    else:
        fraud, legitimate, valid_labels = 0, 0, 0
    return {
        "total_transactions": total,
        "total_amount": total_amount,
        "average_amount": average_amount,
        "fraudulent_transactions": fraud,
        "legitimate_transactions": legitimate,
        "fraud_rate": (fraud / valid_labels * 100) if valid_labels else None,
        "legitimate_percentage": (legitimate / valid_labels * 100) if valid_labels else None,
    }


def filter_transactions(
    frame: pd.DataFrame,
    minimum_amount: float | None = None,
    maximum_amount: float | None = None,
    fraud_status: str = "All",
    minimum_time: float | None = None,
    maximum_time: float | None = None,
) -> pd.DataFrame:
    filtered = frame.copy()
    if "Amount" in filtered.columns:
        amounts = pd.to_numeric(filtered["Amount"], errors="coerce")
        if minimum_amount is not None:
            filtered = filtered.loc[amounts >= minimum_amount]
            amounts = pd.to_numeric(filtered["Amount"], errors="coerce")
        if maximum_amount is not None:
            filtered = filtered.loc[amounts <= maximum_amount]
    if "Time" in filtered.columns:
        elapsed = pd.to_numeric(filtered["Time"], errors="coerce")
        if minimum_time is not None:
            filtered = filtered.loc[elapsed >= minimum_time]
            elapsed = pd.to_numeric(filtered["Time"], errors="coerce")
        if maximum_time is not None:
            filtered = filtered.loc[elapsed <= maximum_time]
    if fraud_status != "All" and TARGET_COLUMN in filtered.columns:
        label = 1 if fraud_status == "Fraud" else 0
        filtered = filtered.loc[pd.to_numeric(filtered[TARGET_COLUMN], errors="coerce") == label]
    return filtered


def evidence_based_insights(frame: pd.DataFrame) -> list[str]:
    kpis = calculate_kpis(frame)
    labeled_count = kpis["fraudulent_transactions"] + kpis["legitimate_transactions"]
    if kpis["fraud_rate"] is None:
        insights = ["Fraud rate is unavailable because no valid Class 0/1 labels were found."]
    else:
        insights = [
            f"Fraud rate is {kpis['fraud_rate']:.4f}% "
            f"({kpis['fraudulent_transactions']:,} of {labeled_count:,} labeled records)."
        ]
    if "Amount" in frame and TARGET_COLUMN in frame:
        valid = frame[["Amount", TARGET_COLUMN]].apply(pd.to_numeric, errors="coerce").dropna()
        fraud_amounts = valid.loc[valid[TARGET_COLUMN] == 1, "Amount"]
        legit_amounts = valid.loc[valid[TARGET_COLUMN] == 0, "Amount"]
        if not fraud_amounts.empty and not legit_amounts.empty:
            insights.append(
                f"Median transaction amount: fraud ${fraud_amounts.median():,.2f}; "
                f"legitimate ${legit_amounts.median():,.2f}. "
                "This is a descriptive difference, not evidence of causation."
            )
    if "Time" in frame:
        time = pd.to_numeric(frame["Time"], errors="coerce").dropna()
        if not time.empty:
            insights.append(
                f"The observed Time field spans {time.min():,.0f}–{time.max():,.0f} seconds "
                "since the dataset's reference point; it is not a calendar timestamp."
            )
    return insights
