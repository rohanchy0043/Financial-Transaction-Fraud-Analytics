import pandas as pd
import streamlit as st

from dashboard.components.ui import require_dataset
from dashboard.utils.analytics import calculate_kpis, evidence_based_insights
from dashboard.utils.report_generator import dataframe_to_excel, text_report_to_pdf
from dashboard.utils.validation import validate_dataset


def render() -> None:
    st.title("Reports & Exports")
    frame = require_dataset()
    if frame is None:
        return
    report_type = st.selectbox("Report", [
        "Transaction Analysis Report", "Fraud Analysis Report",
        "Data Quality Report", "Machine Learning Evaluation Report",
        "High-Risk Transaction Report",
    ])
    trained = st.session_state.get("trained_models")
    if report_type == "Machine Learning Evaluation Report":
        if trained is None or trained.get("test_evaluation") is None:
            st.info("Select a model and evaluate it on the reserved test set on Model Performance before exporting.")
            return
        result = trained["test_evaluation"]
        report = pd.DataFrame([{
            "Model": trained["test_model_name"],
            **{key: result[key] for key in
               ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]},
        }])
    elif report_type == "High-Risk Transaction Report":
        high_risk = st.session_state.get("high_risk_records")
        if high_risk is None:
            st.info("No high-risk results are available. Create predictions on the Fraud Prediction page.")
            return
        report = high_risk.copy()
    elif report_type == "Data Quality Report":
        quality = validate_dataset(frame)
        report = pd.DataFrame([{"Metric": key, "Value": str(value)} for key, value in quality.items()])
    elif report_type == "Fraud Analysis Report":
        if not {"Class", "Amount"}.issubset(frame.columns):
            st.warning("Map both fraud label and Amount before generating this report.")
            return
        report = frame.groupby("Class", dropna=False).agg(
            transaction_count=("Class", "size"),
            total_amount=("Amount", "sum") if "Amount" in frame else ("Class", "size"),
        ).reset_index()
    else:
        report = pd.DataFrame([{"Metric": name, "Value": value} for name, value in calculate_kpis(frame).items()])
        report["Dataset"] = st.session_state.get("dataset_name", "data/raw/creditcard.csv")

    st.caption(f"Dataset: {st.session_state.get('dataset_name', 'data/raw/creditcard.csv')} | Records: {len(frame):,}")
    st.dataframe(report, width="stretch", hide_index=True)
    csv = report.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "analytics_report.csv", "text/csv")
    st.download_button("Download Excel", dataframe_to_excel(report, "Report"),
                       "analytics_report.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    lines = [f"Dataset: {st.session_state.get('dataset_name', 'data/raw/creditcard.csv')}"]
    lines.extend(f"{row[0]}: {row[1]}" for row in report.astype(str).itertuples(index=False, name=None))
    lines.extend(evidence_based_insights(frame))
    st.download_button("Download PDF", text_report_to_pdf(report_type, lines),
                       "analytics_report.pdf", "application/pdf")
