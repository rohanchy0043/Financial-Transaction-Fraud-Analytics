from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from dashboard.utils.analytics import calculate_kpis, filter_transactions
from dashboard.utils.data_loader import load_bundled_dataset


def current_dataset() -> pd.DataFrame | None:
    return st.session_state.get("active_dataset")


def require_dataset() -> pd.DataFrame | None:
    frame = current_dataset()
    if frame is None:
        st.markdown(
            "<div class='empty-state'><div class='empty-mark'>FA</div>"
            "<h3>No dataset loaded</h3>"
            "<p>Upload the public Credit Card Fraud Detection dataset or a compatible "
            "CSV/Excel file to begin exploring actual transaction activity.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button(
            "Upload dataset",
            key="empty_state_upload",
            type="primary",
            width="content",
        ):
            st.session_state["active_page"] = "Upload Dataset"
            st.rerun()
        return None
    return frame


def render_kpis(frame: pd.DataFrame) -> None:
    metrics = calculate_kpis(frame)

    def card(column, label: str, value: str, note: str) -> None:
        with column:
            st.metric(label, value)
            st.markdown(f"<div class='metric-note'>{note}</div>", unsafe_allow_html=True)

    columns = st.columns(3)
    card(columns[0], "Total Transactions", f"{metrics['total_transactions']:,}", "Records matching active filters.")
    total_amount = metrics["total_amount"]
    average_amount = metrics["average_amount"]
    card(columns[1], "Total Transaction Amount", f"${total_amount:,.2f}" if total_amount is not None else "Unavailable", "Sum of available numeric amounts.")
    card(columns[2], "Average Transaction Amount", f"${average_amount:,.2f}" if average_amount is not None else "Unavailable", "Mean of available numeric amounts.")
    columns = st.columns(3)
    card(
        columns[0],
        "Fraudulent Transactions",
        f"{metrics['fraudulent_transactions']:,}" if "Class" in frame.columns else "Unavailable",
        "Observed records mapped to Class 1." if "Class" in frame.columns else "No fraud-label field is mapped.",
    )
    card(
        columns[1],
        "Legitimate Transactions",
        f"{metrics['legitimate_transactions']:,}" if "Class" in frame.columns else "Unavailable",
        "Observed records mapped to Class 0." if "Class" in frame.columns else "No fraud-label field is mapped.",
    )
    card(
        columns[2],
        "Fraud Rate",
        f"{metrics['fraud_rate']:.4f}%" if metrics["fraud_rate"] is not None else "Unavailable",
        "Class 1 ÷ all records labeled 0 or 1." if "Class" in frame.columns else "No fraud-label field is mapped.",
    )


def apply_overview_filters(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the dashboard's active filters to a copy of the active dataset."""
    amount_range = st.session_state.get("overview_amount_range")
    time_range = st.session_state.get("overview_time_range")
    return filter_transactions(
        frame,
        minimum_amount=amount_range[0] if amount_range else None,
        maximum_amount=amount_range[1] if amount_range else None,
        fraud_status=st.session_state.get("overview_fraud_status", "All"),
        minimum_time=time_range[0] if time_range else None,
        maximum_time=time_range[1] if time_range else None,
    )


def reset_overview_filters() -> None:
    for key in (
        "overview_fraud_status",
        "overview_amount_range",
        "overview_time_range",
    ):
        st.session_state.pop(key, None)


def select_overview_dataset() -> None:
    dataset_name = st.session_state.get("overview_dataset")
    datasets = st.session_state.get("datasets", {})
    if dataset_name not in datasets:
        return
    st.session_state["active_dataset"] = datasets[dataset_name]
    st.session_state["dataset_name"] = dataset_name
    for key in ("trained_models", "anomaly_result", "high_risk_records", "sql_fingerprint"):
        st.session_state.pop(key, None)
    reset_overview_filters()


def render_page_topbar(page_name: str, frame: pd.DataFrame | None) -> None:
    breadcrumb, actions = st.columns([3, 2])
    breadcrumb.markdown(
        f"<div class='page-breadcrumb'>Workspace <span>/</span> {escape(page_name)}</div>",
        unsafe_allow_html=True,
    )
    if frame is None:
        dataset_name = "No dataset loaded"
    else:
        dataset_name = st.session_state.get("dataset_name", "Uploaded dataset")
    with actions:
        name, status, refresh = st.columns([3, 2, 1.4])
        safe_name = escape(str(dataset_name))
        name.markdown(f"<div class='dataset-name' title='{safe_name}'>DATASET<br><strong>{safe_name}</strong></div>", unsafe_allow_html=True)
        status.markdown(
            "<div class='dataset-status'><span></span> Loaded</div>"
            if frame is not None
            else "<div class='dataset-status dataset-status-empty'><span></span> No data</div>",
            unsafe_allow_html=True,
        )
        if refresh.button("Refresh", key="refresh_dashboard", help="Refresh current dataset", type="secondary"):
            bundled = load_bundled_dataset()
            if bundled is not None and dataset_name == "data/raw/creditcard.csv":
                st.session_state["active_dataset"] = bundled
                st.session_state.setdefault("datasets", {})[dataset_name] = bundled
            for key in ("trained_models", "anomaly_result", "high_risk_records", "sql_fingerprint"):
                st.session_state.pop(key, None)
            st.rerun()
        if frame is not None:
            report = pd.DataFrame([{
                "Dataset": dataset_name,
                **calculate_kpis(frame),
            }])
            refresh.download_button(
                "Export report",
                report.to_csv(index=False).encode("utf-8"),
                "financial_overview_report.csv",
                "text/csv",
                key="topbar_export",
                help="Export actual KPI calculations with the active dataset name.",
            )


def anomaly_and_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach existing analysis results by original row index without fabricating IDs."""
    result = frame.copy()
    anomaly = st.session_state.get("anomaly_result")
    active = current_dataset()
    if anomaly:
        detected = anomaly.get("frame")
        if (
            active is not None
            and detected is not None
            and anomaly.get("fingerprint") == _dataset_fingerprint(active)
            and detected.index.is_unique
        ):
            result = result.join(
                detected["Anomaly"].map({True: "Anomaly", False: "Typical"})
                .rename("Anomaly Status"),
                how="left",
            )
    high_risk = st.session_state.get("high_risk_records")
    if high_risk is not None and not high_risk.empty:
        prediction_columns = [
            column for column in ("Predicted Class", "Predicted Fraud Probability")
            if column in high_risk.columns
        ]
        if prediction_columns:
            result = result.join(high_risk[prediction_columns], how="left")
    return result


def _dataset_fingerprint(frame: pd.DataFrame) -> str:
    from dashboard.utils.ml_models import dataset_fingerprint

    return dataset_fingerprint(frame)


def model_required(frame: pd.DataFrame) -> bool:
    from dashboard.utils.data_loader import REQUIRED_COLUMNS

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        st.warning(
            "This feature requires the original model columns "
            "(Time, Amount, V1–V28, Class). No result is shown for an incompatible dataset."
        )
        return False
    return True
