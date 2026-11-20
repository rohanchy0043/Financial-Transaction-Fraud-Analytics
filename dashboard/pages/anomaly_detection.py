import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.ui import require_dataset
from dashboard.utils.anomaly import detect_anomalies
from dashboard.utils.data_loader import REQUIRED_COLUMNS
from dashboard.utils.ml_models import dataset_fingerprint


def _render_iqr(frame: pd.DataFrame) -> None:
    numeric = frame.select_dtypes(include="number").columns.tolist()
    features = [column for column in numeric if column != "Class"]
    st.subheader("Statistical outliers · IQR")
    if not features:
        st.info("IQR outlier detection needs at least one numeric feature.")
        return
    feature = st.selectbox("Feature for IQR analysis", features, key="iqr_feature")
    multiplier = st.slider(
        "IQR multiplier",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.25,
        key="iqr_multiplier",
        help="An outlier is below Q1 − multiplier × IQR or above Q3 + multiplier × IQR.",
    )
    values = pd.to_numeric(frame[feature], errors="coerce")
    valid = values.loc[np.isfinite(values)]
    if valid.empty:
        st.info(f"No finite numeric values are available for {feature}.")
        return
    q1, q3 = valid.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    flagged = values.lt(lower) | values.gt(upper)
    flagged = flagged.fillna(False)
    outliers = frame.loc[flagged].copy()
    metrics = st.columns(3)
    metrics[0].metric("IQR outliers", f"{len(outliers):,}")
    metrics[1].metric("Outlier rate", f"{100 * len(outliers) / max(len(frame), 1):.3f}%")
    metrics[2].metric("Observed bounds", f"{lower:,.3f} to {upper:,.3f}")
    if "Class" in frame:
        labels = pd.to_numeric(frame["Class"], errors="coerce")
        valid_labels = labels.isin([0, 1])
        metrics_row = pd.DataFrame(
            {
                "IQR flagged": flagged.loc[valid_labels],
                "Observed class": labels.loc[valid_labels].map({0: "Legitimate", 1: "Fraud"}),
            }
        )
        comparison = (
            metrics_row.groupby(["IQR flagged", "Observed class"])
            .size()
            .reset_index(name="Transactions")
        )
        st.plotly_chart(
            px.bar(
                comparison,
                x="Observed class",
                y="Transactions",
                color="IQR flagged",
                barmode="group",
                color_discrete_map={True: "#111111", False: "#A3A3A3"},
                title="IQR outliers compared with observed labels",
            ),
            width="stretch",
        )
    display = outliers.copy()
    display["IQR Outlier"] = True
    st.dataframe(display.head(500), width="stretch", hide_index=True)
    st.caption("IQR outliers are statistical extremes and are not automatically fraudulent.")


def _render_isolation_forest(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    st.subheader("Isolation Forest")
    if missing:
        st.info(
            "Isolation Forest on this project uses the original dataset schema "
            "(Time, Amount, V1–V28, Class). IQR analysis above works on the current "
            "dataset's numeric columns. Missing model columns: "
            + ", ".join(missing)
        )
        return
    contamination = st.slider(
        "Expected anomaly share (algorithm setting)",
        0.001,
        0.10,
        0.01,
        0.001,
        key="isolation_contamination",
    )
    st.caption(
        "Contamination is an Isolation Forest configuration assumption, not an estimate "
        "of fraud prevalence."
    )
    if st.button("Run Isolation Forest", type="primary"):
        try:
            result, _ = detect_anomalies(frame, contamination)
            st.session_state["anomaly_result"] = {
                "fingerprint": dataset_fingerprint(frame),
                "contamination": contamination,
                "frame": result,
            }
        except ValueError as exc:
            st.error(str(exc))
    cached = st.session_state.get("anomaly_result")
    if (
        not cached
        or cached["fingerprint"] != dataset_fingerprint(frame)
        or cached["contamination"] != contamination
    ):
        st.info("Run Isolation Forest to calculate anomaly scores from the current dataset.")
        return
    result = cached["frame"]
    anomalies = result.loc[result["Anomaly"]]
    metrics = st.columns(3)
    metrics[0].metric("Detected anomalies", f"{len(anomalies):,}")
    metrics[1].metric(
        "Anomaly percentage",
        f"{100 * len(anomalies) / max(len(result), 1):.3f}%",
    )
    actual_labels = pd.to_numeric(result["Class"], errors="coerce")
    metrics[2].metric(
        "Detected anomalies labeled fraud",
        f"{int((result['Anomaly'] & actual_labels.eq(1)).sum()):,}",
    )
    table = result.groupby(["Anomaly", "Class"], dropna=False).size().reset_index(name="Transactions")
    st.plotly_chart(
        px.bar(
            table,
            x="Class",
            y="Transactions",
            color="Anomaly",
            barmode="group",
            color_discrete_map={True: "#111111", False: "#A3A3A3"},
            title="Isolation Forest result vs. observed label",
        ),
        width="stretch",
    )
    st.plotly_chart(
        px.histogram(
            result,
            x="Anomaly Score",
            color="Anomaly",
            nbins=48,
            color_discrete_map={True: "#111111", False: "#A3A3A3"},
            title="Isolation Forest anomaly score distribution",
        ),
        width="stretch",
    )
    anomaly_filter = st.selectbox(
        "Show records",
        ["Detected anomalies", "All records"],
        key="anomaly_records_filter",
    )
    display = anomalies if anomaly_filter == "Detected anomalies" else result
    st.dataframe(display.head(500), width="stretch", hide_index=True)
    st.caption("Anomaly detection and fraud classification answer different questions; unusual records are not automatically fraud.")


def render() -> None:
    st.title("Anomaly Detection")
    frame = require_dataset()
    if frame is None:
        return
    st.caption("Statistical outliers describe feature extremes; Isolation Forest identifies records unusual across the original model feature set.")
    _render_iqr(frame)
    _render_isolation_forest(frame)
