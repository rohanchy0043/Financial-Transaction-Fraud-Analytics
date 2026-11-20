import pandas as pd
import streamlit as st
from zipfile import BadZipFile

from dashboard.utils.data_loader import prepare_analysis_dataset, read_dataset
from dashboard.utils.validation import validate_dataset


def render() -> None:
    st.title("Upload Dataset")
    st.write(
        "Upload the public Credit Card Fraud Detection data or another CSV/Excel file. "
        "Original data is retained; mapped fields are added to an analysis copy."
    )
    st.markdown(
        "Expected model columns: `Time`, `Amount`, `V1`–`V28`, and `Class`. "
        "Other datasets may be explored after mapping Amount and an optional fraud label, "
        "but the original model cannot be used unless all required fields are present."
    )
    uploaded = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded is None:
        st.caption("You can also place the original `creditcard.csv` at `data/raw/creditcard.csv`.")
        return
    try:
        raw = read_dataset(uploaded.getvalue(), uploaded.name)
    except (
        ValueError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        OSError,
        UnicodeDecodeError,
        BadZipFile,
    ) as exc:
        st.error(f"Could not read the uploaded dataset: {exc}")
        return
    if raw.empty:
        st.error("The uploaded file contains no records.")
        return
    st.subheader("Preview")
    st.dataframe(raw.head(10), width="stretch", hide_index=True)
    columns = raw.columns.tolist()
    default_amount = columns.index("Amount") if "Amount" in columns else 0
    amount_column = st.selectbox("Column containing transaction amount", columns, index=default_amount)
    label_options = ["None"] + columns
    default_label = columns.index("Class") + 1 if "Class" in columns else 0
    label_column = st.selectbox("Column containing fraud label (optional)", label_options, index=default_label)
    fraud_value, legitimate_value = None, None
    valid_mapping = True
    if label_column != "None":
        label_values = raw[label_column].dropna().unique().tolist()
        if len(label_values) < 2:
            st.warning("At least two distinct non-missing label values are required for fraud/legitimate mapping.")
            valid_mapping = False
        else:
            fraud_default = label_values.index(1) if 1 in label_values else len(label_values) - 1
            legitimate_default = label_values.index(0) if 0 in label_values else 0
            fraud_value = st.selectbox(
                "Value representing fraud", label_values, index=fraud_default,
                format_func=str,
            )
            legitimate_options = [value for value in label_values if value != fraud_value]
            if not legitimate_options:
                valid_mapping = False
            else:
                preferred_legitimate = label_values[legitimate_default]
                legitimate_index = (
                    legitimate_options.index(preferred_legitimate)
                    if preferred_legitimate in legitimate_options else 0
                )
                legitimate_value = st.selectbox(
                    "Value representing legitimate", legitimate_options,
                    index=legitimate_index, format_func=str,
                )
    if st.button("Use this dataset", type="primary", disabled=not valid_mapping):
        mapped = prepare_analysis_dataset(
            raw,
            amount_column,
            None if label_column == "None" else label_column,
            fraud_value,
            legitimate_value,
        )
        st.session_state["active_dataset"] = mapped
        st.session_state["dataset_name"] = uploaded.name
        st.session_state.setdefault("datasets", {})[uploaded.name] = mapped
        st.session_state["overview_dataset"] = uploaded.name
        st.session_state.pop("trained_models", None)
        st.session_state.pop("anomaly_result", None)
        st.session_state.pop("high_risk_records", None)
        st.success(f"Loaded {len(mapped):,} records from {uploaded.name}.")
    report = validate_dataset(raw)
    st.subheader("Validation summary")
    st.write(
        f"{report['rows']:,} rows, {report['columns']:,} columns, "
        f"{report['missing_cells']:,} missing values, "
        f"{report['duplicate_rows']:,} duplicate rows."
    )
    if report["missing_columns"]:
        st.caption("Original model columns not present: " + ", ".join(report["missing_columns"]))
    st.caption("Validation flags issues without deleting or changing rows.")
    active = st.session_state.get("active_dataset")
    if active is not None:
        st.download_button("Download current analysis dataset (CSV)",
                           active.to_csv(index=False).encode("utf-8"),
                           "processed_transactions.csv", "text/csv")
