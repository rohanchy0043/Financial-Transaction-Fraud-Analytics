import pandas as pd
import streamlit as st

from dashboard.components.ui import require_dataset
from dashboard.utils.validation import clean_for_analysis, validate_dataset


def render() -> None:
    st.title("Data Quality")
    frame = require_dataset()
    if frame is None:
        return
    result = validate_dataset(frame)
    metric = st.columns(4)
    metric[0].metric("Rows", f"{result['rows']:,}")
    metric[1].metric("Columns", f"{result['columns']:,}")
    metric[2].metric("Missing cells", f"{result['missing_cells']:,}")
    metric[3].metric("Duplicate rows", f"{result['duplicate_rows']:,}")
    st.write("Dataset validation:", "Passed" if result["is_valid"] else "Issues found")
    if not result["is_model_compatible"]:
        st.warning("Dataset does not contain every original model column and cannot be used by the pretrained-dataset workflow.")
        if result["missing_columns"]:
            st.write("Missing required columns:", ", ".join(result["missing_columns"]))
    if result["invalid_class_labels"]:
        st.write(f"Invalid or missing fraud labels: {result['invalid_class_labels']:,}")
    invalid = {name: count for name, count in result["invalid_numeric"].items() if count}
    if invalid:
        st.write("Invalid numeric values:", invalid)
    st.subheader("Column types")
    st.dataframe(pd.DataFrame({"Column": result["dtypes"].keys(),
                               "Data type": result["dtypes"].values()}),
                 width="stretch", hide_index=True)
    numeric = frame.select_dtypes(include="number")
    if not numeric.empty:
        st.subheader("Numerical summary")
        st.dataframe(numeric.describe().T, width="stretch")
    if "Class" in frame:
        st.subheader("Class distribution")
        st.dataframe(frame["Class"].value_counts(dropna=False).rename_axis("Class").reset_index(name="Records"),
                     width="stretch", hide_index=True)
    if st.checkbox("Preview a numeric-coerced analysis copy"):
        cleaned = clean_for_analysis(frame)
        st.caption("This preview does not replace or modify the original dataset.")
        st.dataframe(cleaned.head(100), width="stretch", hide_index=True)
