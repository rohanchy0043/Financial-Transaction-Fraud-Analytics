from __future__ import annotations

import numpy as np
import pandas as pd

from dashboard.utils.data_loader import FEATURE_COLUMNS, REQUIRED_COLUMNS, TARGET_COLUMN


def validate_dataset(frame: pd.DataFrame) -> dict:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    numeric_columns = [column for column in REQUIRED_COLUMNS if column in frame.columns]
    invalid_numeric = {}
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid_count = int((values.isna() & frame[column].notna()).sum())
        non_finite = int((~np.isfinite(values.dropna())).sum())
        invalid_numeric[column] = invalid_count + non_finite

    class_values = (
        pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        if TARGET_COLUMN in frame.columns
        else pd.Series(dtype=float)
    )
    invalid_class = int(
        (~class_values.isin([0, 1]) & class_values.notna()).sum()
    ) + int(class_values.isna().sum()) if TARGET_COLUMN in frame.columns else 0

    missing_cells = int(frame.isna().sum().sum())
    duplicate_rows = int(frame.duplicated().sum())
    quality_issues = (
        missing_columns
        or missing_cells
        or duplicate_rows
        or any(invalid_numeric.values())
        or invalid_class
    )
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_columns": missing_columns,
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "invalid_numeric": invalid_numeric,
        "invalid_class_labels": invalid_class,
        "dtypes": frame.dtypes.astype(str).to_dict(),
        "is_model_compatible": not missing_columns,
        "is_valid": not bool(quality_issues),
        "feature_columns": [name for name in FEATURE_COLUMNS if name in frame.columns],
    }


def clean_for_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a separate analysis copy; never silently alter the source frame."""
    cleaned = frame.copy()
    for column in REQUIRED_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned
