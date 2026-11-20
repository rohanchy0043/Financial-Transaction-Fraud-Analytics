from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_FILENAME = "creditcard.csv"
FEATURE_COLUMNS = ["Time", "Amount", *[f"V{i}" for i in range(1, 29)]]
TARGET_COLUMN = "Class"
REQUIRED_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]


def read_dataset(file: str | Path | bytes, filename: str | None = None) -> pd.DataFrame:
    """Read CSV or Excel data without changing its original columns or values."""
    if isinstance(file, bytes):
        source = BytesIO(file)
        name = (filename or "").lower()
    else:
        source = Path(file)
        name = source.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(source)
    if name.endswith(".xlsx"):
        return pd.read_excel(source)
    raise ValueError("Unsupported file type. Upload a CSV or Excel (.xlsx) file.")


def load_bundled_dataset() -> pd.DataFrame | None:
    """Load the original dataset only when the user has placed it in data/raw."""
    path = PROJECT_ROOT / "data" / "raw" / DATASET_FILENAME
    if not path.is_file():
        return None
    return pd.read_csv(path)


def prepare_analysis_dataset(
    frame: pd.DataFrame,
    amount_column: str,
    class_column: str | None = None,
    fraud_value: object | None = None,
    legitimate_value: object | None = None,
) -> pd.DataFrame:
    """Map selected source fields to Amount/Class without mutating source records."""
    if amount_column not in frame.columns:
        raise ValueError("Choose a valid amount column.")
    if class_column is not None and class_column not in frame.columns:
        raise ValueError("Choose a valid fraud-label column.")
    if class_column is not None and fraud_value == legitimate_value:
        raise ValueError("Fraud and legitimate labels must be different values.")

    mapped = frame.copy()

    def preserve_column(column: str) -> None:
        preserved_name = f"Source {column}"
        while preserved_name in mapped.columns:
            preserved_name = f"Original {preserved_name}"
        mapped[preserved_name] = frame[column]

    if amount_column != "Amount":
        if "Amount" in mapped.columns:
            preserve_column("Amount")
        mapped["Amount"] = mapped[amount_column]

    if class_column is None:
        if "Class" in mapped.columns:
            preserve_column("Class")
            mapped = mapped.drop(columns=["Class"])
        return mapped

    if class_column == "Class":
        labels = set(frame[class_column].dropna().unique().tolist())
        if labels == {0, 1} and fraud_value == 1 and legitimate_value == 0:
            mapped["Class"] = frame[class_column]
            return mapped
        preserve_column("Class")
    elif "Class" in mapped.columns:
        preserve_column("Class")

    mapped["Class"] = frame[class_column].map(
        {legitimate_value: 0, fraud_value: 1}
    )
    return mapped
