from io import BytesIO

import pandas as pd

from dashboard.utils.data_loader import prepare_analysis_dataset, read_dataset


def test_csv_and_excel_loading_preserve_rows():
    source = pd.DataFrame({"value": [1, 2], "amount": [12.5, 7.0]})
    csv = source.to_csv(index=False).encode("utf-8")
    excel_buffer = BytesIO()
    source.to_excel(excel_buffer, index=False)
    assert read_dataset(csv, "transactions.csv").equals(source)
    assert read_dataset(excel_buffer.getvalue(), "transactions.xlsx").equals(source)


def test_custom_column_mapping_keeps_source_values():
    source = pd.DataFrame({
        "transaction_total": [12.5, 7.0],
        "risk_label": ["fraudulent", "cleared"],
        "Class": ["original-a", "original-b"],
    })
    mapped = prepare_analysis_dataset(
        source, "transaction_total", "risk_label", "fraudulent", "cleared"
    )
    assert mapped["Amount"].tolist() == [12.5, 7.0]
    assert mapped["Class"].tolist() == [1, 0]
    assert mapped["Source Class"].tolist() == ["original-a", "original-b"]
    assert source["Class"].tolist() == ["original-a", "original-b"]
