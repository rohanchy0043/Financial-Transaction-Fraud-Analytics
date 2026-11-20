import pandas as pd

from dashboard.utils.report_generator import dataframe_to_excel, text_report_to_pdf


def test_excel_and_pdf_exports_produce_files():
    report = pd.DataFrame({"Metric": ["transactions"], "Value": [5]})
    excel = dataframe_to_excel(report)
    pdf = text_report_to_pdf("Data report", ["Transactions: 5"])
    assert excel.startswith(b"PK")
    assert pdf.startswith(b"%PDF")
