import pytest

from dashboard.utils.analytics import calculate_kpis, filter_transactions
from dashboard.utils.validation import clean_for_analysis, validate_dataset


def test_kpis_are_derived_from_input(transactions):
    kpis = calculate_kpis(transactions)
    assert kpis["total_transactions"] == 6
    assert kpis["total_amount"] == 210
    assert kpis["average_amount"] == 35
    assert kpis["fraudulent_transactions"] == 2
    assert kpis["fraud_rate"] == pytest.approx(100 / 3)


def test_filters_combine_amount_and_status(transactions):
    result = filter_transactions(transactions, 20, 50, "Fraud")
    assert result["Amount"].tolist() == [50.0]


def test_filters_combine_amount_status_and_elapsed_time(transactions):
    result = filter_transactions(
        transactions,
        minimum_amount=20,
        maximum_amount=60,
        fraud_status="Fraud",
        minimum_time=55,
    )
    assert result[["Time", "Amount", "Class"]].to_dict("records") == [
        {"Time": 60, "Amount": 60.0, "Class": 1}
    ]


def test_kpis_ignore_non_finite_amounts(transactions):
    transactions.loc[0, "Amount"] = float("inf")
    metrics = calculate_kpis(transactions)
    assert metrics["total_amount"] == 200
    assert metrics["average_amount"] == 40


def test_validation_reports_missing_fields_and_duplicates(transactions):
    duplicated = transactions.iloc[[0, 0]].copy()
    report = validate_dataset(duplicated)
    assert report["duplicate_rows"] == 1
    assert "V2" in report["missing_columns"]
    assert not report["is_model_compatible"]


def test_cleaning_returns_a_copy_and_coerces_invalid_numeric_values(transactions):
    original = transactions.copy()
    transactions["Amount"] = transactions["Amount"].astype(object)
    transactions.loc[0, "Amount"] = "not-a-number"
    cleaned = clean_for_analysis(transactions)
    assert cleaned["Amount"].isna().iloc[0]
    assert transactions.loc[0, "Amount"] == "not-a-number"
    assert original.loc[0, "Amount"] == 10.0
