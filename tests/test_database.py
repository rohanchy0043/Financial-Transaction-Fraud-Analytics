import pandas as pd

from dashboard.utils.database import create_database_engine, replace_transactions, run_sql


def test_sql_summary_matches_pandas_calculations(transactions):
    engine = create_database_engine()
    replace_transactions(engine, transactions)
    result = run_sql(engine, "Summary KPIs").iloc[0]
    assert result["transaction_count"] == len(transactions)
    assert result["total_amount"] == transactions["Amount"].sum()
    assert result["fraud_count"] == transactions["Class"].eq(1).sum()
    assert result["legitimate_count"] == transactions["Class"].eq(0).sum()


def test_sql_deciles_cover_all_records(transactions):
    engine = create_database_engine()
    replace_transactions(engine, transactions)
    result = run_sql(engine, "Amount deciles")
    assert result["transaction_count"].sum() == len(transactions)
