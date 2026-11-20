from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool


def create_database_engine():
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def replace_transactions(engine, frame: pd.DataFrame) -> None:
    frame.to_sql("transactions", engine, if_exists="replace", index=False)


SQL_QUERIES = {
    "Summary KPIs": """
        SELECT COUNT(*) AS transaction_count,
               SUM(Amount) AS total_amount,
               AVG(Amount) AS average_amount,
               SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraud_count,
               SUM(CASE WHEN Class = 0 THEN 1 ELSE 0 END) AS legitimate_count,
               100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END)
                 / NULLIF(SUM(CASE WHEN Class IN (0, 1) THEN 1 ELSE 0 END), 0)
                 AS fraud_percentage
        FROM transactions
    """,
    "Amount by fraud status": """
        SELECT Class, COUNT(*) AS transaction_count,
               SUM(Amount) AS total_amount, AVG(Amount) AS average_amount,
               MIN(Amount) AS minimum_amount, MAX(Amount) AS maximum_amount
        FROM transactions GROUP BY Class ORDER BY Class
    """,
    "Amount deciles": """
        WITH ranked AS (
            SELECT Amount, Class,
                   NTILE(10) OVER (ORDER BY Amount) AS amount_decile
            FROM transactions
        )
        SELECT amount_decile, COUNT(*) AS transaction_count,
               AVG(Amount) AS average_amount,
               SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraud_count
        FROM ranked GROUP BY amount_decile ORDER BY amount_decile
    """,
    "Time ranges": """
        SELECT CAST(Time / 3600 AS INTEGER) AS elapsed_hour,
               COUNT(*) AS transaction_count,
               SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraud_count
        FROM transactions GROUP BY elapsed_hour ORDER BY elapsed_hour
    """,
}


def run_sql(engine, query_name: str) -> pd.DataFrame:
    if query_name not in SQL_QUERIES:
        raise ValueError("Choose one of the supported analytics queries.")
    return pd.read_sql_query(text(SQL_QUERIES[query_name]), engine)
