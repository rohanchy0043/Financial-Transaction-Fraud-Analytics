from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.ui import (
    anomaly_and_prediction_columns,
    apply_overview_filters,
    current_dataset,
    render_kpis,
    require_dataset,
    reset_overview_filters,
    select_overview_dataset,
)


STATUS_NAMES = {0: "Legitimate", 1: "Fraud"}


def _finite_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.loc[np.isfinite(values)]


def _render_filter_toolbar(frame: pd.DataFrame) -> pd.DataFrame:
    with st.container(border=True):
        controls = st.columns([1.25, 1.2, 2.4, 2.1, 0.9])
        dataset_name = st.session_state.get("dataset_name", "Uploaded dataset")
        dataset_options = list(st.session_state.get("datasets", {})) or [dataset_name]
        if dataset_name not in dataset_options:
            dataset_options.append(dataset_name)
        controls[0].selectbox(
            "Dataset",
            dataset_options,
            key="overview_dataset",
            help="The active dataset is selected on Upload Dataset.",
            on_change=select_overview_dataset,
        )
        if "Class" in frame.columns:
            controls[1].selectbox(
                "Transaction status",
                ["All", "Legitimate", "Fraud"],
                key="overview_fraud_status",
            )
        else:
            controls[1].selectbox(
                "Transaction status",
                ["All"],
                key="overview_fraud_status",
                disabled=True,
                help="This dataset has no mapped fraud-label column.",
            )

        if "Amount" in frame.columns:
            amount = _finite_numeric(frame, "Amount")
            if not amount.empty and amount.min() < amount.max():
                controls[2].slider(
                    "Transaction amount",
                    min_value=float(amount.min()),
                    max_value=float(amount.max()),
                    value=(float(amount.min()), float(amount.max())),
                    key="overview_amount_range",
                    help="Applies to dashboard metrics, charts, and recent records.",
                )
            else:
                controls[2].caption("Amount range unavailable")
        else:
            controls[2].caption("Amount unavailable")

        if "Time" in frame.columns:
            elapsed = _finite_numeric(frame, "Time")
            if not elapsed.empty and elapsed.min() < elapsed.max():
                controls[3].slider(
                    "Elapsed time (seconds)",
                    min_value=float(elapsed.min()),
                    max_value=float(elapsed.max()),
                    value=(float(elapsed.min()), float(elapsed.max())),
                    key="overview_time_range",
                    help="Time is elapsed seconds in the source dataset, not calendar time.",
                )
            else:
                controls[3].caption("Time range unavailable")
        else:
            controls[3].caption("Time unavailable")

        controls[4].button(
            "Reset",
            key="reset_overview_filters",
            on_click=reset_overview_filters,
            help="Clear status, amount, and time filters.",
            width="stretch",
        )
    return apply_overview_filters(frame)


def _chart_title(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        height=330,
        margin={"l": 16, "r": 16, "t": 56, "b": 24},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"color": "#111111"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
    )
    fig.update_xaxes(showgrid=False, linecolor="#E5E5E5", zerolinecolor="#E5E5E5")
    fig.update_yaxes(gridcolor="#E5E5E5", linecolor="#FFFFFF", zerolinecolor="#E5E5E5")
    return fig


def _render_status_chart(frame: pd.DataFrame, container) -> None:
    if "Class" not in frame.columns:
        container.info("Fraud labels are not available in this dataset.")
        return
    labels = pd.to_numeric(frame["Class"], errors="coerce")
    counts = (
        labels.loc[labels.isin([0, 1])]
        .map(STATUS_NAMES)
        .value_counts()
        .reindex(["Legitimate", "Fraud"], fill_value=0)
        .rename_axis("Status")
        .reset_index(name="Transactions")
    )
    fig = px.pie(
        counts,
        names="Status",
        values="Transactions",
        hole=0.68,
        color="Status",
        color_discrete_map={"Legitimate": "#A3A3A3", "Fraud": "#111111"},
    )
    fig.update_traces(
        textinfo="percent",
        hovertemplate="%{label}<br>Transactions: %{value:,}<br>Share: %{percent}<extra></extra>",
    )
    container.plotly_chart(
        _chart_title(fig, "Legitimate vs. fraudulent transactions"),
        width="stretch",
    )


def _render_amount_distribution(frame: pd.DataFrame, container) -> None:
    if "Amount" not in frame.columns:
        container.info("The transaction amount field is unavailable.")
        return
    plot_data = frame.copy()
    plot_data["Amount"] = pd.to_numeric(plot_data["Amount"], errors="coerce")
    plot_data = plot_data.loc[np.isfinite(plot_data["Amount"])]
    if plot_data.empty:
        container.info("No numeric transaction amounts are available in the filtered records.")
        return
    if "Class" in plot_data.columns:
        plot_data["Status"] = pd.to_numeric(plot_data["Class"], errors="coerce").map(STATUS_NAMES)
        fig = px.histogram(
            plot_data,
            x="Amount",
            color="Status",
            nbins=48,
            barmode="overlay",
            opacity=0.78,
            color_discrete_map={"Legitimate": "#A3A3A3", "Fraud": "#111111"},
            labels={"Amount": "Transaction amount", "count": "Transactions"},
        )
    else:
        fig = px.histogram(
            plot_data,
            x="Amount",
            nbins=48,
            color_discrete_sequence=["#737373"],
            labels={"Amount": "Transaction amount", "count": "Transactions"},
        )
    container.plotly_chart(_chart_title(fig, "Transaction amount distribution"), width="stretch")


def _render_amount_comparison(frame: pd.DataFrame, container) -> None:
    if not {"Amount", "Class"}.issubset(frame.columns):
        container.info("Amount comparison requires both Amount and Class columns.")
        return
    plot_data = frame[["Amount", "Class"]].copy()
    plot_data["Amount"] = pd.to_numeric(plot_data["Amount"], errors="coerce")
    plot_data["Status"] = pd.to_numeric(plot_data["Class"], errors="coerce").map(STATUS_NAMES)
    plot_data = plot_data.loc[np.isfinite(plot_data["Amount"]) & plot_data["Status"].notna()]
    if plot_data.empty:
        container.info("No labeled numeric amounts are available in the filtered records.")
        return
    fig = px.box(
        plot_data,
        x="Status",
        y="Amount",
        color="Status",
        points=False,
        color_discrete_map={"Legitimate": "#A3A3A3", "Fraud": "#111111"},
        labels={"Status": "Observed class", "Amount": "Transaction amount"},
    )
    container.plotly_chart(_chart_title(fig, "Fraud vs. legitimate amount comparison"), width="stretch")


def _render_time_distribution(frame: pd.DataFrame, container) -> None:
    if "Time" not in frame.columns:
        container.info("The source dataset does not include an elapsed-time field.")
        return
    plot_data = frame.copy()
    plot_data["Elapsed time (seconds)"] = pd.to_numeric(plot_data["Time"], errors="coerce")
    plot_data = plot_data.loc[np.isfinite(plot_data["Elapsed time (seconds)"])]
    if plot_data.empty:
        container.info("No numeric elapsed-time values are available.")
        return
    if "Class" in plot_data.columns:
        plot_data["Status"] = pd.to_numeric(plot_data["Class"], errors="coerce").map(STATUS_NAMES)
        fig = px.histogram(
            plot_data,
            x="Elapsed time (seconds)",
            color="Status",
            nbins=48,
            barmode="overlay",
            opacity=0.78,
            color_discrete_map={"Legitimate": "#A3A3A3", "Fraud": "#111111"},
        )
    else:
        fig = px.histogram(
            plot_data,
            x="Elapsed time (seconds)",
            nbins=48,
            color_discrete_sequence=["#737373"],
        )
    container.plotly_chart(_chart_title(fig, "Transaction time distribution"), width="stretch")


def _render_fraud_rate(frame: pd.DataFrame, container) -> None:
    if "Class" not in frame.columns:
        container.info("Fraud rate is unavailable without a mapped fraud-label column.")
        return
    labels = pd.to_numeric(frame["Class"], errors="coerce")
    valid = labels.loc[labels.isin([0, 1])]
    if valid.empty:
        container.info("No valid fraud labels are present in the filtered records.")
        return
    rate = float(valid.eq(1).mean() * 100)
    fig = go.Figure(
        go.Indicator(
            mode="number+gauge",
            value=rate,
            number={"suffix": "%", "valueformat": ".4f", "font": {"color": "#111111"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#737373"},
                "bar": {"color": "#111111"},
                "bgcolor": "#F5F5F5",
                "bordercolor": "#E5E5E5",
                "steps": [{"range": [0, 100], "color": "#F5F5F5"}],
            },
            title={"text": f"{int(valid.eq(1).sum()):,} fraud records / {len(valid):,} labeled records"},
        )
    )
    container.plotly_chart(_chart_title(fig, "Fraud rate overview"), width="stretch")


def _render_feature_distribution(frame: pd.DataFrame, container) -> None:
    numeric = frame.select_dtypes(include="number").columns.tolist()
    features = [column for column in numeric if column not in {"Class", "Amount", "Time"}]
    if not features:
        container.info("No additional numerical features are available in this dataset.")
        return
    feature = container.selectbox("Feature", features, key="overview_feature")
    plot_data = frame.copy()
    plot_data[feature] = pd.to_numeric(plot_data[feature], errors="coerce")
    plot_data = plot_data.loc[np.isfinite(plot_data[feature])]
    if "Class" in plot_data.columns:
        plot_data["Status"] = pd.to_numeric(plot_data["Class"], errors="coerce").map(STATUS_NAMES)
        fig = px.histogram(
            plot_data,
            x=feature,
            color="Status",
            nbins=48,
            barmode="overlay",
            opacity=0.78,
            color_discrete_map={"Legitimate": "#A3A3A3", "Fraud": "#111111"},
        )
    else:
        fig = px.histogram(
            plot_data,
            x=feature,
            nbins=48,
            color_discrete_sequence=["#737373"],
        )
    container.plotly_chart(_chart_title(fig, f"{feature} distribution"), width="stretch")


def _render_recent_transactions(frame: pd.DataFrame) -> None:
    st.subheader("Recent transactions")
    st.caption(
        "Actual source records only. Class is the observed label; model predictions and "
        "anomaly status are separate calculated fields."
    )
    if frame.empty:
        st.info("No records match the selected filters. Reset the filters to see available transactions.")
        return

    result = anomaly_and_prediction_columns(frame)
    query = st.text_input("Search transactions", key="overview_search")
    if query:
        contains_query = result.astype(str).apply(
            lambda column: column.str.contains(query, case=False, regex=False, na=False)
        ).any(axis=1)
        result = result.loc[contains_query]

    row_id_column = next(
        (column for column in ("TransactionID", "Transaction ID", "transaction_id") if column in result),
        None,
    )
    if row_id_column is None:
        result.insert(0, "Source row", result.index)
        row_id_column = "Source row"
    preferred = [
        row_id_column,
        "Time",
        "Amount",
        "Class",
        "Predicted Class",
        "Predicted Fraud Probability",
        "Anomaly Status",
    ]
    visible_columns = [column for column in preferred if column in result.columns]
    result = result[visible_columns]
    if "Time" in result.columns:
        result = result.sort_values("Time", ascending=False, na_position="last")
    else:
        result = result.iloc[::-1]

    controls = st.columns([1, 4])
    page_size = controls[0].selectbox(
        "Rows per page", [10, 25, 50], key="overview_page_size"
    )
    page_count = max(1, (len(result) + page_size - 1) // page_size)
    page_number = controls[1].number_input(
        "Page",
        min_value=1,
        max_value=page_count,
        value=min(st.session_state.get("overview_page", 1), page_count),
        key="overview_page",
    )
    start = (int(page_number) - 1) * page_size
    st.dataframe(
        result.iloc[start : start + page_size],
        width="stretch",
        hide_index=True,
        height=390,
    )
    st.caption(f"{len(result):,} matching records · page {page_number} of {page_count}")
    st.download_button(
        "Download filtered transactions",
        result.to_csv(index=False).encode("utf-8"),
        "filtered_transactions.csv",
        "text/csv",
        key="overview_download_csv",
    )


def render() -> None:
    st.title("Financial Overview")
    st.caption("Monitor transaction activity and analyze potential fraudulent behavior.")
    frame = require_dataset()
    if frame is None:
        return

    filtered = _render_filter_toolbar(frame)
    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)
    render_kpis(filtered)
    if filtered.empty:
        st.info("No records match the selected filters. Reset filters to restore the full dataset.")
        _render_recent_transactions(filtered)
        return

    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)
    row_one = st.columns(2, gap="large")
    _render_status_chart(filtered, row_one[0])
    _render_amount_distribution(filtered, row_one[1])

    row_two = st.columns(2, gap="large")
    _render_amount_comparison(filtered, row_two[0])
    _render_time_distribution(filtered, row_two[1])

    row_three = st.columns(2, gap="large")
    _render_fraud_rate(filtered, row_three[0])
    _render_feature_distribution(filtered, row_three[1])

    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)
    _render_recent_transactions(filtered)
