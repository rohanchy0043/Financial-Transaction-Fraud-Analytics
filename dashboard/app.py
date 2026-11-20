from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dashboard.components.ui import current_dataset, render_page_topbar
from dashboard.pages import (
    anomaly_detection,
    business_insights,
    data_quality,
    fraud_analysis,
    fraud_prediction,
    model_performance,
    overview,
    reports,
    sql_analytics,
    transaction_analytics,
    transaction_explorer,
    upload_dataset,
)
from dashboard.utils.data_loader import load_bundled_dataset

st.set_page_config(
    page_title="Financial Transaction & Fraud Analytics",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.set_option("client.showSidebarNavigation", False)

monochrome = go.layout.Template(pio.templates["plotly_white"])
monochrome.layout.colorway = ["#111111", "#737373", "#A3A3A3", "#000000"]
pio.templates["financial_monochrome"] = monochrome
pio.templates.default = "financial_monochrome"

if "active_dataset" not in st.session_state:
    st.session_state["active_dataset"] = load_bundled_dataset()
    if st.session_state["active_dataset"] is not None:
        st.session_state["dataset_name"] = "data/raw/creditcard.csv"
if "datasets" not in st.session_state:
    st.session_state["datasets"] = {}
    if st.session_state.get("active_dataset") is not None:
        dataset_name = st.session_state.get("dataset_name", "data/raw/creditcard.csv")
        st.session_state["datasets"][dataset_name] = st.session_state["active_dataset"]

collapsed = st.session_state.get("sidebar_collapsed", False)
sidebar_width = 68 if collapsed else 250
collapsed_navigation_style = """
    html body div[data-testid="stAppViewContainer"] section[data-testid="stSidebar"] [data-testid="stButton"] button { justify-content:center !important; padding-left:8px; padding-right:8px; }
    html body div[data-testid="stAppViewContainer"] section[data-testid="stSidebar"] [data-testid="stButton"] button [data-testid="stMarkdownContainer"] { position:absolute !important; width:1px !important; height:1px !important; padding:0 !important; margin:-1px !important; overflow:hidden !important; clip-path:inset(50%) !important; white-space:nowrap !important; border:0 !important; }
""" if collapsed else ""
st.markdown(
    f"""
    <style>
    :root {{ --sidebar-width:{sidebar_width}px; }}
    html body div[data-testid="stAppViewContainer"] section[data-testid="stSidebar"] {{ width:{sidebar_width}px !important; min-width:{sidebar_width}px !important; max-width:{sidebar_width}px !important; flex:0 0 {sidebar_width}px !important; transition:width .18s ease,min-width .18s ease,max-width .18s ease; }}
    html body div[data-testid="stAppViewContainer"] section[data-testid="stSidebar"] > div {{ width:{sidebar_width}px !important; min-width:{sidebar_width}px !important; max-width:{sidebar_width}px !important; transition:width .18s ease,min-width .18s ease,max-width .18s ease; }}
    section[data-testid="stSidebar"] {{ background:#FAFAFA; border-right:1px solid #E5E5E5; }}
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding-top:8px !important; padding-bottom:16px !important; }}
    section[data-testid="stSidebar"] hr {{ margin:12px 0 !important; }}
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {{ display:none !important; }}
    section[data-testid="stSidebar"] nav {{ display:none !important; }}
    section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {{ display:none !important; }}
    section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] {{ display:none !important; }}
    section[data-testid="stSidebar"] [data-testid="stSidebarNavSeparator"] {{ display:none !important; }}
    section[data-testid="stSidebar"] button[aria-label="Collapse sidebar"],section[data-testid="stSidebar"] button[aria-label="Expand sidebar"] {{ display:none !important; }}
    section[data-testid="stSidebar"] [data-testid="stSidebarResizeHandle"] {{ display:none !important; }}
    section[data-testid="stSidebar"] .st-key-toggle_navigation {{ position:fixed !important; top:12px !important; left:calc(var(--sidebar-width) - 16px) !important; right:auto !important; width:32px !important; z-index:1002; }}
    section[data-testid="stSidebar"] .st-key-toggle_navigation button {{ width:32px !important; height:44px !important; min-height:44px !important; padding:0 !important; border-radius:8px !important; background:#FFFFFF; }}
    section[data-testid="stSidebar"] .st-key-toggle_navigation button [data-testid="stMarkdownContainer"] {{ position:absolute !important; width:1px !important; height:1px !important; padding:0 !important; margin:-1px !important; overflow:hidden !important; clip-path:inset(50%) !important; white-space:nowrap !important; border:0 !important; }}
    div[data-testid="stToolbar"] {{ display:none !important; }}
    div[data-testid="stToolbarActions"] {{ display:none !important; }}
    div[data-testid="stStatusWidget"] {{ display:none !important; }}
    div[data-testid="stAppDeployButton"] {{ display:none !important; }}
    header[data-testid="stHeader"] {{ display:none !important; height:0 !important; }}
    section[data-testid="stSidebar"] [data-testid="stButton"] button {{ color:#111111 !important; }}
    section[data-testid="stSidebar"] [data-testid="stButton"] button svg {{ color:#737373 !important; fill:#737373 !important; stroke:#737373 !important; }}
    section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] svg {{ color:#111111 !important; fill:#111111 !important; stroke:#111111 !important; }}
    {collapsed_navigation_style}
    section[data-testid="stSidebar"] [data-testid="stButton"] button {{ min-height:42px; border-radius:8px; transition:background-color .15s ease,border-color .15s ease; }}
    section[data-testid="stSidebar"] [data-testid="stButton"] button:hover {{ border-color:#A3A3A3; background:#F5F5F5; }}
    .sidebar-brand {{ display:flex; align-items:center; gap:11px; padding:4px 4px 8px; }}
    .brand-mark {{ width:36px; height:36px; display:flex; align-items:center; justify-content:center; border:1px solid #111111; background:#111111; color:#FFFFFF; font-size:11px; font-weight:700; letter-spacing:.06em; }}
    .brand-name {{ color:#111111; font-size:11px; font-weight:750; letter-spacing:.11em; white-space:nowrap; }}
    .brand-subtitle {{ color:#737373; font-size:11px; margin-top:3px; white-space:nowrap; }}
    .sidebar-section {{ color:#737373; font-size:10px; font-weight:700; letter-spacing:.13em; margin:12px 0 4px; }}
    .page-breadcrumb {{ color:#737373; font-size:12px; padding:8px 0; }}
    .page-breadcrumb span {{ color:#A3A3A3; padding:0 7px; }}
    .dataset-name {{ color:#737373; font-size:9px; letter-spacing:.1em; line-height:1.55; overflow:hidden; padding:3px 4px; }}
    .dataset-name strong {{ color:#111111; font-size:11px; letter-spacing:0; display:inline-block; max-width:180px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .dataset-status {{ white-space:nowrap; color:#111111; font-size:11px; padding-top:11px; }}
    .dataset-status span {{ display:inline-block; width:7px; height:7px; background:#737373; border-radius:50%; margin-right:5px; }}
    .dataset-status-empty {{ color:#737373; }}
    .dataset-status-empty span {{ background:#A3A3A3; }}
    .section-spacer {{ height:12px; }}
    [data-testid="stMetric"] {{ background:#FFFFFF; border:1px solid #E5E5E5; border-bottom:0; border-radius:10px 10px 0 0; padding:12px 16px 10px; min-height:82px; }}
    [data-testid="stMetricLabel"] p {{ color:#737373; font-size:11px; letter-spacing:.02em; }}
    [data-testid="stMetricValue"] {{ color:#111111; font-size:25px; font-weight:650; }}
    .metric-note {{ background:#FFFFFF; border:1px solid #E5E5E5; border-top:0; border-radius:0 0 10px 10px; color:#737373; font-size:10px; line-height:1.45; margin-top:-8px; min-height:34px; padding:0 16px 11px; }}
    [data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:10px; border-color:#E5E5E5; }}
    [data-testid="stAlert"] {{ background:#FAFAFA; border:1px solid #E5E5E5; color:#111111; }}
    [data-testid="stAlert"] svg {{ color:#737373; fill:#737373; }}
    [data-testid="stToast"] {{ background:#FAFAFA; border:1px solid #E5E5E5; color:#111111; }}
    [data-testid="stPlotlyChart"] {{ border:1px solid #E5E5E5; border-radius:10px; padding:4px; }}
    [data-testid="stDataFrame"] {{ border:1px solid #E5E5E5; border-radius:10px; overflow:hidden; }}
    .empty-state {{ max-width:640px; margin:9vh auto 0; padding:30px 24px 12px; border:1px solid #E5E5E5; border-bottom:0; border-radius:12px 12px 0 0; text-align:center; background:#FFFFFF; }}
    .empty-mark {{ width:48px; height:48px; margin:0 auto 18px; display:flex; align-items:center; justify-content:center; background:#111111; color:#FFFFFF; font-size:13px; font-weight:700; letter-spacing:.08em; }}
    .empty-state h3 {{ color:#111111; font-size:22px; }}
    .empty-state p {{ color:#737373; line-height:1.7; margin:0 auto; max-width:480px; }}
    div.st-key-empty_state_upload {{ position:relative; display:block; width:100% !important; max-width:640px; height:64px; margin:0 auto 9vh; border:1px solid #E5E5E5; border-top:0; border-radius:0 0 12px 12px; background:#FFFFFF; box-sizing:border-box; }}
    div.st-key-empty_state_upload button {{ position:absolute !important; top:0; left:50%; width:160px !important; transform:translateX(-50%); }}
    .block-container {{ padding-top:3rem; }}
    button[kind="primary"] {{ background:#111111; border-color:#111111; }}
    section[data-testid="stSidebar"] button[kind="primary"] {{ background:#EAEAEA !important; border-color:#E5E5E5 !important; color:#111111 !important; }}
    section[data-testid="stSidebar"] button[kind="primary"] p {{ color:#111111 !important; }}
    @media (max-width:700px) {{
      .block-container {{ padding-left:1rem; padding-right:1rem; }}
      section[data-testid="stSidebar"] {{ width:min(var(--sidebar-width),85vw) !important; }}
      .dataset-name strong {{ max-width:110px; }}
      .empty-state {{ margin:4vh 8px 0; padding:24px 14px 10px; }}
      div.st-key-empty_state_upload {{ max-width:none; width:calc(100% - 16px) !important; margin:0 8px 4vh; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

pages = {
    "Dashboard": (overview.render, ":material/dashboard:"),
    "Transaction Analytics": (transaction_analytics.render, ":material/monitoring:"),
    "Fraud Analysis": (fraud_analysis.render, ":material/analytics:"),
    "Anomaly Detection": (anomaly_detection.render, ":material/troubleshoot:"),
    "Fraud Prediction": (fraud_prediction.render, ":material/policy:"),
    "Model Performance": (model_performance.render, ":material/query_stats:"),
    "Upload Dataset": (upload_dataset.render, ":material/upload_file:"),
    "Transaction Explorer": (transaction_explorer.render, ":material/search:"),
    "Data Quality": (data_quality.render, ":material/fact_check:"),
    "SQL Analytics": (sql_analytics.render, ":material/database:"),
    "Business Insights": (business_insights.render, ":material/lightbulb:"),
    "Reports & Exports": (reports.render, ":material/description:"),
}
groups = {
    "OVERVIEW": ["Dashboard"],
    "ANALYTICS": ["Transaction Analytics", "Fraud Analysis", "Anomaly Detection"],
    "MACHINE LEARNING": ["Fraud Prediction", "Model Performance"],
    "DATA MANAGEMENT": ["Upload Dataset", "Transaction Explorer", "Data Quality", "SQL Analytics"],
    "REPORTING": ["Business Insights", "Reports & Exports"],
}

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Dashboard"
elif st.session_state["active_page"] == "Overview":
    st.session_state["active_page"] = "Dashboard"

with st.sidebar:
    if collapsed:
        st.markdown(
            "<div class='sidebar-brand'><div class='brand-mark'>FA</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='sidebar-brand'><div class='brand-mark'>FA</div>"
            "<div><div class='brand-name'>FINANCE ANALYTICS</div>"
            "<div class='brand-subtitle'>Transaction Intelligence</div></div></div>",
            unsafe_allow_html=True,
        )
    toggle_label = "Expand navigation" if collapsed else "Collapse navigation"
    toggle_icon = ":material/chevron_right:" if collapsed else ":material/chevron_left:"
    if st.button(
        toggle_label,
        key="toggle_navigation",
        icon=toggle_icon,
        help="Expand navigation" if collapsed else "Collapse navigation",
        width="stretch",
    ):
        st.session_state["sidebar_collapsed"] = not collapsed
        st.rerun()
    if not collapsed:
        st.divider()
    for group_name, group_pages in groups.items():
        if not collapsed:
            st.markdown(
                f"<div class='sidebar-section'>{group_name}</div>",
                unsafe_allow_html=True,
            )
        for page_name in group_pages:
            if st.button(
                page_name,
                key=f"navigate_{page_name}",
                icon=pages[page_name][1],
                help=page_name,
                type="primary" if st.session_state["active_page"] == page_name else "secondary",
                width="stretch",
            ):
                st.session_state["active_page"] = page_name
                st.rerun()
    if not collapsed:
        st.divider()
        st.caption("Source data is never synthesized.")
        dataset_name = st.session_state.get("dataset_name")
        if dataset_name:
            st.caption(f"Loaded: {dataset_name}")

selected_page = st.session_state["active_page"]
render_page_topbar(selected_page, current_dataset())
pages[selected_page][0]()
