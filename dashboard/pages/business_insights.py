import streamlit as st

from dashboard.components.ui import require_dataset
from dashboard.utils.analytics import evidence_based_insights


def render() -> None:
    st.title("Business Insights")
    frame = require_dataset()
    if frame is None:
        return
    for insight in evidence_based_insights(frame):
        st.markdown(f"- {insight}")
    if "Class" not in frame:
        st.info("Map a fraud-label column to calculate label-based insights.")
    else:
        st.caption("Insights are descriptive calculations over the loaded records, not causal findings or claims about real-world criminal behavior.")
    trained = st.session_state.get("trained_models")
    if trained:
        st.subheader("Model validation comparison")
        for name, scores in trained["validation_results"].items():
            st.write(
                f"{name}: recall {scores['recall']:.4f}, precision {scores['precision']:.4f}, "
                f"F1 {scores['f1']:.4f}, validation average precision {scores['average_precision']:.4f}."
            )
