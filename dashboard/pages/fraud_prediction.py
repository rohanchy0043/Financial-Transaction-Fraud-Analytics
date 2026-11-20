import pandas as pd
import streamlit as st

from dashboard.components.ui import model_required, require_dataset
from dashboard.utils.ml_models import MODEL_NAMES, dataset_fingerprint, predict_one


def render() -> None:
    st.title("Fraud Prediction")
    frame = require_dataset()
    if frame is None or not model_required(frame):
        return
    fingerprint = dataset_fingerprint(frame)
    model_name = st.selectbox("Trained model", MODEL_NAMES)
    state = st.session_state.get("trained_models")
    if not state or state["fingerprint"] != fingerprint:
        st.info("Train the models from Model Performance before requesting predictions.")
        return
    threshold = st.slider("Prediction threshold", 0.05, 0.95, 0.50, 0.01)
    record_position = st.number_input("Dataset row position (0-based)", 0, len(frame) - 1, 0)
    row = frame.iloc[int(record_position)]
    features = pd.DataFrame([row])
    try:
        probability, predicted = predict_one(state["models"][model_name], features, threshold)
    except (ValueError, KeyError) as exc:
        st.error(f"Could not score this record: {exc}")
        return
    result = st.columns(3)
    result[0].metric("Predicted fraud probability", f"{probability:.2%}")
    result[1].metric("Predicted class", "Fraud" if predicted else "Legitimate")
    result[2].metric("Decision threshold", f"{threshold:.0%}")
    st.write("The prediction uses the selected trained pipeline and this dataset's original feature values.")
    st.dataframe(row.to_frame("Value"), width="stretch")
    if st.button("Score all transactions for high-risk report"):
        try:
            probabilities = state["models"][model_name].predict_proba(frame[state["feature_names"]])[:, 1]
            high_risk = frame.loc[probabilities >= threshold].copy()
            high_risk["Predicted Fraud Probability"] = probabilities[probabilities >= threshold]
            high_risk["Predicted Class"] = 1
            st.session_state["high_risk_records"] = high_risk
            st.success(f"Scored {len(frame):,} records; {len(high_risk):,} met the selected threshold.")
        except (ValueError, KeyError) as exc:
            st.error(f"Could not score the full dataset: {exc}")
    st.warning("Predictions are model estimates, not guarantees. The model was trained on the anonymized public dataset and is not validated for other organizations.")
