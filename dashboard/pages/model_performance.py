import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.ui import model_required, require_dataset
from dashboard.utils.ml_models import (
    MODEL_NAMES,
    dataset_fingerprint,
    evaluate_holdout,
    feature_importances,
    train_models,
)


def render() -> None:
    st.title("Model Performance")
    frame = require_dataset()
    if frame is None or not model_required(frame):
        return
    fingerprint = dataset_fingerprint(frame)
    existing_training = st.session_state.get("trained_models")
    training_complete = bool(
        existing_training and existing_training["fingerprint"] == fingerprint
    )
    if st.button(
        "Train and evaluate all models",
        type="primary",
        disabled=training_complete,
    ):
        with st.spinner("Training models on the stratified training split..."):
            try:
                st.session_state["trained_models"] = train_models(frame)
            except (ValueError, MemoryError) as exc:
                st.error(f"Training could not be completed: {exc}")
    trained = st.session_state.get("trained_models")
    if not trained or trained["fingerprint"] != fingerprint:
        st.info("Training uses separate stratified training, validation, and test splits.")
        return
    rows = []
    for name, result in trained["validation_results"].items():
        rows.append({"Model": name, "Accuracy": result["accuracy"], "Precision": result["precision"],
                     "Recall": result["recall"], "F1": result["f1"], "ROC-AUC": result["roc_auc"],
                     "Average Precision": result["average_precision"]})
    st.caption(
        f"Training records: {trained['train_size']:,} | Validation records: "
        f"{trained['validation_size']:,} | Reserved test records: {trained['test_size']:,} "
        f"| threshold: {trained['threshold']:.0%}"
    )
    st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:.4f}"), width="stretch")
    st.caption("All models are fit on training records and compared on validation records. Choose a model using validation results before evaluating the reserved test partition.")
    test_result = trained.get("test_evaluation")
    selected = st.selectbox(
        "Choose model from validation results",
        MODEL_NAMES,
        disabled=test_result is not None,
    )
    if test_result is None:
        if st.button("Evaluate chosen model on reserved test set", type="primary"):
            try:
                test_result = evaluate_holdout(trained, selected)
            except ValueError as exc:
                st.error(str(exc))
                return
        else:
            st.info("The test partition is reserved for one final, selected-model evaluation.")
            return
    result = test_result
    st.subheader(f"Reserved test evaluation: {trained['test_model_name']}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{result['accuracy']:.4f}")
    c2.metric("Precision", f"{result['precision']:.4f}")
    c3.metric("Recall", f"{result['recall']:.4f}")
    c4.metric("F1-score", f"{result['f1']:.4f}")
    c5.metric("ROC-AUC", f"{result['roc_auc']:.4f}")
    c6.metric("PR-AUC / AP", f"{result['average_precision']:.4f}")
    matrix = result["confusion_matrix"]
    left, right = st.columns(2)
    left.plotly_chart(px.imshow(matrix, text_auto=True, x=["Predicted legitimate", "Predicted fraud"],
                                y=["Actual legitimate", "Actual fraud"], color_continuous_scale="Greys",
                                title="Confusion matrix"),
                      width="stretch")
    fpr, tpr, _ = result["roc_curve"]
    right.plotly_chart(go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
                       .add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline",
                                             line={"dash": "dash"}))
                       .update_layout(title="ROC curve", xaxis_title="False positive rate",
                                      yaxis_title="True positive rate"),
                       width="stretch")
    precision, recall, _ = result["precision_recall_curve"]
    st.plotly_chart(go.Figure(go.Scatter(x=recall, y=precision, mode="lines"))
                    .update_layout(title="Precision–recall curve", xaxis_title="Recall",
                                   yaxis_title="Precision"),
                    width="stretch")
    report = pd.DataFrame(result["classification_report"]).T
    st.subheader("Classification report")
    st.dataframe(report.style.format("{:.4f}", na_rep=""), width="stretch")
    importance = feature_importances(trained["models"][trained["test_model_name"]])
    if importance is not None:
        st.subheader("Feature importance")
        st.plotly_chart(px.bar(importance.head(15).sort_values("Importance"), x="Importance",
                               y="Feature", orientation="h", title=f"Top features: {selected}"),
                        width="stretch")
    st.info("False positives are legitimate transactions flagged for review. False negatives are fraudulent transactions the model misses. With severe class imbalance, accuracy alone is not informative.")
