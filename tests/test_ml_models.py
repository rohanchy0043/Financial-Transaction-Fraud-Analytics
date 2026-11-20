import pandas as pd
import pytest

from dashboard.utils.data_loader import FEATURE_COLUMNS
from dashboard.utils.ml_models import (
    MODEL_NAMES,
    build_model,
    evaluate_holdout,
    predict_one,
    train_models,
)


def _model_transactions():
    rows = []
    for index in range(40):
        row = {column: float(index + offset) for offset, column in enumerate(FEATURE_COLUMNS)}
        row["Class"] = index % 2
        rows.append(row)
    return pd.DataFrame(rows)


def test_model_training_excludes_target_and_returns_real_evaluations(tmp_path, monkeypatch):
    from dashboard.utils import ml_models

    monkeypatch.setattr(ml_models, "PROJECT_ROOT", tmp_path)
    results = train_models(_model_transactions())
    assert set(results["models"]) == set(MODEL_NAMES)
    assert "Class" not in results["feature_names"]
    assert results["test_size"] == 8
    probability, predicted = predict_one(
        results["models"]["Logistic Regression"],
        _model_transactions().drop(columns="Class").iloc[[0]],
        0.5,
    )
    assert 0 <= probability <= 1
    assert predicted in (0, 1)
    evaluated = evaluate_holdout(results, "Logistic Regression")
    assert 0 <= evaluated["average_precision"] <= 1
    with pytest.raises(ValueError):
        evaluate_holdout(results, "Random Forest")
    assert all((tmp_path / "models").glob("*.joblib"))


def test_model_pipeline_uses_supported_name():
    assert build_model("Logistic Regression").named_steps["model"].class_weight == "balanced"
    with pytest.raises(ValueError):
        build_model("Unrecognized")


def test_training_rejects_too_few_records_for_stratified_partitions():
    with pytest.raises(ValueError, match="five records"):
        train_models(_model_transactions().iloc[:4])
