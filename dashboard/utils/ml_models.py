from __future__ import annotations

import hashlib

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from dashboard.utils.data_loader import FEATURE_COLUMNS, PROJECT_ROOT, TARGET_COLUMN


MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    ).hexdigest()[:16]


def build_model(name: str) -> Pipeline:
    if name not in MODEL_NAMES:
        raise ValueError("Select one of the supported model types.")
    estimator = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=12, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            n_estimators=200,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=2, random_state=42
        ),
    }[name]
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if name == "Logistic Regression":
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _measure(model: Pipeline, features: pd.DataFrame, labels: pd.Series) -> dict:
    probability = model.predict_proba(features)[:, 1]
    predicted = (probability >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, predicted),
        "precision": precision_score(labels, predicted, zero_division=0),
        "recall": recall_score(labels, predicted, zero_division=0),
        "f1": f1_score(labels, predicted, zero_division=0),
        "roc_auc": roc_auc_score(labels, probability),
        "average_precision": average_precision_score(labels, probability),
    }


def train_models(frame: pd.DataFrame) -> dict:
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing or TARGET_COLUMN not in frame.columns:
        raise ValueError(
            "Model training requires Time, Amount, V1–V28, and Class. "
            f"Missing: {', '.join(missing + ([TARGET_COLUMN] if TARGET_COLUMN not in frame else []))}"
        )
    model_data = frame.loc[:, [*FEATURE_COLUMNS, TARGET_COLUMN]].copy()
    model_data = model_data.apply(pd.to_numeric, errors="coerce")
    model_data = model_data.dropna(subset=[TARGET_COLUMN])
    model_data.loc[:, FEATURE_COLUMNS] = model_data[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )
    if not model_data[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 (legitimate) and 1 (fraud).")
    if model_data[TARGET_COLUMN].nunique() != 2 or model_data[TARGET_COLUMN].value_counts().min() < 5:
        raise ValueError(
            "Stratified training, validation, and test splits require at least "
            "five records from each class."
        )

    x = model_data[FEATURE_COLUMNS]
    y = model_data[TARGET_COLUMN].astype(int)
    x_train_validation, x_test, y_train_validation, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_validation, y_train_validation, test_size=0.25,
        stratify=y_train_validation, random_state=42,
    )
    fingerprint = dataset_fingerprint(model_data)
    validation_results = {}
    fitted_models = {}

    for name in MODEL_NAMES:
        pipeline = build_model(name)
        fit_params = {}
        if name == "Gradient Boosting":
            class_counts = y_train.value_counts()
            weights = y_train.map(
                {label: len(y_train) / (2 * count) for label, count in class_counts.items()}
            )
            fit_params["model__sample_weight"] = weights
        pipeline.fit(x_train, y_train, **fit_params)
        validation_results[name] = _measure(pipeline, x_validation, y_validation)
        fitted_models[name] = pipeline
        model_path = PROJECT_ROOT / "models" / f"{fingerprint}_{name.lower().replace(' ', '_')}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)

    return {
        "fingerprint": fingerprint,
        "validation_results": validation_results,
        "models": fitted_models,
        "feature_names": FEATURE_COLUMNS,
        "test_features": x_test,
        "test_labels": y_test,
        "validation_size": len(y_validation),
        "test_size": len(y_test),
        "train_size": len(y_train),
        "threshold": 0.5,
    }


def evaluate_holdout(state: dict, model_name: str) -> dict:
    """Evaluate one chosen model exactly once against the reserved test partition."""
    if model_name not in state["models"]:
        raise ValueError("Select one of the trained models.")
    if state.get("test_evaluation") is not None:
        raise ValueError("The reserved test partition has already been evaluated.")
    model = state["models"][model_name]
    features = state["test_features"]
    labels = state["test_labels"]
    probability = model.predict_proba(features)[:, 1]
    predicted = (probability >= state["threshold"]).astype(int)
    result = _measure(model, features, labels)
    result.update({
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]),
        "classification_report": classification_report(
            labels, predicted, labels=[0, 1], target_names=["Legitimate", "Fraud"],
            output_dict=True, zero_division=0,
        ),
        "y_test": labels.to_numpy(),
        "probability": probability,
        "predicted": predicted,
        "roc_curve": roc_curve(labels, probability),
        "precision_recall_curve": precision_recall_curve(labels, probability),
        "model_name": model_name,
    })
    state["test_evaluation"] = result
    state["test_model_name"] = model_name
    return result


def predict_one(model: Pipeline, features: pd.DataFrame, threshold: float) -> tuple[float, int]:
    probability = float(model.predict_proba(features[FEATURE_COLUMNS])[:, 1][0])
    return probability, int(probability >= threshold)


def feature_importances(model: Pipeline) -> pd.DataFrame | None:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance = np.abs(estimator.coef_[0])
    else:
        return None
    return (
        pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": importance})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
