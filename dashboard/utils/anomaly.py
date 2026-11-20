from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from dashboard.utils.data_loader import FEATURE_COLUMNS


def detect_anomalies(
    frame: pd.DataFrame, contamination: float = 0.01
) -> tuple[pd.DataFrame, IsolationForest]:
    features = [column for column in FEATURE_COLUMNS if column in frame.columns]
    if not features:
        raise ValueError("No supported numeric features are available for anomaly detection.")
    values = frame[features].apply(pd.to_numeric, errors="coerce")
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        ),
    )
    model.fit(values)
    predictions = model.predict(values)
    scores = -model.score_samples(values)
    result = frame.copy()
    result["Anomaly"] = predictions == -1
    result["Anomaly Score"] = scores
    return result, model
