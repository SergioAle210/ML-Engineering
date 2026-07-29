"""Modeling stage: wraps the preprocessing pipeline with a final classifier.

CRISP-DM mapping: Modeling — kept as its own module, separate from
`pipeline.py` (Data Preparation), because the two phases have different
concerns and different owners in practice. `build_model_pipeline` reuses
`build_preprocessing_pipeline` as its first stage so preprocessing and the
classifier are always fit together: the resulting Pipeline accepts raw
(cleaned) X rows directly, with no risk of forgetting a preprocessing step
when the model is reused elsewhere.
"""
from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .pipeline import build_preprocessing_pipeline


def build_model_pipeline(estimator: ClassifierMixin | None = None) -> Pipeline:
    """Preprocessing + classifier as a single fit/predict-able Pipeline.

    Defaults to `LogisticRegression` as a simple, interpretable baseline for
    binary churn classification. Pass a different estimator to swap it out
    without touching the preprocessing stage.
    """
    if estimator is None:
        estimator = LogisticRegression(max_iter=1000)
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            ("classifier", estimator),
        ]
    )
