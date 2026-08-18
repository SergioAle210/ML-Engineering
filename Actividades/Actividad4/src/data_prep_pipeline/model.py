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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from .pipeline import build_preprocessing_pipeline

# Winner of the hyperparameter search in `tuning.py` (see
# scripts/tune_models.py / notebooks/pipeline_diagram.ipynb section 6):
# gradient_boosting, cv_roc_auc=0.8486, beating logistic_regression and
# random_forest. Kept as the default so installing the package already
# gives the calibrated model, without re-running the search each time.
_DEFAULT_ESTIMATOR_PARAMS = {
    "learning_rate": 0.1,
    "max_depth": 2,
    "n_estimators": 100,
    "random_state": 42,
}


def build_model_pipeline(estimator: ClassifierMixin | None = None) -> Pipeline:
    """Preprocessing + classifier as a single fit/predict-able Pipeline.

    Defaults to the `GradientBoostingClassifier` hyperparameters selected by
    `tune_all_models` (see `tuning.py`). Pass a different estimator to swap
    it out without touching the preprocessing stage — e.g. to re-run
    `tune_all_models` and pick a new winner as the dataset evolves.
    """
    if estimator is None:
        estimator = GradientBoostingClassifier(**_DEFAULT_ESTIMATOR_PARAMS)
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            ("classifier", estimator),
        ]
    )
