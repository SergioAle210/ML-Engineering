"""Hyperparameter tuning stage: search grids per estimator, best model selection.

CRISP-DM mapping: Modeling — hyperparameter calibration for the estimators
built in `model.py`. Kept as its own module so the search strategy (grid vs
random) and the set of candidate models can change without touching pipeline
assembly.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from .model import build_model_pipeline

CLASSIFIER_STEP = "classifier"

# One entry per candidate model: (default estimator, param grid keyed by the
# `classifier__` prefix so it applies inside the full model Pipeline).
MODEL_SEARCH_SPACES: dict[str, tuple[ClassifierMixin, dict]] = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000),
        {
            f"{CLASSIFIER_STEP}__C": [0.01, 0.1, 1, 10],
        },
    ),
    "random_forest": (
        RandomForestClassifier(random_state=42),
        {
            f"{CLASSIFIER_STEP}__n_estimators": [100, 200, 400],
            f"{CLASSIFIER_STEP}__max_depth": [None, 5, 10, 20],
            f"{CLASSIFIER_STEP}__min_samples_leaf": [1, 2, 4],
        },
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            f"{CLASSIFIER_STEP}__n_estimators": [100, 200],
            f"{CLASSIFIER_STEP}__learning_rate": [0.01, 0.1, 0.2],
            f"{CLASSIFIER_STEP}__max_depth": [2, 3, 4],
        },
    ),
}


@dataclass
class TuningResult:
    model_name: str
    best_estimator: Pipeline
    best_params: dict
    best_cv_score: float


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    search_type: str = "grid",
    n_iter: int = 20,
    cv: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
) -> TuningResult:
    """Runs hyperparameter search for one registered model.

    `search_type="grid"` exhausts the full grid in `MODEL_SEARCH_SPACES`;
    `"random"` samples `n_iter` combinations instead, useful once a grid gets
    large (random_forest, gradient_boosting).
    """
    estimator, param_grid = MODEL_SEARCH_SPACES[model_name]
    pipeline = build_model_pipeline(estimator)

    if search_type == "grid":
        search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    elif search_type == "random":
        search = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown search_type: {search_type!r}, expected 'grid' or 'random'")

    search.fit(X_train, y_train)
    return TuningResult(
        model_name=model_name,
        best_estimator=search.best_estimator_,
        best_params=search.best_params_,
        best_cv_score=search.best_score_,
    )


def tune_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    search_type: str = "grid",
    **kwargs,
) -> list[TuningResult]:
    """Runs `tune_model` for every registered model, sorted best-first by CV score."""
    results = [
        tune_model(name, X_train, y_train, search_type=search_type, **kwargs)
        for name in MODEL_SEARCH_SPACES
    ]
    return sorted(results, key=lambda result: result.best_cv_score, reverse=True)
