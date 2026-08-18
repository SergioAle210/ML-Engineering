"""Evaluation stage: classification metrics for a fitted model pipeline.

CRISP-DM mapping: Evaluation — how well the model achieves the business goal
(flagging likely churners) before it is considered ready to hand off.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def __str__(self) -> str:
        return (
            f"accuracy={self.accuracy:.4f} precision={self.precision:.4f} "
            f"recall={self.recall:.4f} f1={self.f1:.4f} roc_auc={self.roc_auc:.4f}"
        )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> ClassificationMetrics:
    """Scores a fitted model pipeline against a held-out test set.

    `roc_auc` uses predicted probabilities rather than hard labels, since it
    measures ranking quality independent of the classification threshold.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return ClassificationMetrics(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        roc_auc=roc_auc_score(y_test, y_proba),
    )
