"""Extraction stage: pull only the fields relevant to the churn business question.

CRISP-DM mapping: Data Understanding — deciding which attributes of the raw
source are worth carrying into modelling.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class DataExtractor(BaseEstimator, TransformerMixin):
    """Selects the business-relevant columns from the raw Kaggle export.

    Keeps the identifier (needed by the filtering stage to de-duplicate
    customers) and the target alongside the features, and fixes the one
    known type issue in the source: `TotalCharges` arrives as text with a
    handful of blank values (new customers with zero tenure) and must be
    coerced to numeric before anything downstream can use it.
    """

    def __init__(self, columns: list[str] | None = None, include_target: bool = True):
        self.columns = columns
        self.include_target = include_target

    def fit(self, X: pd.DataFrame, y=None):
        base_columns = self.columns or FEATURE_COLUMNS
        self._columns = [ID_COLUMN] + base_columns
        if self.include_target and TARGET_COLUMN not in self._columns:
            self._columns = self._columns + [TARGET_COLUMN]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self._columns if c not in X.columns]
        if missing:
            raise KeyError(f"Source data is missing expected columns: {missing}")
        result = X[self._columns].copy()
        result["TotalCharges"] = pd.to_numeric(result["TotalCharges"], errors="coerce")
        return result
