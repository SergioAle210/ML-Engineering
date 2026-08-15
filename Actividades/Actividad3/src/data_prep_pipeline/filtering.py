"""Filtering stage: enforce data-quality/business rules before feature engineering.

CRISP-DM mapping: Data Understanding — verifying data quality and dropping
records that would mislead the model (duplicate customers, impossible values).
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .extraction import ID_COLUMN

MIN_PLAUSIBLE_TENURE = 0
MAX_PLAUSIBLE_TENURE = 100


class DataFilter(BaseEstimator, TransformerMixin):
    """Drops duplicate customers and rows with implausible values.

    Rows are filtered, never imputed, at this stage: filtering removes
    records that shouldn't exist in the business domain (a customer
    duplicated across extracts, a negative tenure), while missing values
    that are legitimate (e.g. `TotalCharges` for a brand-new customer) are
    left for the imputers in the preprocessing ColumnTransformer. The
    identifier column, only needed for de-duplication, is dropped here.
    """

    def __init__(self, min_tenure: int = MIN_PLAUSIBLE_TENURE, max_tenure: int = MAX_PLAUSIBLE_TENURE):
        self.min_tenure = min_tenure
        self.max_tenure = max_tenure

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.drop_duplicates(subset=[ID_COLUMN])
        tenure_in_range = result["tenure"].between(self.min_tenure, self.max_tenure)
        result = result.loc[tenure_in_range]
        return result.drop(columns=[ID_COLUMN]).copy()
