"""Assembles the full data-preparation workflow.

Two sklearn Pipelines are used deliberately instead of one:

1. `cleaning_pipeline`  (extraction -> filtering): operates on the raw
   dataframe, target column included, and may drop rows (duplicate
   customers, implausible values). Row-dropping steps are safe here because
   the target travels alongside the features in the same dataframe.
2. `preprocessing_pipeline` (ColumnTransformer: numeric + categorical
   handling): fit only on X after the train/test split, never drops rows,
   so it is safe to use directly with `fit(X, y)` / inside a model Pipeline.

Splitting X/y and train/test happens between the two, via `split_dataset`.
This mirrors CRISP-DM: extraction/filtering close out Data Understanding
(is the data fit to use?), while the preprocessing pipeline and split
belong to Data Preparation for modelling.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .extraction import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN, DataExtractor
from .filtering import DataFilter


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Extraction source: read the raw Kaggle Telco Customer Churn export from disk."""
    return pd.read_csv(path)


def build_cleaning_pipeline() -> Pipeline:
    """Extraction + filtering stage, row-preserving guarantees not required."""
    return Pipeline(
        steps=[
            ("extraction", DataExtractor()),
            ("filtering", DataFilter()),
        ]
    )


def build_preprocessing_pipeline() -> Pipeline:
    """Type-aware column handling: numeric impute+scale, categorical impute+encode."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocessing", preprocessor)])


def build_pipeline() -> Pipeline:
    """Full data-preparation pipeline: extraction -> filtering -> type handling.

    Returned as a single Pipeline for display/diagram purposes (see the
    notebook). For actual model fitting, use `build_cleaning_pipeline` and
    `build_preprocessing_pipeline` separately around `split_dataset`, so
    the row-dropping filtering step never runs on a train/test split that
    already has X/y aligned by position.
    """
    return Pipeline(
        steps=[
            ("extraction", DataExtractor()),
            ("filtering", DataFilter()),
            ("preprocessing", build_preprocessing_pipeline()),
        ]
    )


def split_dataset(
    clean_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separates the cleaned dataframe into train/test X and y.

    The target arrives as `Yes`/`No` text from the source and is encoded
    to 1/0 here, at the boundary between cleaning and modelling.
    """
    X = clean_df[FEATURE_COLUMNS]
    y = clean_df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
