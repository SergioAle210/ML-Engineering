"""Run this after `pip install -e .` on any machine to confirm the package works.

Intended to be screenshotted on multiple computers as evidence the packaged
pipeline runs outside the original dev environment.
"""
import platform
import socket
from pathlib import Path

from data_prep_pipeline import build_cleaning_pipeline, build_preprocessing_pipeline, load_raw_data, split_dataset

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "telco_customer_churn.csv"


def main() -> None:
    raw_df = load_raw_data(DATA_PATH)
    print(f"Loaded raw data: {raw_df.shape}")

    cleaning_pipeline = build_cleaning_pipeline()
    clean_df = cleaning_pipeline.fit_transform(raw_df)
    print(f"Cleaned data: {clean_df.shape}")

    X_train, X_test, y_train, y_test = split_dataset(clean_df)
    print(f"Train/test split: X_train={X_train.shape}, X_test={X_test.shape}")

    preprocessing_pipeline = build_preprocessing_pipeline()
    X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
    X_test_transformed = preprocessing_pipeline.transform(X_test)
    print(f"Transformed features: train={X_train_transformed.shape}, test={X_test_transformed.shape}")

    print()
    print("PIPELINE OK")
    print(f"host: {socket.gethostname()}")
    print(f"platform: {platform.platform()}")
    print(f"python: {platform.python_version()}")


if __name__ == "__main__":
    main()
