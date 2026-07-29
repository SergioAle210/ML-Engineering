"""Trains the baseline classifier and prints evaluation metrics.

Covers Modeling + Evaluation on top of the Data Preparation pipeline, same
spirit as verify_pipeline.py: run this after installing the package (editable
or from the wheel) to reproduce metrics identically on any machine.
"""
from pathlib import Path

from data_prep_pipeline import build_cleaning_pipeline, build_model_pipeline, evaluate_model, load_raw_data, split_dataset

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "telco_customer_churn.csv"


def main() -> None:
    raw_df = load_raw_data(DATA_PATH)
    clean_df = build_cleaning_pipeline().fit_transform(raw_df)
    X_train, X_test, y_train, y_test = split_dataset(clean_df)
    print(f"Train/test split: X_train={X_train.shape}, X_test={X_test.shape}")

    model = build_model_pipeline()
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    print()
    print("EVALUATION METRICS")
    print(metrics)


if __name__ == "__main__":
    main()
