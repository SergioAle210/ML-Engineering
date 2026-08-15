"""Runs hyperparameter search across all registered models and reports the best.

Same spirit as train_and_evaluate.py: run this after installing the package
to reproduce the model comparison identically on any machine.
"""
from pathlib import Path

from data_prep_pipeline import build_cleaning_pipeline, evaluate_model, load_raw_data, split_dataset, tune_all_models

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "telco_customer_churn.csv"


def main() -> None:
    raw_df = load_raw_data(DATA_PATH)
    clean_df = build_cleaning_pipeline().fit_transform(raw_df)
    X_train, X_test, y_train, y_test = split_dataset(clean_df)
    print(f"Train/test split: X_train={X_train.shape}, X_test={X_test.shape}")

    results = tune_all_models(X_train, y_train, search_type="grid", cv=5, scoring="roc_auc")

    print()
    print("HYPERPARAMETER SEARCH RESULTS (sorted by CV roc_auc)")
    for result in results:
        print(f"{result.model_name}: cv_roc_auc={result.best_cv_score:.4f} params={result.best_params}")

    best = results[0]
    test_metrics = evaluate_model(best.best_estimator, X_test, y_test)
    print()
    print(f"BEST MODEL: {best.model_name}")
    print(f"Best params: {best.best_params}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
