from pathlib import Path

from data_prep_pipeline import build_cleaning_pipeline, load_raw_data, split_dataset, tune_model

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "telco_customer_churn.csv"


def test_tune_model_random_search_returns_best_estimator():
    raw_df = load_raw_data(DATA_PATH)
    clean_df = build_cleaning_pipeline().fit_transform(raw_df)
    X_train, _, y_train, _ = split_dataset(clean_df)

    result = tune_model(
        "logistic_regression",
        X_train,
        y_train,
        search_type="random",
        n_iter=2,
        cv=2,
    )

    assert result.model_name == "logistic_regression"
    assert 0.0 <= result.best_cv_score <= 1.0
    assert "classifier__C" in result.best_params
