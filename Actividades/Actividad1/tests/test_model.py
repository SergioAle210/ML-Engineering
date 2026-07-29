from pathlib import Path

from data_prep_pipeline import build_cleaning_pipeline, build_model_pipeline, evaluate_model, load_raw_data, split_dataset

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "telco_customer_churn.csv"


def test_model_pipeline_trains_and_evaluates():
    raw_df = load_raw_data(DATA_PATH)
    clean_df = build_cleaning_pipeline().fit_transform(raw_df)
    X_train, X_test, y_train, y_test = split_dataset(clean_df)

    model = build_model_pipeline()
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0
