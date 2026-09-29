from pathlib import Path

from data_prep_pipeline import build_cleaning_pipeline, build_preprocessing_pipeline, load_raw_data, split_dataset

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "telco_customer_churn.csv"


def test_full_workflow_runs_end_to_end():
    raw_df = load_raw_data(DATA_PATH)

    clean_df = build_cleaning_pipeline().fit_transform(raw_df)
    assert clean_df["tenure"].between(0, 100).all()
    assert "customerID" not in clean_df.columns
    assert len(clean_df) <= len(raw_df)

    X_train, X_test, y_train, y_test = split_dataset(clean_df)
    assert len(X_train) + len(X_test) == len(clean_df)

    preprocessing = build_preprocessing_pipeline()
    X_train_transformed = preprocessing.fit_transform(X_train)
    X_test_transformed = preprocessing.transform(X_test)

    assert X_train_transformed.shape[0] == len(X_train)
    assert X_test_transformed.shape[1] == X_train_transformed.shape[1]
