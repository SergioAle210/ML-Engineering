"""Training microservice: silver -> gold.

Gold = model-ready artifacts: the fitted model pipeline, its evaluation
metrics, and a held-out test sample the serving microservice can use to
validate its inputs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd

from data_prep_pipeline import build_model_pipeline, evaluate_model, split_dataset

SILVER_DIR = Path(os.environ.get("SILVER_DIR", "/data/silver"))
GOLD_DIR = Path(os.environ.get("GOLD_DIR", "/data/gold"))
INPUT_PATH = SILVER_DIR / "clean_customers.csv"
MODEL_PATH = GOLD_DIR / "model.joblib"
METRICS_PATH = GOLD_DIR / "metrics.json"
TEST_SAMPLE_PATH = GOLD_DIR / "test_sample.csv"


def main() -> None:
    clean_df = pd.read_csv(INPUT_PATH)
    X_train, X_test, y_train, y_test = split_dataset(clean_df)
    print(f"[training] train/test split: X_train={X_train.shape}, X_test={X_test.shape}")

    model = build_model_pipeline()
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    print(f"[training] {metrics}")

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(vars(metrics), indent=2))
    X_test.head(20).to_csv(TEST_SAMPLE_PATH, index=False)

    print(f"[training] wrote {MODEL_PATH}, {METRICS_PATH}, {TEST_SAMPLE_PATH}")


if __name__ == "__main__":
    main()
