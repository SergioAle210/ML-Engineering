"""Cleaning microservice: bronze -> silver.

Silver = data that has passed extraction (relevant columns, type fixes) and
filtering (deduplication, business-rule validation). Still row-level,
not yet split or model-ready.
"""
from __future__ import annotations

import os
from pathlib import Path

from data_prep_pipeline import build_cleaning_pipeline, load_raw_data

BRONZE_DIR = Path(os.environ.get("BRONZE_DIR", "/data/bronze"))
SILVER_DIR = Path(os.environ.get("SILVER_DIR", "/data/silver"))
INPUT_PATH = BRONZE_DIR / "telco_customer_churn.csv"
OUTPUT_PATH = SILVER_DIR / "clean_customers.csv"


def main() -> None:
    raw_df = load_raw_data(INPUT_PATH)
    print(f"[cleaning] loaded raw data: {raw_df.shape}")

    clean_df = build_cleaning_pipeline().fit_transform(raw_df)
    print(f"[cleaning] cleaned data: {clean_df.shape}")

    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[cleaning] wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
