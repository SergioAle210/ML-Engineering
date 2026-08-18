"""Ingestion microservice: fetches the raw dataset into the bronze layer.

Bronze = raw, unmodified source data, exactly as pulled from the origin.
Writes to a shared volume so downstream microservices never need network
access to the original source.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
BRONZE_DIR = Path(os.environ.get("BRONZE_DIR", "/data/bronze"))
OUTPUT_PATH = BRONZE_DIR / "telco_customer_churn.csv"


def main() -> None:
    dataset = pd.read_csv(SOURCE_URL)
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"[ingestion] wrote {len(dataset)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
