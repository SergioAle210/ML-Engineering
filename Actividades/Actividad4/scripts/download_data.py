"""Downloads the raw dataset used across the project.

Source: Kaggle "Telco Customer Churn" by blastchar
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Fetched from IBM's public mirror of the same file (identical content, no
Kaggle account/API token required), so any teammate can reproduce
`data/telco_customer_churn.csv` with a single command.
"""
from pathlib import Path

import pandas as pd

SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "telco_customer_churn.csv"


def download_raw_data(url: str = SOURCE_URL) -> pd.DataFrame:
    return pd.read_csv(url)


if __name__ == "__main__":
    dataset = download_raw_data()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(dataset)} rows to {OUTPUT_PATH}")
