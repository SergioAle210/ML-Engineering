"""Where the raw dataset comes from, resolved per execution environment.

The scripts in this repo locate the CSV relative to `__file__`, which works
on a laptop but not inside a Databricks notebook: notebook cells have no
`__file__`, and the package itself is installed from a wheel into
site-packages, far away from any `data/` directory.

`resolve_data_source()` therefore tries, in order:

1. an explicit path passed by the caller,
2. the `CHURN_DATA_PATH` environment variable (set it in a Databricks job or
   cluster env to point at a Unity Catalog Volume, e.g.
   `/Volumes/main/default/raw/telco_customer_churn.csv`),
3. `data/telco_customer_churn.csv` next to the repo checkout, if present,
4. the public mirror URL — `pandas.read_csv` reads it directly, so a cluster
   with outbound internet needs no uploaded file at all.
"""
from __future__ import annotations

import os
from pathlib import Path

# Kaggle "Telco Customer Churn" by blastchar, served from IBM's public mirror
# (identical content, no Kaggle account or API token required).
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

DATA_PATH_ENV_VAR = "CHURN_DATA_PATH"
DATA_FILENAME = "telco_customer_churn.csv"


def _repo_data_path() -> Path | None:
    """`<repo>/data/telco_customer_churn.csv`, or None when unavailable.

    Returns None both when the file is missing and when the package was
    installed from a wheel (no repo checkout above it, e.g. on Databricks).
    """
    try:
        candidate = Path(__file__).resolve().parents[2] / "data" / DATA_FILENAME
    except IndexError:  # pragma: no cover - only on exotic install layouts
        return None
    return candidate if candidate.is_file() else None


def resolve_data_source(path: str | Path | None = None) -> str:
    """Returns a path or URL that `load_raw_data` can read from."""
    if path is not None:
        return str(path)

    from_env = os.environ.get(DATA_PATH_ENV_VAR)
    if from_env:
        return from_env

    local = _repo_data_path()
    if local is not None:
        return str(local)

    return DATA_URL
