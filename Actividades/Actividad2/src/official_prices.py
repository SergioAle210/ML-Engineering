"""Official Guatemala Ministry of Energy (MEM/DGH) daily average fuel
prices — used two ways in this project:

1. To backfill price_diesel_gtq/price_regular_gtq for a photo when the
   on-site OCR of the price board couldn't confirm them (see
   src/pipeline.py's _reset_price_board_to_baseline). There is no official
   category for V-Power -- it's a Shell-branded premium grade the ministry
   doesn't track separately -- so that column can never be backfilled this
   way and stays unconfirmed.
2. As exogenous reference features for a future predictive model (see
   build_public_export in src/pipeline.py): the exchange rate and national
   average prices are real drivers of what this station charges, since
   Guatemala's fuel prices are import/exchange-rate driven.

These are national daily averages monitored in Ciudad Capital, not this
specific station's price -- kept as separate `official_*` columns rather
than presented as an on-site reading, and any row that backfills
price_diesel_gtq/price_regular_gtq from them is labeled
board_data_source="auto+official_mem" so the provenance stays traceable
(never silently conflated with a real confirmed reading).

Source file: data/official-data/*.xlsx, sheet "PUBLICACIÓN WEB", published by
Dirección General de Hidrocarburos (DGH). Re-download and re-run
clean_official_prices() to refresh.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_DATA_DIR = ROOT / "data" / "official-data"
OFFICIAL_PRICES_CSV = OFFICIAL_DATA_DIR / "precios_mem_2021_presente.csv"

SHEET_NAME = "PUBLICACIÓN WEB"
HEADER_ROW = 6  # row 7 in the raw sheet ("FECHA", "Tipo de Cambio", ...)

# Only the fuels/columns this project needs, renamed to match the project's
# English snake_case convention (see gasolina_dataset.csv's price_*_gtq).
# Bunker (industrial fuel oil), GLP (bottled gas, different unit), and the
# two aviation fuels aren't sold at a car pump, so they're dropped.
RAW_COLUMN_MAP = {
    "FECHA": "date",
    "Tipo de Cambio": "exchange_rate_usd_gtq",
    "Gasolina Superior": "official_super_gtq",
    "Gasolina Regular": "official_regular_gtq",
    "Aceite Combustible Diésel": "official_diesel_gtq",
}

MIN_DATE = "2021-01-01"


def clean_official_prices(source_xlsx: Path, out_csv: Path = OFFICIAL_PRICES_CSV) -> pd.DataFrame:
    """Read the raw MEM publication and produce a clean, minimal CSV: one
    row per day from MIN_DATE onward, only the columns this project uses.
    Drops the "Unidades:" sub-header row and the source-note footer rows the
    same way -- both have a FECHA value that isn't a real date, so they fall
    out via the date parse + dropna below rather than needing to be
    special-cased by row position (which would break if MEM adds/removes a
    row in a future publication).
    """
    raw = pd.read_excel(source_xlsx, sheet_name=SHEET_NAME, header=HEADER_ROW)
    raw = raw.rename(columns=lambda c: str(c).strip())
    df = raw[list(RAW_COLUMN_MAP)].rename(columns=RAW_COLUMN_MAP)

    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df = df.dropna(subset=["date"])
    df = df[df["date"] >= MIN_DATE]

    for col in RAW_COLUMN_MAP.values():
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


_cache: pd.DataFrame | None = None


def load_official_prices() -> pd.DataFrame:
    """Cached load of the cleaned CSV, indexed by date string (YYYY-MM-DD)."""
    global _cache
    if _cache is None:
        if not OFFICIAL_PRICES_CSV.exists():
            raise FileNotFoundError(
                f"{OFFICIAL_PRICES_CSV} not found — run clean_official_prices() on the "
                "MEM xlsx in data/official-data/ first."
            )
        _cache = pd.read_csv(OFFICIAL_PRICES_CSV, dtype={"date": str}).set_index("date")
    return _cache


def lookup_official_prices(date_str: str) -> dict | None:
    """date_str: "YYYY-MM-DD". Returns the official reference row for that
    exact day, or None if it falls outside the published range."""
    try:
        official = load_official_prices()
    except FileNotFoundError:
        return None
    if date_str not in official.index:
        return None
    return official.loc[date_str].to_dict()


if __name__ == "__main__":
    import sys

    xlsx_files = sorted(OFFICIAL_DATA_DIR.glob("*.xlsx"))
    if not xlsx_files:
        sys.exit(f"No .xlsx file found in {OFFICIAL_DATA_DIR}")
    cleaned = clean_official_prices(xlsx_files[0])
    print(f"Wrote {len(cleaned)} rows -> {OFFICIAL_PRICES_CSV}")
