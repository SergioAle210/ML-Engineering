"""End-to-end pipeline: HEIC photos of the pump display -> structured dataset.

Run with: python -m src.pipeline [--force]

Incremental by design: each raw photo is identified by filename, and stages
are skipped when their cached output is already newer than the source file.
Drop new HEIC files into data/raw/ and re-run — only the new ones get processed.

Each photo captures two things at once: the pump's own "Esta Venta" display
(total paid + gallons for that fill-up) and, further down in the same frame,
the station's posted price-per-gallon board for all four fuels. Both are read
from the same JPEG and land as columns on the same row — there's no separate
photo or dataset for the price board.
"""
import argparse
import dataclasses
from pathlib import Path

import pandas as pd

from src.convert import heic_to_jpeg
from src.exif_extract import dump_metadata, extract_selected_fields
from src.geocode import reverse_geocode
from src.ocr_common import PRICE_BOARD_TOLERANCE_GTQ, parse_price_from_digits
from src.ocr_price_board import FUEL_TYPES, read_price_board
from src.ocr_pump import digits_only, parse_gallons_from_digits, parse_total_from_digits, read_pump_display
from src.official_prices import lookup_official_prices
from src.outliers import flag_board_prices, flag_dataset

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CONVERTED_DIR = ROOT / "data" / "interim" / "converted"
EXIF_DIR = ROOT / "data" / "interim" / "exif"
PRICE_BOARD_CROPS_DIR = ROOT / "data" / "interim" / "price_board_crops"
GEOCODE_CACHE = ROOT / "data" / "interim" / "geocode_cache.json"
PROCESSED_DIR = ROOT / "data" / "processed"
DATASET_CSV = PROCESSED_DIR / "gasolina_dataset.csv"
DATASET_JSON = PROCESSED_DIR / "gasolina_dataset.json"
DATASET_PUBLIC_CSV = PROCESSED_DIR / "gasolina_dataset_public.csv"
MANUAL_OVERRIDES_CSV = ROOT / "data" / "manual_overrides.csv"
PRICE_BOARD_OVERRIDES_CSV = ROOT / "data" / "price_board_overrides.csv"

FUEL_COLUMNS = [fuel.replace("-", "") for fuel in FUEL_TYPES]

# Internal-only columns (raw EXIF noise and review/audit bookkeeping) that
# the app and dashboard need but a public/modeling download doesn't.
PUBLIC_EXPORT_DROP_COLUMNS = [
    "offset_time_original",
    "subsec_time_original",
    "camera_make",
    "camera_model",
    "lens_model",
    "software",
    "image_width",
    "image_height",
    "orientation",
    "exposure_time",
    "f_number",
    "iso",
    "board_review_reason",
    "board_needs_review",
    "review_reason",
    "needs_review",
    "board_override_note",
    "board_data_source",
    "override_note",
    # GPS/geo columns: every row is the same station, so these are constant
    # (zero variance) -- no signal for a model, only useful once photos from
    # more than one station exist.
    "gps_latitude",
    "gps_longitude",
    "gps_altitude_m",
    "gps_horizontal_error_m",
    "gps_datetime",  # duplicate of datetime_original (UTC vs local)
    "geo_display_name",
    "geo_station_name",
    "geo_road",
    "geo_neighbourhood",
    "geo_city",
    "geo_county",
    "geo_state",
    "geo_country",
    "geo_postcode",
    # Camera metadata: irrelevant to price.
    "focal_length_mm",
    "file_size_bytes",
    # total_gtq is always a fixed Q150 (zero variance); gallons is a
    # deterministic function of price_per_gallon_gtq given that fixed total
    # (gallons = 150 / price), so keeping it alongside the price columns
    # would leak the target into a feature.
    "total_gtq",
    "gallons",
    # OCR audit trail (raw digit strings, per-pass notes, confidence flags)
    # and crop image paths: useful for debugging the pipeline, not features
    # for a model.
    "total_raw",
    "gallons_raw",
    "anchors_found",
    "parse_ok",
    "notes",
    "board_notes",
    "price_diesel_raw",
    "price_regular_raw",
    "price_super_raw",
    "price_vpower_raw",
    "price_diesel_crop",
    "price_regular_crop",
    "price_super_crop",
    "price_vpower_crop",
    "data_source",
]


def build_public_export(df: pd.DataFrame) -> pd.DataFrame:
    """Drop internal EXIF/review columns for the downloadable dataset, and
    add features a future predictive model would need but this dataset
    doesn't otherwise carry. Assumes `df` is already sorted by
    datetime_original (both call sites do this before calling here), since
    the lag features below are only meaningful in chronological order.

    - fecha/day_of_week/month/is_weekend: calendar features -- prices move
      with the exchange rate over time, and fill-ups aren't evenly spaced.
    - days_since_previous_fill, price_change_gtq/_pct: how much time passed
      and how much Súper's price moved since this user's previous fill-up --
      the natural lag features for forecasting the next one.
    - exchange_rate_usd_gtq, official_super/regular/diesel_gtq: same-day
      Ministry of Energy reference prices (src/official_prices.py) -- real
      exogenous drivers, since Guatemala's fuel prices are largely
      import/exchange-rate driven, not just a function of this dataset's
      own history.
    - station_markup_super_gtq: this station's Súper price minus that day's
      official national average, isolating the station's own margin from
      nationwide price movements.
    """
    df = df.copy()
    df["fecha"] = df["datetime_original"].astype(str).str.slice(0, 10).str.replace(":", "-", regex=False)
    dt = pd.to_datetime(df["fecha"], errors="coerce")
    df["day_of_week"] = dt.dt.day_name()
    df["month"] = dt.dt.month
    df["is_weekend"] = dt.dt.dayofweek >= 5

    df["days_since_previous_fill"] = dt.diff().dt.days
    df["price_change_gtq"] = df["price_per_gallon_gtq"].diff()
    df["price_change_pct"] = df["price_per_gallon_gtq"].pct_change() * 100

    official = df["fecha"].apply(lookup_official_prices)
    df["exchange_rate_usd_gtq"] = official.apply(lambda o: o["exchange_rate_usd_gtq"] if o else None)
    df["official_super_gtq"] = official.apply(lambda o: o["official_super_gtq"] if o else None)
    df["official_regular_gtq"] = official.apply(lambda o: o["official_regular_gtq"] if o else None)
    df["official_diesel_gtq"] = official.apply(lambda o: o["official_diesel_gtq"] if o else None)
    df["station_markup_super_gtq"] = df["price_per_gallon_gtq"] - df["official_super_gtq"]

    return df.drop(columns=PUBLIC_EXPORT_DROP_COLUMNS, errors="ignore")


def _price_board_columns(fuel_col: str) -> tuple[str, str, str]:
    return f"price_{fuel_col}_gtq", f"price_{fuel_col}_raw", f"price_{fuel_col}_crop"


def process_one(heic_path: Path) -> dict:
    stem = heic_path.stem
    jpeg_path = CONVERTED_DIR / f"{stem}.jpg"
    exif_json_path = EXIF_DIR / f"{stem}.json"

    heic_to_jpeg(heic_path, jpeg_path)
    raw_exif = dump_metadata(heic_path, exif_json_path)
    exif = extract_selected_fields(raw_exif)
    pump = read_pump_display(jpeg_path)
    board = read_price_board(jpeg_path, PRICE_BOARD_CROPS_DIR)

    row = {
        "filename": heic_path.name,
        **exif,
        **dataclasses.asdict(pump),
    }
    for fuel, col in zip(FUEL_TYPES, FUEL_COLUMNS):
        gtq_col, raw_col, crop_col = _price_board_columns(col)
        row[gtq_col] = None
        row[raw_col] = board.prices_raw[fuel]
        row[crop_col] = board.crop_paths[fuel]
    row["board_notes"] = ";".join(board.notes)

    if exif.get("gps_latitude") and exif.get("gps_longitude"):
        geo = reverse_geocode(exif["gps_latitude"], exif["gps_longitude"], GEOCODE_CACHE)
        row.update({f"geo_{k}": v for k, v in geo.items()})

    return row


def _clean_digit_string(value) -> str:
    # A round-tripped CSV read coerces a numeric-looking raw-OCR string
    # (e.g. "3938") to a float (3938.0); str() on that reintroduces a
    # bogus trailing "0" via ".0". Normalize whole floats to int first.
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return digits_only(str(value))


def _reset_to_ocr_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute total/gallons/parse_ok purely from the raw OCR text columns
    (total_raw, gallons_raw), which are never mutated by an override. This
    makes apply_manual_overrides idempotent: re-running it after an override
    row is edited or removed always starts from the true OCR result instead
    of compounding on a previously-overridden value.
    """
    df = df.copy()

    def recompute(row):
        total = None
        if pd.notna(row.get("total_raw")):
            total = parse_total_from_digits(_clean_digit_string(row["total_raw"]))
        gallons = None
        if pd.notna(row.get("gallons_raw")):
            gallons = parse_gallons_from_digits(_clean_digit_string(row["gallons_raw"]))
        row["total_gtq"] = total
        row["gallons"] = gallons
        row["price_per_gallon_gtq"] = round(total / gallons, 4) if total and gallons else None
        row["parse_ok"] = total is not None and bool(gallons)
        row["data_source"] = "ocr"
        row["override_note"] = ""
        return row

    return df.apply(recompute, axis=1)


def apply_manual_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Reset every row to its raw OCR baseline, then merge in human-verified
    corrections for readings the pipeline could not confidently resolve on
    its own (e.g. glare occluding a digit). Rows with no override keep the
    automated OCR result.
    """
    df = _reset_to_ocr_baseline(df)
    if not MANUAL_OVERRIDES_CSV.exists():
        return df
    overrides = pd.read_csv(MANUAL_OVERRIDES_CSV)
    df = df.set_index("filename")
    overrides = overrides.set_index("filename")
    for filename, row in overrides.iterrows():
        if filename not in df.index:
            continue
        df.loc[filename, "total_gtq"] = row["total_gtq"]
        df.loc[filename, "gallons"] = row["gallons"]
        df.loc[filename, "price_per_gallon_gtq"] = round(row["total_gtq"] / row["gallons"], 4)
        df.loc[filename, "parse_ok"] = True
        df.loc[filename, "data_source"] = "manual_override"
        df.loc[filename, "override_note"] = row.get("override_note", "")
    return df.reset_index()


# Diesel and Regular have an official national-average equivalent to fall
# back on (see src/official_prices.py); V-Power is a Shell-branded premium
# grade the Ministry of Energy doesn't track, so it has no such fallback and
# stays unconfirmed until either OCR corroborates it or a human reviews it.
OFFICIAL_FUEL_COLUMN = {"diesel": "official_diesel_gtq", "regular": "official_regular_gtq"}


def _exif_date_to_iso(datetime_original) -> str | None:
    if pd.isna(datetime_original):
        return None
    # EXIF format is "YYYY:MM:DD HH:MM:SS"; official_prices looks up "YYYY-MM-DD".
    return str(datetime_original)[:10].replace(":", "-")


def _reset_price_board_to_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute price_*_gtq from scratch, same idempotency reasoning as
    _reset_to_ocr_baseline: never compound on a previous run's result.

    Súper's price is never OCR'd at all -- the user only ever fills up on
    Súper for a fixed Q150.00 (see README, Fase 1), so the board's Súper
    price *is* price_per_gallon_gtq, already known with full confidence from
    the pump's own display. For the other grades, two tiers are tried in
    order:
    1. OCR (price_*_raw, see _price_digit_candidates in ocr_backend_vision.py)
       -- an actual on-site reading, but only trusted if it lands within
       PRICE_BOARD_TOLERANCE_GTQ of Súper's now-known price (our one anchor
       of ground truth on this board).
    2. The official Ministry of Energy national average for that same date
       (see src/official_prices.py) -- not this station's real price, but a
       same-day reference close enough to be useful when OCR has nothing.
    If neither tier resolves a fuel, it's left unconfirmed for manual review,
    same as before.
    """
    df = df.copy()

    def recompute(row):
        super_price = row.get("price_per_gallon_gtq")
        has_super = pd.notna(super_price)
        row["price_super_gtq"] = super_price if has_super else None

        official = lookup_official_prices(_exif_date_to_iso(row.get("datetime_original")) or "")
        used_official = False
        for col in FUEL_COLUMNS:
            if col == "super":
                continue
            gtq_col, raw_col, _ = _price_board_columns(col)
            candidate = None
            if pd.notna(row.get(raw_col)):
                candidate = parse_price_from_digits(_clean_digit_string(row[raw_col]))
            if candidate is not None and has_super and abs(candidate - super_price) <= PRICE_BOARD_TOLERANCE_GTQ:
                row[gtq_col] = candidate
            elif official is not None and col in OFFICIAL_FUEL_COLUMN:
                row[gtq_col] = official[OFFICIAL_FUEL_COLUMN[col]]
                used_official = True
            else:
                row[gtq_col] = None
        if used_official:
            row["board_data_source"] = "auto+official_mem"
        else:
            row["board_data_source"] = "auto" if has_super else "unconfirmed"
        row["board_override_note"] = ""
        return row

    return df.apply(recompute, axis=1)


def apply_price_board_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Reset the board to its auto baseline (see _reset_price_board_to_baseline),
    then merge in human-confirmed readings for whatever a photo's review
    still needs -- rows with no override keep the auto baseline.
    """
    df = _reset_price_board_to_baseline(df)

    if not PRICE_BOARD_OVERRIDES_CSV.exists():
        return df
    overrides = pd.read_csv(PRICE_BOARD_OVERRIDES_CSV)
    df = df.set_index("filename")
    overrides = overrides.set_index("filename")
    for filename, row in overrides.iterrows():
        if filename not in df.index:
            continue
        for col in FUEL_COLUMNS:
            gtq_col, _, _ = _price_board_columns(col)
            df.loc[filename, gtq_col] = row.get(gtq_col)
        df.loc[filename, "board_data_source"] = "manual"
        df.loc[filename, "board_override_note"] = row.get("note", "")
    return df.reset_index()


def refresh_overrides() -> pd.DataFrame:
    """Reapply the manual-override CSVs onto the already-built dataset and
    recompute review flags, without touching HEIC/EXIF/OCR at all.

    This is the function the dashboard calls after a human corrects a
    reading: it's pure pandas, so it works in a deploy environment that
    doesn't have macOS/Vision available (unlike `run()`, which needs it
    to process new photos).
    """
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"{DATASET_CSV} not found — run the full pipeline first.")
    df = pd.read_csv(DATASET_CSV)
    df = apply_manual_overrides(df)
    df = apply_price_board_overrides(df)
    df = df.sort_values("datetime_original").reset_index(drop=True)
    df = flag_dataset(df)
    df = flag_board_prices(df)
    df.to_csv(DATASET_CSV, index=False)
    df.to_json(DATASET_JSON, orient="records", indent=2, force_ascii=False)
    build_public_export(df).to_csv(DATASET_PUBLIC_CSV, index=False)
    return df


def save_manual_override(filename: str, total_gtq: float, gallons: float, note: str) -> pd.DataFrame:
    """Upsert a human-verified correction for one photo's total/gallons and refresh the dataset."""
    if MANUAL_OVERRIDES_CSV.exists():
        overrides = pd.read_csv(MANUAL_OVERRIDES_CSV)
    else:
        overrides = pd.DataFrame(columns=["filename", "total_gtq", "gallons", "override_note"])
    overrides = overrides[overrides["filename"] != filename]
    new_row = pd.DataFrame([{
        "filename": filename, "total_gtq": total_gtq, "gallons": gallons, "override_note": note,
    }])
    overrides = pd.concat([overrides, new_row], ignore_index=True)
    overrides.to_csv(MANUAL_OVERRIDES_CSV, index=False)
    return refresh_overrides()


def save_price_board_override(filename: str, prices_gtq: dict[str, float], note: str) -> pd.DataFrame:
    """Upsert a human-confirmed reading of the four price-board LCDs for one
    photo and refresh the dataset. `prices_gtq` keys are the plain fuel
    columns (e.g. "diesel", "vpower"), matching FUEL_COLUMNS.
    """
    if PRICE_BOARD_OVERRIDES_CSV.exists():
        overrides = pd.read_csv(PRICE_BOARD_OVERRIDES_CSV)
    else:
        overrides = pd.DataFrame(columns=["filename", *[f"price_{c}_gtq" for c in FUEL_COLUMNS], "note"])
    overrides = overrides[overrides["filename"] != filename]
    new_row = {"filename": filename, "note": note}
    new_row.update({f"price_{col}_gtq": prices_gtq[col] for col in FUEL_COLUMNS})
    overrides = pd.concat([overrides, pd.DataFrame([new_row])], ignore_index=True)
    overrides.to_csv(PRICE_BOARD_OVERRIDES_CSV, index=False)
    return refresh_overrides()


def run(force: bool = False) -> pd.DataFrame:
    for d in (CONVERTED_DIR, EXIF_DIR, PROCESSED_DIR, PRICE_BOARD_CROPS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    heic_files = sorted(RAW_DIR.glob("*.HEIC")) + sorted(RAW_DIR.glob("*.heic"))
    if not heic_files:
        raise SystemExit(f"No HEIC files found in {RAW_DIR}")

    existing = pd.DataFrame()
    if DATASET_CSV.exists() and not force:
        existing = pd.read_csv(DATASET_CSV)

    rows = []
    for heic_path in heic_files:
        already_done = not existing.empty and heic_path.name in existing["filename"].values
        if already_done and not force:
            rows.append(existing[existing["filename"] == heic_path.name].iloc[0].to_dict())
            continue
        print(f"Processing {heic_path.name} ...")
        rows.append(process_one(heic_path))

    df = pd.DataFrame(rows)
    df = apply_manual_overrides(df)
    df = apply_price_board_overrides(df)
    df = df.sort_values("datetime_original").reset_index(drop=True)
    df = flag_dataset(df)
    df = flag_board_prices(df)

    df.to_csv(DATASET_CSV, index=False)
    df.to_json(DATASET_JSON, orient="records", indent=2, force_ascii=False)
    build_public_export(df).to_csv(DATASET_PUBLIC_CSV, index=False)
    print(f"\nWrote {len(df)} rows -> {DATASET_CSV}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Reprocess every file, ignoring cache")
    args = parser.parse_args()
    run(force=args.force)
