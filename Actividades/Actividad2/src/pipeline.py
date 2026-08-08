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
from src.ocr_price_board import FUEL_TYPES, read_price_board
from src.ocr_pump import digits_only, parse_gallons_from_digits, parse_total_from_digits, read_pump_display
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
MANUAL_OVERRIDES_CSV = ROOT / "data" / "manual_overrides.csv"
PRICE_BOARD_OVERRIDES_CSV = ROOT / "data" / "price_board_overrides.csv"

FUEL_COLUMNS = [fuel.replace("-", "") for fuel in FUEL_TYPES]


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


def _reset_to_ocr_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute total/gallons/parse_ok purely from the raw OCR text columns
    (total_raw, gallons_raw), which are never mutated by an override. This
    makes apply_manual_overrides idempotent: re-running it after an override
    row is edited or removed always starts from the true OCR result instead
    of compounding on a previously-overridden value.
    """
    df = df.copy()

    def clean_digits(value) -> str:
        # A round-tripped CSV read coerces a numeric-looking raw-OCR string
        # (e.g. "3938") to a float (3938.0); str() on that reintroduces a
        # bogus trailing "0" via ".0". Normalize whole floats to int first.
        if pd.isna(value):
            return ""
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return digits_only(str(value))

    def recompute(row):
        total = None
        if pd.notna(row.get("total_raw")):
            total = parse_total_from_digits(clean_digits(row["total_raw"]))
        gallons = None
        if pd.notna(row.get("gallons_raw")):
            gallons = parse_gallons_from_digits(clean_digits(row["gallons_raw"]))
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


def apply_price_board_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """The price-board LCDs are never auto-read (see PriceBoardReading), so
    every price_*_gtq column resets to unconfirmed here and is only filled
    in for photos a human has reviewed via data/price_board_overrides.csv.
    """
    df = df.copy()
    for col in FUEL_COLUMNS:
        gtq_col, _, _ = _price_board_columns(col)
        df[gtq_col] = None
    df["board_data_source"] = "unconfirmed"
    df["board_override_note"] = ""

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
    print(f"\nWrote {len(df)} rows -> {DATASET_CSV}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Reprocess every file, ignoring cache")
    args = parser.parse_args()
    run(force=args.force)
