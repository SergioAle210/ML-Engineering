"""End-to-end pipeline: HEIC photos of the pump display -> structured dataset.

Run with: python -m src.pipeline [--force]

Incremental by design: each raw photo is identified by filename, and stages
are skipped when their cached output is already newer than the source file.
Drop new HEIC files into data/raw/ and re-run — only the new ones get processed.
"""
import argparse
import dataclasses
from pathlib import Path

import pandas as pd

from src.convert import heic_to_jpeg
from src.exif_extract import dump_metadata, extract_selected_fields
from src.geocode import reverse_geocode
from src.ocr_pump import digits_only, parse_gallons_from_digits, parse_total_from_digits, read_pump_display
from src.outliers import flag_dataset

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CONVERTED_DIR = ROOT / "data" / "interim" / "converted"
EXIF_DIR = ROOT / "data" / "interim" / "exif"
GEOCODE_CACHE = ROOT / "data" / "interim" / "geocode_cache.json"
PROCESSED_DIR = ROOT / "data" / "processed"
DATASET_CSV = PROCESSED_DIR / "gasolina_dataset.csv"
DATASET_JSON = PROCESSED_DIR / "gasolina_dataset.json"
MANUAL_OVERRIDES_CSV = ROOT / "data" / "manual_overrides.csv"


def process_one(heic_path: Path) -> dict:
    stem = heic_path.stem
    jpeg_path = CONVERTED_DIR / f"{stem}.jpg"
    exif_json_path = EXIF_DIR / f"{stem}.json"

    heic_to_jpeg(heic_path, jpeg_path)
    raw_exif = dump_metadata(heic_path, exif_json_path)
    exif = extract_selected_fields(raw_exif)
    pump = read_pump_display(jpeg_path)

    row = {
        "filename": heic_path.name,
        **exif,
        **dataclasses.asdict(pump),
    }

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


def refresh_overrides() -> pd.DataFrame:
    """Reapply data/manual_overrides.csv onto the already-built dataset and
    recompute review flags, without touching HEIC/EXIF/OCR at all.

    This is the function the dashboard calls after a human corrects a
    reading: it's pure pandas, so it works in a deploy environment that
    doesn't have macOS/Vision available (unlike `run()`, which needs them
    to process new photos).
    """
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"{DATASET_CSV} not found — run the full pipeline first.")
    df = pd.read_csv(DATASET_CSV)
    df = apply_manual_overrides(df)
    df = df.sort_values("datetime_original").reset_index(drop=True)
    df = flag_dataset(df)
    df.to_csv(DATASET_CSV, index=False)
    df.to_json(DATASET_JSON, orient="records", indent=2, force_ascii=False)
    return df


def save_manual_override(filename: str, total_gtq: float, gallons: float, note: str) -> pd.DataFrame:
    """Upsert a human-verified correction for one photo and refresh the dataset."""
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


def run(force: bool = False) -> pd.DataFrame:
    for d in (CONVERTED_DIR, EXIF_DIR, PROCESSED_DIR):
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
    df = df.sort_values("datetime_original").reset_index(drop=True)
    df = flag_dataset(df)

    df.to_csv(DATASET_CSV, index=False)
    df.to_json(DATASET_JSON, orient="records", indent=2, force_ascii=False)
    print(f"\nWrote {len(df)} rows -> {DATASET_CSV}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Reprocess every file, ignoring cache")
    args = parser.parse_args()
    run(force=args.force)
