"""Flags OCR readings that are structurally valid but statistically suspect
(e.g. a digit misread due to glare) so they surface for manual review instead
of silently polluting the dataset. Gets more effective as more photos accumulate.
"""
import pandas as pd

from src.ocr_common import EXPECTED_TOTAL_GTQ

TOTAL_TOLERANCE_GTQ = 0.01
MIN_ROWS_FOR_STATS = 3
DEVIATION_THRESHOLD = 0.15  # 15% away from the median price/gallon


def flag_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["needs_review"] = False
    df["review_reason"] = ""

    def add_reason(mask, reason):
        df.loc[mask, "needs_review"] = True
        df.loc[mask, "review_reason"] = df.loc[mask, "review_reason"].apply(
            lambda existing: f"{existing};{reason}" if existing else reason
        )

    add_reason(~df["parse_ok"], "ocr_parse_failed")

    total_off = df["total_gtq"].notna() & (
        (df["total_gtq"] - EXPECTED_TOTAL_GTQ).abs() > TOTAL_TOLERANCE_GTQ
    )
    add_reason(total_off, "total_not_150gtq")

    # A human-verified override already resolves whatever the raw OCR notes
    # say, so exclude those rows -- the notes column is an OCR audit trail,
    # not live state, and isn't cleared when an override is applied.
    total_assumed = (
        df["notes"].astype(str).str.contains("total_assumed_default_not_confirmed_by_ocr")
        & (df["data_source"] != "manual_override")
    )
    add_reason(total_assumed, "total_assumed_not_confirmed_by_ocr")

    valid_prices = df.loc[df["parse_ok"], "price_per_gallon_gtq"]
    if len(valid_prices) >= MIN_ROWS_FOR_STATS:
        median_price = valid_prices.median()
        deviation = (df["price_per_gallon_gtq"] - median_price).abs() / median_price
        statistical_outlier = df["parse_ok"] & (deviation > DEVIATION_THRESHOLD)
        add_reason(statistical_outlier, "price_deviates_from_median")

    return df


PRICE_BOARD_GTQ_COLUMNS = ["price_diesel_gtq", "price_regular_gtq", "price_super_gtq", "price_vpower_gtq"]


def flag_board_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where any of the four price-board fuels hasn't been
    confirmed by a human yet (see PriceBoardReading -- these are never
    auto-read). Independent of `needs_review`/`review_reason`, which only
    cover the user's own pump total/gallons."""
    df = df.copy()
    df["board_needs_review"] = df[PRICE_BOARD_GTQ_COLUMNS].isna().any(axis=1)
    df["board_review_reason"] = df["board_needs_review"].map(
        {True: "board_prices_unconfirmed", False: ""}
    )
    return df
