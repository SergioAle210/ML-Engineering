"""Cross-platform OCR backend (Windows/Linux/macOS): reads the pump's
'Esta Venta' LCD off the converted JPEG using Tesseract, with a 7-segment
display model instead of the general-purpose English one.

Used automatically on any non-macOS machine — see src/ocr_pump.py for the
platform dispatcher and src/ocr_backend_vision.py for the macOS backend
(which reads noticeably more reliably; expect a higher manual-review rate
here, by design — see the corroboration logic below and in ocr_common.py).

Strategy, entirely deterministic computer vision, no OCR needed for setup:
1. Locate the LCD screen itself by color, not brightness: its cool-white
   backlight has more blue than red (B-R > threshold), while the pump's
   plastic body is warm/cream colored (B-R < 0). This finds the display
   reliably across different framing/lighting without needing to OCR any
   anchor text first.
2. Split that region into the total (top) and gallons (bottom) lines by
   finding the row with the fewest dark pixels in the middle third — the
   gap between the two digit rows.
3. Run Tesseract (with the 7-segment "letsgodigital" model bundled in
   models/) on each line at a couple of PSM settings, and only accept a
   digit reading if two independent passes corroborate each other
   (src.ocr_common.pick_corroborated) — same principle as the Vision
   backend: don't silently trust a single noisy pass.
4. The total is (almost) always exactly Q150.00 — a real, stated constraint
   of this dataset, not a guess (see EXPECTED_TOTAL_GTQ in ocr_common.py).
   If no corroborated OCR reading is found for it, we fall back to that
   known constant rather than leaving the row empty — but we say so in
   `notes`, and outliers.py treats that as needing review just like any
   other low-confidence read, so nothing is silently assumed.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from src.ocr_common import (
    EXPECTED_TOTAL_GTQ,
    PumpReading,
    digits_only,
    parse_gallons_from_digits,
    parse_total_from_digits,
    pick_corroborated,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TESSDATA_DIR = MODELS_DIR
LANG = "letsgodigital"

BLUE_MINUS_RED_THRESHOLD = 20
BLUE_MIN_BRIGHTNESS = 120
DARK_PIXEL_OFFSET = 25  # below the row's mean to count as "digit segment"


def _require_tesseract() -> str:
    path = shutil.which("tesseract")
    if not path:
        raise RuntimeError(
            "tesseract not found on PATH. Install it (e.g. `apt install tesseract-ocr`, "
            "`choco install tesseract`, or `brew install tesseract`) to process new photos "
            "on this machine."
        )
    return path


def locate_lcd_bbox(im: Image.Image) -> tuple[int, int, int, int] | None:
    """Find the LCD screen as the largest patch where blue-channel intensity
    clearly exceeds red — the display's cool backlight vs. the warm plastic
    body. Returns (x0, y0, x1, y1) in pixels, or None if nothing qualifies.
    """
    arr = np.asarray(im.convert("RGB")).astype(int)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mask = ((b - r) > BLUE_MINUS_RED_THRESHOLD) & (b > BLUE_MIN_BRIGHTNESS)
    labeled, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    ys, xs = np.where(labeled == biggest)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def split_total_and_gallons(im: Image.Image, bbox: tuple[int, int, int, int]) -> tuple[Image.Image, Image.Image]:
    """Split the LCD crop into (total_line, gallons_line) images by finding
    the row with the fewest dark pixels in the middle third of the screen —
    the blank gap between the two digit rows.
    """
    x0, y0, x1, y1 = bbox
    pad_x = int((x1 - x0) * 0.08)
    x0, x1 = x0 + pad_x, x1 - pad_x

    gray = np.asarray(im.convert("L"))[y0:y1, x0:x1]
    dark = gray < (gray.mean() - DARK_PIXEL_OFFSET)
    row_counts = dark.sum(axis=1)
    h = len(row_counts)
    mid_lo, mid_hi = int(h * 0.35), int(h * 0.75)
    gap = mid_lo + int(np.argmin(row_counts[mid_lo:mid_hi]))

    top = im.crop((x0, y0, x1, y0 + gap))
    bot = im.crop((x0, y0 + gap, x1, y1))
    return top, bot


def _upscale(im: Image.Image, factor: int = 3) -> Image.Image:
    return im.resize((im.width * factor, im.height * factor), Image.LANCZOS)


def _tesseract_text(tesseract_bin: str, image_path: Path, psm: int) -> str:
    proc = subprocess.run(
        [
            tesseract_bin, str(image_path), "stdout",
            "-l", LANG, "--psm", str(psm),
            "--tessdata-dir", str(TESSDATA_DIR),
        ],
        capture_output=True, text=True,
    )
    return proc.stdout


def _digit_candidates(tesseract_bin: str, line_image: Image.Image) -> list[str]:
    """Run a couple of PSM variants over the same line crop and return every
    non-empty digit string found — corroboration happens downstream."""
    candidates = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "line.png"
        _upscale(line_image).save(path)
        for psm in (7, 8):
            digits = digits_only(_tesseract_text(tesseract_bin, path, psm))
            if digits:
                candidates.append(digits)
    return candidates


def read_pump_display(jpeg_path: Path) -> PumpReading:
    reading = PumpReading()
    tesseract_bin = _require_tesseract()

    im = Image.open(jpeg_path)
    bbox = locate_lcd_bbox(im)
    if not bbox:
        reading.notes.append("lcd_screen_not_located")
        return reading
    reading.anchors_found = True

    top_line, bottom_line = split_total_and_gallons(im, bbox)

    total_candidates = _digit_candidates(tesseract_bin, top_line)
    gallons_candidates = _digit_candidates(tesseract_bin, bottom_line)
    reading.notes.append(f"total_ocr_candidates={total_candidates}")
    reading.notes.append(f"gallons_ocr_candidates={gallons_candidates}")

    total_accepted = pick_corroborated(total_candidates, min_len=4)
    parsed_total = parse_total_from_digits(total_accepted) if total_accepted else None
    if parsed_total is not None:
        reading.total_raw = total_accepted
        reading.total_gtq = parsed_total
    else:
        # Fall back to the dataset's known constant instead of leaving the
        # row empty -- but flag it, since it wasn't actually confirmed by OCR.
        reading.total_gtq = EXPECTED_TOTAL_GTQ
        reading.notes.append("total_assumed_default_not_confirmed_by_ocr")

    gallons_accepted = pick_corroborated(gallons_candidates, min_len=3)
    parsed_gallons = parse_gallons_from_digits(gallons_accepted) if gallons_accepted else None
    if parsed_gallons is not None:
        reading.gallons_raw = gallons_accepted
        reading.gallons = parsed_gallons
    elif gallons_accepted:
        reading.notes.append("gallons_digits_incomplete_no_corroboration")
    elif gallons_candidates:
        reading.notes.append("gallons_digits_disagree")
    else:
        reading.notes.append("gallons_not_found")

    if reading.total_gtq is not None and reading.gallons:
        reading.price_per_gallon_gtq = round(reading.total_gtq / reading.gallons, 4)
        reading.parse_ok = True

    return reading
