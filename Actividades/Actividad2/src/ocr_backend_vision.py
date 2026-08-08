"""macOS OCR backend: reads the pump's 'Esta Venta' LCD (total GTQ + gallons
dispensed) off the converted JPEG using Apple's Vision framework.

Only importable on macOS (needs pyobjc-framework-Vision/Quartz). See
src/ocr_pump.py for the platform dispatcher and src/ocr_backend_tesseract.py
for the cross-platform fallback used on Windows/Linux.

Strategy:
1. Fast-level OCR on the full image locates the "Esta Venta" / "Galones"
   printed labels (high accuracy on normal fonts) and grabs whatever numeric
   text Vision finds between them (the 7-segment LCD digits — lower accuracy).
2. For the gallons reading specifically, re-run OCR (fast + accurate) on a
   tight crop around its bounding box and take the candidate with exactly 4
   digits, since that recovers cases the whole-image pass truncates.
3. Everything is returned alongside the raw OCR strings so a downstream
   sanity/outlier check can flag readings that are still suspect (e.g. glare
   over a digit) instead of silently trusting a misread.
"""
import re
from pathlib import Path

import Quartz
import Vision
from Foundation import NSURL
from PIL import Image

from src.ocr_common import (
    FUEL_TYPES,
    PriceBoardReading,
    PumpReading,
    digits_only,
    parse_gallons_from_digits,
    parse_total_from_digits,
    pick_corroborated,
)

FAST, ACCURATE = 0, 1

TOTAL_NUM_RE = re.compile(r"\d[\d.]{2,6}\d")
HAS_ENOUGH_DIGITS_RE = re.compile(r"(?:\D*\d){2,}")  # >=2 digits anywhere in the text


def _load_cg_image(path: Path):
    url = NSURL.fileURLWithPath_(str(path))
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    return Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)


def _run_ocr(cg_image, level=FAST, lang_correction=False):
    results = []

    def handler(request, error):
        for obs in request.results():
            results.append((obs.text(), obs.confidence(), obs.boundingBox()))

    req = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(handler)
    req.setRecognitionLevel_(level)
    req.setUsesLanguageCorrection_(lang_correction)
    handler_obj = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
    handler_obj.performRequests_error_([req], None)
    return results


def _ocr_path(path: Path, level=FAST, lang_correction=False):
    return _run_ocr(_load_cg_image(path), level=level, lang_correction=lang_correction)


def _tight_crop(im: Image.Image, bbox, pad_frac=(0.4, 0.7)) -> Image.Image:
    W, H = im.size
    x, y, w, h = bbox.origin.x, bbox.origin.y, bbox.size.width, bbox.size.height
    padx, pady = w * pad_frac[0], h * pad_frac[1]
    x0n, x1n = x - padx, x + w + padx
    y0n, y1n = y - pady, y + h + pady
    px0, px1 = max(int(x0n * W), 0), min(int(x1n * W), W)
    py0, py1 = max(int((1 - y1n) * H), 0), min(int((1 - y0n) * H), H)
    crop = im.crop((px0, py0, px1, py1))
    return crop.resize((crop.width * 5, crop.height * 5), Image.LANCZOS)


def _pil_to_cg(im: Image.Image):
    """Round-trip a PIL image through a temp file to get a CGImage (simplest
    reliable path with PyObjC's Vision/Quartz bindings)."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        im.save(f.name, "PNG")
        return _load_cg_image(Path(f.name))


def read_pump_display(jpeg_path: Path) -> PumpReading:
    reading = PumpReading()
    results = _ocr_path(jpeg_path, level=FAST)

    ev = next((r for r in results if "venta" in r[0].lower()), None)
    ga = next((r for r in results if "galones" in r[0].lower()), None)
    if not ev or not ga:
        reading.notes.append("anchor_labels_not_found")
        return reading
    reading.anchors_found = True

    y_lo, y_hi = ga[2].origin.y, ev[2].origin.y
    # Total line: strict match, it's always clean ("15000" / "150.00").
    total_matches = [
        r for r in results
        if TOTAL_NUM_RE.fullmatch(r[0]) and y_lo < r[2].origin.y < y_hi
    ]
    total_matches.sort(key=lambda r: -r[2].origin.y)
    total_candidate = total_matches[0] if total_matches else None

    # Gallons line: below the total line, loosely require >=2 digits since
    # OCR noise (stray "/", missing digits) is common on this smaller text.
    total_y = total_candidate[2].origin.y if total_candidate else y_hi
    gallons_pool = [
        r for r in results
        if HAS_ENOUGH_DIGITS_RE.search(r[0])
        and y_lo < r[2].origin.y < total_y
        and r is not total_candidate
    ]
    gallons_pool.sort(key=lambda r: -r[2].origin.y)
    gallons_candidate = gallons_pool[0] if gallons_pool else None

    if total_candidate:
        reading.total_raw = total_candidate[0]
        reading.total_gtq = parse_total_from_digits(digits_only(total_candidate[0]))
    else:
        reading.notes.append("total_not_found")

    # Gallons: pool digit strings from the whole-image pass plus a tight
    # recrop at both OCR quality levels. Photographed 7-segment digits are
    # noisy enough that a single pass can silently misread one digit (e.g.
    # "6" -> "5"), so we don't just trust the first 4-digit hit — we require
    # it to be corroborated (as a prefix) by another, independent partial
    # reading before accepting it.
    gallons_digit_candidates = []
    if gallons_candidate:
        gallons_digit_candidates.append(digits_only(gallons_candidate[0]))
        im = Image.open(jpeg_path)
        crop = _tight_crop(im, gallons_candidate[2])
        for level in (FAST, ACCURATE):
            for text, _, _ in _run_ocr(_pil_to_cg(crop), level=level):
                digits = digits_only(text)
                if digits:
                    gallons_digit_candidates.append(digits)
    else:
        reading.notes.append("gallons_not_found")

    reading.notes.append(f"gallons_ocr_candidates={gallons_digit_candidates}")

    accepted = pick_corroborated(gallons_digit_candidates, min_len=3)
    parsed_gallons = parse_gallons_from_digits(accepted) if accepted else None

    if parsed_gallons is not None:
        reading.gallons_raw = accepted
        reading.gallons = parsed_gallons
    elif accepted:
        reading.notes.append("gallons_digits_incomplete_no_corroboration")
    elif gallons_digit_candidates:
        reading.notes.append("gallons_digits_disagree")

    if reading.total_gtq is not None and reading.gallons:
        reading.price_per_gallon_gtq = round(reading.total_gtq / reading.gallons, 4)
        reading.parse_ok = True

    return reading


# The price LCD for each fuel sits a short distance directly above its
# printed fuel-name label, in the same column -- these offsets (as a
# fraction of image size) were measured by hand against a real photo of
# this station's board (see README, Fase 2). They won't transfer to a
# differently laid out board.
PRICE_LCD_ABOVE_LABEL = (0.09, 0.21)  # (dy_min, dy_max) above the label's y
PRICE_LCD_X_PAD = 0.035


def _bbox_center(bbox) -> tuple[float, float]:
    return bbox.origin.x + bbox.size.width / 2, bbox.origin.y + bbox.size.height / 2


def _price_lcd_crop_box(im: Image.Image, label_bbox) -> tuple[int, int, int, int]:
    W, H = im.size
    x, y, w = label_bbox.origin.x, label_bbox.origin.y, label_bbox.size.width
    dy_min, dy_max = PRICE_LCD_ABOVE_LABEL
    x0n, x1n = x - PRICE_LCD_X_PAD, x + w + PRICE_LCD_X_PAD
    y0n, y1n = y + dy_min, y + dy_max
    px0, px1 = max(int(x0n * W), 0), min(int(x1n * W), W)
    py0, py1 = max(int((1 - y1n) * H), 0), min(int((1 - y0n) * H), H)
    return px0, py0, px1, py1


def read_price_board(jpeg_path: Path, crops_dir: Path) -> PriceBoardReading:
    """Locate the station's posted price-per-gallon board (one small LCD per
    fuel type: Diesel/Regular/Super/V-Power) in a photo that also contains
    the main pump display, and save a close-up crop of each price LCD for a
    human to read.

    These LCDs are far smaller and blurrier than the main "Esta Venta"
    display, and neither Vision nor Tesseract read them reliably -- so
    unlike read_pump_display, this never parses a digit value on its own.
    It only finds each fuel's printed name label (accurate-level OCR over
    the whole image, matched case-insensitively) and crops the LCD expected
    directly above it. A best-effort single OCR pass on that crop is kept
    as `prices_raw` purely as a hint for the reviewer, not a trusted result.
    """
    reading = PriceBoardReading()
    results = _run_ocr(_load_cg_image(jpeg_path), level=ACCURATE, lang_correction=True)
    im = Image.open(jpeg_path)
    crops_dir.mkdir(parents=True, exist_ok=True)

    for fuel in FUEL_TYPES:
        label = next(
            (r for r in results if fuel.replace("-", "") in r[0].lower().replace("-", "").replace(" ", "")),
            None,
        )
        if not label:
            reading.notes.append(f"{fuel}_label_not_found")
            continue

        crop = im.crop(_price_lcd_crop_box(im, label[2]))
        crop_path = crops_dir / f"{jpeg_path.stem}_{fuel}.jpg"
        crop.resize((crop.width * 3, crop.height * 3), Image.LANCZOS).save(crop_path, "JPEG", quality=92)
        reading.crop_paths[fuel] = str(crop_path.relative_to(crops_dir.parent.parent.parent))

        hint = next((text for text, _, _ in _run_ocr(_pil_to_cg(crop), level=FAST) if digits_only(text)), None)
        reading.prices_raw[fuel] = hint

    return reading
