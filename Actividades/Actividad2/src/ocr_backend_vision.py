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

import numpy as np
import Quartz
import Vision
from Foundation import NSURL
from PIL import Image
from scipy import ndimage

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


# Fallback offset when a price LCD isn't found by its visual signature (see
# _detect_lcd_row below) and we have to guess from the printed fuel-name
# label instead. Measured by hand against real photos of this station's
# board (see README, Fase 2); won't transfer to a differently laid out board.
PRICE_LCD_ABOVE_LABEL = (0.0, 0.20)  # (dy_min, dy_max) above the label's y
PRICE_LCD_X_PAD = 0.04


def _price_lcd_crop_box(im: Image.Image, label_bbox) -> tuple[int, int, int, int]:
    W, H = im.size
    x, y, w = label_bbox.origin.x, label_bbox.origin.y, label_bbox.size.width
    dy_min, dy_max = PRICE_LCD_ABOVE_LABEL
    x0n, x1n = x - PRICE_LCD_X_PAD, x + w + PRICE_LCD_X_PAD
    y0n, y1n = y + dy_min, y + dy_max
    px0, px1 = max(int(x0n * W), 0), min(int(x1n * W), W)
    py0, py1 = max(int((1 - y1n) * H), 0), min(int((1 - y0n) * H), H)
    return px0, py0, px1, py1


def _detect_lcd_row(im: Image.Image, downscale: int = 4) -> list[tuple[float, float, float, float]]:
    """Find the row of small price-board LCDs by their visual signature --
    a bright, low-saturation, faintly blue-white rectangle -- instead of by
    reading the (often illegible) printed fuel name next to them.

    Every LCD on this pump (the big "Esta Venta" display and the four price
    displays) has that same glow, sharply different from the yellow/red/
    green painted panels around it, so a simple color+shape threshold finds
    them reliably even in photos where OCR can't read a single label (e.g.
    a hose crossing in front of the print). The four price LCDs always sit
    in one horizontal row below the pump's keypad, evenly spaced -- so we
    cluster all candidate blobs by y and keep the largest same-row cluster,
    ordered left to right. That left-to-right order matches FUEL_TYPES
    (Diesel/Regular/Super/V-Power) on this station's board; a differently
    laid out board would need reordering here.
    """
    W0, H0 = im.size
    small = im.resize((max(W0 // downscale, 1), max(H0 // downscale, 1)), Image.BILINEAR)
    W, H = small.size
    arr = np.asarray(small.convert("RGB")).astype(np.float32) / 255.0
    r, _, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx, mn = arr.max(axis=-1), arr.min(axis=-1)
    brightness, saturation = mx, np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0)
    is_lcd_glow = (brightness > 0.55) & (saturation < 0.35) & (b >= r - 0.02)

    labelled, count = ndimage.label(is_lcd_glow)
    total_area = W * H
    boxes = []
    for i, sl in enumerate(ndimage.find_objects(labelled), start=1):
        if sl is None:
            continue
        ys, xs = sl
        area = int((labelled[sl] == i).sum())
        if not (0.00015 <= area / total_area <= 0.006):
            continue
        y0, y1, x0, x1 = ys.start, ys.stop, xs.start, xs.stop
        w, h = x1 - x0, y1 - y0
        if h == 0 or not (1.3 < w / h < 4.5):
            continue
        if area / (w * h) < 0.5:
            continue
        if y0 / H < 0.4:  # price panel row is always in the lower part of the frame
            continue
        boxes.append((x0 / W, y0 / H, x1 / W, y1 / H))

    if not boxes:
        return []
    centers = sorted((((b[1] + b[3]) / 2, b) for b in boxes), key=lambda c: c[0])
    groups, current = [], [centers[0]]
    for c in centers[1:]:
        if c[0] - current[-1][0] <= 0.035:
            current.append(c)
        else:
            groups.append(current)
            current = [c]
    groups.append(current)
    best_row = max(groups, key=len)
    return [c[1] for c in sorted(best_row, key=lambda c: c[1][0])]


def _binarize(im: Image.Image) -> Image.Image:
    arr = np.asarray(im.convert("L")).astype(np.float32)
    threshold = arr.mean() - 0.15 * arr.std()
    return Image.fromarray(((arr > threshold) * 255).astype(np.uint8))


def _price_digit_candidates(crop: Image.Image) -> list[str]:
    """Gather independent digit readings of a price LCD crop from every OCR
    engine available, to give pick_corroborated the best chance of agreeing
    on the real 4-digit price. Vision's general text recognizer is the
    baseline (always available on macOS); Tesseract's "letsgodigital"
    7-segment model -- the same one src/ocr_backend_tesseract.py uses on
    Windows/Linux -- is tried too when the binary happens to be installed,
    since it's purpose-built for this exact font and sometimes catches
    digits Vision misses. It's optional here (unlike on Windows/Linux): the
    README promises macOS users don't need Tesseract installed, so its
    absence must never be an error, only a missed corroboration opportunity.
    """
    candidates = [digits_only(text) for text, _, _ in _run_ocr(_pil_to_cg(crop), level=FAST)]
    candidates += [digits_only(text) for text, _, _ in _run_ocr(_pil_to_cg(crop), level=ACCURATE)]
    try:
        from src.ocr_backend_tesseract import _digit_candidates, _require_tesseract

        tesseract_bin = _require_tesseract()
        candidates += _digit_candidates(tesseract_bin, crop)
        candidates += _digit_candidates(tesseract_bin, _binarize(crop))
    except RuntimeError:
        pass  # tesseract not installed on this machine -- Vision-only is fine
    return [c for c in candidates if c]


def _crop_from_lcd_box(im: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    W, H = im.size
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    padx, pad_top, pad_bottom = w * 0.3, h * 1.0, h * 0.4
    px0 = max(int((x0 - padx) * W), 0)
    px1 = min(int((x1 + padx) * W), W)
    py0 = max(int((y0 - pad_top) * H), 0)
    py1 = min(int((y1 + pad_bottom) * H), H)
    return im.crop((px0, py0, px1, py1))


def read_price_board(jpeg_path: Path, crops_dir: Path) -> PriceBoardReading:
    """Locate the station's posted price-per-gallon board (one small LCD per
    fuel type: Diesel/Regular/Super/V-Power) in a photo that also contains
    the main pump display, and save a close-up crop of each price LCD for a
    human to read.

    These LCDs are far smaller and blurrier than the main "Esta Venta"
    display, and neither Vision nor Tesseract read their digits reliably --
    so unlike read_pump_display, this never parses a digit value on its own.
    Primary strategy is _detect_lcd_row: find the four LCDs by their visual
    glow, not by reading print. If that doesn't find all four (e.g. only 2-3
    line up within the row tolerance), whichever fuels are missing fall back
    to the old approach of finding the printed fuel-name label and cropping
    the LCD expected above it -- worse, but better than nothing. A
    multi-engine, corroborated OCR pass on the final crop (see
    _price_digit_candidates) is kept as `prices_raw`, still just a hint for
    the reviewer -- src/pipeline.py decides whether it's trustworthy enough
    to prefill price_*_gtq, by checking it against the one price on this
    board that's always known for free (Súper's, from price_per_gallon_gtq).
    """
    reading = PriceBoardReading()
    im = Image.open(jpeg_path)
    crops_dir.mkdir(parents=True, exist_ok=True)

    lcd_row = _detect_lcd_row(im)
    lcd_by_fuel = dict(zip(FUEL_TYPES, lcd_row)) if len(lcd_row) == len(FUEL_TYPES) else {}

    results = None
    for fuel in FUEL_TYPES:
        box = lcd_by_fuel.get(fuel)
        if box is not None:
            crop = _crop_from_lcd_box(im, box)
        else:
            if results is None:
                results = _run_ocr(_load_cg_image(jpeg_path), level=ACCURATE, lang_correction=True)
            # Each fuel name is printed twice in frame: once on a decorative
            # sticker on the pump's side pillar (no LCD nearby) and once on
            # the actual price panel next to its LCD, lower in the photo.
            # Vision returns matches in no guaranteed order, so picking the
            # first one would often grab the decorative sticker instead --
            # the bottom-most match (smallest bounding-box origin.y, since
            # Vision's y-axis starts at the image bottom) is the real
            # price-panel label.
            candidates = [
                r for r in results if fuel.replace("-", "") in r[0].lower().replace("-", "").replace(" ", "")
            ]
            label = min(candidates, key=lambda r: r[2].origin.y) if candidates else None
            if not label:
                reading.notes.append(f"{fuel}_label_not_found")
                continue
            crop = im.crop(_price_lcd_crop_box(im, label[2]))

        crop_path = crops_dir / f"{jpeg_path.stem}_{fuel}.jpg"
        crop.resize((crop.width * 3, crop.height * 3), Image.LANCZOS).save(crop_path, "JPEG", quality=92)
        reading.crop_paths[fuel] = str(crop_path.relative_to(crops_dir.parent.parent.parent))

        # Unlike read_pump_display's gallons hint, an uncorroborated single
        # guess is not kept as a fallback here: src/pipeline.py treats
        # prices_raw as trustworthy enough to auto-fill price_*_gtq once it
        # also passes a plausibility check against Súper's known price, and
        # a single OCR pass that nothing else agrees with is exactly the
        # kind of misread (one wrong digit) that plausibility check can't
        # reliably catch, since the wrong value can still land in range by
        # coincidence. Only a reading two independent passes agree on goes
        # in here; anything else stays None and waits for manual review.
        reading.prices_raw[fuel] = pick_corroborated(_price_digit_candidates(crop), min_len=3)

    return reading
