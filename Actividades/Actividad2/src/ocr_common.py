"""Shared types and digit-parsing helpers used by every OCR backend
(src/ocr_backend_vision.py on macOS, src/ocr_backend_tesseract.py elsewhere).
"""
import re
from dataclasses import dataclass, field

DIGITS_ONLY_RE = re.compile(r"\d+")

# The user always dispenses exactly Q150.00 of Súper. It's a real, stated
# constraint of this dataset (see outliers.py), not a guess — used as a
# documented fallback when OCR can't confidently read the total digits.
EXPECTED_TOTAL_GTQ = 150.00


@dataclass
class PumpReading:
    total_gtq: float | None = None
    gallons: float | None = None
    price_per_gallon_gtq: float | None = None
    total_raw: str | None = None
    gallons_raw: str | None = None
    anchors_found: bool = False
    parse_ok: bool = False
    notes: list[str] = field(default_factory=list)


def digits_only(text: str) -> str:
    return "".join(DIGITS_ONLY_RE.findall(text))


def parse_total_from_digits(digits: str) -> float | None:
    # Display has no visible decimal point when OCR drops it; last two
    # digits are always cents (e.g. "15000" -> 150.00).
    if len(digits) < 3:
        return None
    return int(digits) / 100


def parse_gallons_from_digits(digits: str) -> float | None:
    # Format is always d.ddd (one whole digit, three decimals).
    if len(digits) != 4:
        return None
    return int(digits) / 1000


def pick_corroborated(candidates: list[str], min_len: int = 3) -> str | None:
    """Pick the digit string best supported by independent OCR passes.

    Photographed 7-segment digits are noisy enough that any single pass can
    silently misread one digit, so we never trust the first plausible hit —
    a candidate is only accepted if another, independent candidate shares
    its first `min_len` digits (or is a prefix of it). Ties prefer the
    longer (more complete) reading. Returns None if nothing is corroborated.
    """
    candidates = [c for c in candidates if len(c) >= min_len]
    if not candidates:
        return None

    def score(candidate: str, source_index: int) -> int:
        s = 0
        for i, d in enumerate(candidates):
            if i == source_index:
                continue
            if candidate.startswith(d[:min_len]) or d.startswith(candidate[:min_len]):
                s += 1
        return s

    scored = [(c, score(c, i), len(c)) for i, c in enumerate(candidates)]
    scored.sort(key=lambda t: (t[1], t[2]), reverse=True)
    best, best_score, _ = scored[0]
    return best if best_score >= 1 else None
