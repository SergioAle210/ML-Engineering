"""Platform dispatcher for reading the pump's LCD display off a converted
JPEG. macOS uses Apple's Vision framework (src/ocr_backend_vision.py) since
it reads these photographed 7-segment digits noticeably more reliably;
Windows/Linux fall back to Tesseract + a 7-segment model
(src/ocr_backend_tesseract.py). Both expose the same read_pump_display(path)
-> PumpReading interface, so src/pipeline.py doesn't care which one ran.
"""
import platform

from src.ocr_common import (  # noqa: F401 -- re-exported for src/pipeline.py
    PumpReading,
    digits_only,
    parse_gallons_from_digits,
    parse_total_from_digits,
    pick_corroborated,
)

if platform.system() == "Darwin":
    from src.ocr_backend_vision import read_pump_display
else:
    from src.ocr_backend_tesseract import read_pump_display

__all__ = [
    "PumpReading",
    "digits_only",
    "parse_gallons_from_digits",
    "parse_total_from_digits",
    "pick_corroborated",
    "read_pump_display",
]
