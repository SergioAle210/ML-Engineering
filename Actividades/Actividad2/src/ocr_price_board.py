"""Platform dispatcher for reading the station's posted price board (one LCD
per fuel type: Diesel/Regular/Super/V-Power). macOS uses Apple's Vision
framework (src/ocr_backend_vision.py); Windows/Linux currently raise
NotImplementedError (src/ocr_backend_tesseract.py) -- see that module's
read_price_board docstring for why. Both expose the same
read_price_board(path) -> PriceBoardReading interface, so src/pipeline.py
doesn't care which one ran.
"""
import platform

from src.ocr_common import FUEL_TYPES, PriceBoardReading  # noqa: F401 -- re-exported

if platform.system() == "Darwin":
    from src.ocr_backend_vision import read_price_board
else:
    from src.ocr_backend_tesseract import read_price_board

__all__ = ["FUEL_TYPES", "PriceBoardReading", "read_price_board"]
