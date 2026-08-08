"""Full EXIF/GPS metadata extraction via exiftool (installed through Homebrew)."""
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _require_exiftool() -> str:
    path = shutil.which("exiftool")
    if not path:
        raise RuntimeError(
            "exiftool not found on PATH. Install it with `brew install exiftool`."
        )
    return path


def extract_all_metadata(image_path: Path) -> dict[str, Any]:
    """Return every EXIF/GPS/QuickTime tag exiftool can pull from the file."""
    exiftool = _require_exiftool()
    proc = subprocess.run(
        [exiftool, "-j", "-a", "-n", str(image_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)[0]
    return data


def extract_selected_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Pull out the fields the dataset actually needs, tolerating missing keys."""

    def g(*keys, default=None):
        for k in keys:
            if k in raw:
                return raw[k]
        return default

    lat = g("GPSLatitude")
    lon = g("GPSLongitude")
    # exiftool -n gives signed decimal degrees directly.
    return {
        "datetime_original": g("DateTimeOriginal"),
        "offset_time_original": g("OffsetTimeOriginal"),
        "subsec_time_original": g("SubSecTimeOriginal"),
        "gps_latitude": lat,
        "gps_longitude": lon,
        "gps_altitude_m": g("GPSAltitude"),
        "gps_datetime": g("GPSDateTime"),
        "gps_horizontal_error_m": g("GPSHPositioningError"),
        "camera_make": g("Make"),
        "camera_model": g("Model"),
        "lens_model": g("LensModel"),
        "software": g("Software"),
        "image_width": g("ImageWidth", "ExifImageWidth"),
        "image_height": g("ImageHeight", "ExifImageHeight"),
        "orientation": g("Orientation"),
        "exposure_time": g("ExposureTime"),
        "f_number": g("FNumber"),
        "iso": g("ISO"),
        "focal_length_mm": g("FocalLength"),
        "file_size_bytes": g("FileSize"),
    }


def dump_metadata(image_path: Path, out_json_path: Path) -> dict[str, Any]:
    """Extract full metadata and cache it to disk, skipping if already cached."""
    if out_json_path.exists() and out_json_path.stat().st_mtime >= image_path.stat().st_mtime:
        return json.loads(out_json_path.read_text())

    raw = extract_all_metadata(image_path)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False))
    return raw
