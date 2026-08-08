"""HEIC -> JPEG conversion, with EXIF orientation applied so pixels are upright."""
from pathlib import Path

import pillow_heif
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()


def heic_to_jpeg(src_path: Path, dst_path: Path) -> Path:
    """Convert a HEIC image to an upright JPEG. Skips work if dst is newer than src."""
    if dst_path.exists() and dst_path.stat().st_mtime >= src_path.stat().st_mtime:
        return dst_path

    with Image.open(src_path) as im:
        im = ImageOps.exif_transpose(im)  # bakes rotation into pixels
        im = im.convert("RGB")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst_path, "JPEG", quality=92)
    return dst_path
