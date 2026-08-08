"""Reverse geocoding of GPS coordinates via OpenStreetMap Nominatim, with a
disk cache so repeated photos from the same station don't hit the API again."""
import json
import time
from pathlib import Path
from typing import Any

import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "gasolina-price-pipeline/1.0 (personal research project)"
CACHE_KEY_PRECISION = 4  # ~11m grid, station-scale


def _cache_key(lat: float, lon: float) -> str:
    return f"{round(lat, CACHE_KEY_PRECISION)},{round(lon, CACHE_KEY_PRECISION)}"


def _load_cache(cache_path: Path) -> dict[str, Any]:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def _save_cache(cache_path: Path, cache: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def reverse_geocode(lat: float, lon: float, cache_path: Path) -> dict[str, Any]:
    cache = _load_cache(cache_path)
    key = _cache_key(lat, lon)
    if key in cache:
        return cache[key]

    resp = requests.get(
        NOMINATIM_URL,
        params={"format": "json", "lat": lat, "lon": lon, "zoom": 18, "addressdetails": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    address = data.get("address", {})
    result = {
        "display_name": data.get("display_name"),
        "station_name": address.get("amenity"),
        "road": address.get("road"),
        "neighbourhood": address.get("neighbourhood"),
        "city": address.get("city") or address.get("town") or address.get("village"),
        "county": address.get("county"),
        "state": address.get("state"),
        "country": address.get("country"),
        "postcode": address.get("postcode"),
    }
    cache[key] = result
    _save_cache(cache_path, cache)
    time.sleep(1)  # respect Nominatim's 1 req/sec usage policy
    return result
