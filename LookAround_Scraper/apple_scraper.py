"""Apple Look Around debug scraper — pure logic, no web framework."""
from __future__ import annotations

from typing import Callable, List, Tuple, Optional
from streetlevel.geo import wgs84_to_tile_coord


def _polygon_to_tiles(polygon: List[List[float]]) -> List[Tuple[int, int]]:
    """
    Return the list of z=17 (x, y) tiles covering the bbox of `polygon`.

    `polygon` is [[lon, lat], ...]. Tiles are emitted in row-major order
    (y outer ascending, x inner ascending) so iteration is top-left → bottom-right.
    """
    if not polygon:
        return []
    lons = [pt[0] for pt in polygon]
    lats = [pt[1] for pt in polygon]
    x_min, y_max = wgs84_to_tile_coord(min(lats), min(lons), 17)
    x_max, y_min = wgs84_to_tile_coord(max(lats), max(lons), 17)
    tiles: List[Tuple[int, int]] = []
    for ty in range(y_min, y_max + 1):
        for tx in range(x_min, x_max + 1):
            tiles.append((tx, ty))
    return tiles


import io

import pillow_heif
from PIL import Image
from shapely.geometry import Point, Polygon as ShapelyPolygon

pillow_heif.register_heif_opener()


def _decode_heic_to_jpg(heic_bytes: bytes, out_path: str, quality: int = 92) -> None:
    """Decode HEIC bytes to a JPEG file at `out_path`. Overwrites if exists."""
    img = Image.open(io.BytesIO(heic_bytes))
    img = img.convert("RGB")
    img.save(out_path, format="JPEG", quality=quality)


def _pick_pano(panos, polygon: List[List[float]]):
    """Return first pano inside `polygon`, else first pano in list, else None."""
    if not panos:
        return None
    poly = ShapelyPolygon(polygon)
    for pano in panos:
        if poly.contains(Point(pano.lon, pano.lat)):
            return pano
    return panos[0]


import csv

# Index → human name. Order matches streetlevel.lookaround.Face enum.
FACE_NAMES = ["back", "left", "front", "right", "top", "bottom"]


def _write_meta_csv(pano, face_paths: List[str], csv_path: str) -> None:
    """Write one row per face. `face_paths` is 6 entries in FACE_NAMES order."""
    cols = [
        "pano_id", "build_id", "lat", "lon", "capture_date",
        "heading", "pitch", "roll", "coverage_type",
        "face_index", "face_name", "image_path",
    ]
    capture_date = pano.date.isoformat() if pano.date else ""
    coverage_type = pano.coverage_type.name if pano.coverage_type else ""
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, path in enumerate(face_paths):
            w.writerow({
                "pano_id": pano.id,
                "build_id": pano.build_id,
                "lat": pano.lat,
                "lon": pano.lon,
                "capture_date": capture_date,
                "heading": getattr(pano, "heading", ""),
                "pitch": getattr(pano, "pitch", ""),
                "roll": getattr(pano, "roll", ""),
                "coverage_type": coverage_type,
                "face_index": i,
                "face_name": FACE_NAMES[i],
                "image_path": path,
            })
