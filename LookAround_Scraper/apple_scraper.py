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
