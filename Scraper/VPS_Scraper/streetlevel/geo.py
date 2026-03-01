"""Geographic coordinate utilities. Stripped to pure-math functions only."""

import math
from typing import Tuple


def tile_coord_to_wgs84(x: float, y: float, zoom: int) -> Tuple[float, float]:
    """Converts XYZ tile coordinates to WGS84 (lat, lon)."""
    scale = 1 << zoom
    lon_deg = x / scale * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / scale)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def wgs84_to_tile_coord(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Converts WGS84 (lat, lon) to XYZ tile coordinates."""
    lat_rad = math.radians(lat)
    scale = 1 << zoom
    x = (lon + 180.0) / 360.0 * scale
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * scale
    return int(x), int(y)
