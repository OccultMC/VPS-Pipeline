"""Async tile fetcher — downloads GSV panorama tiles and stitches them.

Adapted from core_optimized.py for the VPR Pipeline Docker container.
Uses zoom level 2 (2048x1024 / 1664x832) by default — sufficient for 322px views.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import aiohttp
import cv2
import numpy as np

from constants import (
    ZOOM_SIZES,
    OLD_ZOOM_SIZES,
    TILES_AXIS_COUNT,
    TILE_SIZE,
)

ZOOM_LEVEL = 2


async def fetch_tile(
    session: aiohttp.ClientSession,
    panoid: str,
    x: int,
    y: int,
    retries: int = 2,
    backoff: float = 0.15,
) -> Optional[Tuple[int, int, bytes]]:
    """Fetch a single panorama tile with retry."""
    host = (x + y) % 4
    url = f"https://cbk{host}.google.com/cbk?output=tile&panoid={panoid}&zoom={ZOOM_LEVEL}&x={x}&y={y}"

    BLACK_TILE_BYTE_SIZE = 1184

    for attempt in range(1, retries + 1):
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                size = int(resp.headers.get("Content-Length", 0))
                if size == BLACK_TILE_BYTE_SIZE:
                    return None
                data = await resp.read()
                return (x, y, data)
        except Exception:
            if attempt < retries:
                await asyncio.sleep(backoff * (2 ** (attempt - 1)))
    return None


def _check_black_bottom(tile_data: bytes) -> bool:
    """Check if tile has black bottom rows (old panorama detection)."""
    arr = cv2.imdecode(np.frombuffer(tile_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return False
    bottom = arr[-5:]
    return bool(np.all(bottom <= 10))


async def fetch_panorama(
    session: aiohttp.ClientSession,
    panoid: str,
    executor: ThreadPoolExecutor,
) -> Optional[np.ndarray]:
    """Fetch all tiles for a panorama, stitch into a single BGR image.

    Returns:
        Stitched BGR numpy array, or None on failure.
    """
    tiles_x, tiles_y = TILES_AXIS_COUNT[ZOOM_LEVEL]

    tasks = [
        fetch_tile(session, panoid, x, y)
        for x in range(tiles_x + 1)
        for y in range(tiles_y + 1)
    ]
    results = await asyncio.gather(*tasks)
    tiles = [t for t in results if t is not None]

    if not tiles:
        return None

    # Determine dimensions
    if ZOOM_LEVEL <= 2:
        # Check second tile for black bottom to detect old vs new pano
        sorted_tiles = sorted(tiles, key=lambda t: (t[1], t[0]))
        if len(sorted_tiles) > 1:
            is_old = await asyncio.get_running_loop().run_in_executor(
                executor, _check_black_bottom, sorted_tiles[1][2]
            )
        else:
            is_old = False
        w, h = OLD_ZOOM_SIZES[ZOOM_LEVEL] if is_old else ZOOM_SIZES[ZOOM_LEVEL]
    else:
        x_count = len({x for x, _, _ in tiles})
        y_count = len({y for _, y, _ in tiles})
        from constants import TILE_COUNT_TO_SIZE
        dims = TILE_COUNT_TO_SIZE.get((x_count, y_count))
        if dims is None:
            return None
        w, h = dims

    # Stitch in thread (numpy/cv2 release GIL)
    loop = asyncio.get_running_loop()
    panorama = await loop.run_in_executor(executor, _stitch_tiles, tiles, w, h)
    return panorama


def _stitch_tiles(
    tiles: List[Tuple[int, int, bytes]], width: int, height: int
) -> np.ndarray:
    """Stitch tile bytes into a single BGR image."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    for x, y, data in tiles:
        tile = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if tile is None:
            continue
        th, tw = tile.shape[:2]
        y0 = y * TILE_SIZE
        x0 = x * TILE_SIZE
        y1 = min(y0 + th, height)
        x1 = min(x0 + tw, width)
        h_t = y1 - y0
        w_t = x1 - x0
        if h_t > 0 and w_t > 0:
            canvas[y0:y1, x0:x1] = tile[:h_t, :w_t]

    return canvas
