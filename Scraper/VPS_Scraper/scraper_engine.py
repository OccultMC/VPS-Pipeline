"""Async panorama metadata scraper using Google coverage tiles.

Adapted from scraper.py Phase 1 tile discovery — discovers all panorama IDs
and coordinates within a polygon, outputs CSV rows (PanoID,Lat,Lon).
"""

import asyncio
import csv
import io
import logging
import math
from typing import List, Tuple, Callable, Optional

import aiohttp
from aiohttp import ClientSession, TCPConnector, ClientTimeout
from shapely.geometry import Polygon, Point

from streetlevel.geo import wgs84_to_tile_coord
from streetlevel.streetview.api import get_coverage_tile_async
from streetlevel.streetview.parse import parse_coverage_tile_response, PanoMetadata

logger = logging.getLogger(__name__)

# Scraper configuration
TILE_ZOOM_LEVEL = 17
CONCURRENCY = 1000
SESSION_POOLS = 4
TIMEOUT_TOTAL = 30.0
TIMEOUT_CONNECT = 5.0


def truncate_coord(value: float, decimals: int = 6) -> str:
    """Truncate (not round) a coordinate to N decimal places."""
    factor = 10 ** decimals
    truncated = math.trunc(value * factor) / factor
    return f"{truncated:.{decimals}f}"


async def scrape_polygon(
    polygon_coords: List[Tuple[float, float]],
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
    cancel_event: Optional[asyncio.Event] = None,
) -> List[PanoMetadata]:
    """Discover all Street View panoramas within a polygon.

    Args:
        polygon_coords: List of (lat, lon) tuples defining the polygon.
        progress_callback: Optional callback(tiles_done, total_tiles, panos_found).
        cancel_event: Optional event to signal cancellation.

    Returns:
        List of PanoMetadata objects found within the polygon.
    """
    # Build Shapely polygon (lon, lat order for Shapely)
    polygon = Polygon([(lon, lat) for lat, lon in polygon_coords])

    # Calculate bounding tiles
    bounds = polygon.bounds  # (minx=minlon, miny=minlat, maxx=maxlon, maxy=maxlat)
    min_tile_x, max_tile_y = wgs84_to_tile_coord(bounds[1], bounds[0], TILE_ZOOM_LEVEL)
    max_tile_x, min_tile_y = wgs84_to_tile_coord(bounds[3], bounds[2], TILE_ZOOM_LEVEL)

    tiles = [
        (x, y)
        for x in range(min_tile_x, max_tile_x + 1)
        for y in range(min_tile_y, max_tile_y + 1)
    ]
    total_tiles = len(tiles)
    logger.info(f"Scraping {total_tiles} tiles at zoom {TILE_ZOOM_LEVEL}")

    found_panos = {}  # panoid -> PanoMetadata (dedup)
    tiles_processed = 0
    consecutive_errors = 0

    timeout = ClientTimeout(total=TIMEOUT_TOTAL, connect=TIMEOUT_CONNECT)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Create multiple session pools for throughput
    connectors = []
    sessions = []
    for _ in range(SESSION_POOLS):
        conn = TCPConnector(limit=0, ttl_dns_cache=300, enable_cleanup_closed=True)
        connectors.append(conn)
        sessions.append(ClientSession(connector=conn, timeout=timeout))

    pool_idx = [0]

    async def fetch_tile(x: int, y: int) -> List[PanoMetadata]:
        nonlocal tiles_processed, consecutive_errors

        if cancel_event and cancel_event.is_set():
            return []

        try:
            sess = sessions[pool_idx[0] % SESSION_POOLS]
            pool_idx[0] += 1

            async with semaphore:
                raw_response = await get_coverage_tile_async(x, y, sess)

            panos = parse_coverage_tile_response(raw_response)
            found = []
            for p in panos:
                if p.id not in found_panos:
                    point = Point(p.lon, p.lat)
                    if polygon.contains(point):
                        found_panos[p.id] = p
                        found.append(p)

            tiles_processed += 1
            if consecutive_errors > 0:
                consecutive_errors = max(0, consecutive_errors - 1)

            if progress_callback and tiles_processed % 50 == 0:
                progress_callback(tiles_processed, total_tiles, len(found_panos))

            return found

        except Exception as e:
            tiles_processed += 1
            consecutive_errors += 1
            if consecutive_errors > 50:
                await asyncio.sleep(2.0)
            return []

    try:
        # Process tiles in batches
        all_panos = []
        batch_size = CONCURRENCY
        for i in range(0, len(tiles), batch_size):
            if cancel_event and cancel_event.is_set():
                break
            batch = tiles[i : i + batch_size]
            tasks = [asyncio.create_task(fetch_tile(x, y)) for x, y in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, list):
                    all_panos.extend(res)

    finally:
        for sess in sessions:
            await sess.close()
        for conn in connectors:
            await conn.close()

    # Final progress callback
    if progress_callback:
        progress_callback(total_tiles, total_tiles, len(found_panos))

    logger.info(f"Discovered {len(found_panos)} unique panoramas in {total_tiles} tiles")
    return list(found_panos.values())


def generate_csv_bytes(panos: List[PanoMetadata]) -> bytes:
    """Generate CSV bytes from panorama metadata.

    Format: PanoID,Lat,Lon (no header, 6 decimal truncation).
    """
    output = io.StringIO()
    writer = csv.writer(output)
    for p in panos:
        writer.writerow([p.id, truncate_coord(p.lat), truncate_coord(p.lon)])
    return output.getvalue().encode("utf-8")


def split_csv_chunks(
    panos: List[PanoMetadata],
    city_name: str,
    chunk_size: int = 25000,
    worker_override: Optional[int] = None,
) -> List[Tuple[str, bytes]]:
    """Split panoramas into CSV chunks.

    Args:
        panos: List of panorama metadata.
        city_name: Name for file naming (e.g., Sacramento).
        chunk_size: Panoramas per chunk (default 25000).
        worker_override: If set, split into this many chunks regardless.

    Returns:
        List of (filename, csv_bytes) tuples.
    """
    if worker_override and worker_override > 0:
        # Split evenly into N chunks
        total_chunks = worker_override
        chunk_size = max(1, len(panos) // total_chunks)
        # Adjust so last chunk gets remainder
    else:
        total_chunks = max(1, math.ceil(len(panos) / chunk_size))

    chunks = []
    for i in range(total_chunks):
        start = i * chunk_size if not worker_override else i * (len(panos) // total_chunks)
        if i == total_chunks - 1:
            end = len(panos)
        else:
            end = start + chunk_size if not worker_override else (i + 1) * (len(panos) // total_chunks)

        chunk_panos = panos[start:end]
        if not chunk_panos:
            continue

        filename = f"{city_name}_{i + 1}_{total_chunks}.csv"
        csv_data = generate_csv_bytes(chunk_panos)
        chunks.append((filename, csv_data))

    return chunks
