"""Producer: download panoramas, extract views, push to queue.

Adapted from Image Downloading reference (gsvpd/core_optimized.py).
Fetches GSV tiles via aiohttp, stitches panoramas, extracts 8 directional
views (322x322, 60deg FOV, cubic interpolation, no antialias), preprocesses
to normalized FP16 tensors, and pushes (panoid, view_idx, tensor) into an
asyncio.Queue for the consumer.

Download config (hardcoded per spec Section 6.4):
  zoom_level=2, max_threads=150, workers=8, 8 views, 322x322, 60deg FOV,
  no_antialias=True, interpolation=cubic, keep_panorama=False
"""

import asyncio
import csv
import io
import logging
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as tvf

from tile_fetcher import fetch_panorama
from directional_views import extract_views

logger = logging.getLogger(__name__)

# ImageNet normalization (same as MegaLoc training — matches reference build_megaloc_index.py)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _preprocess_view(view_bgr: np.ndarray) -> torch.Tensor:
    """Convert a 322x322 BGR view to a normalized float16 tensor [3, 322, 322].

    Matches the reference transform pipeline:
        transforms.Resize((322, 322))  -- already 322x322 from view extraction
        transforms.ToTensor()          -- HWC uint8 -> CHW float32 [0,1]
        transforms.Normalize(...)      -- ImageNet mean/std
    """
    rgb = cv2.cvtColor(view_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    tensor = tvf.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor.half()


async def run_producer(
    queue: asyncio.Queue,
    csv_bytes: bytes,
    max_concurrent: int = 150,
    on_progress=None,
):
    """Parse CSV, fetch panoramas, extract views, push to queue.

    Args:
        queue: asyncio.Queue to push (panoid, view_idx, tensor) items.
        csv_bytes: Raw CSV bytes (PanoID,Lat,Lon — no header).
        max_concurrent: Max concurrent panorama downloads (spec: 150).
        on_progress: Optional callback(panos_done, panos_total, panos_failed).
    """
    # Parse CSV: PanoID,Lat,Lon (no header)
    reader = csv.reader(io.StringIO(csv_bytes.decode("utf-8")))
    pano_rows = [
        (row[0].strip(), float(row[1]), float(row[2]))
        for row in reader
        if len(row) >= 3 and row[0].strip()
    ]

    total = len(pano_rows)
    logger.info(f"Loaded {total} panoramas from CSV")

    if total == 0:
        await queue.put(None)
        return

    done = 0
    failed = 0
    sem = asyncio.Semaphore(max_concurrent)
    executor = ThreadPoolExecutor(max_workers=8)

    # Connection pool matching reference core_optimized.py (limit=600, limit_per_host=200)
    connector = aiohttp.TCPConnector(limit=600, limit_per_host=200, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=15, connect=8)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        async def process_one(panoid: str, lat: float, lon: float):
            nonlocal done, failed
            async with sem:
                try:
                    panorama = await fetch_panorama(session, panoid, executor)
                    if panorama is None:
                        failed += 1
                        done += 1
                        if on_progress:
                            on_progress(done, total, failed)
                        return

                    # Extract 8 directional views in thread (cv2 releases GIL)
                    loop = asyncio.get_running_loop()
                    views = await loop.run_in_executor(executor, extract_views, panorama)
                    del panorama

                    # Preprocess and push each view
                    for view_idx, view_bgr in enumerate(views):
                        tensor = await loop.run_in_executor(executor, _preprocess_view, view_bgr)
                        await queue.put((panoid, view_idx, tensor))
                    del views

                    done += 1
                    if on_progress:
                        on_progress(done, total, failed)

                except Exception as e:
                    logger.warning(f"Failed {panoid}: {e}")
                    failed += 1
                    done += 1
                    if on_progress:
                        on_progress(done, total, failed)

        # Process in chunks to avoid OOM from too many pending coroutines
        # (same pattern as reference core_optimized.py CHUNK_SIZE=10000)
        CHUNK_SIZE = 5000
        for i in range(0, total, CHUNK_SIZE):
            chunk = pano_rows[i:i + CHUNK_SIZE]
            tasks = [process_one(pid, lat, lon) for pid, lat, lon in chunk]
            await asyncio.gather(*tasks)

    executor.shutdown(wait=False)

    # Signal consumer that production is done
    await queue.put(None)
    logger.info(f"Producer done: {done - failed}/{total} succeeded, {failed} failed")
