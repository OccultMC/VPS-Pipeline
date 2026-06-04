"""
Google Street View server-rendered thumbnail fetcher.

One function: fetch_thumbnail_view(). Used by pipeline.py to pull
each pre-projected perspective view directly from Google's thumbnail
endpoint (no tile fetch, no client-side stitching, no equirect reproject).

Replaces the old gsvpd/ tile pipeline entirely.
"""
import asyncio
import random
from typing import Optional

# Uses curl_cffi instead of aiohttp so the TLS + HTTP/2/3 fingerprint matches
# real Chrome — Google's streetviewpixels abuse gate inspects the handshake
# (JA3/JA4) and HTTP/2 SETTINGS frame, not just headers. Vanilla aiohttp gets
# 403 PERMISSION_DENIED at scale even with a Chrome User-Agent.
from curl_cffi.requests import AsyncSession
import cv2
import numpy as np


# ── Endpoint ──────────────────────────────────────────────────────────────
THUMB_HOST = "streetviewpixels-pa.googleapis.com"
THUMB_BASE = f"https://{THUMB_HOST}/v1/thumbnail"

# ── Retry policy ──────────────────────────────────────────────────────────
THUMB_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# ── Black-placeholder detection ───────────────────────────────────────────
# Empirical: Google returns a near-solid-black JPEG (~uniform <5 mean) for
# missing yaws or "no imagery here" placeholders. Real outdoor views even at
# night sit comfortably above these thresholds because of compression noise.
BLACK_THUMB_MEAN = 5.0
BLACK_THUMB_STD = 5.0


async def fetch_thumbnail_view(
    session: AsyncSession,
    panoid: str,
    yaw: float,
    pitch: float,
    fov: float,
    w: int,
    h: int,
    retries: int = 5,
    backoff: float = 0.3,
    stats: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """Fetch one server-rendered perspective view from Google's thumbnail endpoint.

    Returns numpy uint8 HWC RGB array, or None on any failure
    (4xx, persistent 5xx, decode error, black/garbage placeholder).

    Failure handling:
      * 200 + valid decode + non-black     -> return RGB array
      * 200 + decode fail or near-black    -> None (count black via stats)
      * 403                                -> retry with backoff (Google
                                              occasionally flags a single
                                              request for a split second);
                                              only after every retry also
                                              returns 403 do we conclude the
                                              IP is blocked and set
                                              stats['ip_blocked_403']
      * 5xx / 429                          -> retry with Retry-After + jittered
                                              exponential backoff
      * other 4xx                          -> None (no retry — permanent)
      * exception (timeout / reset / DNS)  -> retry with backoff
    """
    url = (
        f"{THUMB_BASE}?panoid={panoid}&cb_client=maps_sv.tactile.gps"
        f"&w={w}&h={h}&yaw={yaw:.2f}&pitch={pitch:.2f}&thumbfov={int(round(fov))}"
    )
    for attempt in range(1, retries + 1):
        try:
            # curl_cffi's await session.get() returns the fully-loaded Response.
            # No context manager; response.content is already bytes (no await).
            response = await session.get(url, timeout=15)
            status = response.status_code

            if status == 200:
                data = response.content
                arr_bgr = cv2.imdecode(
                    np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR
                )
                if arr_bgr is None:
                    if stats is not None:
                        stats['decode_fail'] = stats.get('decode_fail', 0) + 1
                    return None

                # "No imagery here" placeholder: solid black JPEG.
                # Detected via low mean AND low std so legitimate dark
                # scenes (night, tunnels) with any texture still pass.
                if (arr_bgr.mean() < BLACK_THUMB_MEAN
                        and arr_bgr.std() < BLACK_THUMB_STD):
                    if stats is not None:
                        stats['black_views'] = stats.get('black_views', 0) + 1
                    return None

                return cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)

            if status == 403:
                # Two flavours of 403 from this endpoint:
                #   (a) transient — Google flags a single request for a
                #       split second; the next attempt succeeds.
                #   (b) persistent IP block — every request from this IP
                #       returns 403 until the cooldown lifts (hours).
                # Retry with backoff to absorb (a). If we exhaust retries
                # the 403 is persistent → flag so the outer downloader
                # short-circuits and the worker self-destructs instead of
                # poisoning the queue with half-extracted features.
                if stats is not None:
                    stats['http_403'] = stats.get('http_403', 0) + 1

                if attempt < retries:
                    delay = (backoff * (2 ** (attempt - 1))
                             + random.uniform(0, backoff))
                    await asyncio.sleep(delay)
                    continue

                if stats is not None:
                    stats['ip_blocked_403'] = True
                return None

            if status not in THUMB_RETRYABLE_STATUS or attempt >= retries:
                if stats is not None:
                    stats[f'http_{status}'] = stats.get(f'http_{status}', 0) + 1
                return None

            # Retryable: respect Retry-After if present, else jittered backoff.
            ra = response.headers.get('retry-after')
            try:
                delay = float(ra) if ra else backoff * (2 ** (attempt - 1))
            except ValueError:
                delay = backoff * (2 ** (attempt - 1))
            delay += random.uniform(0, backoff)
            await asyncio.sleep(delay)

        except Exception:
            if attempt < retries:
                await asyncio.sleep(
                    backoff * (2 ** (attempt - 1)) + random.uniform(0, backoff)
                )

    return None
