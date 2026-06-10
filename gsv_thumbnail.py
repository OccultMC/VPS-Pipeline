"""
Google Street View server-rendered thumbnail fetcher.

One function: fetch_thumbnail_view(). Used by pipeline.py to pull
each pre-projected perspective view directly from Google's thumbnail
endpoint (no tile fetch, no client-side stitching, no equirect reproject).

Replaces the old gsvpd/ tile pipeline entirely.
"""
import asyncio
import random
import threading
import time
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

# ── Throttled exception logging ───────────────────────────────────────────
_exc_log_lock = threading.Lock()
_exc_log_last = [0.0]


def _log_fetch_exc(exc: BaseException):
    """Print fetch exception types at most once per ~30s so network
    pathologies (DNS failures, resets, TLS errors) are visible without
    generating gigabytes of duplicate log lines."""
    now = time.time()
    with _exc_log_lock:
        if now - _exc_log_last[0] < 30.0:
            return
        _exc_log_last[0] = now
    print(f"[THUMB] fetch exception {type(exc).__name__}: {str(exc)[:120]} "
          f"(throttled — at most one line per 30s)", flush=True)


def _decode_and_check(body: bytes, out_size: Optional[int] = None):
    """Decode JPEG bytes → RGB array plus a status tag.

    Runs in a thread-pool executor: cv2.imdecode + the black-placeholder
    mean/std check done inline on the asyncio event loop across hundreds of
    concurrent fetches visibly starves the loop and stalls in-flight requests.

    out_size: when set and the decoded image is larger, bicubic-downscale to
    out_size × out_size. The endpoint's JPEGs carry heavy compression
    artifacts at small sizes — fetching at 2× and supersampling down
    averages the blocking/ringing away. The array stays raw RGB end-to-end
    (no re-encode, PNG or otherwise) on its way to the GPU.

    Returns ('ok', rgb) | ('decode_fail', None) | ('black', None).
    """
    arr_bgr = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if arr_bgr is None:
        return 'decode_fail', None
    # "No imagery here" placeholder: solid black JPEG. Detected via low
    # mean AND low std so legitimate dark scenes (night, tunnels) with
    # any texture still pass. Checked at native resolution, pre-resize.
    if arr_bgr.mean() < BLACK_THUMB_MEAN and arr_bgr.std() < BLACK_THUMB_STD:
        return 'black', None
    if out_size and (arr_bgr.shape[0] != out_size or arr_bgr.shape[1] != out_size):
        arr_bgr = cv2.resize(arr_bgr, (out_size, out_size),
                             interpolation=cv2.INTER_CUBIC)
    return 'ok', cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)


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
    out_size: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Fetch one server-rendered perspective view from Google's thumbnail endpoint.

    out_size: bicubic-downscale the decoded image to out_size × out_size
    (fetch at w×h = 2× for a supersampled, artifact-free result).

    Returns numpy uint8 HWC RGB array, or None on any failure
    (4xx, persistent 5xx, decode error, black/garbage placeholder).

    Failure handling:
      * 200 + valid decode + non-black     -> return RGB array
      * 200 + decode fail or near-black    -> None (count black via stats)
      * 403                                -> retry with backoff (Google
                                              occasionally flags a single
                                              request for a split second);
                                              if every retry returns 403 the
                                              pano is counted as suspect
                                              (stats['suspect_403']) and
                                              dropped. The global
                                              stats['ip_blocked_403'] flag is
                                              ONLY set by the known-good-URL
                                              probe machinery in pipeline.py
                                              — a single pano 403ing forever
                                              must not poison the whole chunk.
      * 5xx / 429                          -> retry with Retry-After (capped
                                              at 30s) + jittered exponential
                                              backoff
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
                # Decode + black-check off the event loop — done inline it
                # blocks every in-flight request for the imdecode duration.
                tag, rgb = await asyncio.get_running_loop().run_in_executor(
                    None, _decode_and_check, data, out_size
                )
                if tag == 'decode_fail':
                    if stats is not None:
                        stats['decode_fail'] = stats.get('decode_fail', 0) + 1
                    return None
                if tag == 'black':
                    if stats is not None:
                        stats['black_views'] = stats.get('black_views', 0) + 1
                    return None
                return rgb

            if status == 403:
                # Two flavours of 403 from this endpoint:
                #   (a) transient — Google flags a single request for a
                #       split second; the next attempt succeeds.
                #   (b) persistent IP block — every request from this IP
                #       returns 403 until the cooldown lifts (hours).
                # Retry with backoff to absorb (a). If we exhaust retries,
                # count the pano as suspect and drop it — do NOT set the
                # global ip_blocked_403 flag here. A single pano can 403
                # forever (pulled imagery, region lock) while the IP is
                # perfectly fine; flipping the global flag from one pano
                # poison-pills the whole chunk. Only the known-good-URL
                # probe machinery in pipeline.py decides IP-level blocks.
                if stats is not None:
                    stats['http_403'] = stats.get('http_403', 0) + 1

                if attempt < retries:
                    delay = (backoff * (2 ** (attempt - 1))
                             + random.uniform(0, backoff))
                    await asyncio.sleep(delay)
                    continue

                if stats is not None:
                    stats['suspect_403'] = stats.get('suspect_403', 0) + 1
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
            # Cap honoured Retry-After — Google occasionally sends absurd
            # values that would park a downloader slot for minutes.
            delay = min(delay, 30.0)
            delay += random.uniform(0, backoff)
            await asyncio.sleep(delay)

        except Exception as e:
            if stats is not None:
                stats['fetch_exc'] = stats.get('fetch_exc', 0) + 1
            _log_fetch_exc(e)
            if attempt < retries:
                await asyncio.sleep(
                    backoff * (2 ** (attempt - 1)) + random.uniform(0, backoff)
                )

    return None
