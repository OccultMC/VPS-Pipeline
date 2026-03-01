"""Directional view extraction from equirectangular panoramas.

Hardcoded for VPR Pipeline: 8 views, 322x322px, 60deg FOV, cubic interpolation, no antialias.
"""

import numpy as np
import cv2
from typing import List, Tuple

# Pipeline constants
NUM_VIEWS = 8
OUTPUT_SIZE = 322
FOV_DEGREES = 60.0

# Remap matrix cache — shared across calls, never evicted.
# 8 entries for 8 yaw directions * ~1 pano size = small footprint.
_remap_cache: dict = {}


def extract_views(panorama: np.ndarray) -> List[np.ndarray]:
    """Extract 8 directional views from an equirectangular panorama.

    Args:
        panorama: BGR equirectangular panorama (H, W, 3).

    Returns:
        List of 8 BGR images, each 322x322, evenly spaced around the horizon.
    """
    if panorama is None or panorama.size == 0:
        return []

    views = []
    angle_step = 360.0 / NUM_VIEWS
    fov_rad = np.radians(FOV_DEGREES)

    for i in range(NUM_VIEWS):
        yaw_rad = np.radians(i * angle_step)
        map_x, map_y = _get_remap_matrices(
            panorama.shape[1], panorama.shape[0], OUTPUT_SIZE, yaw_rad, fov_rad
        )
        view = cv2.remap(panorama, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        views.append(view)

    return views


def _get_remap_matrices(
    pano_w: int, pano_h: int, out_size: int, yaw: float, fov_rad: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Get or create cached remap matrices for equirect -> rectilinear projection."""
    cache_key = (pano_w, pano_h, out_size, round(yaw, 6), round(fov_rad, 6))
    cached = _remap_cache.get(cache_key)
    if cached is not None:
        return cached

    hfov_rad = 2.0 * np.arctan(np.tan(fov_rad / 2.0))

    x_coords, y_coords = np.meshgrid(
        np.arange(out_size, dtype=np.float32),
        np.arange(out_size, dtype=np.float32),
    )

    nx = (2.0 * x_coords / out_size) - 1.0
    ny = 1.0 - (2.0 * y_coords / out_size)

    tan_hfov_half = np.tan(hfov_rad / 2.0)
    tan_fov_half = np.tan(fov_rad / 2.0)

    ray_x = tan_hfov_half * nx
    ray_y = tan_fov_half * ny
    ray_z = np.ones_like(ray_x)

    ray_len = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2)
    ray_x /= ray_len
    ray_y /= ray_len
    ray_z /= ray_len

    # Yaw rotation (around Y axis), pitch=0, roll=0
    c, s = np.cos(yaw), np.sin(yaw)
    rx = ray_x * c + ray_z * s
    rz = -ray_x * s + ray_z * c
    ray_x, ray_z = rx, rz

    theta = np.arctan2(ray_x, ray_z)
    phi = np.arcsin(np.clip(ray_y, -1.0, 1.0))

    map_x = (theta / (2.0 * np.pi) + 0.5) * pano_w
    map_y = (0.5 - phi / np.pi) * pano_h

    result = (map_x.astype(np.float32), map_y.astype(np.float32))
    _remap_cache[cache_key] = result
    return result
