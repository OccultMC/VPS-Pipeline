"""
Convert a Look Around stitched 4-face strip into an equirectangular pano,
then render 6 perspective views from it.

Pipeline:
  1. Read stitched.jpg + the 4 source face JPGs.
  2. Trim the wrap-around overlap from `right`'s trailing edge (same
     overlap value as the internal seams — derived from the lens metadata
     stored alongside the stitched output).
  3. Pad the strip to a 2:1 aspect ratio with black bars top + bottom —
     that IS the equirectangular panorama.
  4. Render 6 perspective views from the equirect: yaw 0/90/180/270 with
     pitch=0, plus pure up (pitch=+90) and down (pitch=-90).

All outputs land next to the input stitched.jpg.

Usage:
    python cylindrical_project.py <pano_dir>
    python cylindrical_project.py downloads/_calibration/<id>/zoom_6
    python cylindrical_project.py <pano_dir> --view-fov 90 --view-size 768
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Tuple

import numpy as np
from PIL import Image


# Apple's standard CAR-rig FOVs (radians) — back/front wide, left/right narrow
FOV_S_BACK = math.radians(123.7)
FOV_S_LEFT = math.radians(67.5)
FOV_S_FRONT = math.radians(123.7)
FOV_S_RIGHT = math.radians(67.5)
SEAM_ANGLE = math.radians(5.6)  # overlap between every adjacent pair


def trim_wrap_overlap(strip: Image.Image, right_face_width: int) -> Image.Image:
    """
    Crop the rightmost N pixels of `strip` where N is the wrap-around overlap
    expressed in pixels of the RIGHT face. Same per-pixel value as the
    matching internal seam: SEAM_ANGLE / FOV_S_RIGHT * right_face_width.
    """
    overlap_px = round(SEAM_ANGLE / FOV_S_RIGHT * right_face_width)
    if overlap_px <= 0 or overlap_px >= strip.width:
        return strip
    return strip.crop((0, 0, strip.width - overlap_px, strip.height))


def pad_to_2to1(img: Image.Image) -> Image.Image:
    """Pad img with black top + bottom bars until aspect = 2:1 (W = 2H)."""
    w, h = img.width, img.height
    target_h = w // 2
    if h >= target_h:
        return img
    canvas = Image.new("RGB", (w, target_h), (0, 0, 0))
    pad_top = (target_h - h) // 2
    canvas.paste(img, (0, pad_top))
    return canvas


def _sample(equi: np.ndarray, eq_x: np.ndarray, eq_y: np.ndarray) -> np.ndarray:
    """Nearest-neighbor lookup with horizontal wrap."""
    H, W, _ = equi.shape
    xi = np.mod(eq_x.astype(np.int32), W)
    yi = np.clip(eq_y.astype(np.int32), 0, H - 1)
    return equi[yi, xi]


def render_cylindrical(equi: np.ndarray, out_w: int, out_h: int,
                       fov_h_deg: float = 360.0) -> np.ndarray:
    """Cylindrical projection: full 360° wrap horizontally, perfect verticals."""
    H, W, _ = equi.shape
    fov_h = math.radians(fov_h_deg)
    yaw = (np.arange(out_w, dtype=np.float64) / out_w - 0.5) * fov_h
    focal = out_w / fov_h  # pixels per radian
    # v=0 at TOP of output → +pitch (sky)
    v_centered = out_h / 2.0 - np.arange(out_h, dtype=np.float64)
    pitch = np.arctan(v_centered / focal)
    uu, _ = np.meshgrid(yaw, np.zeros(out_h))
    _, vv = np.meshgrid(np.zeros(out_w), pitch)
    eq_x = (uu / (2 * math.pi) + 0.5) * W
    eq_y = (0.5 - vv / math.pi) * H
    return _sample(equi, eq_x, eq_y)


def render_mercator(equi: np.ndarray, out_w: int, out_h: int,
                    fov_h_deg: float = 360.0) -> np.ndarray:
    """
    Mercator: full 360° wrap, vertical stretches near poles. Same horizontal
    geometry as cylindrical, different vertical mapping.
    """
    H, W, _ = equi.shape
    fov_h = math.radians(fov_h_deg)
    yaw = (np.arange(out_w, dtype=np.float64) / out_w - 0.5) * fov_h
    scale = out_w / (2 * math.pi)
    # v=0 at TOP → +pitch (sky)
    v_centered = (out_h / 2.0 - np.arange(out_h, dtype=np.float64)) / scale
    pitch = 2 * np.arctan(np.exp(v_centered)) - math.pi / 2
    uu, _ = np.meshgrid(yaw, np.zeros(out_h))
    _, vv = np.meshgrid(np.zeros(out_w), pitch)
    eq_x = (uu / (2 * math.pi) + 0.5) * W
    eq_y = (0.5 - vv / math.pi) * H
    return _sample(equi, eq_x, eq_y)


def render_stereographic(equi: np.ndarray, out_size: int, pole: str = "down",
                         zoom: float = 0.33) -> np.ndarray:
    """
    Stereographic projection. pole='down' is a "little planet"; pole='up'
    is a "tunnel". `zoom` controls how much of the sphere fits inside the
    disk (smaller zoom = more sphere included).
    """
    H, W, _ = equi.shape
    cx = cy = out_size / 2.0
    u_c = np.arange(out_size, dtype=np.float64) - cx
    v_c = np.arange(out_size, dtype=np.float64) - cy
    uu, vv = np.meshgrid(u_c, v_c)
    r = np.sqrt(uu * uu + vv * vv)
    f = cx * zoom
    theta_pole = 2 * np.arctan(r / (2 * f))
    az = np.arctan2(vv, uu)
    if pole == "down":
        pitch = -math.pi / 2 + theta_pole
        yaw = az
    else:  # up
        pitch = math.pi / 2 - theta_pole
        yaw = az + math.pi
    pitch = np.clip(pitch, -math.pi / 2, math.pi / 2)
    eq_x = (yaw / (2 * math.pi) + 0.5) * W
    eq_y = (0.5 - pitch / math.pi) * H
    return _sample(equi, eq_x, eq_y)


def render_fisheye(equi: np.ndarray, out_size: int,
                   fov_deg: float = 180.0, pole: str = "front") -> np.ndarray:
    """
    Equidistant fisheye (circular). Renders content inside a disk; outside
    the disk is black. `pole='front'` looks straight ahead at yaw=0,pitch=0.
    """
    H, W, _ = equi.shape
    cx = cy = out_size / 2.0
    u_c = np.arange(out_size, dtype=np.float64) - cx
    v_c = np.arange(out_size, dtype=np.float64) - cy
    uu, vv = np.meshgrid(u_c, v_c)
    r = np.sqrt(uu * uu + vv * vv)
    fov = math.radians(fov_deg)
    theta = r / cx * (fov / 2)        # angle from optical axis
    az = np.arctan2(vv, uu)
    # Optical axis = +Z (forward). Theta is angle from +Z.
    rx = np.sin(theta) * np.cos(az)
    ry = np.sin(theta) * np.sin(az)
    rz = np.cos(theta)
    if pole == "down":
        ry, rz = -rz, ry
    elif pole == "up":
        ry, rz = rz, -ry
    yaw = np.arctan2(rx, rz)
    pitch = np.arcsin(np.clip(-ry, -1, 1))
    eq_x = (yaw / (2 * math.pi) + 0.5) * W
    eq_y = (0.5 - pitch / math.pi) * H
    out = _sample(equi, eq_x, eq_y)
    # Mask outside the disk
    mask = (r > cx).astype(np.uint8)
    out = out * (1 - mask[:, :, None])
    return out


def render_perspective(equi: np.ndarray, yaw_deg: float, pitch_deg: float,
                       fov_deg: float, out_w: int, out_h: int) -> np.ndarray:
    """
    Sample a perspective (rectilinear) view from an equirectangular image.

    equi:        (H, W, 3) uint8 array. The pano covers yaw [-180, 180]
                 left→right and pitch [+90, -90] top→bottom.
    yaw_deg:     view center yaw (0 = +Z forward)
    pitch_deg:   view center pitch (0 = horizon, +90 = straight up)
    fov_deg:     horizontal field of view of the rendered view
    out_w/out_h: output resolution
    """
    H, W, _ = equi.shape
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    fov = math.radians(fov_deg)
    f = (out_w / 2.0) / math.tan(fov / 2.0)

    # Build per-pixel ray directions in camera space (camera looks +Z, up=+Y)
    u = np.arange(out_w, dtype=np.float64) - out_w / 2.0
    v = np.arange(out_h, dtype=np.float64) - out_h / 2.0
    uu, vv = np.meshgrid(u, v)
    rx = uu
    ry = -vv
    rz = np.full_like(uu, f)
    norm = np.sqrt(rx * rx + ry * ry + rz * rz)
    rx /= norm; ry /= norm; rz /= norm

    # Rotate camera: pitch (around X), then yaw (around Y).
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    ry2 = ry * cp - rz * sp
    rz2 = ry * sp + rz * cp
    rx3 = rx * cy + rz2 * sy
    rz3 = -rx * sy + rz2 * cy
    ry3 = ry2

    # Cartesian → spherical (yaw, pitch)
    yaw_out = np.arctan2(rx3, rz3)            # [-π, π]
    pitch_out = np.arcsin(np.clip(ry3, -1, 1))  # [-π/2, π/2]

    # Spherical → equirect pixel
    eq_x = (yaw_out / (2 * math.pi) + 0.5) * W
    eq_y = (0.5 - pitch_out / math.pi) * H

    # Nearest-neighbor sampling (cheap, fine for debug)
    eq_xi = np.clip(eq_x.astype(np.int32), 0, W - 1)
    eq_yi = np.clip(eq_y.astype(np.int32), 0, H - 1)
    return equi[eq_yi, eq_xi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pano_dir", help="folder with stitched.jpg + face jpgs")
    ap.add_argument("--strip-name", default="stitched.jpg")
    ap.add_argument("--equi-name", default="equirect.jpg")
    ap.add_argument("--view-fov", type=float, default=90.0)
    ap.add_argument("--view-size", type=int, default=768)
    args = ap.parse_args()

    strip_path = os.path.join(args.pano_dir, args.strip_name)
    if not os.path.exists(strip_path):
        raise SystemExit(f"missing strip: {strip_path}")
    right_path = os.path.join(args.pano_dir, "right.jpg")
    if not os.path.exists(right_path):
        raise SystemExit(f"missing right.jpg in {args.pano_dir}")

    strip = Image.open(strip_path).convert("RGB")
    right_w = Image.open(right_path).width

    # 1. Trim wrap-around overlap
    trimmed = trim_wrap_overlap(strip, right_w)
    print(f"strip:   {strip.width}x{strip.height}")
    print(f"trimmed: {trimmed.width}x{trimmed.height}  "
          f"(removed {strip.width - trimmed.width} px from right edge)")

    # 2. Pad to 2:1 → equirectangular pano
    equi_img = pad_to_2to1(trimmed)
    equi_path = os.path.join(args.pano_dir, args.equi_name)
    equi_img.save(equi_path, format="JPEG", quality=92)
    print(f"equirect: {equi_img.width}x{equi_img.height}  -> {equi_path}")

    equi_arr = np.asarray(equi_img)
    eq_w, eq_h = equi_img.size

    # 3a. 6 perspective (rectilinear) cube views
    cube_views = [
        ("front",  0,    0),
        ("right",  90,   0),
        ("back",   180,  0),
        ("left",   270,  0),
        ("up",     0,    90),
        ("down",   0,   -90),
    ]
    for name, yaw, pitch in cube_views:
        out = render_perspective(
            equi_arr, yaw_deg=yaw, pitch_deg=pitch,
            fov_deg=args.view_fov,
            out_w=args.view_size, out_h=args.view_size,
        )
        out_path = os.path.join(args.pano_dir, f"view_{name}.jpg")
        Image.fromarray(out).save(out_path, format="JPEG", quality=92)
        print(f"  perspective {name:<6} (yaw={yaw}, pitch={pitch}) -> {out_path}")

    # 3b. Whole-pano renders in different projections
    proj_w, proj_h = eq_w, eq_h  # match equirect dims for fair comparison

    out = render_cylindrical(equi_arr, proj_w, proj_h)
    Image.fromarray(out).save(os.path.join(args.pano_dir, "proj_cylindrical.jpg"),
                              format="JPEG", quality=92)
    print(f"  proj cylindrical    -> proj_cylindrical.jpg ({proj_w}x{proj_h})")

    out = render_mercator(equi_arr, proj_w, proj_h)
    Image.fromarray(out).save(os.path.join(args.pano_dir, "proj_mercator.jpg"),
                              format="JPEG", quality=92)
    print(f"  proj mercator       -> proj_mercator.jpg   ({proj_w}x{proj_h})")

    sphere_size = min(proj_w, proj_h * 2)
    out = render_stereographic(equi_arr, sphere_size, pole="down", zoom=0.33)
    Image.fromarray(out).save(os.path.join(args.pano_dir, "proj_littleplanet.jpg"),
                              format="JPEG", quality=92)
    print(f"  proj little planet  -> proj_littleplanet.jpg ({sphere_size}x{sphere_size})")

    out = render_stereographic(equi_arr, sphere_size, pole="up", zoom=0.33)
    Image.fromarray(out).save(os.path.join(args.pano_dir, "proj_tunnel.jpg"),
                              format="JPEG", quality=92)
    print(f"  proj tunnel         -> proj_tunnel.jpg     ({sphere_size}x{sphere_size})")

    out = render_fisheye(equi_arr, sphere_size, fov_deg=180, pole="front")
    Image.fromarray(out).save(os.path.join(args.pano_dir, "proj_fisheye.jpg"),
                              format="JPEG", quality=92)
    print(f"  proj fisheye 180    -> proj_fisheye.jpg    ({sphere_size}x{sphere_size})")


if __name__ == "__main__":
    main()
