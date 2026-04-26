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

    # 3. Render 6 perspective views
    equi_arr = np.asarray(equi_img)
    views = [
        ("front",  0,    0),
        ("right",  90,   0),
        ("back",   180,  0),
        ("left",   270,  0),
        ("up",     0,    90),
        ("down",   0,   -90),
    ]
    for name, yaw, pitch in views:
        out = render_perspective(
            equi_arr, yaw_deg=yaw, pitch_deg=pitch,
            fov_deg=args.view_fov,
            out_w=args.view_size, out_h=args.view_size,
        )
        out_path = os.path.join(args.pano_dir, f"view_{name}.jpg")
        Image.fromarray(out).save(out_path, format="JPEG", quality=92)
        print(f"  view {name:<6} (yaw={yaw}, pitch={pitch}) -> {out_path}")


if __name__ == "__main__":
    main()
