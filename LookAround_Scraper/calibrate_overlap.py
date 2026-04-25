"""
Find the optimal stitch overlap for each Look Around zoom level.

For a given pano (selected by lat/lon, optionally pano-id+build-id), this
downloads back/left/front/right at every requested zoom level, then runs a
brute-force sliding-window match on each of the 3 seams: it picks the
overlap d (in pixels) that minimizes mean-squared pixel error between the
last d columns of the left face and the first d columns of the right face.

The result is reported in pixels and as a percentage of the left face's
width — the percentage is what the stitcher actually uses, and it should
be approximately constant across zoom levels.

NOTE: Apple's side faces are independent lens projections, not pure cube
faces, so adjacent faces don't share identical content. The matched
overlap is therefore approximate; treat the percentage column as a
debugging signal, not ground truth.

Usage:
    python calibrate_overlap.py                  # uses default SF pano
    python calibrate_overlap.py --lat 37.7749 --lon -122.4194
    python calibrate_overlap.py --zooms 2,3,4    # subset of zooms
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
from PIL import Image

from streetlevel.lookaround import Face, get_coverage_tile_by_latlon, get_panorama_face
from streetlevel.lookaround.auth import Authenticator

from apple_scraper import _decode_heic_to_jpg, FACE_NAMES


SEAM_LABELS = ["back->left", "left->front", "front->right"]
RESIZE_W = 256  # downscale faces to this width for matching (keeps it fast)


def angular_overlap_from_metadata(pano, left_idx: int, right_idx: int) -> float:
    """
    Return the angular overlap (radians) between two adjacent faces, using
    the camera_metadata yaw + lens_projection.fov_s. The overlap is
    (left_yaw + fov_left/2) - (right_yaw - fov_right/2), modulo 2π.
    """
    import math
    cm_l = pano.camera_metadata[left_idx]
    cm_r = pano.camera_metadata[right_idx]
    yaw_l = cm_l.position.yaw
    yaw_r = cm_r.position.yaw
    fov_l = cm_l.lens_projection.fov_s
    fov_r = cm_r.lens_projection.fov_s
    left_edge = yaw_l + fov_l / 2.0       # right edge of LEFT face (in yaw)
    right_edge = yaw_r - fov_r / 2.0      # left edge of RIGHT face (in yaw)
    delta = left_edge - right_edge
    # normalize to [-π, π]
    while delta > math.pi:
        delta -= 2 * math.pi
    while delta < -math.pi:
        delta += 2 * math.pi
    return delta


def overlap_px_from_geometry(pano, left_idx: int, right_idx: int,
                             left_img: Image.Image) -> Tuple[int, float]:
    """
    Compute the seam overlap in *pixels of the LEFT face*, using the lens
    metadata. Returns (px, pct_of_left_width).
    """
    cm_l = pano.camera_metadata[left_idx]
    fov_l = cm_l.lens_projection.fov_s
    overlap_rad = angular_overlap_from_metadata(pano, left_idx, right_idx)
    if overlap_rad <= 0:
        return (0, 0.0)
    px = round(overlap_rad / fov_l * left_img.width)
    pct = overlap_rad / fov_l * 100.0
    return (px, pct)


def find_optimal_overlap(left_img: Image.Image, right_img: Image.Image,
                         max_pct: float = 30.0) -> Tuple[int, int, float, float]:
    """
    Return (d_orig_px, d_resized_px, pct_of_left_w, mse_at_best).

    Operates on a downscaled copy (width=RESIZE_W) for speed; the percentage
    is invariant to the scale, so it's accurate at full resolution too.
    """
    w_l_orig = left_img.width
    if w_l_orig == 0:
        return (0, 0, 0.0, float("inf"))

    # Downscale both to common scale
    h_l = round(left_img.height * RESIZE_W / left_img.width)
    L = left_img.resize((RESIZE_W, h_l)).convert("L")
    h_r = round(right_img.height * RESIZE_W / right_img.width)
    R = right_img.resize((RESIZE_W, h_r)).convert("L")
    h = min(h_l, h_r)
    L_arr = np.asarray(L, dtype=np.int16)[:h]
    R_arr = np.asarray(R, dtype=np.int16)[:h]

    max_d = max(1, int(RESIZE_W * max_pct / 100.0))
    best_d, best_err = 1, float("inf")
    for d in range(1, max_d + 1):
        a = L_arr[:, -d:]
        b = R_arr[:, :d]
        err = float(np.mean((a - b) ** 2))
        if err < best_err:
            best_err = err
            best_d = d

    pct = best_d / RESIZE_W * 100.0
    d_orig_px = round(w_l_orig * pct / 100.0)
    return (d_orig_px, best_d, pct, best_err)


def stitch_with_per_seam_overlaps(imgs: List[Image.Image],
                                  seam_overlaps_px: List[int],
                                  out_path: str) -> None:
    """Stitch left-to-right using one pixel overlap per seam (n-1 values)."""
    n = len(imgs)
    if len(seam_overlaps_px) != n - 1:
        raise ValueError("need n-1 seam overlaps")
    h = min(im.height for im in imgs)
    out_w = sum(im.width for im in imgs) - sum(seam_overlaps_px)
    canvas = Image.new("RGB", (out_w, h))
    x = 0
    for i, im in enumerate(imgs):
        canvas.paste(im.crop((0, 0, im.width, h)), (x, 0))
        if i < n - 1:
            x += im.width - seam_overlaps_px[i]
        else:
            x += im.width
    canvas.save(out_path, format="JPEG", quality=92)


def get_or_download_face(pano, face_idx: int, zoom: int, auth: Authenticator,
                         out_path: str) -> str:
    if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
        return out_path
    heic = get_panorama_face(pano, Face(face_idx), zoom, auth)
    _decode_heic_to_jpg(heic, out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=37.77508)
    ap.add_argument("--lon", type=float, default=-122.41912)
    ap.add_argument("--zooms", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--out-root", default="downloads/_calibration")
    ap.add_argument("--max-pct", type=float, default=30.0)
    args = ap.parse_args()

    panos = get_coverage_tile_by_latlon(args.lat, args.lon).panos
    if not panos:
        raise SystemExit(f"no panos at ({args.lat}, {args.lon})")
    pano = panos[0]
    print(f"Using pano: {pano.id} / build {pano.build_id} @ "
          f"({pano.lat:.5f}, {pano.lon:.5f})")
    print(f"Coverage type: {pano.coverage_type}, captured: {pano.date}")
    print(f"Resize for matching: {RESIZE_W}px wide")
    print()

    auth = Authenticator()
    out_root = os.path.join(args.out_root, str(pano.id))
    rows = []
    for z_str in args.zooms.split(","):
        z = int(z_str)
        z_dir = os.path.join(out_root, f"zoom_{z}")
        os.makedirs(z_dir, exist_ok=True)
        face_paths: List[str] = []
        try:
            for i, name in enumerate(FACE_NAMES[:4]):
                p = os.path.join(z_dir, f"{name}.jpg")
                get_or_download_face(pano, i, z, auth, p)
                face_paths.append(p)
        except Exception as e:
            print(f"zoom {z}: download failed — {e}")
            continue

        imgs = [Image.open(p) for p in face_paths]
        widths = [im.width for im in imgs]

        # GEOMETRIC overlap (truth, from lens metadata)
        face_indices = [0, 1, 2, 3]  # back, left, front, right
        geom = []
        for i in range(3):
            px, pct = overlap_px_from_geometry(
                pano, face_indices[i], face_indices[i + 1], imgs[i]
            )
            geom.append((px, pct))

        # MATCHED overlap (pixel MSE — kept for reference)
        match = []
        for i in range(3):
            d_px, d_resized, pct, err = find_optimal_overlap(
                imgs[i], imgs[i + 1], max_pct=args.max_pct
            )
            match.append((d_px, pct, err))

        rows.append((z, widths, geom, match))
        g_str = "  ".join(
            f"{lbl}={px}px({pct:.2f}%)"
            for lbl, (px, pct) in zip(["b->l", "l->f", "f->r"], geom)
        )
        m_str = "  ".join(
            f"{lbl}={px}px({pct:.2f}%)"
            for lbl, (px, pct, _) in zip(["b->l", "l->f", "f->r"], match)
        )
        print(f"zoom {z}: w={widths}")
        print(f"   geom : {g_str}")
        print(f"   match: {m_str}")

        # Stitch using GEOMETRIC overlaps
        seam_px = [g[0] for g in geom]
        stitched_path = os.path.join(z_dir, "stitched.jpg")
        try:
            stitch_with_per_seam_overlaps(imgs, seam_px, stitched_path)
            sw, sh = Image.open(stitched_path).size
            print(f"   stitched -> {stitched_path}  ({sw}x{sh})")
        except Exception as e:
            print(f"   stitch failed: {e}")

    print()
    print("=" * 100)
    print("GEOMETRIC overlaps (from pano.camera_metadata yaw + fov_s)")
    print(f"{'zoom':<5} {'widths (b/l/f/r)':<24} "
          f"{'b->l':<14} {'l->f':<14} {'f->r':<14}")
    print("-" * 100)
    for z, ws, geom, _ in rows:
        wstr = "/".join(map(str, ws))
        cells = [f"{px}px ({pct:.2f}%)" for (px, pct) in geom]
        print(f"{z:<5} {wstr:<24} {cells[0]:<14} {cells[1]:<14} {cells[2]:<14}")
    print("-" * 100)
    print()
    print("MATCHED overlaps (pixel MSE — naive, kept for comparison)")
    print(f"{'zoom':<5} {'widths (b/l/f/r)':<24} "
          f"{'b->l':<14} {'l->f':<14} {'f->r':<14}")
    print("-" * 100)
    for z, ws, _, match in rows:
        wstr = "/".join(map(str, ws))
        cells = [f"{px}px ({pct:.2f}%)" for (px, pct, _) in match]
        print(f"{z:<5} {wstr:<24} {cells[0]:<14} {cells[1]:<14} {cells[2]:<14}")


if __name__ == "__main__":
    main()
