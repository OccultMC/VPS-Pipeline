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
        seams = []
        for i in range(3):
            d_px, d_resized, pct, err = find_optimal_overlap(
                imgs[i], imgs[i + 1], max_pct=args.max_pct
            )
            seams.append((d_px, pct, err))

        rows.append((z, widths, seams))
        s_str = "  ".join(
            f"{lbl}={d}px({pct:.2f}%)"
            for lbl, (d, pct, _) in zip(["b->l", "l->f", "f->r"], seams)
        )
        print(f"zoom {z}: w={widths}  {s_str}")

    print()
    print("=" * 90)
    print(f"{'zoom':<5} {'face widths (b/l/f/r)':<24} "
          f"{'b->l %':<8} {'l->f %':<8} {'f->r %':<8} {'avg %':<8} "
          f"{'avg px (vs back)':<18}")
    print("-" * 90)
    pcts_per_zoom = []
    for z, ws, sr in rows:
        avg_pct = sum(s[1] for s in sr) / 3
        avg_px = round(ws[0] * avg_pct / 100)
        pcts_per_zoom.append(avg_pct)
        wstr = "/".join(map(str, ws))
        print(f"{z:<5} {wstr:<24} "
              f"{sr[0][1]:<8.2f} {sr[1][1]:<8.2f} {sr[2][1]:<8.2f} "
              f"{avg_pct:<8.2f} {avg_px:<18}")
    print("-" * 90)
    if pcts_per_zoom:
        overall = sum(pcts_per_zoom) / len(pcts_per_zoom)
        print(f"\nMean optimal overlap across all zoom levels: {overall:.2f}%")
        print(f"(this is what the stitcher should default to)")


if __name__ == "__main__":
    main()
