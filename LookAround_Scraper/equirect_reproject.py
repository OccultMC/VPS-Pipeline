"""
Server-side port of lookmap.eu's LookaroundAdapter.js geometry.

For each output equirectangular pixel we figure out (yaw, pitch), find which
face covers that direction (using the lens metadata in metadata.json), and
sample that face. Adjacent overlap between sides 0..3 is removed exactly the
way the JS adapter does it: each side's right edge is shrunk to the next
side's left edge.

Top/bottom faces are intentionally not used for now (caller can flip the
flag when we figure them out).

Usage:
    python equirect_reproject.py <pano_dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import List

import numpy as np
from PIL import Image


SIDE_NAMES = ["back", "left", "front", "right"]


def _load_face(pano_dir: str, name: str) -> np.ndarray:
    path = os.path.join(pano_dir, f"{name}.jpg")
    return np.asarray(Image.open(path).convert("RGB"))


def reproject_to_equirect(pano_dir: str,
                          out_w: int | None = None,
                          out_h: int | None = None,
                          out_name: str = "equirect_reproj.jpg") -> str:
    """
    Reproject the 4 side faces in `pano_dir` to a single equirectangular
    image written to `pano_dir/out_name`. Requires `metadata.json` next to
    the face JPGs (saved by `apple_scraper.scrape_polygon`).

    `out_w` / `out_h` default to 4 × widest face × 1, paired 2:1.
    """
    with open(os.path.join(pano_dir, "metadata.json")) as f:
        meta = json.load(f)
    faces_meta = meta["faces"][:4]  # sides only

    face_imgs = [_load_face(pano_dir, n) for n in SIDE_NAMES]

    if out_w is None:
        out_w = max(im.shape[1] for im in face_imgs) * 4
    if out_h is None:
        out_h = out_w // 2

    yaws = (np.arange(out_w, dtype=np.float64) / out_w) * 2 * math.pi - math.pi
    pitches = math.pi / 2 - (np.arange(out_h, dtype=np.float64) / out_h) * math.pi
    yaw_2d, pitch_2d = np.meshgrid(yaws, pitches)

    # Per-face yaw bounds, then shrink prev's right edge to next's left edge
    # (mirrors LookaroundAdapter.createMesh's overlap-removal loop).
    #
    # streetlevel returns yaws in [-π, π], but the overlap math assumes the 4
    # side faces are listed in monotonically increasing yaw. Add 2π to any
    # face whose yaw goes backwards relative to the previous one.
    yaw_centers = [m["yaw"] for m in faces_meta]
    for i in range(1, 4):
        while yaw_centers[i] < yaw_centers[i - 1]:
            yaw_centers[i] += 2 * math.pi
    fov_s = [m["fov_s"] for m in faces_meta]
    fov_h = [m["fov_h"] for m in faces_meta]
    cy = [m["cy"] for m in faces_meta]

    eff_min = [yaw_centers[i] - fov_s[i] / 2 for i in range(4)]
    eff_max = [yaw_centers[i] + fov_s[i] / 2 for i in range(4)]
    for i in range(1, 4):
        if eff_max[i - 1] > eff_min[i]:
            eff_max[i - 1] = eff_min[i]

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for i, img in enumerate(face_imgs):
        # Angular distance from face center, normalized to [-π, π]
        dyaw = ((yaw_2d - yaw_centers[i] + math.pi) % (2 * math.pi)) - math.pi
        dyaw_min = eff_min[i] - yaw_centers[i]
        dyaw_max = eff_max[i] - yaw_centers[i]
        in_yaw = (dyaw >= dyaw_min) & (dyaw <= dyaw_max)

        pitch_min = -fov_h[i] / 2 + cy[i]
        pitch_max = fov_h[i] / 2 + cy[i]
        in_pitch = (pitch_2d >= pitch_min) & (pitch_2d <= pitch_max)

        in_face = in_yaw & in_pitch
        if not in_face.any():
            continue

        # UV inside the face (matching the SphereGeometry / texture mapping)
        u = (dyaw - dyaw_min) / fov_s[i]            # 0..1, left→right
        v = (pitch_max - pitch_2d) / fov_h[i]       # 0..1, top→bottom

        u_pix = np.clip((u * img.shape[1]).astype(np.int32), 0, img.shape[1] - 1)
        v_pix = np.clip((v * img.shape[0]).astype(np.int32), 0, img.shape[0] - 1)

        sampled = img[v_pix, u_pix]
        out = np.where(in_face[..., None], sampled, out)

    out_path = os.path.join(pano_dir, out_name)
    Image.fromarray(out).save(out_path, format="JPEG", quality=92)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pano_dir")
    ap.add_argument("--out-w", type=int, default=None)
    ap.add_argument("--out-h", type=int, default=None)
    ap.add_argument("--out-name", default="equirect_reproj.jpg")
    args = ap.parse_args()
    out = reproject_to_equirect(args.pano_dir, args.out_w, args.out_h, args.out_name)
    img = Image.open(out)
    print(f"wrote {out}  ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
