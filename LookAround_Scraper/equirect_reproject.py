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
ALL_NAMES = ["back", "left", "front", "right", "top", "bottom"]


def _load_face(pano_dir: str, name: str) -> np.ndarray:
    path = os.path.join(pano_dir, f"{name}.jpg")
    return np.asarray(Image.open(path).convert("RGB"))


def _render_top_or_bottom(face_idx, m, img, rx_w, ry_w, rz_w, out, out_h, out_w):
    """
    Render top/bottom face into `out`. Same approach as
    LookaroundAdapter.createMesh: the SphereGeometry is placed in its default
    equator position using (yaw, fov_s, fov_h, cy) and then rotated by
    `rotateX(-pitch)` and `rotateZ(-roll)` to bring it to the pole. We do
    the inverse: take each output pixel's world direction and rotate it back
    into the default frame, then sample the face by phi/theta.
    """
    face_yaw = m["yaw"]
    face_pitch = m["pitch"]
    face_roll = m["roll"]
    fov_s = m["fov_s"]
    fov_h = m["fov_h"]
    cy = m["cy"]

    cp, sp = math.cos(face_pitch), math.sin(face_pitch)
    cr, sr = math.cos(face_roll), math.sin(face_roll)

    # Forward chain on the geometry (in lookmap.eu's adapter):
    #     v_world = R_z(-roll) . R_x(-pitch) . S . v_default
    # where S = diag(-1, 1, 1) comes from the .scale(-1, 1, 1) the JS calls
    # on every face's SphereGeometry (so it can be viewed from inside).
    # Inverse for our backward sampling:
    #     v_default = S . R_x(+pitch) . R_z(+roll) . v_world
    rx1 = cr * rx_w - sr * ry_w
    ry1 = sr * rx_w + cr * ry_w
    rz1 = rz_w
    rx2 = rx1
    ry2 = cp * ry1 - sp * rz1
    rz2 = sp * ry1 + cp * rz1
    rx2 = -rx2  # S = diag(-1, 1, 1)

    # Three.js SphereGeometry parameterization:
    #   x = -cos(phi) sin(theta), y = cos(theta), z = sin(phi) sin(theta)
    # → phi = atan2(z, -x), theta = acos(y)
    phi = np.arctan2(rz2, -rx2)
    theta = np.arccos(np.clip(ry2, -1.0, 1.0))

    phi_start = face_yaw - fov_s / 2 - math.pi / 2
    theta_start = math.pi / 2 - fov_h / 2 - cy

    phi_norm = (phi - phi_start) % (2 * math.pi)  # [0, 2π)
    in_phi = phi_norm < fov_s
    in_theta = (theta >= theta_start) & (theta <= theta_start + fov_h)
    in_face = in_phi & in_theta
    if not in_face.any():
        return out

    u = phi_norm / fov_s
    v = (theta - theta_start) / fov_h
    u_pix = np.clip((u * img.shape[1]).astype(np.int32), 0, img.shape[1] - 1)
    v_pix = np.clip((v * img.shape[0]).astype(np.int32), 0, img.shape[0] - 1)
    sampled = img[v_pix, u_pix]
    return np.where(in_face[..., None], sampled, out)


def reproject_to_equirect(pano_dir: str,
                          out_w: int | None = None,
                          out_h: int | None = None,
                          out_name: str = "equirect_reproj.jpg",
                          include_top_bottom: bool = True,
                          max_out_w: int = 4096) -> str:
    """
    Reproject the side (and optionally top/bottom) faces in `pano_dir` to a
    single equirectangular image written to `pano_dir/out_name`. Requires
    `metadata.json` next to the face JPGs.

    Top and bottom are rendered first, then sides on top — matching
    lookmap.eu's behaviour where sides win in any side/cap overlap.
    """
    with open(os.path.join(pano_dir, "metadata.json")) as f:
        meta = json.load(f)

    faces_meta_sides = meta["faces"][:4]
    face_imgs_sides = [_load_face(pano_dir, n) for n in SIDE_NAMES]

    cap_imgs = []
    cap_meta = []
    if include_top_bottom and len(meta["faces"]) >= 6:
        for idx, name in [(4, "top"), (5, "bottom")]:
            face_path = os.path.join(pano_dir, f"{name}.jpg")
            if os.path.exists(face_path):
                cap_imgs.append(_load_face(pano_dir, name))
                cap_meta.append((idx, meta["faces"][idx]))

    if out_w is None:
        out_w = max(im.shape[1] for im in face_imgs_sides) * 4
    out_w = min(out_w, max_out_w)
    if out_h is None:
        out_h = out_w // 2

    # Use float32 — at 4096x2048 the per-pixel arrays are ~33 MB each (float32)
    # vs 67 MB (float64); at 22528x11264 they're ~1 GB vs 2 GB. Cuts memory in half.
    yaws = (np.arange(out_w, dtype=np.float32) / out_w) * 2 * np.float32(math.pi) - np.float32(math.pi)
    pitches = np.float32(math.pi) / 2 - (np.arange(out_h, dtype=np.float32) / out_h) * np.float32(math.pi)
    yaw_2d, pitch_2d = np.meshgrid(yaws, pitches)
    cos_p = np.cos(pitch_2d)
    rx_w = np.sin(yaw_2d) * cos_p
    ry_w = np.sin(pitch_2d)
    rz_w = np.cos(yaw_2d) * cos_p

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Step 1: top/bottom (drawn first so sides overwrite them in any overlap)
    for (idx, m), img in zip(cap_meta, cap_imgs):
        out = _render_top_or_bottom(idx, m, img, rx_w, ry_w, rz_w, out, out_h, out_w)

    # Step 2: side faces (existing logic — overlap shrunk so they tile cleanly)
    yaw_centers = [m["yaw"] for m in faces_meta_sides]
    for i in range(1, 4):
        while yaw_centers[i] < yaw_centers[i - 1]:
            yaw_centers[i] += 2 * math.pi
    fov_s = [m["fov_s"] for m in faces_meta_sides]
    fov_h = [m["fov_h"] for m in faces_meta_sides]
    cy = [m["cy"] for m in faces_meta_sides]

    eff_min = [yaw_centers[i] - fov_s[i] / 2 for i in range(4)]
    eff_max = [yaw_centers[i] + fov_s[i] / 2 for i in range(4)]
    for i in range(1, 4):
        if eff_max[i - 1] > eff_min[i]:
            eff_max[i - 1] = eff_min[i]

    for i, img in enumerate(face_imgs_sides):
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
        u = (dyaw - dyaw_min) / fov_s[i]
        v = (pitch_max - pitch_2d) / fov_h[i]
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
