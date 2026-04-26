"""Apple Look Around debug scraper — pure logic, no web framework."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional

from streetlevel.geo import wgs84_to_tile_coord
from streetlevel.lookaround import (
    Face,
    get_coverage_tile,
    get_panorama_face,
)
from streetlevel.lookaround.auth import Authenticator


def _polygon_to_tiles(polygon: List[List[float]]) -> List[Tuple[int, int]]:
    """
    Return the list of z=17 (x, y) tiles covering the bbox of `polygon`.

    `polygon` is [[lon, lat], ...]. Tiles are emitted in row-major order
    (y outer ascending, x inner ascending) so iteration is top-left → bottom-right.
    """
    if not polygon:
        return []
    lons = [pt[0] for pt in polygon]
    lats = [pt[1] for pt in polygon]
    x_min, y_max = wgs84_to_tile_coord(min(lats), min(lons), 17)
    x_max, y_min = wgs84_to_tile_coord(max(lats), max(lons), 17)
    tiles: List[Tuple[int, int]] = []
    for ty in range(y_min, y_max + 1):
        for tx in range(x_min, x_max + 1):
            tiles.append((tx, ty))
    return tiles


import io

import pillow_heif
from PIL import Image
from shapely.geometry import Point, Polygon as ShapelyPolygon

pillow_heif.register_heif_opener()


def _decode_heic_to_jpg(heic_bytes: bytes, out_path: str, quality: int = 92) -> None:
    """Decode HEIC bytes to a JPEG file at `out_path`. Overwrites if exists."""
    img = Image.open(io.BytesIO(heic_bytes))
    img = img.convert("RGB")
    img.save(out_path, format="JPEG", quality=quality)


def _pick_pano(panos, polygon: List[List[float]]):
    """Return first pano inside `polygon`, else first pano in list, else None."""
    if not panos:
        return None
    poly = ShapelyPolygon(polygon)
    for pano in panos:
        if poly.contains(Point(pano.lon, pano.lat)):
            return pano
    return panos[0]


import csv

# Index → human name. Order matches streetlevel.lookaround.Face enum.
FACE_NAMES = ["back", "left", "front", "right", "top", "bottom"]


def _write_meta_csv(pano, face_paths: List[str], csv_path: str) -> None:
    """Write one row per face. `face_paths` is 6 entries in FACE_NAMES order."""
    cols = [
        "pano_id", "build_id", "lat", "lon", "capture_date",
        "heading", "pitch", "roll", "coverage_type",
        "face_index", "face_name", "image_path",
    ]
    capture_date = pano.date.isoformat() if pano.date else ""
    coverage_type = pano.coverage_type.name if pano.coverage_type else ""
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, path in enumerate(face_paths):
            w.writerow({
                "pano_id": pano.id,
                "build_id": pano.build_id,
                "lat": pano.lat,
                "lon": pano.lon,
                "capture_date": capture_date,
                "heading": getattr(pano, "heading", ""),
                "pitch": getattr(pano, "pitch", ""),
                "roll": getattr(pano, "roll", ""),
                "coverage_type": coverage_type,
                "face_index": i,
                "face_name": FACE_NAMES[i],
                "image_path": path,
            })


def stitch_faces(pano_dir: str, overlap_pct: float = 3.0, out_name: str = "stitched.jpg") -> str:
    """
    Stitch back/left/front/right JPGs in `pano_dir` left-to-right with a
    horizontal overlap between adjacent faces. The overlap on each seam is
    `overlap_pct` percent of the *left* face's width on that seam, so the
    overlap scales with zoom level (face resolution). Writes
    `<pano_dir>/<out_name>` and returns its path.

    Naive paste — no blending in the overlap region; the right face wins.
    """
    names = ["back", "left", "front", "right"]
    paths = [os.path.join(pano_dir, f"{n}.jpg") for n in names]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing face for stitching: {p}")
    imgs = [Image.open(p).convert("RGB") for p in paths]
    h = imgs[0].height
    if any(im.height != h for im in imgs):
        raise ValueError("face heights differ — cannot stitch")
    n = len(imgs)
    # Per-seam overlap: i in 0..n-2 contributes overlap on its right edge.
    seam_overlaps = [round(imgs[i].width * overlap_pct / 100.0) for i in range(n - 1)]
    out_w = sum(im.width for im in imgs) - sum(seam_overlaps)
    canvas = Image.new("RGB", (out_w, h))
    x = 0
    for i, im in enumerate(imgs):
        canvas.paste(im, (x, 0))
        # Advance by this face's width minus its right-seam overlap (last face has none).
        if i < n - 1:
            x += im.width - seam_overlaps[i]
        else:
            x += im.width
    out_path = os.path.join(pano_dir, out_name)
    canvas.save(out_path, format="JPEG", quality=92)
    return out_path


def scrape_all_in_polygon(
    polygon: List[List[float]],
    progress_cb: Optional[Callable[[str, dict], None]] = None,
) -> List[dict]:
    """
    Enumerate every Look Around panorama whose lat/lon falls inside `polygon`.

    Iterates all z=17 tiles covering the polygon's bbox, fetches each
    coverage tile, filters by point-in-polygon. Returns a list of dicts
    suitable for CSV export to Stage_2_Apple_Image_Scraper.

    progress_cb is called as
        ('progress', {'step': 'tile', 'i': k, 'total': n, 'panos_so_far': m})
    """
    def _emit(event: str, data: dict):
        if progress_cb:
            progress_cb(event, data)

    tiles = _polygon_to_tiles(polygon)
    if not tiles:
        return []

    poly = ShapelyPolygon(polygon)
    seen_ids = set()
    out_records: List[dict] = []

    for i, (tx, ty) in enumerate(tiles):
        try:
            coverage = get_coverage_tile(tx, ty)
        except Exception:
            coverage = None

        if coverage and coverage.panos:
            for pano in coverage.panos:
                if pano.id in seen_ids:
                    continue
                if not poly.contains(Point(pano.lon, pano.lat)):
                    continue
                seen_ids.add(pano.id)
                out_records.append({
                    "panoid": str(pano.id),
                    "build_id": str(pano.build_id),
                    "lat": pano.lat,
                    "lon": pano.lon,
                    "heading_deg": float(getattr(pano, "heading", 0.0) or 0.0),
                    "capture_date": pano.date.isoformat() if getattr(pano, "date", None) else "",
                    "coverage_type": pano.coverage_type.name if pano.coverage_type else "",
                })

        _emit("progress", {
            "step": "tile",
            "i": i + 1,
            "total": len(tiles),
            "panos_so_far": len(out_records),
        })

    return out_records


def write_bulk_csv(records: List[dict], csv_path: str,
                   country_code: str = "", address_label: str = "") -> str:
    """Write the bulk scrape result as a Stage_2-compatible CSV."""
    cols = [
        "panoid", "build_id", "lat", "lon",
        "heading_deg", "capture_date", "coverage_type",
        "country_code", "address_label",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow({
                "panoid": r["panoid"],
                "build_id": r["build_id"],
                "lat": r["lat"],
                "lon": r["lon"],
                "heading_deg": r.get("heading_deg", ""),
                "capture_date": r.get("capture_date", ""),
                "coverage_type": r.get("coverage_type", ""),
                "country_code": country_code,
                "address_label": address_label,
            })
    return csv_path


def _write_metadata_json(pano, pano_dir: str) -> str:
    """Persist per-face lens + position metadata for downstream reprojection."""
    import json
    data = {
        "pano_id": str(pano.id),
        "build_id": str(pano.build_id),
        "lat": pano.lat,
        "lon": pano.lon,
        "heading": float(getattr(pano, "heading", 0.0) or 0.0),
        "pitch": float(getattr(pano, "pitch", 0.0) or 0.0),
        "roll": float(getattr(pano, "roll", 0.0) or 0.0),
        "faces": [],
    }
    for cm in (pano.camera_metadata or []):
        data["faces"].append({
            "yaw": cm.position.yaw,
            "pitch": cm.position.pitch,
            "roll": cm.position.roll,
            "fov_s": cm.lens_projection.fov_s,
            "fov_h": cm.lens_projection.fov_h,
            "cx": cm.lens_projection.cx,
            "cy": cm.lens_projection.cy,
        })
    out_path = os.path.join(pano_dir, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path


@dataclass
class ScrapeResult:
    pano_id: str
    build_id: str
    lat: float
    lon: float
    capture_date: str
    heading: float
    face_paths: List[str] = field(default_factory=list)
    csv_path: str = ""


def scrape_polygon(
    polygon: List[List[float]],
    zoom: int = 0,
    out_root: str = "downloads",
    progress_cb: Optional[Callable[[str, dict], None]] = None,
) -> ScrapeResult:
    """
    Scrape a single panorama from inside `polygon`. Downloads 6 faces, decodes
    HEIC -> JPG, writes meta.csv. Returns ScrapeResult.

    Raises:
        RuntimeError: if no Look Around coverage exists in any covering tile,
                      or face fetch / decode fails.
    """
    def _emit(event: str, data: dict):
        if progress_cb:
            progress_cb(event, data)

    _emit("progress", {"step": "finding-tile"})
    tiles = _polygon_to_tiles(polygon)
    if not tiles:
        raise RuntimeError("polygon has no vertices")

    chosen_pano = None
    for tx, ty in tiles:
        coverage = get_coverage_tile(tx, ty)
        if coverage.panos:
            chosen_pano = _pick_pano(coverage.panos, polygon)
            if chosen_pano is not None:
                break
    if chosen_pano is None:
        raise RuntimeError("no Look Around coverage in polygon")

    _emit("progress", {
        "step": "pano-selected",
        "pano_id": str(chosen_pano.id),
        "lat": chosen_pano.lat,
        "lon": chosen_pano.lon,
    })

    pano_dir = os.path.join(out_root, str(chosen_pano.id))
    os.makedirs(pano_dir, exist_ok=True)
    auth = Authenticator()
    face_paths: List[str] = []
    # All 6 faces (back/left/front/right + top/bottom). The reprojector folds
    # top/bottom into the equirect via their lens metadata.
    for i, name in enumerate(FACE_NAMES):
        out_path = os.path.join(pano_dir, f"{name}.jpg")
        try:
            heic = get_panorama_face(chosen_pano, Face(i), zoom, auth)
        except Exception:
            auth = Authenticator()
            heic = get_panorama_face(chosen_pano, Face(i), zoom, auth)
        try:
            _decode_heic_to_jpg(heic, out_path)
        except Exception as e:
            raise RuntimeError(
                f"HEIC decode failed on face {name} (bytes={len(heic)}): {e}"
            ) from e
        face_paths.append(out_path)
        _emit("progress", {"step": "face", "i": i + 1, "total": 6, "name": name})

    csv_path = os.path.join(pano_dir, "meta.csv")
    _write_meta_csv(chosen_pano, face_paths, csv_path)
    _write_metadata_json(chosen_pano, pano_dir)

    capture_date = chosen_pano.date.isoformat() if getattr(chosen_pano, "date", None) else ""
    return ScrapeResult(
        pano_id=str(chosen_pano.id),
        build_id=str(chosen_pano.build_id),
        lat=chosen_pano.lat,
        lon=chosen_pano.lon,
        capture_date=capture_date,
        heading=float(getattr(chosen_pano, "heading", 0.0) or 0.0),
        face_paths=face_paths,
        csv_path=csv_path,
    )
