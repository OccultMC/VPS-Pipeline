import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apple_scraper import _polygon_to_tiles


def test_polygon_to_tiles_single_point_returns_one_tile():
    polygon = [
        [-122.4194, 37.7749],
        [-122.4193, 37.7749],
        [-122.4193, 37.7750],
        [-122.4194, 37.7750],
    ]
    tiles = _polygon_to_tiles(polygon)
    assert len(tiles) >= 1
    for tx, ty in tiles:
        assert isinstance(tx, int) and isinstance(ty, int)
    tx, ty = tiles[0]
    assert 20000 < tx < 22000
    assert 50000 < ty < 52000


def test_polygon_to_tiles_iterates_row_major():
    polygon = [
        [0.000, 0.000],
        [0.010, 0.000],
        [0.010, 0.010],
        [0.000, 0.010],
    ]
    tiles = _polygon_to_tiles(polygon)
    assert len(tiles) >= 4
    ys = [ty for _, ty in tiles]
    assert tiles[0][1] == min(ys)


from types import SimpleNamespace
from apple_scraper import _pick_pano


def _pano(lat, lon, pid=1):
    return SimpleNamespace(id=pid, lat=lat, lon=lon)


def test_pick_pano_prefers_inside_polygon():
    polygon = [[-122.42, 37.77], [-122.41, 37.77], [-122.41, 37.78], [-122.42, 37.78]]
    panos = [
        _pano(37.7749, -122.4050, pid=2),  # outside (east of -122.41)
        _pano(37.7749, -122.4150, pid=3),  # inside
    ]
    picked = _pick_pano(panos, polygon)
    assert picked.id == 3


def test_pick_pano_falls_back_to_first_when_none_inside():
    polygon = [[-122.42, 37.77], [-122.41, 37.77], [-122.41, 37.78], [-122.42, 37.78]]
    panos = [
        _pano(40.0, -100.0, pid=10),
        _pano(45.0, -95.0, pid=11),
    ]
    picked = _pick_pano(panos, polygon)
    assert picked.id == 10


def test_pick_pano_returns_none_for_empty_list():
    assert _pick_pano([], [[-122, 37], [-121, 37], [-121, 38]]) is None


import os
from PIL import Image
from apple_scraper import _decode_heic_to_jpg

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_face.heic")


def test_decode_heic_to_jpg_writes_valid_image(tmp_path):
    with open(FIXTURE, "rb") as f:
        heic = f.read()
    out = tmp_path / "front.jpg"
    _decode_heic_to_jpg(heic, str(out))
    assert out.exists()
    assert out.stat().st_size > 1000
    img = Image.open(out)
    img.verify()
    img2 = Image.open(out)
    assert img2.format == "JPEG"
    assert img2.width > 100 and img2.height > 100


import csv
from datetime import datetime, timezone
from apple_scraper import _write_meta_csv, FACE_NAMES


def _full_pano():
    return SimpleNamespace(
        id=12345,
        build_id=678,
        lat=37.7749,
        lon=-122.4194,
        date=datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        heading=1.5,
        pitch=0.05,
        roll=0.0,
        coverage_type=SimpleNamespace(name="CAR"),
    )


def test_write_meta_csv_emits_six_rows_with_schema(tmp_path):
    pano = _full_pano()
    face_paths = [str(tmp_path / f"{name}.jpg") for name in FACE_NAMES]
    csv_path = tmp_path / "meta.csv"
    _write_meta_csv(pano, face_paths, str(csv_path))

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    expected_cols = {
        "pano_id", "build_id", "lat", "lon", "capture_date",
        "heading", "pitch", "roll", "coverage_type",
        "face_index", "face_name", "image_path",
    }
    assert set(rows[0].keys()) == expected_cols
    assert rows[0]["pano_id"] == "12345"
    assert rows[0]["coverage_type"] == "CAR"
    assert rows[0]["face_index"] == "0"
    assert rows[0]["face_name"] == "back"
    assert rows[5]["face_name"] == "bottom"
    assert rows[2]["image_path"].endswith("front.jpg")


import pytest
from apple_scraper import scrape_polygon, ScrapeResult


@pytest.mark.network
def test_scrape_polygon_downloads_four_faces_and_csv(tmp_path):
    # Downtown SF — known dense Look Around coverage
    polygon = [
        [-122.4200, 37.7745],
        [-122.4188, 37.7745],
        [-122.4188, 37.7755],
        [-122.4200, 37.7755],
    ]
    result = scrape_polygon(polygon, zoom=6, out_root=str(tmp_path))
    assert isinstance(result, ScrapeResult)
    assert result.pano_id
    assert len(result.face_paths) == 4
    for p in result.face_paths:
        assert p.endswith(".jpg")
        assert os.path.getsize(p) > 1000
    # top/bottom must NOT exist
    pano_dir = os.path.dirname(result.face_paths[0])
    assert not os.path.exists(os.path.join(pano_dir, "top.jpg"))
    assert not os.path.exists(os.path.join(pano_dir, "bottom.jpg"))
    assert result.csv_path.endswith("meta.csv")
    assert os.path.exists(result.csv_path)


from apple_scraper import stitch_faces


def test_stitch_faces_overlaps_correctly(tmp_path):
    # Mimic Apple's varying widths: back/front wider than left/right
    widths = {"back": 100, "left": 60, "front": 100, "right": 60}
    for name, w in widths.items():
        img = Image.new("RGB", (w, 80), color=(0, 0, 0))
        img.save(tmp_path / f"{name}.jpg", "JPEG")
    # 10% overlap. seams use left-face widths in order: back(100), left(60), front(100).
    # seam overlaps = round(100*.1)=10, round(60*.1)=6, round(100*.1)=10 → total 26
    # out_w = (100+60+100+60) - 26 = 294
    out = stitch_faces(str(tmp_path), overlap_pct=10.0)
    assert os.path.exists(out)
    out_img = Image.open(out)
    assert out_img.width == 294
    assert out_img.height == 80


def test_stitch_faces_zero_pct_yields_concatenation(tmp_path):
    for name, w in {"back": 100, "left": 60, "front": 100, "right": 60}.items():
        Image.new("RGB", (w, 80)).save(tmp_path / f"{name}.jpg", "JPEG")
    out = stitch_faces(str(tmp_path), overlap_pct=0.0)
    assert Image.open(out).width == 100 + 60 + 100 + 60


def test_stitch_faces_rejects_mismatched_heights(tmp_path):
    Image.new("RGB", (100, 80)).save(tmp_path / "back.jpg", "JPEG")
    Image.new("RGB", (60, 80)).save(tmp_path / "left.jpg", "JPEG")
    Image.new("RGB", (100, 90)).save(tmp_path / "front.jpg", "JPEG")  # wrong height
    Image.new("RGB", (60, 80)).save(tmp_path / "right.jpg", "JPEG")
    import pytest as _pt
    with _pt.raises(ValueError, match="heights differ"):
        stitch_faces(str(tmp_path))


def test_stitch_faces_raises_when_faces_missing(tmp_path):
    img = Image.new("RGB", (50, 50), color=(0, 0, 0))
    img.save(tmp_path / "back.jpg", "JPEG")
    # left/front/right missing
    import pytest as _pt
    with _pt.raises(FileNotFoundError):
        stitch_faces(str(tmp_path))
