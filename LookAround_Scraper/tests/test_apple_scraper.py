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
