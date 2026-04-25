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
