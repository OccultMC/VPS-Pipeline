# Apple Look Around Debug Scraper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-only Flask + Socket.IO web app that, on a polygon click, downloads exactly one Apple Look Around panorama (6 faces decoded HEIC→JPG) plus a metadata CSV, and renders the result in the browser for visual debugging.

**Architecture:** Fork of `VPS_Scraper/web_app.py` and `VPS_Scraper/templates/index.html` into a new sibling folder `LookAround_Scraper/`. All deployment infra (Vast.ai, R2, Redis, Ray, log monitors) is removed. A new `apple_scraper.py` module wraps `streetlevel.lookaround` to handle tile discovery → pano selection → face download/decode → CSV write. A SocketIO event drives the scrape and streams progress.

**Tech Stack:** Python 3.11+, Flask, Flask-SocketIO, geopandas, shapely, Pillow, pillow-heif, streetlevel (Apple Look Around client), pycryptodome, pytest.

**Spec:** `docs/superpowers/specs/2026-04-25-lookaround-debug-scraper-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `LookAround_Scraper/__init__.py` | Empty package marker. |
| `LookAround_Scraper/requirements.txt` | Pinned deps (Flask, Flask-SocketIO, geopandas, shapely, pillow-heif, streetlevel, etc). |
| `LookAround_Scraper/.gitignore` | Ignore `downloads/*` (except `.gitkeep`). |
| `LookAround_Scraper/apple_scraper.py` | Pure scrape logic. `scrape_polygon()` orchestrator + private helpers (`_polygon_to_tiles`, `_pick_pano`, `_download_face_jpg`, `_write_meta_csv`). No Flask, no SocketIO. |
| `LookAround_Scraper/web_app.py` | Flask + SocketIO server. Routes (`/`, `/api/load-shapes`, `/downloads/<pano_id>/<file>`) and SocketIO event (`scrape_polygon`). Spawns a daemon thread per scrape, streams progress. |
| `LookAround_Scraper/templates/index.html` | Single-page UI: Leaflet map, shape upload, polygon click, progress strip, 6-face gallery, CSV link. |
| `LookAround_Scraper/downloads/.gitkeep` | Placeholder so the empty folder commits. |
| `LookAround_Scraper/tests/__init__.py` | Empty package marker. |
| `LookAround_Scraper/tests/test_apple_scraper.py` | Unit tests for `_polygon_to_tiles`, `_pick_pano`, `_write_meta_csv`. Plus an opt-in integration test for `scrape_polygon` (real network, marker `network`). |

---

## Task 1: Project skeleton

**Files:**
- Create: `LookAround_Scraper/__init__.py` (empty)
- Create: `LookAround_Scraper/tests/__init__.py` (empty)
- Create: `LookAround_Scraper/requirements.txt`
- Create: `LookAround_Scraper/.gitignore`
- Create: `LookAround_Scraper/downloads/.gitkeep` (empty)
- Create: `LookAround_Scraper/templates/.gitkeep` (empty placeholder until Task 8 fills it)

- [ ] **Step 1: Create folder + empty markers**

```bash
mkdir -p LookAround_Scraper/tests LookAround_Scraper/templates LookAround_Scraper/downloads
touch LookAround_Scraper/__init__.py LookAround_Scraper/tests/__init__.py
touch LookAround_Scraper/downloads/.gitkeep LookAround_Scraper/templates/.gitkeep
```

- [ ] **Step 2: Write `LookAround_Scraper/requirements.txt`**

```
Flask>=2.3,<4
Flask-SocketIO>=5.3,<6
python-engineio>=4.8
python-socketio>=5.10
geopandas>=0.14
shapely>=2.0
Pillow>=10
pillow-heif>=0.16
streetlevel>=0.13
requests>=2.31
aiohttp>=3.9
pycryptodome>=3.20
protobuf>=4.25
pytest>=8
```

- [ ] **Step 3: Write `LookAround_Scraper/.gitignore`**

```
downloads/*
!downloads/.gitkeep
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 4: Verify import path**

Run: `python -c "from streetlevel.lookaround import get_coverage_tile_by_latlon; print('ok')"`
Expected: `ok` (if it errors with ModuleNotFoundError, run `pip install -r LookAround_Scraper/requirements.txt`).

- [ ] **Step 5: Commit**

```bash
git add LookAround_Scraper/
git commit -m "feat(lookaround): scaffold LookAround_Scraper project skeleton"
```

---

## Task 2: `_polygon_to_tiles` (TDD)

Compute the set of z=17 tiles that any vertex of a polygon falls on, plus all integer tiles inside its bounding box. This is the search space for finding coverage.

**Files:**
- Test: `LookAround_Scraper/tests/test_apple_scraper.py`
- Create: `LookAround_Scraper/apple_scraper.py`

- [ ] **Step 1: Write the failing test**

Append to `LookAround_Scraper/tests/test_apple_scraper.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apple_scraper import _polygon_to_tiles


def test_polygon_to_tiles_single_point_returns_one_tile():
    # Tiny polygon around a point in San Francisco
    polygon = [
        [-122.4194, 37.7749],
        [-122.4193, 37.7749],
        [-122.4193, 37.7750],
        [-122.4194, 37.7750],
    ]
    tiles = _polygon_to_tiles(polygon)
    assert len(tiles) >= 1
    # Each tile is (x, y) integer pair
    for tx, ty in tiles:
        assert isinstance(tx, int) and isinstance(ty, int)
    # SF at z=17 is around tile (20984, 50749) — sanity check range
    tx, ty = tiles[0]
    assert 20000 < tx < 22000
    assert 50000 < ty < 52000


def test_polygon_to_tiles_iterates_row_major():
    # 0.01° wide polygon at the equator — spans multiple tiles at z=17
    polygon = [
        [0.000, 0.000],
        [0.010, 0.000],
        [0.010, 0.010],
        [0.000, 0.010],
    ]
    tiles = _polygon_to_tiles(polygon)
    assert len(tiles) >= 4
    # First tile must be the top-left of the bbox (smallest y, smallest x)
    ys = [ty for _, ty in tiles]
    assert tiles[0][1] == min(ys)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd LookAround_Scraper && pytest tests/test_apple_scraper.py -v`
Expected: ImportError or "module apple_scraper has no attribute _polygon_to_tiles".

- [ ] **Step 3: Implement `_polygon_to_tiles`**

Create `LookAround_Scraper/apple_scraper.py`:

```python
"""Apple Look Around debug scraper — pure logic, no web framework."""
from __future__ import annotations

from typing import Callable, List, Tuple, Optional
from streetlevel.geo import wgs84_to_tile_coord


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
    # Note: y increases southward, so min lat → max y, max lat → min y
    x_min, y_max = wgs84_to_tile_coord(min(lats), min(lons), 17)
    x_max, y_min = wgs84_to_tile_coord(max(lats), max(lons), 17)
    tiles: List[Tuple[int, int]] = []
    for ty in range(y_min, y_max + 1):
        for tx in range(x_min, x_max + 1):
            tiles.append((tx, ty))
    return tiles
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd LookAround_Scraper && pytest tests/test_apple_scraper.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add LookAround_Scraper/apple_scraper.py LookAround_Scraper/tests/test_apple_scraper.py
git commit -m "feat(lookaround): polygon->tiles bbox enumeration at z=17"
```

---

## Task 3: `_pick_pano` (TDD)

Given a tile's panoramas and a polygon, return the pano whose lat/lon falls inside the polygon (preferred), else the first pano. This keeps the picked pano "in" the user-clicked shape when possible.

**Files:**
- Test: `LookAround_Scraper/tests/test_apple_scraper.py` (extend)
- Modify: `LookAround_Scraper/apple_scraper.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `LookAround_Scraper/tests/test_apple_scraper.py`:

```python
from types import SimpleNamespace
from apple_scraper import _pick_pano


def _pano(lat, lon, pid=1):
    return SimpleNamespace(id=pid, lat=lat, lon=lon)


def test_pick_pano_prefers_inside_polygon():
    polygon = [[-122.42, 37.77], [-122.41, 37.77], [-122.41, 37.78], [-122.42, 37.78]]
    panos = [
        _pano(37.7749, -122.4150, pid=2),  # outside (lon > -122.41)
        _pano(37.7749, -122.4194, pid=3),  # inside
    ]
    picked = _pick_pano(panos, polygon)
    assert picked.id == 3


def test_pick_pano_falls_back_to_first_when_none_inside():
    polygon = [[-122.42, 37.77], [-122.41, 37.77], [-122.41, 37.78], [-122.42, 37.78]]
    panos = [
        _pano(40.0, -100.0, pid=10),  # nowhere near polygon
        _pano(45.0, -95.0, pid=11),
    ]
    picked = _pick_pano(panos, polygon)
    assert picked.id == 10


def test_pick_pano_returns_none_for_empty_list():
    assert _pick_pano([], [[-122, 37], [-121, 37], [-121, 38]]) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd LookAround_Scraper && pytest tests/test_apple_scraper.py -v -k pick_pano`
Expected: 3 errors / fails (function not defined).

- [ ] **Step 3: Implement `_pick_pano`**

Append to `LookAround_Scraper/apple_scraper.py`:

```python
from shapely.geometry import Point, Polygon as ShapelyPolygon


def _pick_pano(panos, polygon: List[List[float]]):
    """Return first pano inside `polygon`, else first pano in list, else None."""
    if not panos:
        return None
    poly = ShapelyPolygon(polygon)
    for pano in panos:
        if poly.contains(Point(pano.lon, pano.lat)):
            return pano
    return panos[0]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd LookAround_Scraper && pytest tests/test_apple_scraper.py -v`
Expected: 5 passed total (2 from Task 2, 3 from Task 3).

- [ ] **Step 5: Commit**

```bash
git add LookAround_Scraper/apple_scraper.py LookAround_Scraper/tests/test_apple_scraper.py
git commit -m "feat(lookaround): pano selector prefers panos inside polygon"
```

---

## Task 4: `_decode_heic_to_jpg` (TDD)

Decode a HEIC byte string to a PIL Image and save as JPG. Test uses a fixture HEIC file generated from a real Apple face the first time the test is run, then committed under `tests/fixtures/`.

**Files:**
- Test: `LookAround_Scraper/tests/test_apple_scraper.py` (extend)
- Modify: `LookAround_Scraper/apple_scraper.py` (extend)
- Create: `LookAround_Scraper/tests/fixtures/sample_face.heic` (binary, ≤200 KB)

- [ ] **Step 1: Generate the HEIC fixture**

Run a one-off script (you can drop this in a Python REPL — do NOT commit the script):

```python
from streetlevel.lookaround import (
    get_coverage_tile_by_latlon, get_panorama_face, Face,
)
from streetlevel.lookaround.auth import Authenticator

panos = get_coverage_tile_by_latlon(37.7749, -122.4194).panos
pano = panos[0]
auth = Authenticator()
heic = get_panorama_face(pano, Face.FRONT, zoom=6, auth=auth)  # zoom 6 → small file
import os; os.makedirs("LookAround_Scraper/tests/fixtures", exist_ok=True)
with open("LookAround_Scraper/tests/fixtures/sample_face.heic", "wb") as f:
    f.write(heic)
print("bytes:", len(heic))
```

Expected output: `bytes: 30000` (roughly — anywhere in 5 KB–200 KB is fine for zoom 6).

- [ ] **Step 2: Write the failing test**

Append to `LookAround_Scraper/tests/test_apple_scraper.py`:

```python
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
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd LookAround_Scraper && pytest tests/test_apple_scraper.py -v -k decode_heic`
Expected: ImportError on `_decode_heic_to_jpg`.

- [ ] **Step 4: Implement `_decode_heic_to_jpg`**

Append to `LookAround_Scraper/apple_scraper.py`:

```python
import io

import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()


def _decode_heic_to_jpg(heic_bytes: bytes, out_path: str, quality: int = 92) -> None:
    """Decode HEIC bytes to a JPEG file at `out_path`. Overwrites if exists."""
    img = Image.open(io.BytesIO(heic_bytes))
    img = img.convert("RGB")
    img.save(out_path, format="JPEG", quality=quality)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd LookAround_Scraper && pytest tests/test_apple_scraper.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add LookAround_Scraper/apple_scraper.py LookAround_Scraper/tests/test_apple_scraper.py LookAround_Scraper/tests/fixtures/sample_face.heic
git commit -m "feat(lookaround): HEIC->JPG face decoder via pillow-heif"
```

---

## Task 5: `_write_meta_csv` (TDD)

Write a CSV with one row per face containing pano metadata + image path.

**Files:**
- Test: `LookAround_Scraper/tests/test_apple_scraper.py` (extend)
- Modify: `LookAround_Scraper/apple_scraper.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `LookAround_Scraper/tests/test_apple_scraper.py`:

```python
import csv
from datetime import datetime, timezone
from types import SimpleNamespace
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd LookAround_Scraper && pytest tests/test_apple_scraper.py -v -k write_meta_csv`
Expected: ImportError on `_write_meta_csv` / `FACE_NAMES`.

- [ ] **Step 3: Implement `_write_meta_csv` + `FACE_NAMES`**

Append to `LookAround_Scraper/apple_scraper.py`:

```python
import csv

# Index → human name. Order matches streetlevel.lookaround.Face enum.
FACE_NAMES = ["back", "left", "front", "right", "top", "bottom"]


def _write_meta_csv(pano, face_paths: List[str], csv_path: str) -> None:
    """
    Write one row per face. `face_paths` is 6 entries in FACE_NAMES order.
    """
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd LookAround_Scraper && pytest tests/test_apple_scraper.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add LookAround_Scraper/apple_scraper.py LookAround_Scraper/tests/test_apple_scraper.py
git commit -m "feat(lookaround): per-pano meta.csv writer (6 rows, full schema)"
```

---

## Task 6: `scrape_polygon` orchestrator (integration test, real network)

Compose the helpers into the public entry point. Single integration test hits the real Apple API; mark it `network` so unit-test runs can skip it.

**Files:**
- Test: `LookAround_Scraper/tests/test_apple_scraper.py` (extend)
- Modify: `LookAround_Scraper/apple_scraper.py` (extend)
- Create: `LookAround_Scraper/conftest.py`

- [ ] **Step 1: Configure pytest marker**

Create `LookAround_Scraper/conftest.py`:

```python
def pytest_configure(config):
    config.addinivalue_line("markers", "network: hits the real Apple Look Around API")
```

- [ ] **Step 2: Write the failing test**

Append to `LookAround_Scraper/tests/test_apple_scraper.py`:

```python
import pytest
from apple_scraper import scrape_polygon, ScrapeResult


@pytest.mark.network
def test_scrape_polygon_downloads_six_faces_and_csv(tmp_path):
    # Tight polygon over Apple Park (well-covered Look Around area)
    polygon = [
        [-122.0091, 37.3349],
        [-122.0081, 37.3349],
        [-122.0081, 37.3359],
        [-122.0091, 37.3359],
    ]
    result = scrape_polygon(polygon, zoom=6, out_root=str(tmp_path))
    assert isinstance(result, ScrapeResult)
    assert result.pano_id
    assert len(result.face_paths) == 6
    for p in result.face_paths:
        assert p.endswith(".jpg")
        import os
        assert os.path.getsize(p) > 1000
    assert result.csv_path.endswith("meta.csv")
    import os
    assert os.path.exists(result.csv_path)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd LookAround_Scraper && pytest tests/test_apple_scraper.py -v -k scrape_polygon -m network`
Expected: ImportError on `scrape_polygon` / `ScrapeResult`.

- [ ] **Step 4: Implement `scrape_polygon` + `ScrapeResult`**

Append to `LookAround_Scraper/apple_scraper.py`:

```python
import os
from dataclasses import dataclass, field

from streetlevel.lookaround import (
    Face,
    get_coverage_tile,
    get_panorama_face,
)
from streetlevel.lookaround.auth import Authenticator


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
    for i, name in enumerate(FACE_NAMES):
        out_path = os.path.join(pano_dir, f"{name}.jpg")
        try:
            heic = get_panorama_face(chosen_pano, Face(i), zoom, auth)
        except Exception:
            # one retry with a fresh authenticator (covers transient 403)
            auth = Authenticator()
            heic = get_panorama_face(chosen_pano, Face(i), zoom, auth)
        try:
            _decode_heic_to_jpg(heic, out_path)
        except Exception as e:
            raise RuntimeError(f"HEIC decode failed on face {name} (bytes={len(heic)}): {e}") from e
        face_paths.append(out_path)
        _emit("progress", {"step": "face", "i": i + 1, "total": 6, "name": name})

    csv_path = os.path.join(pano_dir, "meta.csv")
    _write_meta_csv(chosen_pano, face_paths, csv_path)

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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd LookAround_Scraper && pytest tests/test_apple_scraper.py -v -m network`
Expected: 1 passed (the integration test). May take 30–90 s at zoom 6.

Also run unit-test-only mode to confirm marker works:

Run: `cd LookAround_Scraper && pytest tests/test_apple_scraper.py -v -m "not network"`
Expected: 7 passed, 1 deselected.

- [ ] **Step 6: Commit**

```bash
git add LookAround_Scraper/apple_scraper.py LookAround_Scraper/tests/test_apple_scraper.py LookAround_Scraper/conftest.py
git commit -m "feat(lookaround): scrape_polygon orchestrator + integration test"
```

---

## Task 7: `web_app.py` — Flask + SocketIO server

Forked-and-gutted Flask app. We write it from scratch (cleaner than incremental gutting of 1700 lines), keeping only the `/api/load-shapes` body verbatim.

**Files:**
- Create: `LookAround_Scraper/web_app.py`

- [ ] **Step 1: Write `LookAround_Scraper/web_app.py`**

```python
"""
LookAround Scraper — local debug Web GUI.

Forked from VPS_Scraper/web_app.py with all deployment infrastructure removed.

Usage:
    cd LookAround_Scraper
    python web_app.py          # http://localhost:5000
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict

from flask import (
    Flask, jsonify, render_template, request, send_from_directory, url_for,
)
from flask_socketio import SocketIO, emit, join_room

from apple_scraper import scrape_polygon

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT = Path(__file__).resolve().parent
DOWNLOADS_DIR = ROOT / "downloads"
DOWNLOADS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET", "dev-secret-key")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

sessions: Dict[str, Dict[str, Any]] = {}


def _get_session(sid: str) -> Dict[str, Any]:
    if not sid:
        sid = str(uuid.uuid4())
    if sid not in sessions:
        sessions[sid] = {"shapes_data": None}
    return sessions[sid]


# ───────────────────────────────────────── Routes ─────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/load-shapes", methods=["POST"])
def load_shapes():
    """Load shapes from GeoJSON, Shapefile (.shp/.zip), KML, or GeoPackage."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    all_files = request.files.getlist("file")
    sid = request.form.get("session_id", "")

    main_file = None
    for uf in all_files:
        ext_ = (uf.filename or "").rsplit(".", 1)[-1].lower()
        if ext_ in ("shp", "geojson", "json", "zip", "kml", "gpkg"):
            main_file = uf
            break
    if main_file is None:
        main_file = all_files[0]

    fname = main_file.filename or ""
    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""

    try:
        if ext in ("geojson", "json"):
            content = main_file.read().decode("utf-8")
            geojson = json.loads(content)
        elif ext in ("shp", "dbf", "zip", "kml", "gpkg"):
            import geopandas as gpd
            os.environ["SHAPE_RESTORE_SHX"] = "YES"
            tmpdir = tempfile.mkdtemp()
            try:
                for uf in all_files:
                    uf.save(os.path.join(tmpdir, uf.filename or "upload"))
                if ext == "zip":
                    main_path = f"zip://{os.path.join(tmpdir, fname)}"
                else:
                    main_path = os.path.join(tmpdir, fname)
                gdf = gpd.read_file(main_path)
                if gdf.crs is None:
                    gdf = gdf.set_crs(epsg=4326)
                elif gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs(epsg=4326)
                geojson = json.loads(gdf.to_json())
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            return jsonify({"error": f"Unsupported file type: .{ext}"}), 400

        sess = _get_session(sid)
        sess["shapes_data"] = geojson
        return jsonify({"geojson": geojson, "filename": fname})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/downloads/<pano_id>/<path:filename>")
def serve_download(pano_id: str, filename: str):
    pano_dir = DOWNLOADS_DIR / pano_id
    return send_from_directory(str(pano_dir), filename)


# ───────────────────────────────────────── SocketIO ───────────────────────────

@socketio.on("connect")
def on_connect():
    sid = request.args.get("session_id") or str(uuid.uuid4())
    join_room(sid)
    emit("connected", {"session_id": sid})


@socketio.on("scrape_polygon")
def on_scrape_polygon(data):
    sid = data.get("session_id") or request.sid
    coords = data.get("coords") or []
    zoom = int(data.get("zoom", 0))

    if not coords or len(coords) < 3:
        emit("scrape_error", {"message": "polygon needs at least 3 vertices"}, room=sid)
        return

    def _run():
        def progress_cb(event, payload):
            socketio.emit(event, payload, room=sid)
        try:
            result = scrape_polygon(
                coords, zoom=zoom, out_root=str(DOWNLOADS_DIR),
                progress_cb=progress_cb,
            )
            with app.app_context():
                face_urls = [
                    url_for("serve_download", pano_id=result.pano_id, filename=os.path.basename(p))
                    for p in result.face_paths
                ]
                csv_url = url_for("serve_download", pano_id=result.pano_id, filename="meta.csv")
            socketio.emit("scrape_done", {
                "pano_id": result.pano_id,
                "lat": result.lat,
                "lon": result.lon,
                "capture_date": result.capture_date,
                "heading": result.heading,
                "face_urls": face_urls,
                "csv_url": csv_url,
            }, room=sid)
        except Exception as e:
            logger.exception("scrape failed")
            socketio.emit("scrape_error", {"message": str(e)}, room=sid)

    threading.Thread(target=_run, daemon=True).start()


# ───────────────────────────────────────── Main ───────────────────────────────

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)
```

- [ ] **Step 2: Smoke-test imports**

Run: `cd LookAround_Scraper && python -c "import web_app; print('ok')"`
Expected: `ok` (no template error yet — `index()` only fires on request).

- [ ] **Step 3: Smoke-test server boot**

Run: `cd LookAround_Scraper && timeout 3 python web_app.py || true`
Expected: log line "Running on http://0.0.0.0:5000" before the timeout kills it.

- [ ] **Step 4: Commit**

```bash
git add LookAround_Scraper/web_app.py
git commit -m "feat(lookaround): Flask+SocketIO server with /api/load-shapes and scrape_polygon event"
```

---

## Task 8: `templates/index.html` — single-page UI

Slim Leaflet UI: dark theme, shape upload, polygon click → emit `scrape_polygon`, progress strip, 6-face gallery, CSV link.

**Files:**
- Create: `LookAround_Scraper/templates/index.html`
- Delete: `LookAround_Scraper/templates/.gitkeep`

- [ ] **Step 1: Write `LookAround_Scraper/templates/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LookAround Debug Scraper</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
  :root { --bg:#111; --panel:#1c1c1c; --fg:#eee; --muted:#888; --accent:#4ea1ff; --err:#ff5c5c; }
  * { box-sizing: border-box; }
  html, body { margin:0; height:100%; background:var(--bg); color:var(--fg); font-family:system-ui,sans-serif; font-size:13px; }
  #app { display:grid; grid-template-columns: 320px 1fr; height:100vh; }
  #sidebar { padding:16px; overflow:auto; border-right:1px solid #222; background:var(--panel); }
  #map { height:100%; }
  h1 { font-size:14px; margin:0 0 12px; letter-spacing:0.04em; text-transform:uppercase; color:var(--muted); }
  label { display:block; font-size:11px; color:var(--muted); margin:12px 0 4px; text-transform:uppercase; letter-spacing:0.05em; }
  input[type=file], input[type=number], select { width:100%; background:#000; color:var(--fg); border:1px solid #333; padding:6px 8px; font-size:12px; }
  .progress { margin-top:16px; padding:8px; background:#000; border:1px solid #333; min-height:32px; font-family:ui-monospace,monospace; font-size:11px; white-space:pre-wrap; }
  .gallery { margin-top:12px; display:grid; grid-template-columns: repeat(2, 1fr); gap:6px; }
  .gallery .face { background:#000; border:1px solid #333; aspect-ratio:1/1; display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:10px; overflow:hidden; }
  .gallery .face img { width:100%; height:100%; object-fit:cover; }
  .meta { margin-top:12px; font-family:ui-monospace,monospace; font-size:11px; color:var(--muted); }
  .meta b { color:var(--fg); }
  a.csv-link { color:var(--accent); display:inline-block; margin-top:8px; }
  .err { color:var(--err); }
</style>
</head>
<body>
<div id="app">
  <aside id="sidebar">
    <h1>LookAround Debug</h1>
    <label>Shape file</label>
    <input type="file" id="shape-file" accept=".geojson,.json,.shp,.dbf,.shx,.prj,.zip,.kml,.gpkg" multiple>
    <label>Zoom (0=full, 7=lowest)</label>
    <input type="number" id="zoom" value="0" min="0" max="7">
    <div class="progress" id="progress">click a polygon to scrape one panorama</div>
    <div class="meta" id="meta"></div>
    <div class="gallery" id="gallery">
      <div class="face" data-name="back">back</div>
      <div class="face" data-name="left">left</div>
      <div class="face" data-name="front">front</div>
      <div class="face" data-name="right">right</div>
      <div class="face" data-name="top">top</div>
      <div class="face" data-name="bottom">bottom</div>
    </div>
    <a class="csv-link" id="csv-link" href="#" style="display:none">download meta.csv</a>
  </aside>
  <div id="map"></div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<script>
  const sessionId = crypto.randomUUID();
  const socket = io({ query: { session_id: sessionId } });
  const map = L.map('map').setView([37.7749, -122.4194], 13);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png', {
    attribution: '&copy; OSM &copy; CARTO', maxZoom: 19
  }).addTo(map);

  const progressEl = document.getElementById('progress');
  const metaEl = document.getElementById('meta');
  const csvLink = document.getElementById('csv-link');
  const galleryFaces = {};
  document.querySelectorAll('#gallery .face').forEach(el => galleryFaces[el.dataset.name] = el);

  let shapesLayer = null;

  function setProgress(text, cls = '') {
    progressEl.textContent = text;
    progressEl.className = 'progress ' + cls;
  }

  function clearGallery() {
    Object.entries(galleryFaces).forEach(([name, el]) => {
      el.innerHTML = name;
    });
    metaEl.innerHTML = '';
    csvLink.style.display = 'none';
  }

  function attachClickHandler(layer) {
    layer.on('click', (e) => {
      L.DomEvent.stopPropagation(e);
      const geom = layer.feature && layer.feature.geometry;
      if (!geom) return;
      let ring;
      if (geom.type === 'Polygon') {
        ring = geom.coordinates[0];
      } else if (geom.type === 'MultiPolygon') {
        ring = geom.coordinates[0][0];
      } else {
        setProgress('unsupported geometry: ' + geom.type, 'err');
        return;
      }
      clearGallery();
      setProgress('submitting polygon (' + ring.length + ' vertices)...');
      socket.emit('scrape_polygon', {
        session_id: sessionId,
        coords: ring,
        zoom: parseInt(document.getElementById('zoom').value, 10) || 0,
      });
    });
  }

  document.getElementById('shape-file').addEventListener('change', async (e) => {
    const files = e.target.files;
    if (!files.length) return;
    const fd = new FormData();
    for (const f of files) fd.append('file', f);
    fd.append('session_id', sessionId);
    setProgress('uploading shape...');
    const res = await fetch('/api/load-shapes', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error) { setProgress('shape load error: ' + data.error, 'err'); return; }
    if (shapesLayer) map.removeLayer(shapesLayer);
    shapesLayer = L.geoJSON(data.geojson, {
      style: { color: '#4ea1ff', weight: 1, fillOpacity: 0.15 },
      onEachFeature: (feature, layer) => attachClickHandler(layer),
    }).addTo(map);
    map.fitBounds(shapesLayer.getBounds());
    setProgress('loaded ' + data.filename + ' — click a polygon');
  });

  socket.on('progress', (p) => {
    if (p.step === 'finding-tile') setProgress('finding coverage tile...');
    else if (p.step === 'pano-selected') setProgress('pano ' + p.pano_id + ' @ ' + p.lat.toFixed(5) + ',' + p.lon.toFixed(5));
    else if (p.step === 'face') setProgress('face ' + p.i + '/' + p.total + ' (' + p.name + ')...');
  });

  socket.on('scrape_done', (r) => {
    setProgress('done — pano ' + r.pano_id);
    metaEl.innerHTML = '<b>pano:</b> ' + r.pano_id + '<br><b>lat,lon:</b> ' + r.lat.toFixed(6) + ', ' + r.lon.toFixed(6) +
      '<br><b>captured:</b> ' + r.capture_date + '<br><b>heading:</b> ' + r.heading.toFixed(3);
    const names = ['back','left','front','right','top','bottom'];
    r.face_urls.forEach((url, i) => {
      const el = galleryFaces[names[i]];
      el.innerHTML = '';
      const img = document.createElement('img');
      img.src = url; img.alt = names[i];
      el.appendChild(img);
    });
    csvLink.href = r.csv_url;
    csvLink.style.display = 'inline-block';
  });

  socket.on('scrape_error', (e) => setProgress('error: ' + e.message, 'err'));
</script>
</body>
</html>
```

- [ ] **Step 2: Remove the `.gitkeep` placeholder**

```bash
rm LookAround_Scraper/templates/.gitkeep
```

- [ ] **Step 3: Smoke-test render**

Start the server in the background:

```bash
cd LookAround_Scraper && python web_app.py &
SERVER_PID=$!
sleep 2
curl -s http://localhost:5000/ | head -5
kill $SERVER_PID
```

Expected: HTML starting with `<!DOCTYPE html>` and `<title>LookAround Debug Scraper</title>`.

- [ ] **Step 4: Commit**

```bash
git add LookAround_Scraper/templates/index.html
git rm -f LookAround_Scraper/templates/.gitkeep
git commit -m "feat(lookaround): single-page debug UI (Leaflet + 6-face gallery)"
```

---

## Task 9: End-to-end manual smoke test

Verify the whole pipeline in the browser.

**Files:** none (verification only)

- [ ] **Step 1: Create a sample polygon GeoJSON for testing**

Create `LookAround_Scraper/samples/apple_park.geojson` (commit alongside):

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"name": "Apple Park"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [-122.0091, 37.3349],
          [-122.0070, 37.3349],
          [-122.0070, 37.3370],
          [-122.0091, 37.3370],
          [-122.0091, 37.3349]
        ]]
      }
    }
  ]
}
```

- [ ] **Step 2: Start the server**

```bash
cd LookAround_Scraper && python web_app.py
```

Expected: server listening on `http://localhost:5000`.

- [ ] **Step 3: Open the browser and test**

Open http://localhost:5000 in a browser. Then:

1. Click the file picker, choose `samples/apple_park.geojson`.
2. Map should pan/fit to the polygon.
3. Click the polygon. Watch the progress strip cycle through:
   - `finding coverage tile...`
   - `pano <id> @ <lat>,<lon>`
   - `face 1/6 (back)...` … `face 6/6 (bottom)...`
   - `done — pano <id>`
4. The gallery should fill with 6 face images.
5. The "download meta.csv" link should appear.

- [ ] **Step 4: Verify on disk**

```bash
ls LookAround_Scraper/downloads/
```

Expected: a folder named `<pano_id>` containing `back.jpg`, `left.jpg`, `front.jpg`, `right.jpg`, `top.jpg`, `bottom.jpg`, `meta.csv`. At zoom 0 each JPG should be 1–10 MB.

```bash
head -2 LookAround_Scraper/downloads/<pano_id>/meta.csv
wc -l LookAround_Scraper/downloads/<pano_id>/meta.csv
```

Expected: header line + 6 data rows = 7 lines total.

- [ ] **Step 5: Verify error path**

Upload a polygon over open ocean (e.g. mid-Pacific) or any area without Look Around coverage. Click the polygon.
Expected: progress strip shows `error: no Look Around coverage in polygon` in red.

- [ ] **Step 6: Commit sample**

```bash
git add LookAround_Scraper/samples/apple_park.geojson
git commit -m "test(lookaround): add Apple Park sample polygon for manual smoke test"
```

---

## Self-Review (executed by plan author)

**Spec coverage:**
- Architecture (sibling folder, no deployment infra) → Tasks 1, 7
- `apple_scraper.py` orchestrator → Tasks 2–6
- `web_app.py` routes + SocketIO → Task 7
- `templates/index.html` → Task 8
- 6-face decode HEIC→JPG → Task 4
- Per-face CSV schema → Task 5
- Progress events → Tasks 6, 7
- Error handling (no coverage / 403 retry / decode fail) → Task 6
- Manual debug verification → Task 9

**Placeholder scan:** none.

**Type consistency:**
- `FACE_NAMES` defined Task 5, used Tasks 6 and 8 (in same order, hardcoded mirror in HTML).
- `ScrapeResult` defined Task 6 with fields used in Task 7.
- `Face(i)` from streetlevel is the int-indexed enum matching `FACE_NAMES` (verified against `streetlevel/lookaround/lookaround.py:19-28` — BACK=0, LEFT=1, FRONT=2, RIGHT=3, TOP=4, BOTTOM=5).
- Progress event names: `progress`, `scrape_done`, `scrape_error` — consistent across server (Task 7) and client (Task 8).

No further fixes needed.
