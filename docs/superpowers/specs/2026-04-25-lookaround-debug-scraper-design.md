# Apple Look Around Debug Scraper — Design

**Date:** 2026-04-25
**Status:** Approved (awaiting user spec review)

## Goal

A local-only Flask + Socket.IO web app that scrapes a single Apple Look Around panorama from a user-clicked polygon, downloads the 6 face images decoded to JPG, writes a CSV with per-face metadata, and renders the result in the browser for visual debugging.

Forked from `VPS_Scraper/web_app.py` with all deployment infrastructure stripped (no Vast.ai, no R2, no Redis, no Ray, no log monitors).

## Scope

**In scope:**
- Shape file import (geojson / shp / kml / gpkg) — same UX as VPS_Scraper
- Leaflet map render of imported polygons
- Click polygon → scrape **first non-empty Look Around tile covering polygon** → **first pano on that tile**
- Download all 6 faces at zoom 0 (full resolution, ~5632 px wide each)
- Decode HEIC → JPG via `pillow-heif`
- Write per-pano `meta.csv` with standard columns
- Live progress over Socket.IO
- Browser gallery showing the 6 face JPGs + CSV download link

**Out of scope (explicit):**
- Multi-pano / batch / area-fill scraping
- Cloud upload / R2 / S3
- Vast.ai workers
- Equirectangular reprojection
- Concurrent multi-session scraping
- Deduplication / resume / state tracking
- Authentication (local-only, single user)

## Architecture

New folder `LookAround_Scraper/` sibling to `VPS_Scraper/`:

```
LookAround_Scraper/
├── web_app.py              # Flask + SocketIO (forked, gutted)
├── apple_scraper.py        # NEW — wraps streetlevel.lookaround
├── templates/
│   └── index.html          # Forked (gutted)
├── downloads/              # Output, gitignored
│   └── <pano_id>/
│       ├── back.jpg
│       ├── left.jpg
│       ├── front.jpg
│       ├── right.jpg
│       ├── top.jpg
│       ├── bottom.jpg
│       └── meta.csv
└── requirements.txt        # NEW — Flask, flask-socketio, geopandas, shapely, pillow-heif, requests, pycryptodome
```

`streetlevel` library is already vendored in the repo (used as `from streetlevel.lookaround import ...`).

## Components

### `apple_scraper.py`

Single public entry point:

```python
def scrape_polygon(
    polygon_coords: list[list[float]],   # [[lon, lat], ...]
    zoom: int = 0,
    out_root: str = "downloads",
    progress_cb: Callable[[str, dict], None] = None,
) -> ScrapeResult:
    """
    1. Compute z=17 tile bbox covering polygon (geo.wgs84_to_tile_coord on each vertex).
    2. Iterate tiles in row-major order (y outer, x inner, top-left → bottom-right)
       calling lookaround.get_coverage_tile() until first non-empty result.
    3. Pick pano: prefer one inside polygon (shapely point-in-polygon),
       fallback to first pano returned.
    4. mkdir downloads/<pano_id>/.
    5. For face in BACK, LEFT, FRONT, RIGHT, TOP, BOTTOM:
         heic_bytes = lookaround.get_panorama_face(pano, face, zoom, auth)
         jpg = pillow_heif decode → PIL → save downloads/<pano_id>/<name>.jpg
         progress_cb("face", {"i": idx, "total": 6, "name": name})
    6. Write meta.csv (one row per face).
    7. Return ScrapeResult(pano_id, build_id, faces=[6 paths], csv_path,
                          lat, lon, capture_date, heading).
    """
```

CSV schema (one row per face):

```
pano_id, build_id, lat, lon, capture_date, heading, pitch, roll,
coverage_type, face_index, face_name, image_path
```

### `web_app.py` (forked from VPS_Scraper)

**Kept:**
- Flask + SocketIO setup, `app.config['SECRET_KEY']`, `async_mode='threading'`
- Per-session state dict (simplified; only tracks `shapes_data`)
- `GET /` → `render_template('index.html')`
- `POST /api/load-shapes` (geopandas-based shape import — keep verbatim)

**Removed (every reference, every import, every helper):**
- `r2_storage`, `vast_manager`, `redis_queue`, `csv_splitter`, `log_monitor_web`,
  `ray_workers`, `build_tracker`, `abort_multipart`, `gsvpd/`
- `_update_tracker_json`, `_reverse_geocode`, `_geocode_shapes_batch`,
  `_polygon_centroid` (replaced by shapely), `_point_in_polygon` (replaced by shapely),
  `_point_near_polygon_m`, `_polygon_area_km2`
- All routes related to scrape orchestration, vast offers, build, audit, missing-workers
- Threads: `R2StatusMonitorThread`, `RedisQueueMonitorThread`

**Added:**
- `GET /downloads/<pano_id>/<path:filename>` → `send_from_directory("downloads/<pano_id>", filename)`
- SocketIO event `scrape_polygon` — payload `{coords: [[lon,lat],...], session_id}`. Spawns a daemon thread that calls `apple_scraper.scrape_polygon(coords, progress_cb=lambda evt, data: socketio.emit(evt, data, room=session_id))`. On done: emit `scrape_done` with `{pano_id, face_urls: [...], csv_url, lat, lon, capture_date, heading}`. On error: emit `scrape_error` with `{message}`.

### `templates/index.html` (forked)

**Kept:**
- Leaflet map setup, dark theme CSS
- Shape file `<input type="file" accept=".geojson,.json,.shp,.dbf,.shx,.prj,.zip,.kml,.gpkg" multiple>`
- `loadShapesFromGeoJSON()` rendering, polygon styling
- SocketIO client wiring, session_id generation

**Removed:**
- Country/state/city panels, Vast offer modal, build/index buttons, R2 audit, missing-workers modal, all population/area filtering UI

**Added:**
- Polygon click handler: `polygon.on('click', () => socket.emit('scrape_polygon', {coords, session_id}))`
- Result panel:
  - Status line bound to `progress` events
  - 6 `<img>` slots for face JPGs (lazy-filled from `face_urls`)
  - CSV download `<a href="...">`
  - Lat/lon/capture-date/pano-id/heading display
- Error toast on `scrape_error`

## Data flow

```
1. User uploads shape file
       │
       ▼ POST /api/load-shapes
   geopandas → 4326 GeoJSON → JSON to client
       │
       ▼
   Leaflet renders polygons

2. User clicks polygon
       │
       ▼ socket.emit('scrape_polygon', {coords, session_id})
   web_app spawns thread → apple_scraper.scrape_polygon(coords, progress_cb)
       │
       ▼ progress events:
   'progress' {step:"finding-tile"}
   'progress' {step:"pano-selected", pano_id, lat, lon}
   'progress' {step:"face", i:1..6, name}
       │
       ▼ on success:
   'scrape_done' {pano_id, face_urls[6], csv_url, lat, lon, capture_date, heading}
       │
       ▼
   Browser renders 6 imgs + csv link + metadata

   On any failure: 'scrape_error' {message} → toast
```

## Error handling

| Failure | Handling |
|---|---|
| No panos on any covering z=17 tile | `scrape_error: "no Look Around coverage in polygon"` |
| 403 on `get_panorama_face` | Retry once with fresh `Authenticator()`. On second 403, `scrape_error: "403 on face <name>"` |
| HEIC decode failure | `scrape_error: "HEIC decode failed on face <name> (bytes=<n>)"` |
| Polygon outside coverage area (China, etc.) | Same as no-panos |
| Geopandas missing on shape import | Existing VPS handler already returns 400 |
| User clicks before shape loads | Client guards: button disabled until `shapes_data` set |

## Dependencies

`requirements.txt`:
```
Flask>=2.3
Flask-SocketIO>=5.3
geopandas>=0.14
shapely>=2.0
pillow-heif>=0.16
Pillow>=10
requests>=2.31
aiohttp>=3.9
pycryptodome>=3.20   # streetlevel lookaround.auth
protobuf>=4.25       # streetlevel lookaround.proto
```

(No torch — equirectangular reprojection is out of scope.)

## Testing

Manual debug loop (sole verification):

1. `cd LookAround_Scraper && python web_app.py`
2. Open `http://localhost:5000`
3. Upload a `.geojson` polygon over a known Apple-Look-Around-covered area (e.g. SF, NYC, London)
4. Click the polygon on the map
5. Verify in browser:
   - Progress messages stream (finding-tile → pano-selected → face 1/6 … 6/6)
   - 6 face JPGs render in gallery
   - CSV link downloads valid CSV with 6 rows
6. Verify on disk:
   - `LookAround_Scraper/downloads/<pano_id>/{back,left,front,right,top,bottom}.jpg` all >100 KB at zoom 0
   - `LookAround_Scraper/downloads/<pano_id>/meta.csv` parseable, 6 rows, correct schema
7. Edge cases (manual):
   - Upload polygon over ocean / China → expect `scrape_error: "no Look Around coverage in polygon"`
   - Click polygon twice quickly → second click should still work (idempotent: same pano_id folder, files overwritten)

## Open questions / future work (NOT in this spec)

- Pano selection strategy beyond "first" (nearest-to-centroid, newest, etc.)
- Multi-pano grid scraping for an entire polygon
- HEIC kept alongside JPG (currently JPG only)
- Equirectangular reprojection
- Concurrent runs / queue
