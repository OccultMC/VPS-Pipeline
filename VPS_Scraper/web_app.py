"""
VPS Scraper — Web GUI

Flask + Socket.IO server replacing the PyQt6 desktop UI.
Each browser tab gets its own session, enabling concurrent multi-city scraping.

Usage:
    cd VPS_Scraper
    python web_app.py          # Starts on http://localhost:5000
"""
import os
import sys
import json
import uuid
import math
import re
import base64
import asyncio
import datetime
import time
import logging
import threading
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room

# ── Project imports ─────────────────────────────────────────────────────
from scraper import UnifiedScraper, UnifiedScraperConfig

try:
    from r2_storage import R2Client
    from csv_splitter import split_csv, split_csv_chunks, upload_csv_segments
    from vast_manager import VastManager, ContainerNotFoundError
    from log_monitor_web import R2StatusMonitorThread, RedisQueueMonitorThread
    from redis_queue import TaskQueue
    VPS_AVAILABLE = True
except ImportError as e:
    VPS_AVAILABLE = False
    print(f"[WARNING] VPS modules not available — import failed: {e}")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Flask + SocketIO ────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET', 'dev-secret-key')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ── Per-session state ───────────────────────────────────────────────────
sessions: Dict[str, Dict[str, Any]] = {}


# ═════════════════════════════════════════════════════════════════════════
# Utility helpers (ported from ui.py)
# ═════════════════════════════════════════════════════════════════════════

def _sanitize_geo(s):
    import re
    s = "".join(c for c in str(s) if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
    return re.sub(r'-{2,}', '-', s)


TRACKER_KEY = "status/shapes_tracker.json"


def _update_tracker_json(r2_client, country: str, state: str, city: str,
                          field: str, lat: float = None, lon: float = None):
    """
    Read-modify-write the shapes tracker JSON on R2.

    Args:
        r2_client: R2Client instance
        country, state, city: Path components
        field: Which flag to set True ('csv', 'features', 'index')
        lat, lon: Centroid coordinates (required when field='csv')
    """
    tracker = r2_client.download_json(TRACKER_KEY)  # returns {} on failure

    region_key = f"{country}/{state}/{city}"

    if region_key not in tracker:
        tracker[region_key] = {
            'lat': lat or 0,
            'lon': lon or 0,
            'csv': False,
            'features': False,
            'index': False,
        }

    tracker[region_key][field] = True
    if lat is not None and lon is not None:
        tracker[region_key]['lat'] = lat
        tracker[region_key]['lon'] = lon

    r2_client.upload_json(TRACKER_KEY, tracker)


def _reverse_geocode(lat: float, lon: float):
    import urllib.request
    CITY_FIELDS = [
        'city', 'town', 'municipality', 'city_district', 'district',
        'county', 'suburb', 'village', 'hamlet', 'locality',
        'neighbourhood', 'state_district',
    ]
    STATE_FIELDS = ['state', 'state_district', 'region']

    def _try_geocode(zoom: int):
        url = (f"https://nominatim.openstreetmap.org/reverse?"
               f"lat={lat}&lon={lon}&format=json&zoom={zoom}&addressdetails=1")
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Hypervision-VPS-Scraper/1.0 (+https://github.com/OccultMC/GeoaxisImage)'
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())

    best_country = best_state = best_city = None

    for zoom in (10, 8, 6):
        try:
            data = _try_geocode(zoom)
            addr = data.get('address', {})
            if not best_country:
                cc = addr.get('country_code', '')
                if cc:
                    best_country = cc.upper()
            if not best_state:
                for f in STATE_FIELDS:
                    val = addr.get(f, '').strip()
                    if val:
                        best_state = val
                        break
            if not best_city:
                for f in CITY_FIELDS:
                    val = addr.get(f, '').strip()
                    if val:
                        best_city = val
                        break
                if not best_city and 'display_name' in data:
                    for part in [p.strip() for p in data['display_name'].split(',')]:
                        if part and not part.replace(' ', '').isdigit() and len(part) > 1:
                            best_city = part
                            break
            if best_country and best_state and best_city:
                break
            time.sleep(1.1)
        except Exception:
            continue

    country = _sanitize_geo(best_country) if best_country else "Unknown"
    state = _sanitize_geo(best_state) if best_state else "Unknown"
    if best_city:
        city = _sanitize_geo(best_city)
    else:
        city = f"Region_{lat:.4f}_{lon:.4f}".replace('-', 'S').replace('.', 'd')
    return country, state, city


# ── Offline geocoder using worldcities.csv ─────────────────────────────

WORLDCITIES_PATH = Path(__file__).parent.parent / "Local_Stages" / "Stage_1_Dataset_Collection" / "worldcities.csv"
_worldcities_cache = None


def _load_worldcities():
    """Load worldcities.csv into memory (lazy, cached)."""
    global _worldcities_cache
    if _worldcities_cache is not None:
        return _worldcities_cache
    _worldcities_cache = []
    if not WORLDCITIES_PATH.exists():
        logger.warning(f"worldcities.csv not found at {WORLDCITIES_PATH}")
        return _worldcities_cache
    import csv as _csv
    with open(WORLDCITIES_PATH, 'r', encoding='utf-8') as f:
        for row in _csv.DictReader(f):
            try:
                pop_str = row.get('population', '0') or '0'
                pop = int(float(pop_str)) if pop_str else 0
                _worldcities_cache.append((
                    float(row['lat']), float(row['lng']),
                    row.get('city_ascii', ''), row.get('iso2', ''),
                    row.get('admin_name', ''), pop,
                ))
            except (ValueError, KeyError):
                continue
    logger.info(f"Loaded {len(_worldcities_cache)} cities from worldcities.csv")
    return _worldcities_cache


def _haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return 6371 * 2 * math.asin(min(1.0, math.sqrt(a)))


def _geocode_from_worldcities(lat: float, lon: float):
    """Find nearest city from worldcities.csv → (iso2, admin_name, city_ascii)."""
    cities = _load_worldcities()
    if not cities:
        return _reverse_geocode(lat, lon)
    best_dist = float('inf')
    best = None
    for entry in cities:
        clat, clng, ccity, ciso2, cadmin = entry[0], entry[1], entry[2], entry[3], entry[4]
        d = _haversine_km(lat, lon, clat, clng)
        if d < best_dist:
            best_dist = d
            best = (ciso2, cadmin, ccity)
    if best is None:
        return "Unknown", "Unknown", f"Region_{lat:.4f}_{lon:.4f}"
    return (
        best[0] or "Unknown",
        _sanitize_geo(best[1]) if best[1] else "Unknown",
        _sanitize_geo(best[2]) if best[2] else "Unknown",
    )


def _point_in_polygon(px, py, polygon):
    """Ray-casting point-in-polygon test. polygon = [[lng, lat], ...]."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _point_to_segment_dist_m(px, py, ax, ay, bx, by):
    """Min distance in metres from point (px,py) to segment (ax,ay)-(bx,by). All in lng/lat."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        t = 0.0
    else:
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    nx, ny = ax + t * dx, ay + t * dy
    return _haversine_km(py, px, ny, nx) * 1000.0


def _point_near_polygon_m(px, py, coords):
    """Min distance in metres from a point to the polygon boundary."""
    best = float('inf')
    n = len(coords)
    for i in range(n):
        j = (i + 1) % n
        d = _point_to_segment_dist_m(px, py, coords[i][0], coords[i][1], coords[j][0], coords[j][1])
        if d < best:
            best = d
    return best


def _approx_area_sqkm(coords):
    """Fast approximate area of a polygon in sq km (shoelace on cos-projected coords)."""
    if len(coords) < 3:
        return 0.0
    import math as _m
    lats = [c[1] for c in coords]
    mid_lat = _m.radians(sum(lats) / len(lats))
    cos_lat = _m.cos(mid_lat)
    pts = [(c[0] * 111.320 * cos_lat, c[1] * 110.540) for c in coords]
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def _geocode_shapes_batch(all_coords_list):
    """
    Batch-geocode shapes using worldcities.csv with three-pass strategy:
    Pass 1: For each shape find the highest-population city point inside it.
            Track which city indices got claimed.
    Pass 2: For orphan shapes, check unclaimed city points within 150m of the
            polygon boundary (handles rivers/water exclusions).
            Process largest shapes first so big landmasses beat tiny islands.
    Final fallback: nearest city to centroid.
    Returns list of (iso2, admin_name, city_ascii) tuples.
    """
    cities = _load_worldcities()
    n_shapes = len(all_coords_list)
    results = [None] * n_shapes

    if not cities:
        for i, coords in enumerate(all_coords_list):
            lat, lng = _polygon_centroid(coords)
            results[i] = _reverse_geocode(lat, lng)
        return results

    # Precompute bounding boxes and areas
    bboxes = []
    areas = []
    for coords in all_coords_list:
        lngs = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        bboxes.append((min(lngs), max(lngs), min(lats), max(lats)))
        areas.append(_approx_area_sqkm(coords))

    # ── Pass 1: find best city for each shape (inside OR within 150m) ──
    # For each shape, consider all city points that are either inside the polygon
    # or within 150m of its boundary. Pick the one with highest population.
    # This handles cities whose points fall on rivers/water cut out of shapes.
    NEAR_THRESHOLD_M = 150.0
    shape_best = [None] * n_shapes  # (city_index, population) per shape

    for si, coords in enumerate(all_coords_list):
        min_lng, max_lng, min_lat, max_lat = bboxes[si]
        buf = 0.002  # ~200m in degrees for bbox expansion
        best_idx = -1
        best_pop = -1
        for ci, entry in enumerate(cities):
            clat, clng = entry[0], entry[1]
            # Fast bbox reject (with buffer for near-boundary points)
            if clng < min_lng - buf or clng > max_lng + buf or clat < min_lat - buf or clat > max_lat + buf:
                continue
            cpop = entry[5]
            if cpop <= best_pop:
                continue
            # Check if inside (fast) or within 150m of boundary
            if _point_in_polygon(clng, clat, coords):
                best_pop = cpop
                best_idx = ci
            else:
                dist = _point_near_polygon_m(clng, clat, coords)
                if dist <= NEAR_THRESHOLD_M:
                    best_pop = cpop
                    best_idx = ci
        if best_idx >= 0:
            shape_best[si] = (best_idx, best_pop)

    # ── Resolve conflicts: if multiple shapes want the same city, largest shape wins ──
    # Build city_index → list of (shape_index, population) mappings
    city_to_shapes = {}
    for si, match in enumerate(shape_best):
        if match is None:
            continue
        ci, pop = match
        if ci not in city_to_shapes:
            city_to_shapes[ci] = []
        city_to_shapes[ci].append(si)

    claimed_cities = set()
    evicted = set()  # shapes that lost their city in conflict resolution

    for ci, shape_list in city_to_shapes.items():
        if len(shape_list) == 1:
            # No conflict — assign directly
            si = shape_list[0]
            e = cities[ci]
            results[si] = (
                e[3] or "Unknown",
                _sanitize_geo(e[4]) if e[4] else "Unknown",
                _sanitize_geo(e[2]) if e[2] else "Unknown",
            )
            claimed_cities.add(ci)
        else:
            # Conflict — largest shape wins
            winner = max(shape_list, key=lambda s: areas[s])
            e = cities[ci]
            results[winner] = (
                e[3] or "Unknown",
                _sanitize_geo(e[4]) if e[4] else "Unknown",
                _sanitize_geo(e[2]) if e[2] else "Unknown",
            )
            claimed_cities.add(ci)
            for si in shape_list:
                if si != winner:
                    evicted.add(si)

    # ── Pass 2: evicted/unmatched shapes — find next best unclaimed city ──
    retry = [si for si in range(n_shapes) if results[si] is None]
    retry.sort(key=lambda si: areas[si], reverse=True)  # largest first

    for si in retry:
        coords = all_coords_list[si]
        min_lng, max_lng, min_lat, max_lat = bboxes[si]
        buf = 0.002
        best_idx = -1
        best_pop = -1
        for ci, entry in enumerate(cities):
            if ci in claimed_cities:
                continue
            clat, clng = entry[0], entry[1]
            if clng < min_lng - buf or clng > max_lng + buf or clat < min_lat - buf or clat > max_lat + buf:
                continue
            cpop = entry[5]
            if cpop <= best_pop:
                continue
            if _point_in_polygon(clng, clat, coords):
                best_pop = cpop
                best_idx = ci
            else:
                dist = _point_near_polygon_m(clng, clat, coords)
                if dist <= NEAR_THRESHOLD_M:
                    best_pop = cpop
                    best_idx = ci
        if best_idx >= 0:
            e = cities[best_idx]
            results[si] = (
                e[3] or "Unknown",
                _sanitize_geo(e[4]) if e[4] else "Unknown",
                _sanitize_geo(e[2]) if e[2] else "Unknown",
            )
            claimed_cities.add(best_idx)

    # ── Final fallback: nearest city to centroid ──
    for si, coords in enumerate(all_coords_list):
        if results[si] is not None:
            continue
        lat, lng = _polygon_centroid(coords)
        best_dist = float('inf')
        best_entry = None
        for ci, entry in enumerate(cities):
            d = _haversine_km(lat, lng, entry[0], entry[1])
            if d < best_dist:
                best_dist = d
                best_entry = entry
        if best_entry:
            results[si] = (
                best_entry[3] or "Unknown",
                _sanitize_geo(best_entry[4]) if best_entry[4] else "Unknown",
                _sanitize_geo(best_entry[2]) if best_entry[2] else "Unknown",
            )
        else:
            results[si] = ("Unknown", "Unknown", f"Region_{lat:.4f}_{lng:.4f}")

    return results


def _geocode_shape_by_population(coords):
    """Single-shape geocode — delegates to batch with one shape."""
    return _geocode_shapes_batch([coords])[0]


def _polygon_centroid(coords):
    """
    Compute the true geometric centroid (center of mass) of a polygon.
    coords = [[lng, lat], ...].  Returns (lat, lng).
    Uses the signed-area formula so irregular shapes balance correctly.
    """
    if not coords:
        return 0.0, 0.0
    n = len(coords)
    if n < 3:
        return sum(c[1] for c in coords) / n, sum(c[0] for c in coords) / n

    # Ensure ring is closed
    pts = list(coords)
    if pts[0] != pts[-1]:
        pts.append(pts[0])

    signed_area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(len(pts) - 1):
        x0, y0 = pts[i][0], pts[i][1]      # lng, lat
        x1, y1 = pts[i + 1][0], pts[i + 1][1]
        cross = x0 * y1 - x1 * y0
        signed_area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross

    if abs(signed_area) < 1e-12:
        # Degenerate polygon — fall back to arithmetic mean
        return sum(c[1] for c in coords) / n, sum(c[0] for c in coords) / n

    signed_area *= 0.5
    cx /= (6.0 * signed_area)
    cy /= (6.0 * signed_area)
    return cy, cx  # lat, lng


def _get_session(sid: str) -> Dict[str, Any]:
    """Get or create session state."""
    if sid not in sessions:
        sessions[sid] = {
            'scraper': None,
            'scraper_thread': None,
            'vps_context': {},
            'vast_manager': None,
            'log_monitor': None,
            'builder_monitor': None,
            'shapes_data': [],
            'status': 'Ready',
            'pending_offers': None,
            # Job state: idle | scraping | searching_offers | awaiting_offer_selection
            #          | creating_instances | monitoring | building_index | done | error
            'job_state': 'idle',
        }
    return sessions[sid]


# ═════════════════════════════════════════════════════════════════════════
# Thread wrappers (plain threading.Thread instead of QThread)
# ═════════════════════════════════════════════════════════════════════════

class ScraperThread(threading.Thread):
    """Run the async scraper in a background thread."""

    def __init__(self, session_id, scraper, mode, **kwargs):
        super().__init__(daemon=True)
        self.session_id = session_id
        self.scraper = scraper
        self.mode = mode
        self.kwargs = kwargs
        self._cancelled = False

    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            if self.mode == "polygon":
                result = loop.run_until_complete(
                    self.scraper.scrape_area_two_phase(self.kwargs['polygon_coords'])
                )
            elif self.mode == "multi_polygon":
                result = loop.run_until_complete(
                    self.scraper.scrape_multiple_polygons_two_phase(
                        self.kwargs['polygon_list'],
                        csv_paths=self.kwargs.get('csv_paths'),
                        images_dirs=self.kwargs.get('images_dirs'),
                        merge_csv=self.kwargs.get('merge_csv', False)
                    )
                )
            else:
                result = []

            loop.close()
            sess = _get_session(self.session_id)
            if sess.get('job_state') == 'scraping':
                sess['job_state'] = 'idle'
            socketio.emit('scrape_finished', {
                'total_written': self.scraper.total_written,
                'total_images': self.scraper.total_images,
            }, room=self.session_id)
        except Exception as e:
            import traceback
            traceback.print_exc()
            _get_session(self.session_id)['job_state'] = 'error'
            socketio.emit('scrape_error', {'error': str(e)}, room=self.session_id)

    def cancel(self):
        self._cancelled = True
        self.scraper.cancel()


class VPSDeployThread(threading.Thread):
    """Geocode + R2 check in background."""

    def __init__(self, session_id, context, mode="full"):
        super().__init__(daemon=True)
        self.session_id = session_id
        self.context = context
        self.mode = mode

    def run(self):
        try:
            if self.mode == "full":
                self._do_geocode_and_r2_check()
            elif self.mode == "provision":
                self._do_search_offers()
        except Exception as e:
            sess = _get_session(self.session_id)
            sess['job_state'] = 'error'
            socketio.emit('vps_error', {'error': str(e)}, room=self.session_id)

    def _do_geocode_and_r2_check(self):
        ctx = self.context
        if ctx.get('country') and ctx.get('state') and ctx.get('city'):
            country, state, city = ctx['country'], ctx['state'], ctx['city']
        else:
            coords = ctx['first_coords']
            socketio.emit('vps_status', {'message': 'Geocoding region...'}, room=self.session_id)
            lats = [c[0] for c in coords]
            lons = [c[1] for c in coords]
            country, state, city = _reverse_geocode(
                sum(lats) / len(lats), sum(lons) / len(lons)
            )

        ctx['country'] = country
        ctx['state'] = state
        ctx['city'] = city
        socketio.emit('vps_geocode_done', {
            'country': country, 'state': state, 'city': city
        }, room=self.session_id)

        # R2 check
        socketio.emit('vps_status', {'message': 'Checking R2 for existing CSV segments...'}, room=self.session_id)
        try:
            r2 = R2Client(ctx['r2_account'], ctx['r2_access'], ctx['r2_secret'], ctx['r2_bucket'])
            prefix = f"CSV/{country}/{state}/{city}/"
            files = r2.list_files(prefix)
            r2_csv_files = [f for f in files if f['key'].endswith(".csv")]

            if r2_csv_files:
                segment_count = len(r2_csv_files)
                completed = self._scan_completed_features(r2, country, state, city, segment_count)
                ctx['completed_worker_indices'] = completed
                socketio.emit('vps_r2_checked', {
                    'segment_count': segment_count,
                    'completed': completed,
                }, room=self.session_id)
                return

        except Exception as e:
            logger.warning(f"R2 Check failed: {e}")

        ctx['completed_worker_indices'] = []
        socketio.emit('vps_r2_checked', {'segment_count': 0, 'completed': []}, room=self.session_id)

    def _scan_completed_features(self, r2, country, state, city, total_workers):
        features_prefix = f"Features/{country}/{state}/{city}/"
        try:
            files = r2.list_files(features_prefix)
        except Exception:
            return []

        npy_indices = set()
        meta_indices = set()
        for f in files:
            fname = f['key'].split('/')[-1]
            try:
                if fname.startswith("Metadata_") and fname.endswith(".jsonl"):
                    inner = fname[len("Metadata_"):-len(".jsonl")]
                    base, total_str = inner.rsplit('.', 1)
                    _city, idx_str = base.rsplit('_', 1)
                    if idx_str.isdigit() and int(total_str) == total_workers:
                        meta_indices.add(int(idx_str))
                elif fname.endswith(".npy"):
                    base, total_str = fname[:-4].rsplit('.', 1)
                    _city, idx_str = base.rsplit('_', 1)
                    if idx_str.isdigit() and int(total_str) == total_workers:
                        npy_indices.add(int(idx_str))
            except (ValueError, IndexError):
                continue
        return sorted(npy_indices & meta_indices)

    def _do_search_offers(self):
        ctx = self.context
        socketio.emit('vps_status', {'message': 'Searching Vast.ai offers...'}, room=self.session_id)
        manager = VastManager(api_key=ctx['vast_key'])
        geo_filter = ctx.get('geo_filter', '').strip()
        geo_region = None
        if geo_filter:
            codes = [c.strip().upper() for c in geo_filter.replace(',', ' ').split() if c.strip()]
            if codes:
                geo_region = " ".join(codes)
        offers = manager.search_offers(
            gpu_type=ctx.get('gpu_type', ''),
            region=geo_region,
            min_disk_gb=ctx.get('disk_gb', 100),
            max_price=ctx.get('max_price'),
            min_ram_gb=ctx.get('min_vram_gb', 21),
        )
        ctx['_vast_manager'] = manager

        sess = _get_session(self.session_id)
        sess['vast_manager'] = manager

        worker_indices = ctx.get('worker_indices_to_deploy',
                                  list(range(1, ctx.get('actual_workers', ctx.get('num_workers', 10)) + 1)))
        total_workers = ctx.get('total_workers', ctx.get('actual_workers', len(worker_indices)))
        offers_data = {
            'offers': offers if offers else [],
            'worker_indices': worker_indices,
            'total_workers': total_workers,
        }
        # Persist so frontend can recover after websocket reconnect
        sess = _get_session(self.session_id)
        sess['pending_offers'] = offers_data
        sess['job_state'] = 'awaiting_offer_selection'
        socketio.emit('vps_offers_found', offers_data, room=self.session_id)


class InstanceCreateThread(threading.Thread):
    """Create Vast.ai instances in background."""

    def __init__(self, session_id, vast_manager, selected_offers, worker_indices, total_workers, ctx):
        super().__init__(daemon=True)
        self.session_id = session_id
        self.vast_manager = vast_manager
        self.selected_offers = selected_offers
        self.worker_indices = worker_indices
        self.total_workers = total_workers
        self.ctx = ctx

    def run(self):
        try:
            r2 = R2Client(
                account_id=self.ctx['r2_account'],
                access_key_id=self.ctx['r2_access'],
                secret_access_key=self.ctx['r2_secret'],
                bucket_name=self.ctx['r2_bucket'],
            )
            features_prefix = f"Features/{self.ctx['country']}/{self.ctx['state']}/{self.ctx['city']}"
            instance_worker_map = {}

            # Show all workers in the table immediately
            socketio.emit('vps_creation_started', {
                'worker_indices': self.worker_indices,
            }, room=self.session_id)

            # Track offers claimed across ALL workers so each gets a different machine
            globally_used_offer_ids = set()

            for i, worker_idx in enumerate(self.worker_indices):
                env_vars = {
                    'R2_ACCOUNT_ID': self.ctx['r2_account'],
                    'R2_ACCESS_KEY_ID': self.ctx['r2_access'],
                    'R2_SECRET_ACCESS_KEY': self.ctx['r2_secret'],
                    'R2_BUCKET_NAME': self.ctx['r2_bucket'],
                    'CSV_BUCKET_PREFIX': f"CSV/{self.ctx['country']}/{self.ctx['state']}/{self.ctx['city']}",
                    'FEATURES_BUCKET_PREFIX': features_prefix,
                    'CITY_NAME': self.ctx['city'],
                    'VAST_API_KEY': self.ctx['vast_key'],
                    'REDIS_URL': self.ctx.get('redis_url', ''),
                    'REDIS_TOKEN': self.ctx.get('redis_token', ''),
                    'REGION': self.ctx.get('region', f"{self.ctx['country']}/{self.ctx['state']}/{self.ctx['city']}"),
                }

                # Pick an offer not yet claimed by another worker
                offer = None
                for candidate in self.selected_offers:
                    if candidate['id'] not in globally_used_offer_ids:
                        offer = candidate
                        break
                if offer is None:
                    offer = self.selected_offers[i % len(self.selected_offers)]
                globally_used_offer_ids.add(offer['id'])

                label = f"({i+1}/{len(self.worker_indices)})"
                instance_id = self.vast_manager.create_instance(
                    offer_id=offer['id'],
                    docker_image=self.ctx['docker_image'],
                    env_vars=env_vars,
                    disk_gb=self.ctx.get('disk_gb', 100),
                    onstart_cmd="bash /app/entrypoint.sh",
                    template_hash=VastManager.GEOAXIS_TEMPLATE_HASH,
                )

                if instance_id:
                    instance_worker_map[instance_id] = worker_idx
                    socketio.emit('vps_worker_created', {
                        'worker_idx': worker_idx,
                        'instance_id': instance_id,
                    }, room=self.session_id)
                    socketio.emit('vps_status', {
                        'message': f"Worker {worker_idx} {label}: created instance {instance_id} on {offer['gpu_name']}"
                    }, room=self.session_id)
                else:
                    socketio.emit('vps_worker_create_failed', {
                        'worker_idx': worker_idx,
                        'message': f"Failed to create instance on offer {offer['id']}",
                    }, room=self.session_id)
                    socketio.emit('vps_status', {
                        'message': f"Worker {worker_idx} {label} — creation FAILED"
                    }, room=self.session_id)

            socketio.emit('vps_instances_created', {
                'instance_worker_map': instance_worker_map,
                'total_workers': len(self.worker_indices),
            }, room=self.session_id)

            # Auto-start monitoring immediately
            if instance_worker_map:
                self._start_monitoring(r2, instance_worker_map)

        except Exception as e:
            _get_session(self.session_id)['job_state'] = 'error'
            socketio.emit('vps_error', {'error': str(e)}, room=self.session_id)

    def _start_monitoring(self, r2, instance_worker_map):
        """Start Redis queue monitor + Vast.ai status polling after instances are created."""
        sid = self.session_id
        sess = _get_session(sid)

        def on_progress(progress_data):
            socketio.emit('vps_queue_progress', progress_data, room=sid)
            # Emit per-worker progress so the worker table updates
            worker_statuses = progress_data.get('worker_statuses', {})
            for worker_id, ws in worker_statuses.items():
                worker_idx = instance_worker_map.get(worker_id)
                if worker_idx is not None:
                    socketio.emit('vps_worker_progress', {
                        'worker_idx': worker_idx,
                        'processed': ws.get('processed', 0),
                        'total': ws.get('total', 0),
                        'eta': ws.get('eta', 0),
                        'speed': ws.get('speed', 0),
                        'status': ws.get('status', 'UNKNOWN'),
                    }, room=sid)

        def on_complete():
            socketio.emit('vps_job_complete', {}, room=sid)
            try:
                country = self.ctx.get('country', '')
                state = self.ctx.get('state', '')
                city = self.ctx.get('city', '')
                if country and state and city:
                    _update_tracker_json(r2, country, state, city, 'features')
            except Exception as e:
                logger.warning(f"Failed to update tracker after features: {e}")

        def on_log(message):
            socketio.emit('vps_log', {'message': message}, room=sid)

        # Use Redis-based queue monitor
        tq = sess.get('task_queue')
        region = self.ctx.get('region', f"{self.ctx['country']}/{self.ctx['state']}/{self.ctx['city']}")

        if tq is None:
            redis_url = self.ctx.get('redis_url', os.environ.get('REDIS_URL', ''))
            redis_token = self.ctx.get('redis_token', os.environ.get('REDIS_TOKEN', ''))
            if redis_url and redis_token:
                tq = TaskQueue(redis_url, redis_token)
                sess['task_queue'] = tq

        if tq:
            monitor = RedisQueueMonitorThread(
                task_queue=tq,
                region=region,
                instance_ids=list(instance_worker_map.keys()),
                poll_interval=10.0,
                stale_interval=60.0,
                on_progress=on_progress,
                on_complete=on_complete,
                on_log=on_log,
                vast_manager=self.vast_manager,
            )
        else:
            # Fallback to R2-based monitoring if no Redis
            monitor = R2StatusMonitorThread(
                r2_client=r2, city_name=self.ctx['city'],
                instance_worker_map=instance_worker_map,
                poll_interval=10.0,
                on_progress=on_progress,
                on_worker_finished=lambda w, s: None,
                on_log_message=on_log,
                vast_manager=self.vast_manager,
            )

        sess['log_monitor'] = monitor
        sess['job_state'] = 'monitoring'
        monitor.start()


class AutoScrapeThread(threading.Thread):
    """Process multiple shapes: per-shape mode or batch mode (shared workers)."""

    def __init__(self, session_id, shapes_queue, ctx):
        super().__init__(daemon=True)
        self.session_id = session_id
        self.shapes_queue = shapes_queue
        self.ctx = ctx
        self._cancelled = False
        self._offer_event = threading.Event()
        self._selected_offers = None

    def cancel(self):
        self._cancelled = True
        self._offer_event.set()

    def on_offers_selected(self, offers):
        self._selected_offers = offers
        self._offer_event.set()

    def _emit(self, msg, idx=None):
        socketio.emit('auto_scrape_status', {'index': idx, 'message': msg}, room=self.session_id)

    def run(self):
        try:
            if self.ctx.get('share_workers'):
                self._run_batch()
            else:
                self._run_sequential()
            if not self._cancelled:
                socketio.emit('auto_scrape_complete', {
                    'total': len(self.shapes_queue)}, room=self.session_id)
            _get_session(self.session_id)['job_state'] = 'idle'
        except Exception as e:
            import traceback
            traceback.print_exc()
            socketio.emit('auto_scrape_error', {'error': str(e)}, room=self.session_id)
            _get_session(self.session_id)['job_state'] = 'error'

    # ── Helpers ───────────────────────────────────────────────────────────

    def _make_r2(self):
        return R2Client(self.ctx['r2_account'], self.ctx['r2_access'],
                        self.ctx['r2_secret'], self.ctx['r2_bucket'])

    def _get_redis(self):
        redis_url = self.ctx.get('redis_url') or os.environ.get('REDIS_URL', '')
        redis_token = self.ctx.get('redis_token') or os.environ.get('REDIS_TOKEN', '')
        if not redis_url or not redis_token:
            socketio.emit('auto_scrape_error', {
                'error': 'REDIS_URL/REDIS_TOKEN not set in .env'
            }, room=self.session_id)
            return None, None, None
        self.ctx['redis_url'] = redis_url
        self.ctx['redis_token'] = redis_token
        return TaskQueue(redis_url, redis_token), redis_url, redis_token

    def _scrape_shape_csv(self, idx, shape, r2):
        """Scrape a single shape and upload chunks to R2.

        Returns (country, state, city, total_chunks, total_panos) or None if skipped.
        """
        sid = self.session_id
        total = len(self.shapes_queue)
        country, state, city = shape['country'], shape['state'], shape['city']
        region = f"{country}/{state}/{city}"
        coords = shape['coords']
        tag = f"[{idx + 1}/{total}]"

        socketio.emit('auto_scrape_shape_start', {
            'index': idx, 'total': total, 'region': region,
        }, room=sid)

        # ── R2 check ─────────────────────────────────────────────────────
        csv_prefix = f"CSV/{country}/{state}/{city}/"
        existing = []
        try:
            existing = [f for f in r2.list_files(csv_prefix) if f['key'].endswith('.csv')]
        except Exception:
            pass

        if existing:
            chunk_count = len(existing)
            self._emit(f'{tag} Found {chunk_count} existing chunks for {region}', idx)
            socketio.emit('auto_scrape_shape_done', {
                'index': idx, 'total': total, 'region': region,
            }, room=sid)
            return (country, state, city, chunk_count, chunk_count * 1000)

        # ── Scrape locally ────────────────────────────────────────────────
        self._emit(f'{tag} Scraping CSV for {region}...', idx)
        config = UnifiedScraperConfig(concurrency=1000, proxy_file=None)
        scraper = UnifiedScraper(config)
        scraper.set_progress_callback(
            lambda c, t: self._emit(f'{tag} Scraping {region}: {c}/{t} tiles', idx))
        scraper.set_point_callback(
            lambda lat, lon, pid: socketio.emit(
                'point_found', {'lat': lat, 'lon': lon, 'panoid': pid}, room=sid))

        project_root = Path(__file__).parent.parent
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_dir = project_root / "Output" / "CSV" / country / state / city
        csv_path = str(csv_dir / f"{ts}.csv")
        scraper.init_csv(csv_path)
        scraper.init_images_dir(
            str(project_root / "Output" / "Images" / country / state / city))

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(scraper.scrape_area_two_phase(coords))
        loop.close()

        if scraper.csv_file_handle and not scraper.csv_file_handle.closed:
            scraper.csv_file_handle.flush()
            scraper.csv_file_handle.close()

        total_panos = scraper.total_written
        if total_panos == 0:
            self._emit(f'{tag} No panoramas in {region}, skipping', idx)
            socketio.emit('auto_scrape_shape_done', {
                'index': idx, 'total': total, 'region': region, 'skipped': True,
            }, room=sid)
            return None

        # ── Split + upload ────────────────────────────────────────────────
        self._emit(f'{tag} Splitting {total_panos} panos...', idx)
        chunks = split_csv_chunks(
            csv_path, chunk_size=1000, city_name=city,
            output_dir=str(Path(csv_path).parent / "chunks"))
        if not chunks:
            self._emit(f'{tag} Split failed, skipping', idx)
            return None

        total_chunks = len(chunks)
        self._emit(f'{tag} Uploading {total_chunks} chunks to R2...', idx)
        upload_csv_segments(chunks, r2, country, state, city)

        try:
            with open(chunks[0], 'r', encoding='utf-8') as fh:
                fh.readline()
                parts = fh.readline().split(',')
                _update_tracker_json(r2, country, state, city, 'csv',
                                     float(parts[1]), float(parts[2]))
        except Exception:
            pass

        socketio.emit('auto_scrape_shape_done', {
            'index': idx, 'total': total, 'region': region,
        }, room=sid)
        return (country, state, city, total_chunks, total_panos)

    def _search_and_deploy(self, num_workers, batch_region, redis_url, redis_token,
                           env_overrides=None):
        """Search offers, show modal, deploy workers. Returns instance_worker_map or None."""
        sid = self.session_id
        manager = VastManager(api_key=self.ctx['vast_key'])
        geo_filter = self.ctx.get('geo_filter', '').strip()
        geo_region = None
        if geo_filter:
            codes = [c.strip().upper() for c in geo_filter.replace(',', ' ').split()
                     if c.strip()]
            if codes:
                geo_region = " ".join(codes)

        offers = manager.search_offers(
            gpu_type=self.ctx.get('gpu_type', ''),
            region=geo_region,
            min_disk_gb=self.ctx.get('disk_gb', 100),
            max_price=self.ctx.get('max_price'),
            min_ram_gb=self.ctx.get('min_vram_gb', 21),
        )
        if not offers:
            socketio.emit('auto_scrape_error', {
                'error': f'No Vast.ai offers found'}, room=sid)
            return None, None

        socketio.emit('auto_scrape_offers', {
            'offers': offers, 'num_needed': num_workers,
            'shape_index': -1, 'region': batch_region,
        }, room=sid)
        _get_session(sid)['job_state'] = 'awaiting_offer_selection'
        self._offer_event.wait()
        self._offer_event.clear()
        if self._cancelled or not self._selected_offers:
            return None, None

        # ── Create instances ──────────────────────────────────────────────
        self._emit(f'Creating {num_workers} workers...')
        base_env = {
            'R2_ACCOUNT_ID': self.ctx['r2_account'],
            'R2_ACCESS_KEY_ID': self.ctx['r2_access'],
            'R2_SECRET_ACCESS_KEY': self.ctx['r2_secret'],
            'R2_BUCKET_NAME': self.ctx['r2_bucket'],
            'VAST_API_KEY': self.ctx['vast_key'],
            'REDIS_URL': redis_url,
            'REDIS_TOKEN': redis_token,
            'REGION': batch_region,
        }
        if env_overrides:
            base_env.update(env_overrides)

        used_offers = set()
        instance_worker_map = {}
        for wi in range(num_workers):
            if self._cancelled:
                return None, None
            offer = None
            for c in self._selected_offers:
                if c['id'] not in used_offers:
                    offer = c
                    break
            if offer is None:
                offer = self._selected_offers[wi % len(self._selected_offers)]
            used_offers.add(offer['id'])

            iid = manager.create_instance(
                offer_id=offer['id'],
                docker_image=self.ctx.get('docker_image', ''),
                env_vars=base_env,
                disk_gb=self.ctx.get('disk_gb', 100),
                onstart_cmd="bash /app/entrypoint.sh",
                template_hash=VastManager.GEOAXIS_TEMPLATE_HASH,
            )
            if iid:
                instance_worker_map[iid] = wi + 1
                self._emit(f'Worker {wi + 1}: instance {iid}')

        if not instance_worker_map:
            socketio.emit('auto_scrape_error', {
                'error': 'Failed to create any instances'}, room=sid)
            return None, None

        return instance_worker_map, manager

    def _monitor_queue(self, tq, region, instance_worker_map, manager, shape_index=-1):
        """Monitor Redis queue until complete."""
        sid = self.session_id
        socketio.emit('auto_scrape_workers_started', {
            'shape_index': shape_index, 'region': region,
            'instance_worker_map': {str(k): v for k, v in instance_worker_map.items()},
            'num_workers': len(instance_worker_map),
        }, room=sid)

        monitor = RedisQueueMonitorThread(
            task_queue=tq, region=region,
            instance_ids=list(instance_worker_map.keys()),
            poll_interval=10.0, stale_interval=60.0,
            on_progress=lambda p: socketio.emit(
                'auto_scrape_progress', {**p, 'shape_index': shape_index}, room=sid),
            on_complete=lambda: None,
            on_log=lambda msg: self._emit(msg),
            vast_manager=manager,
        )
        monitor.start()

        while not self._cancelled:
            time.sleep(15)
            try:
                p = tq.get_progress(region)
                if (p['todo'] == 0 and p['active'] == 0
                        and p['done'] >= p['total'] and p['total'] > 0):
                    break
            except Exception:
                pass

        monitor.stop()

    # ── Sequential mode (share_workers OFF) ───────────────────────────────

    def _run_sequential(self):
        total = len(self.shapes_queue)
        r2 = self._make_r2()
        tq, redis_url, redis_token = self._get_redis()
        if tq is None:
            return

        for i, shape in enumerate(self.shapes_queue):
            if self._cancelled:
                break
            result = self._scrape_shape_csv(i, shape, r2)
            if result is None:
                continue
            country, state, city, total_chunks, total_panos = result
            region = f"{country}/{state}/{city}"

            # Init per-shape Redis queue
            chunk_ids = [f"chunk_{j + 1:04d}" for j in range(total_chunks)]
            tq.init_job(region, chunk_ids, total_panos, city)
            self._emit(f'[{i+1}/{total}] Redis queue: {total_chunks} chunks', i)

            # Worker count
            PANOS_PER_WORKER = 50000
            if self.ctx.get('override_workers'):
                num_workers = self.ctx['num_workers']
            else:
                num_workers = max(1, math.ceil(total_panos / PANOS_PER_WORKER))

            # Deploy
            env = {
                'CSV_BUCKET_PREFIX': f"CSV/{country}/{state}/{city}",
                'FEATURES_BUCKET_PREFIX': f"Features/{country}/{state}/{city}",
                'CITY_NAME': city,
            }
            if i == 0:
                iwm, manager = self._search_and_deploy(
                    num_workers, region, redis_url, redis_token, env)
            else:
                # Auto-pick cheapest for subsequent shapes
                manager = VastManager(api_key=self.ctx['vast_key'])
                geo_filter = self.ctx.get('geo_filter', '').strip()
                geo_region = None
                if geo_filter:
                    codes = [c.strip().upper() for c in geo_filter.replace(',', ' ').split()
                             if c.strip()]
                    if codes:
                        geo_region = " ".join(codes)
                offers = manager.search_offers(
                    gpu_type=self.ctx.get('gpu_type', ''),
                    region=geo_region,
                    min_disk_gb=self.ctx.get('disk_gb', 100),
                    max_price=self.ctx.get('max_price'),
                    min_ram_gb=self.ctx.get('min_vram_gb', 21),
                )
                ranked = sorted(offers, key=lambda o: o.get('price_per_hr', 999))
                self._selected_offers = ranked[:min(num_workers, len(ranked))]
                iwm, manager = self._search_and_deploy(
                    num_workers, region, redis_url, redis_token, env)
            if iwm is None:
                continue

            self._monitor_queue(tq, region, iwm, manager, i)

            try:
                _update_tracker_json(r2, country, state, city, 'features')
            except Exception:
                pass

    # ── Batch mode (share_workers ON) ─────────────────────────────────────

    def _run_batch(self):
        """Scrape all shapes → one Redis queue → shared workers."""
        sid = self.session_id
        total = len(self.shapes_queue)
        r2 = self._make_r2()

        # ── Phase 1: Scrape all shapes ────────────────────────────────────
        self._emit('Phase 1: Scraping all shapes...')
        shape_results = []  # [(country, state, city, total_chunks, total_panos)]

        for i, shape in enumerate(self.shapes_queue):
            if self._cancelled:
                return
            result = self._scrape_shape_csv(i, shape, r2)
            if result is not None:
                shape_results.append(result)

        if not shape_results or self._cancelled:
            self._emit('No shapes to process')
            return

        # ── Phase 2: Init unified Redis queue ─────────────────────────────
        self._emit('Phase 2: Building unified queue...')
        tq, redis_url, redis_token = self._get_redis()
        if tq is None:
            return

        batch_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_region = f"batch/{batch_ts}"

        # Build global chunk list with per-chunk metadata
        global_idx = 0
        all_chunk_ids = []
        chunk_meta = {}
        grand_total_panos = 0

        for country, state, city, chunk_count, pano_count in shape_results:
            csv_prefix = f"CSV/{country}/{state}/{city}"
            features_prefix = f"Features/{country}/{state}/{city}"
            for j in range(1, chunk_count + 1):
                global_idx += 1
                cid = f"chunk_{global_idx:04d}"
                all_chunk_ids.append(cid)
                chunk_meta[cid] = f"{csv_prefix}|{features_prefix}|{city}|{chunk_count}|{j}"
            grand_total_panos += pano_count

        tq.init_job(batch_region, all_chunk_ids, grand_total_panos, f"batch_{batch_ts}")
        tq.set_batch_meta(batch_region, chunk_meta)

        cities_str = ", ".join(f"{c}/{s}/{ci}" for c, s, ci, _, _ in shape_results)
        self._emit(
            f'Queue ready: {len(all_chunk_ids)} chunks from {len(shape_results)} shapes '
            f'({cities_str})')

        # ── Phase 3: Deploy workers ───────────────────────────────────────
        self._emit('Phase 3: Deploying shared workers...')
        PANOS_PER_WORKER = 50000
        if self.ctx.get('override_workers'):
            num_workers = self.ctx['num_workers']
        else:
            num_workers = max(1, math.ceil(grand_total_panos / PANOS_PER_WORKER))

        # Use first shape's paths as env var defaults (overridden per-chunk by batch meta)
        first = shape_results[0]
        env = {
            'CSV_BUCKET_PREFIX': f"CSV/{first[0]}/{first[1]}/{first[2]}",
            'FEATURES_BUCKET_PREFIX': f"Features/{first[0]}/{first[1]}/{first[2]}",
            'CITY_NAME': first[2],
        }
        iwm, manager = self._search_and_deploy(
            num_workers, batch_region, redis_url, redis_token, env)
        if iwm is None:
            return

        # ── Phase 4: Monitor ──────────────────────────────────────────────
        self._emit(f'Phase 4: Monitoring {len(iwm)} workers across {len(all_chunk_ids)} chunks...')
        self._monitor_queue(tq, batch_region, iwm, manager)

        # Update tracker for each shape
        for country, state, city, _, _ in shape_results:
            try:
                _update_tracker_json(r2, country, state, city, 'features')
            except Exception:
                pass


class BuilderSearchThread(threading.Thread):
    """Search for builder-grade VPS offers."""

    def __init__(self, session_id, vast_key, max_price, min_disk_gb):
        super().__init__(daemon=True)
        self.session_id = session_id
        self.vast_key = vast_key
        self.max_price = max_price
        self.min_disk_gb = min_disk_gb
        self._vast_manager = None

    def run(self):
        try:
            manager = VastManager(api_key=self.vast_key)
            self._vast_manager = manager
            query_parts = [
                f"disk_space>={self.min_disk_gb}",
                "cpu_ram>=128",
                "rentable=true",
            ]
            if self.max_price:
                query_parts.append(f"dph_total<={self.max_price}")
            query = " ".join(query_parts)

            from vast_manager import _run_vastai
            raw = _run_vastai(
                "search", "offers", query,
                "--order", "dph_total", "--raw",
                api_key=self.vast_key,
            )
            offers = json.loads(raw)
            results = []
            for o in offers:
                results.append({
                    "id": o.get("id"),
                    "gpu_name": o.get("gpu_name", "CPU-only"),
                    "num_gpus": o.get("num_gpus", 0),
                    "gpu_ram": o.get("gpu_ram", 0),
                    "cpu_name": o.get("cpu_name", "Unknown"),
                    "cpu_cores": o.get("cpu_cores_effective", 0),
                    "cpu_ghz": o.get("cpu_ghz", 0),
                    "ram": o.get("cpu_ram", 0),
                    "disk": o.get("disk_space", 0),
                    "price_per_hr": o.get("dph_total", 0),
                    "inet_down": o.get("inet_down", 0),
                    "inet_up": o.get("inet_up", 0),
                    "reliability": o.get("reliability2", 0),
                    "location": o.get("geolocation", "Unknown"),
                })

            sess = _get_session(self.session_id)
            sess['builder_vast_manager'] = manager

            socketio.emit('builder_offers_found', {'offers': results}, room=self.session_id)
        except Exception as e:
            socketio.emit('builder_error', {'error': str(e)}, room=self.session_id)


class BuilderMonitorThread(threading.Thread):
    """Poll R2 for builder progress. Supports batch (multiple cities)."""

    def __init__(self, session_id, r2_client, city_name, instance_id,
                 batch_cities=None, poll_interval=10.0):
        super().__init__(daemon=True)
        self.session_id = session_id
        self.r2_client = r2_client
        self.city_name = city_name
        self.instance_id = instance_id
        self.batch_cities = batch_cities or [{'city_name': city_name}]
        # Batch builder writes to instance_id-based key; legacy uses city_name
        self.status_key = f"Status/INDEX_{instance_id}.json"
        self.legacy_status_key = f"Status/INDEX_{city_name}_{instance_id}.json"
        self.poll_interval = poll_interval
        self._running = True

    def run(self):
        while self._running:
            try:
                # Try batch status key first, then legacy
                data = self.r2_client.download_json(self.status_key)
                if not data:
                    data = self.r2_client.download_json(self.legacy_status_key)
                if data:
                    step = data.get('step', '')
                    detail = data.get('detail', '')
                    pct = data.get('pct', 0)
                    status = data.get('s', data.get('status', 'UNKNOWN'))
                    socketio.emit('builder_progress', {
                        'step': step, 'detail': detail, 'pct': pct, 'status': status,
                        'batch_count': len(self.batch_cities),
                    }, room=self.session_id)

                    if status in ("COMPLETED", "FAILED") or status.startswith("FAILED"):
                        socketio.emit('builder_finished', {
                            'status': status,
                            'detail': detail,
                        }, room=self.session_id)

                        # Update tracker for all successfully built cities
                        if status == "COMPLETED":
                            self._update_trackers()
                        self._cleanup()
                        return

                    # Per-city completion — update tracker immediately
                    if status == "CITY_DONE":
                        self._update_tracker_for_detail(detail)
            except Exception:
                pass
            time.sleep(self.poll_interval)

    def _update_trackers(self):
        """Update tracker JSON for all batch cities."""
        for city_info in self.batch_cities:
            cn = city_info['city_name']
            fp = city_info.get('features_prefix', '')
            # Parse country/state/city from features_prefix: Features/Country/State/City
            parts = fp.replace('Features/', '').split('/')
            if len(parts) >= 3:
                try:
                    _update_tracker_json(self.r2_client, parts[0], parts[1], parts[2], 'index')
                except Exception as e:
                    logger.warning(f"Failed to update tracker for {cn}: {e}")

    def _update_tracker_for_detail(self, detail):
        """Parse city name from detail string and update its tracker."""
        for city_info in self.batch_cities:
            cn = city_info['city_name']
            if cn in detail:
                fp = city_info.get('features_prefix', '')
                parts = fp.replace('Features/', '').split('/')
                if len(parts) >= 3:
                    try:
                        _update_tracker_json(self.r2_client, parts[0], parts[1], parts[2], 'index')
                    except Exception as e:
                        logger.warning(f"Failed to update tracker for {cn}: {e}")
                break

    def stop(self):
        self._running = False

    def _cleanup(self):
        """Delete builder status and lookup files from R2."""
        for key in [
            self.status_key,
            self.legacy_status_key,
            f"Status/INDEX_{self.city_name}_lookup.json",
        ]:
            try:
                self.r2_client.delete_file(key)
            except Exception:
                pass


# ═════════════════════════════════════════════════════════════════════════
# Socket.IO events
# ═════════════════════════════════════════════════════════════════════════

@socketio.on('connect')
def handle_connect():
    sid = str(uuid.uuid4())
    join_room(sid)
    emit('session_init', {'session_id': sid, 'vps_available': VPS_AVAILABLE})
    logger.info(f"New session connected: {sid}")


@socketio.on('register_session')
def handle_register(data):
    sid = data.get('session_id')
    if sid:
        join_room(sid)
        _get_session(sid)


# ═════════════════════════════════════════════════════════════════════════
# REST API endpoints
# ═════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/load-shapes', methods=['POST'])
def load_shapes():
    """Load shapes from GeoJSON, Shapefile (.shp/.zip), KML, or GeoPackage."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    all_files = request.files.getlist('file')
    sid = request.form.get('session_id', '')

    # Find the main file (pick .shp, .geojson, .zip, .kml, .gpkg — first match)
    main_file = None
    for uf in all_files:
        uf_ext = (uf.filename or '').rsplit('.', 1)[-1].lower()
        if uf_ext in ('shp', 'geojson', 'json', 'zip', 'kml', 'gpkg'):
            main_file = uf
            break
    if main_file is None:
        main_file = all_files[0]

    fname = main_file.filename or ''
    ext = fname.rsplit('.', 1)[-1].lower() if '.' in fname else ''

    try:
        if ext in ('geojson', 'json'):
            content = main_file.read().decode('utf-8')
            geojson = json.loads(content)
        elif ext in ('shp', 'dbf', 'zip', 'kml', 'gpkg'):
            import geopandas as gpd
            import tempfile
            import shutil
            # Allow reading .shp without .shx companion
            os.environ['SHAPE_RESTORE_SHX'] = 'YES'
            tmpdir = tempfile.mkdtemp()
            try:
                for uf in all_files:
                    uf_name = uf.filename or 'upload'
                    uf.save(os.path.join(tmpdir, uf_name))
                if ext == 'zip':
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
            return jsonify({'error': f'Unsupported file type: .{ext}. Use .geojson, .shp, .zip, .kml, or .gpkg'}), 400

        sess = _get_session(sid)
        sess['shapes_data'] = geojson
        return jsonify({'geojson': geojson, 'filename': fname})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/start-scrape', methods=['POST'])
def start_scrape():
    """Start scraping selected shapes."""
    data = request.json
    sid = data.get('session_id', '')
    selected_coords = data.get('selected_coords', [])

    if not selected_coords:
        return jsonify({'error': 'No shapes selected'}), 400

    sess = _get_session(sid)
    config = UnifiedScraperConfig(concurrency=1000, proxy_file=None)
    scraper = UnifiedScraper(config)
    sess['scraper'] = scraper

    # Set up callbacks that emit Socket.IO events
    def on_progress(c, t):
        socketio.emit('scrape_progress', {'current': c, 'total': t}, room=sid)

    def on_status(s):
        socketio.emit('scrape_status', {'message': s}, room=sid)

    def on_point_found(lat, lon, panoid):
        socketio.emit('point_found', {'lat': lat, 'lon': lon, 'panoid': panoid}, room=sid)

    def on_point_completed(panoid):
        socketio.emit('point_completed', {'panoid': panoid}, room=sid)

    scraper.set_progress_callback(on_progress)
    scraper.set_status_callback(on_status)
    scraper.set_point_callback(on_point_found)
    scraper.set_completed_callback(on_point_completed)

    project_root = Path(__file__).parent.parent
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_dir = project_root / "Output" / "CSV"
    images_dir = project_root / "Output" / "Images"

    if len(selected_coords) == 1:
        country, state, city = _reverse_geocode(
            sum(c[0] for c in selected_coords[0]) / len(selected_coords[0]),
            sum(c[1] for c in selected_coords[0]) / len(selected_coords[0])
        )
        csv_path = csv_dir / country / state / city / f"{timestamp}.csv"
        scraper.init_csv(str(csv_path))
        scraper.init_images_dir(str(images_dir / country / state / city))

        thread = ScraperThread(sid, scraper, mode="polygon", polygon_coords=selected_coords[0])
    else:
        geocoded = []
        for sc in selected_coords:
            c, s, ci = _reverse_geocode(
                sum(p[0] for p in sc) / len(sc),
                sum(p[1] for p in sc) / len(sc)
            )
            geocoded.append((c, s, ci))

        first_c, first_s, first_ci = geocoded[0]
        merged_csv = str(csv_dir / first_c / first_s / first_ci / f"Merged_{timestamp}.csv")
        scraper.init_csv(merged_csv)
        scraper.init_images_dir(str(images_dir / first_c / first_s / first_ci))

        thread = ScraperThread(
            sid, scraper, mode="multi_polygon",
            polygon_list=selected_coords,
            csv_paths=[merged_csv] * len(selected_coords),
            images_dirs=[str(images_dir / c / s / ci) for c, s, ci in geocoded],
            merge_csv=True
        )

    sess['scraper_thread'] = thread
    sess['job_state'] = 'scraping'

    # Start speed stats emitter
    thread.start()  # Start scraper first so is_alive() is True

    def speed_stats_loop():
        while thread.is_alive():
            time.sleep(1.0)
            if scraper:
                found = len(scraper.found_panos) if hasattr(scraper, 'found_panos') else 0
                written = scraper.total_written if hasattr(scraper, 'total_written') else 0
                images = scraper.total_images if hasattr(scraper, 'total_images') else 0
                tiles = scraper.tiles_processed if hasattr(scraper, 'tiles_processed') else 0
                total_tiles = scraper.total_tiles if hasattr(scraper, 'total_tiles') else 0
                p2_done = scraper.phase2_completed if hasattr(scraper, 'phase2_completed') else 0
                p2_total = scraper.phase2_total if hasattr(scraper, 'phase2_total') else 0
                socketio.emit('scrape_stats', {
                    'found': found, 'written': written, 'images': images,
                    'tiles': tiles, 'total_tiles': total_tiles,
                    'p2_done': p2_done, 'p2_total': p2_total,
                }, room=sid)
        # Emit one final stats update after thread finishes
        if scraper:
            socketio.emit('scrape_stats', {
                'found': len(scraper.found_panos) if hasattr(scraper, 'found_panos') else 0,
                'written': scraper.total_written if hasattr(scraper, 'total_written') else 0,
                'images': scraper.total_images if hasattr(scraper, 'total_images') else 0,
                'tiles': scraper.tiles_processed if hasattr(scraper, 'tiles_processed') else 0,
                'total_tiles': scraper.total_tiles if hasattr(scraper, 'total_tiles') else 0,
                'p2_done': scraper.phase2_completed if hasattr(scraper, 'phase2_completed') else 0,
                'p2_total': scraper.phase2_total if hasattr(scraper, 'phase2_total') else 0,
            }, room=sid)

    stats_thread = threading.Thread(target=speed_stats_loop, daemon=True)
    stats_thread.start()

    return jsonify({'status': 'started'})


@app.route('/api/cancel-scrape', methods=['POST'])
def cancel_scrape():
    data = request.json
    sid = data.get('session_id', '')
    sess = _get_session(sid)
    thread = sess.get('scraper_thread')
    if thread and thread.is_alive():
        if isinstance(thread, ScraperThread):
            thread.cancel()
        else:
            # Plain threading.Thread (e.g. from vps-scrape-upload) —
            # cancel via the scraper object directly.
            scraper = sess.get('scraper')
            if scraper and hasattr(scraper, 'cancel'):
                scraper.cancel()
    sess['job_state'] = 'idle'
    return jsonify({'status': 'cancelled'})


@app.route('/api/deploy-vps', methods=['POST'])
def deploy_vps():
    """Start VPS deployment pipeline."""
    if not VPS_AVAILABLE:
        return jsonify({'error': 'VPS modules not available'}), 400

    data = request.json
    sid = data.get('session_id', '')
    sess = _get_session(sid)

    r2_account = data.get('r2_account', '').strip()
    r2_access = data.get('r2_access', '').strip()
    r2_secret = data.get('r2_secret', '').strip()
    r2_bucket = data.get('r2_bucket', '').strip()
    vast_key = data.get('vast_key', '').strip()

    if not all([r2_account, r2_access, r2_secret, r2_bucket]):
        return jsonify({'error': 'Missing R2 configuration'}), 400
    if not vast_key:
        return jsonify({'error': 'Missing Vast.ai API key'}), 400

    selected_coords = data.get('selected_coords', [])
    if not selected_coords:
        return jsonify({'error': 'No shapes selected'}), 400

    region_path = data.get('region_path', '').strip().strip('/')
    manual_country = manual_state = manual_city = None
    if region_path:
        parts = [p.strip() for p in region_path.split('/') if p.strip()]
        if len(parts) != 3:
            return jsonify({'error': 'Region path must be Country/State/City'}), 400
        manual_country, manual_state, manual_city = parts

    ctx = {
        'country': manual_country, 'state': manual_state, 'city': manual_city,
        'r2_account': r2_account, 'r2_access': r2_access,
        'r2_secret': r2_secret, 'r2_bucket': r2_bucket,
        'vast_key': vast_key,
        'num_workers': data.get('num_workers', 10),
        'override_workers': data.get('override_workers', False),
        'docker_image': data.get('docker_image', ''),
        'gpu_type': data.get('gpu_type', ''),
        'min_vram_gb': data.get('min_vram_gb', 21),
        'max_price': data.get('max_price', 0.5),
        'geo_filter': data.get('geo_filter', ''),
        'disk_gb': data.get('disk_gb', 100),
        'selected_coords': selected_coords,
        'first_coords': selected_coords[0],
    }
    sess['vps_context'] = ctx

    sess['job_state'] = 'searching_offers'
    sess['pending_offers'] = None

    thread = VPSDeployThread(sid, ctx, mode="full")
    thread.start()

    return jsonify({'status': 'deploying'})


@app.route('/api/vps-scrape-upload', methods=['POST'])
def vps_scrape_upload():
    """Fresh scrape → split CSV → upload to R2 → search offers.

    This is the flow triggered when R2 has no existing CSVs.  The original
    PyQt6 ui.py ran the scraper locally (CSV-only metadata pass), split the
    resulting CSV into per-worker segments, uploaded them to R2, and then
    searched for VPS offers.  The web_app was missing this entire pipeline.
    """
    if not VPS_AVAILABLE:
        return jsonify({'error': 'VPS modules not available'}), 400

    data = request.json
    sid = data.get('session_id', '')
    sess = _get_session(sid)
    ctx = sess.get('vps_context', {})

    if not ctx:
        return jsonify({'error': 'No VPS context — call deploy-vps first'}), 400

    selected_coords = ctx.get('selected_coords', [])
    if not selected_coords:
        return jsonify({'error': 'No shapes in VPS context'}), 400

    country = ctx['country']
    state = ctx['state']
    city = ctx['city']

    # Build scraper (CSV-only, no images)
    config = UnifiedScraperConfig(concurrency=1000, proxy_file=None)
    scraper = UnifiedScraper(config)
    sess['scraper'] = scraper

    # Callbacks that emit Socket.IO events
    def on_progress(c, t):
        socketio.emit('vps_scrape_progress', {'current': c, 'total': t}, room=sid)

    def on_status(s):
        socketio.emit('vps_status', {'message': f'CSV Scrape: {s}'}, room=sid)

    def on_point_found(lat, lon, panoid):
        socketio.emit('point_found', {'lat': lat, 'lon': lon, 'panoid': panoid}, room=sid)

    scraper.set_progress_callback(on_progress)
    scraper.set_status_callback(on_status)
    scraper.set_point_callback(on_point_found)

    project_root = Path(__file__).parent.parent
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_dir = project_root / "Output" / "CSV" / country / state / city
    images_dir = project_root / "Output" / "Images" / country / state / city

    if len(selected_coords) == 1:
        csv_path = str(csv_dir / f"{timestamp}.csv")
        scraper.init_csv(csv_path)
        scraper.init_images_dir(str(images_dir))
        mode = "polygon"
        kwargs = {'polygon_coords': selected_coords[0]}
    else:
        merged_csv_path = str(csv_dir / f"Merged_{timestamp}.csv")
        scraper.init_csv(merged_csv_path)
        scraper.init_images_dir(str(images_dir))
        csv_path = merged_csv_path
        mode = "multi_polygon"
        kwargs = {
            'polygon_list': selected_coords,
            'csv_paths': [merged_csv_path] * len(selected_coords),
            'images_dirs': [str(images_dir)] * len(selected_coords),
            'merge_csv': True,
        }

    def _scrape_split_upload():
        """Background thread: scrape → split → upload → search offers."""
        try:
            # Phase 1: Scrape
            socketio.emit('vps_status', {'message': 'VPS Deploy: Scraping CSV...'}, room=sid)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            if mode == "polygon":
                loop.run_until_complete(
                    scraper.scrape_area_two_phase(kwargs['polygon_coords'])
                )
            else:
                loop.run_until_complete(
                    scraper.scrape_multiple_polygons_two_phase(
                        kwargs['polygon_list'],
                        csv_paths=kwargs.get('csv_paths'),
                        images_dirs=kwargs.get('images_dirs'),
                        merge_csv=kwargs.get('merge_csv', False),
                    )
                )
            loop.close()

            # Close CSV handle
            if scraper.csv_file_handle and not scraper.csv_file_handle.closed:
                scraper.csv_file_handle.flush()
                scraper.csv_file_handle.close()

            socketio.emit('vps_status', {
                'message': f'VPS Deploy: CSV scraping complete ({scraper.total_written} panos). Splitting...'
            }, room=sid)

            # Phase 2: Split CSV into 1K-pano chunks + calculate workers
            CHUNK_SIZE = 1000
            PANOS_PER_WORKER = 50000
            total_panos = scraper.total_written
            if ctx.get('override_workers'):
                num_workers = ctx['num_workers']
            else:
                num_workers = max(1, math.ceil(total_panos / PANOS_PER_WORKER))

            socketio.emit('vps_status', {
                'message': f'VPS Deploy: Splitting {total_panos} panos into {CHUNK_SIZE}-pano chunks, {num_workers} workers'
            }, room=sid)

            chunks = split_csv_chunks(
                csv_path, chunk_size=CHUNK_SIZE, city_name=city,
                output_dir=str(Path(csv_path).parent / "chunks"),
            )
            if not chunks:
                socketio.emit('vps_error', {'error': 'No data in CSV to split'}, room=sid)
                return

            total_chunks = len(chunks)
            ctx['actual_workers'] = num_workers
            ctx['total_workers'] = num_workers
            ctx['total_chunks'] = total_chunks
            ctx['worker_indices_to_deploy'] = list(range(1, num_workers + 1))

            socketio.emit('vps_status', {
                'message': f'VPS Deploy: Split into {total_chunks} chunks. Uploading to R2...'
            }, room=sid)

            # Phase 3: Upload chunks to R2
            r2 = R2Client(ctx['r2_account'], ctx['r2_access'], ctx['r2_secret'], ctx['r2_bucket'])
            uploaded = upload_csv_segments(chunks, r2, country, state, city)
            if len(uploaded) != total_chunks:
                socketio.emit('vps_status', {
                    'message': f'VPS Deploy: Warning — only {len(uploaded)}/{total_chunks} chunks uploaded'
                }, room=sid)

            # Phase 3b: Initialize Redis task queue
            redis_url = os.environ.get('REDIS_URL', '')
            redis_token = os.environ.get('REDIS_TOKEN', '')
            if not redis_url or not redis_token:
                socketio.emit('vps_error', {'error': 'REDIS_URL and REDIS_TOKEN must be set in .env'}, room=sid)
                return

            region = f"{country}/{state}/{city}"
            chunk_ids = [f"chunk_{i+1:04d}" for i in range(total_chunks)]
            tq = TaskQueue(redis_url, redis_token)
            tq.init_job(region, chunk_ids, total_panos, city)
            ctx['redis_url'] = redis_url
            ctx['redis_token'] = redis_token
            ctx['region'] = region
            sess['task_queue'] = tq

            socketio.emit('vps_status', {
                'message': f'VPS Deploy: Redis queue initialized with {total_chunks} chunks'
            }, room=sid)

            # Update tracker JSON with CSV status
            try:
                with open(segments[0], 'r', encoding='utf-8') as fh:
                    fh.readline()  # skip header
                    first_line = fh.readline()
                    parts = first_line.split(',')
                    csv_lat, csv_lon = float(parts[1]), float(parts[2])
                _update_tracker_json(r2, country, state, city, 'csv', csv_lat, csv_lon)
            except Exception as e:
                logger.warning(f"Failed to update tracker after CSV upload: {e}")

            socketio.emit('vps_status', {
                'message': 'VPS Deploy: CSV uploaded. Searching Vast.ai offers...'
            }, room=sid)

            # Phase 4: Search offers (reuse VPSDeployThread logic)
            thread = VPSDeployThread(sid, ctx, mode="provision")
            thread.run()  # run synchronously in this thread

        except Exception as e:
            import traceback
            traceback.print_exc()
            socketio.emit('vps_error', {'error': str(e)}, room=sid)

    sess['job_state'] = 'scraping'
    sess['pending_offers'] = None

    bg = threading.Thread(target=_scrape_split_upload, daemon=True)
    bg.start()
    sess['scraper_thread'] = bg

    return jsonify({'status': 'scraping'})


@app.route('/api/search-offers', methods=['POST'])
def search_offers():
    """Search Vast.ai offers for provisioning."""
    if not VPS_AVAILABLE:
        return jsonify({'error': 'VPS modules not available'}), 400

    data = request.json
    sid = data.get('session_id', '')
    sess = _get_session(sid)
    ctx = sess.get('vps_context', {})
    ctx.update({k: data[k] for k in data if k != 'session_id'})

    # Accept worker subset from frontend (missing-workers dialog)
    if 'worker_indices_to_deploy' in data:
        ctx['worker_indices_to_deploy'] = data['worker_indices_to_deploy']
    if 'total_workers' in data:
        ctx['total_workers'] = data['total_workers']
        ctx['actual_workers'] = len(ctx.get('worker_indices_to_deploy',
                                             list(range(1, data['total_workers'] + 1))))

    sess['job_state'] = 'searching_offers'
    sess['pending_offers'] = None

    thread = VPSDeployThread(sid, ctx, mode="provision")
    thread.start()

    return jsonify({'status': 'searching'})


@app.route('/api/create-instances', methods=['POST'])
def create_instances():
    """Create VPS instances from selected offers."""
    if not VPS_AVAILABLE:
        return jsonify({'error': 'VPS modules not available'}), 400

    data = request.json
    sid = data.get('session_id', '')
    sess = _get_session(sid)
    ctx = sess.get('vps_context', {})

    vast_manager = sess.get('vast_manager')
    if not vast_manager:
        return jsonify({'error': 'No Vast manager — search for offers first'}), 400

    selected_offers = data.get('selected_offers', [])
    worker_indices = data.get('worker_indices', [])
    total_workers = data.get('total_workers', len(worker_indices))

    # Offers have been consumed — clear them so reconnect doesn't re-show the modal
    sess['pending_offers'] = None
    sess['job_state'] = 'creating_instances'

    thread = InstanceCreateThread(sid, vast_manager, selected_offers, worker_indices, total_workers, ctx)
    thread.start()

    return jsonify({'status': 'creating'})


@app.route('/api/start-monitoring', methods=['POST'])
def start_monitoring():
    """Start R2-based VPS worker monitoring."""
    if not VPS_AVAILABLE:
        return jsonify({'error': 'VPS modules not available'}), 400

    data = request.json
    sid = data.get('session_id', '')
    sess = _get_session(sid)
    ctx = sess.get('vps_context', {})
    instance_worker_map = data.get('instance_worker_map', {})

    r2 = R2Client(ctx['r2_account'], ctx['r2_access'], ctx['r2_secret'], ctx['r2_bucket'])
    total_workers = len(instance_worker_map)
    finished_count = [0]

    def on_progress(worker_idx, processed, total, eta, speed, status):
        socketio.emit('vps_worker_progress', {
            'worker_idx': worker_idx, 'processed': processed, 'total': total,
            'eta': eta, 'speed': speed, 'status': status
        }, room=sid)

    def on_finished(worker_idx, status):
        socketio.emit('vps_worker_finished', {
            'worker_idx': worker_idx, 'status': status
        }, room=sid)
        if status == 'COMPLETED':
            finished_count[0] += 1
            if finished_count[0] >= total_workers:
                try:
                    country = ctx.get('country', '')
                    state_name = ctx.get('state', '')
                    city_name = ctx.get('city', '')
                    if country and state_name and city_name:
                        _update_tracker_json(r2, country, state_name, city_name, 'features')
                except Exception as e:
                    logger.warning(f"Failed to update tracker after features: {e}")

    def on_log(message):
        socketio.emit('vps_log', {'message': message}, room=sid)

    monitor = R2StatusMonitorThread(
        r2_client=r2, city_name=ctx['city'],
        instance_worker_map=instance_worker_map,
        poll_interval=10.0,
        on_progress=on_progress,
        on_worker_finished=on_finished,
        on_log_message=on_log,
        vast_manager=sess.get('vast_manager'),
    )
    sess['log_monitor'] = monitor
    sess['job_state'] = 'monitoring'
    monitor.start()

    return jsonify({'status': 'monitoring'})


@app.route('/api/destroy-all', methods=['POST'])
def destroy_all():
    """Destroy all VPS instances."""
    if not VPS_AVAILABLE:
        return jsonify({'error': 'VPS modules not available'}), 400

    data = request.json
    sid = data.get('session_id', '')
    sess = _get_session(sid)

    monitor = sess.get('log_monitor')
    if monitor:
        monitor.stop()

    vm = sess.get('vast_manager')
    count = 0
    if vm:
        count = vm.destroy_all()

    sess['job_state'] = 'idle'
    sess['pending_offers'] = None

    return jsonify({'status': 'destroyed', 'count': count})


@app.route('/api/revive-workers', methods=['POST'])
def revive_workers():
    """Kill dead/idle workers and spin up replacements to finish the job."""
    if not VPS_AVAILABLE:
        return jsonify({'error': 'VPS modules not available'}), 400

    data = request.json
    sid = data.get('session_id', '')
    sess = _get_session(sid)
    ctx = sess.get('vps_context')
    if not ctx:
        return jsonify({'error': 'No active VPS deployment context'}), 400

    vm = sess.get('vast_manager')
    if not vm:
        return jsonify({'error': 'No Vast.ai manager — deploy first'}), 400

    # Check Redis for remaining work
    tq = sess.get('task_queue')
    if not tq:
        redis_url = ctx.get('redis_url', os.environ.get('REDIS_URL', ''))
        redis_token = ctx.get('redis_token', os.environ.get('REDIS_TOKEN', ''))
        if redis_url and redis_token:
            tq = TaskQueue(redis_url, redis_token)
    if not tq:
        return jsonify({'error': 'No Redis connection'}), 400

    region = ctx.get('region', f"{ctx['country']}/{ctx['state']}/{ctx['city']}")
    progress = tq.get_progress(region)
    remaining = progress['todo'] + progress['active']
    if remaining == 0:
        return jsonify({'error': 'Job already complete — nothing to revive'}), 400

    # Reclaim stale tasks first
    reclaimed = tq.reclaim_stale(region, timeout=120)

    # Destroy all current instances
    destroyed = vm.destroy_all()

    # Figure out how many new workers to spin up
    num_workers = min(remaining, ctx.get('num_workers', 10))

    # Search for new offers and create instances in background
    def _revive_thread():
        try:
            socketio.emit('vps_status', {
                'message': f'Reviving: destroyed {destroyed} instances, '
                           f'reclaimed {len(reclaimed)} stale tasks, '
                           f'{remaining} chunks remaining, spinning up {num_workers} workers...'
            }, room=sid)

            offers = vm.search_offers(
                gpu_type=ctx.get('gpu_type', ''),
                region=ctx.get('geo_filter', ''),
                min_disk_gb=ctx.get('disk_gb', 100),
                max_price=ctx.get('max_price', 0.5),
                min_ram_gb=ctx.get('min_vram_gb', 21),
            )
            if not offers:
                socketio.emit('vps_error', {'error': 'No GPU offers found'}, room=sid)
                return

            features_prefix = f"Features/{ctx['country']}/{ctx['state']}/{ctx['city']}"
            instance_worker_map = {}
            used_offer_ids = set()

            for i in range(num_workers):
                env_vars = {
                    'R2_ACCOUNT_ID': ctx['r2_account'],
                    'R2_ACCESS_KEY_ID': ctx['r2_access'],
                    'R2_SECRET_ACCESS_KEY': ctx['r2_secret'],
                    'R2_BUCKET_NAME': ctx['r2_bucket'],
                    'CSV_BUCKET_PREFIX': f"CSV/{ctx['country']}/{ctx['state']}/{ctx['city']}",
                    'FEATURES_BUCKET_PREFIX': features_prefix,
                    'CITY_NAME': ctx['city'],
                    'VAST_API_KEY': ctx['vast_key'],
                    'REDIS_URL': ctx.get('redis_url', ''),
                    'REDIS_TOKEN': ctx.get('redis_token', ''),
                    'REGION': region,
                }

                offer = None
                for candidate in offers:
                    if candidate['id'] not in used_offer_ids:
                        offer = candidate
                        break
                if offer is None:
                    offer = offers[i % len(offers)]
                used_offer_ids.add(offer['id'])

                instance_id = vm.create_instance(
                    offer_id=offer['id'],
                    docker_image=ctx['docker_image'],
                    env_vars=env_vars,
                    disk_gb=ctx.get('disk_gb', 100),
                    onstart_cmd="bash /app/entrypoint.sh",
                    template_hash=VastManager.GEOAXIS_TEMPLATE_HASH,
                )

                if instance_id:
                    instance_worker_map[instance_id] = i + 1
                    socketio.emit('vps_status', {
                        'message': f"Revive worker {i+1}/{num_workers}: "
                                   f"instance {instance_id} on {offer.get('gpu_name', '?')}"
                    }, room=sid)

            if instance_worker_map:
                socketio.emit('vps_status', {
                    'message': f"Revived {len(instance_worker_map)} workers for {remaining} remaining chunks"
                }, room=sid)
                # Show worker table
                socketio.emit('vps_creation_started', {
                    'worker_indices': list(instance_worker_map.values()),
                }, room=sid)

                # Restart monitoring
                old_monitor = sess.get('log_monitor')
                if old_monitor:
                    old_monitor.stop()

                monitor = RedisQueueMonitorThread(
                    task_queue=tq, region=region,
                    instance_ids=list(instance_worker_map.keys()),
                    poll_interval=10.0,
                    on_progress=lambda pd: socketio.emit('vps_queue_progress', pd, room=sid),
                    on_complete=lambda: socketio.emit('vps_status', {'message': 'All chunks completed!'}, room=sid),
                    on_log=lambda msg: socketio.emit('vps_log', {'message': msg}, room=sid),
                    vast_manager=vm,
                )
                monitor.start()
                sess['log_monitor'] = monitor
                sess['job_state'] = 'monitoring'

        except Exception as e:
            socketio.emit('vps_error', {'error': f'Revive failed: {e}'}, room=sid)

    import threading
    threading.Thread(target=_revive_thread, daemon=True).start()

    return jsonify({
        'status': 'reviving',
        'destroyed': destroyed,
        'reclaimed': len(reclaimed),
        'remaining_chunks': remaining,
        'new_workers': num_workers,
    })


@app.route('/api/audit-index', methods=['POST'])
def audit_index():
    """Scan R2 for cities that have features but no index.

    Uses delimiter-based listing to discover city paths quickly,
    then a single head_object per city to check for megaloc.index.
    """
    if not VPS_AVAILABLE:
        return jsonify({'error': 'VPS modules not available'}), 400

    data = request.json
    r2_account = data.get('r2_account', '').strip()
    r2_access = data.get('r2_access', '').strip()
    r2_secret = data.get('r2_secret', '').strip()
    r2_bucket = data.get('r2_bucket', '').strip()

    if not all([r2_account, r2_access, r2_secret, r2_bucket]):
        return jsonify({'error': 'Missing R2 credentials'}), 400

    try:
        r2 = R2Client(
            account_id=r2_account,
            access_key_id=r2_access,
            secret_access_key=r2_secret,
            bucket_name=r2_bucket,
        )
        s3 = r2.s3
        bucket = r2.bucket_name

        # Flat-list Features/ and Index/ in two bulk scans (few API calls)
        feature_city_re = re.compile(r'^Features/(.+?/.+?/.+?)/')
        index_city_re = re.compile(r'^Index/(.+?/.+?/.+?)/')

        # Discover all city paths that have features
        feature_cities = set()
        kwargs = {'Bucket': bucket, 'Prefix': 'Features/', 'MaxKeys': 1000}
        while True:
            resp = s3.list_objects_v2(**kwargs)
            for obj in resp.get('Contents', []):
                m = feature_city_re.match(obj['Key'])
                if m:
                    feature_cities.add(m.group(1))
            if not resp.get('IsTruncated'):
                break
            kwargs['ContinuationToken'] = resp['NextContinuationToken']

        # Discover all city paths that already have an index
        indexed_cities = set()
        kwargs = {'Bucket': bucket, 'Prefix': 'Index/', 'MaxKeys': 1000}
        while True:
            resp = s3.list_objects_v2(**kwargs)
            for obj in resp.get('Contents', []):
                if obj['Key'].endswith('/megaloc.index'):
                    m = index_city_re.match(obj['Key'])
                    if m:
                        indexed_cities.add(m.group(1))
            if not resp.get('IsTruncated'):
                break
            kwargs['ContinuationToken'] = resp['NextContinuationToken']

        # Cities with features but no index
        needs_index = []
        for city_path in sorted(feature_cities - indexed_cities):
            city_name = city_path.rsplit('/', 1)[-1]
            needs_index.append({
                'features_prefix': f"Features/{city_path}",
                'city_name': city_name,
                'city_path': city_path,
            })

        return jsonify({'needs_index': needs_index})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _list_prefixes(s3, bucket, prefix):
    """List immediate sub-prefixes (directories) under a prefix using delimiter."""
    prefixes = []
    kwargs = {'Bucket': bucket, 'Prefix': prefix, 'Delimiter': '/'}
    while True:
        resp = s3.list_objects_v2(**kwargs)
        for cp in resp.get('CommonPrefixes', []):
            prefixes.append(cp['Prefix'])
        if not resp.get('IsTruncated'):
            break
        kwargs['ContinuationToken'] = resp['NextContinuationToken']
    return prefixes


@app.route('/api/build-index', methods=['POST'])
def build_index():
    """Start index builder search. Accepts multiple shapes for batch building."""
    if not VPS_AVAILABLE:
        return jsonify({'error': 'VPS modules not available'}), 400

    data = request.json
    sid = data.get('session_id', '')

    vast_key = data.get('vast_key', '').strip()
    max_price = data.get('max_price', 2.0)
    min_disk = data.get('min_disk', 500)

    if not vast_key:
        return jsonify({'error': 'Missing Vast.ai API key'}), 400

    # Store shapes in session for create-builder to use
    sess = _get_session(sid)
    sess['builder_shapes'] = data.get('shapes', [])
    sess['builder_path_override'] = data.get('path_override', '')
    sess['builder_batch_cities'] = data.get('batch_cities', [])

    thread = BuilderSearchThread(sid, vast_key, max_price, min_disk)
    thread.start()

    return jsonify({'status': 'searching'})


@app.route('/api/create-builder', methods=['POST'])
def create_builder():
    """Create builder instance(s). Supports multi-worker Redis queue mode."""
    if not VPS_AVAILABLE:
        return jsonify({'error': 'VPS modules not available'}), 400

    data = request.json
    sid = data.get('session_id', '')
    sess = _get_session(sid)
    ctx = sess.get('vps_context', {})

    vm = sess.get('builder_vast_manager')
    if not vm:
        return jsonify({'error': 'No builder manager — search for offers first'}), 400

    selected_offers = data.get('selected_offers', [])
    offer = data.get('offer', {})
    if not selected_offers and offer:
        selected_offers = [offer]
    r2_account = data.get('r2_account', '').strip()
    r2_access = data.get('r2_access', '').strip()
    r2_secret = data.get('r2_secret', '').strip()
    r2_bucket = data.get('r2_bucket', '').strip()
    vast_key = data.get('vast_key', '').strip()
    num_workers = int(data.get('num_workers', 1))

    # ── Build batch cities list ──
    audit_batch = data.get('batch_cities', []) or sess.get('builder_batch_cities', [])
    shapes = data.get('shapes', []) or sess.get('builder_shapes', [])
    path_override = data.get('path_override', '').strip().strip('/') or sess.get('builder_path_override', '')

    batch_cities = []
    if audit_batch:
        batch_cities = audit_batch
    elif shapes:
        for s in shapes:
            country = s.get('country', '').replace(' ', '_')
            state = s.get('state', '').replace(' ', '_')
            city = s.get('city', '').replace(' ', '_')
            batch_cities.append({
                'features_prefix': f"Features/{country}/{state}/{city}",
                'city_name': city,
            })
    elif path_override:
        city_name = path_override.split('/')[-1]
        batch_cities.append({
            'features_prefix': f"Features/{path_override}",
            'city_name': city_name,
        })
    else:
        city_name = ctx.get('city', 'Unknown')
        batch_cities.append({
            'features_prefix': f"Features/{ctx.get('country', '')}/{ctx.get('state', '')}/{city_name}",
            'city_name': city_name,
        })

    sess['builder_batch_cities'] = batch_cities

    first_city = batch_cities[0]['city_name']
    first_prefix = batch_cities[0]['features_prefix']

    # Common env vars for all workers
    env_vars = {
        'R2_ACCOUNT_ID': r2_account,
        'R2_ACCESS_KEY_ID': r2_access,
        'R2_SECRET_ACCESS_KEY': r2_secret,
        'R2_BUCKET_NAME': r2_bucket,
        'FEATURES_BUCKET_PREFIX': first_prefix,
        'CITY_NAME': first_city,
        'VAST_API_KEY': vast_key,
        'INDEX_TYPE': data.get('index_type', 'pq'),
        'NLIST': str(data.get('nlist', 4096)),
        'M': str(data.get('m', 256)),
        'NBITS': str(data.get('nbits', 8)),
        'TRAIN_SAMPLES': str(data.get('train_samples', 1000000)),
        'NITER': str(data.get('niter', 100)),
    }

    # ── Multi-worker Redis queue mode ──
    use_redis = num_workers > 1 or len(batch_cities) > 3
    redis_url = os.environ.get('REDIS_URL', '')
    redis_token = os.environ.get('REDIS_TOKEN', '')

    if use_redis and redis_url and redis_token:
        # Initialize Redis build queue with all cities
        from redis_queue import BuildQueue
        build_job = f"idx_{int(time.time())}"
        bq = BuildQueue(redis_url, redis_token)
        bq.init_job(build_job, batch_cities)

        env_vars['REDIS_URL'] = redis_url
        env_vars['REDIS_TOKEN'] = redis_token
        env_vars['BUILD_JOB'] = build_job

        # Spawn N workers across selected offers (round-robin)
        instance_ids = []
        for w in range(num_workers):
            use_offer = selected_offers[w % len(selected_offers)]
            try:
                iid = vm.create_instance(
                    offer_id=use_offer['id'],
                    docker_image=data.get('builder_image', 'ghcr.io/occultmc/vps-builder:latest'),
                    env_vars=env_vars,
                    disk_gb=data.get('disk_gb', 700),
                    onstart_cmd="bash /app/entrypoint.sh",
                )
                if iid:
                    instance_ids.append(iid)
                    print(f"[BUILDER] Worker {w+1}/{num_workers} created: {iid}")
            except Exception as e:
                print(f"[BUILDER] Worker {w+1}/{num_workers} failed: {e}")

        if not instance_ids:
            return jsonify({'error': 'Failed to create any builder instances'}), 500

        r2 = R2Client(r2_account, r2_access, r2_secret, r2_bucket)
        r2.upload_json(
            f"Status/INDEX_{first_city}_lookup.json",
            {'instance_ids': instance_ids, 'build_job': build_job,
             'city_name': first_city, 'num_workers': len(instance_ids)},
        )

        # Monitor first worker (all share same R2 status prefix)
        monitor = BuilderMonitorThread(
            sid, r2, first_city, instance_ids[0],
            batch_cities=batch_cities,
            poll_interval=10.0,
        )
        sess['builder_monitor'] = monitor
        monitor.start()

        city_list = ', '.join(c['city_name'] for c in batch_cities)
        return jsonify({
            'status': 'created',
            'instance_ids': instance_ids,
            'num_workers': len(instance_ids),
            'build_job': build_job,
            'batch_count': len(batch_cities),
            'cities': city_list,
        })

    else:
        # No-Redis mode — spawn num_workers independent instances
        if len(batch_cities) > 1 or shapes or audit_batch:
            env_vars['BATCH_CITIES'] = base64.b64encode(
                json.dumps(batch_cities).encode()
            ).decode()

        try:
            instance_ids = []
            for w in range(num_workers):
                use_offer = selected_offers[w % len(selected_offers)]
                iid = vm.create_instance(
                    offer_id=use_offer['id'],
                    docker_image=data.get('builder_image', 'ghcr.io/occultmc/vps-builder:latest'),
                    env_vars=env_vars,
                    disk_gb=data.get('disk_gb', 700),
                    onstart_cmd="bash /app/entrypoint.sh",
                )
                if iid:
                    instance_ids.append(iid)
                    print(f"[BUILDER] Worker {w+1}/{num_workers} created: {iid}")
                else:
                    print(f"[BUILDER] Worker {w+1}/{num_workers} failed: no instance ID")

            if not instance_ids:
                return jsonify({'error': 'Failed to create any builder instances'}), 500

            r2 = R2Client(r2_account, r2_access, r2_secret, r2_bucket)
            r2.upload_json(
                f"Status/INDEX_{first_city}_lookup.json",
                {'instance_ids': instance_ids, 'city_name': first_city,
                 'num_workers': len(instance_ids)},
            )

            monitor = BuilderMonitorThread(
                sid, r2, first_city, instance_ids[0],
                batch_cities=batch_cities,
                poll_interval=10.0,
            )
            sess['builder_monitor'] = monitor
            monitor.start()

            city_list = ', '.join(c['city_name'] for c in batch_cities)
            return jsonify({
                'status': 'created',
                'instance_ids': instance_ids,
                'num_workers': len(instance_ids),
                'batch_count': len(batch_cities),
                'cities': city_list,
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/api/scan-progress', methods=['POST'])
def scan_progress():
    """Scan for completed shapes using tracker JSON (fast) or local CSV fallback."""
    data = request.json
    sid = data.get('session_id', '')
    shapes_data = data.get('shapes_data', [])

    if not shapes_data:
        return jsonify({'status_map': {}})

    r2_client = None
    r2_config = data.get('r2_config', {})
    if VPS_AVAILABLE and all(r2_config.get(k) for k in ['account', 'access', 'secret', 'bucket']):
        try:
            r2_client = R2Client(
                r2_config['account'], r2_config['access'],
                r2_config['secret'], r2_config['bucket']
            )
        except Exception:
            pass

    def scan_thread():
        try:
            from shapely.geometry import Point, Polygon

            polygons = []
            for shape in shapes_data:
                coords = shape.get('coordinates', [])
                polygons.append(Polygon(coords) if coords else None)

            status_map = {}  # {shape_index_str: 'csv'|'features'|'complete'}

            if r2_client:
                # Fast path: fetch single tracker JSON
                tracker = r2_client.download_json(TRACKER_KEY)
                for region_key, info in tracker.items():
                    lat = info.get('lat', 0)
                    lon = info.get('lon', 0)
                    if lat == 0 and lon == 0:
                        continue
                    point = Point(lon, lat)
                    for idx, poly in enumerate(polygons):
                        if str(idx) in status_map:
                            continue
                        if poly and poly.contains(point):
                            if info.get('index'):
                                status_map[str(idx)] = 'complete'
                            elif info.get('features'):
                                status_map[str(idx)] = 'features'
                            elif info.get('csv'):
                                status_map[str(idx)] = 'csv'
                            break
            else:
                # Local fallback: scan Output/CSV directory
                import glob as _glob
                project_root = Path(__file__).parent.parent
                output_dir = str(project_root / "Output" / "CSV")
                csv_files = _glob.glob(os.path.join(output_dir, "**/*.csv"), recursive=True)
                for csv_path in csv_files:
                    try:
                        with open(csv_path, 'r', encoding='utf-8') as fh:
                            fh.readline()
                            first_line = fh.readline()
                            if not first_line:
                                continue
                            parts = first_line.split(',')
                            if len(parts) < 3:
                                continue
                            lat, lon = float(parts[1]), float(parts[2])
                            point = Point(lon, lat)
                            for idx, poly in enumerate(polygons):
                                if str(idx) in status_map:
                                    continue
                                if poly and poly.contains(point):
                                    status_map[str(idx)] = 'csv'
                                    break
                    except Exception:
                        continue

            socketio.emit('scan_complete', {'status_map': status_map}, room=sid)
        except Exception as e:
            socketio.emit('scan_error', {'error': str(e)}, room=sid)

    t = threading.Thread(target=scan_thread, daemon=True)
    t.start()
    return jsonify({'status': 'scanning'})


@app.route('/api/merge-shapes', methods=['POST'])
def merge_shapes():
    """Merge two selected shapes into their geometric union."""
    data = request.json
    coords_list = data.get('coords_list', [])

    if len(coords_list) != 2:
        return jsonify({'error': 'Select exactly 2 shapes to merge'}), 400

    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        poly1 = Polygon([(lng, lat) for lat, lng in coords_list[0]])
        poly2 = Polygon([(lng, lat) for lat, lng in coords_list[1]])
        merged = unary_union([poly1, poly2])

        if merged.geom_type == 'Polygon':
            js_coords = [[lng, lat] for lng, lat in merged.exterior.coords]
        elif merged.geom_type == 'MultiPolygon':
            hull = merged.convex_hull
            js_coords = [[lng, lat] for lng, lat in hull.exterior.coords]
        else:
            return jsonify({'error': f'Unexpected geometry: {merged.geom_type}'}), 400

        return jsonify({'merged_coords': js_coords})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/validate-features', methods=['POST'])
def validate_features():
    """Validate feature files in R2: check all NPY+Metadata pairs exist."""
    data = request.json
    path_override = (data.get('path_override', '') or '').strip().strip('/')
    r2_account = (data.get('r2_account', '') or '').strip()
    r2_access = (data.get('r2_access', '') or '').strip()
    r2_secret = (data.get('r2_secret', '') or '').strip()
    r2_bucket = (data.get('r2_bucket', '') or '').strip()

    if not path_override:
        return jsonify({'error': 'Path Override is required'}), 400
    if not all([r2_account, r2_access, r2_secret, r2_bucket]):
        return jsonify({'error': 'R2 credentials are required (check Advanced Settings)'}), 400

    try:
        r2 = R2Client(r2_account, r2_access, r2_secret, r2_bucket)
    except Exception as e:
        return jsonify({'error': f'R2 connection failed: {e}'}), 500

    features_prefix = f"Features/{path_override}/"
    try:
        files = r2.list_files(features_prefix)
    except Exception as e:
        return jsonify({'error': f'Failed to list R2 files: {e}'}), 500

    # Parse all feature and metadata files
    npy_files = {}   # {chunk_num: {key, size, name}}
    meta_files = {}  # {chunk_num: {key, size, name}}
    other_files = []
    total_npy = 0
    total_meta = 0
    city_name = path_override.split('/')[-1]

    for f in files:
        fname = f['key'].split('/')[-1]
        size = f.get('size', 0)
        try:
            if fname.startswith("Metadata_") and fname.endswith(".jsonl"):
                total_meta += 1
                inner = fname[len("Metadata_"):-len(".jsonl")]
                base, total_str = inner.rsplit('.', 1)
                _city, idx_str = base.rsplit('_', 1)
                if idx_str.isdigit():
                    n = int(idx_str)
                    meta_files[n] = {'key': f['key'], 'size': size, 'name': fname, 'total': int(total_str)}
            elif fname.endswith(".npy"):
                total_npy += 1
                base, total_str = fname[:-4].rsplit('.', 1)
                _city, idx_str = base.rsplit('_', 1)
                if idx_str.isdigit():
                    n = int(idx_str)
                    npy_files[n] = {'key': f['key'], 'size': size, 'name': fname, 'total': int(total_str)}
            else:
                other_files.append(fname)
        except (ValueError, IndexError):
            other_files.append(fname)

    # Determine expected total from file naming
    all_totals = set()
    for info in list(npy_files.values()) + list(meta_files.values()):
        all_totals.add(info['total'])
    expected_total = max(all_totals) if all_totals else 0

    # Find complete pairs, missing NPY, missing metadata
    all_indices = sorted(set(npy_files.keys()) | set(meta_files.keys()))
    complete_pairs = sorted(set(npy_files.keys()) & set(meta_files.keys()))
    missing_npy = sorted(set(meta_files.keys()) - set(npy_files.keys()))
    missing_meta = sorted(set(npy_files.keys()) - set(meta_files.keys()))

    # Check for gaps in sequence 1..expected_total
    if expected_total > 0:
        expected_set = set(range(1, expected_total + 1))
        missing_both = sorted(expected_set - set(all_indices))
    else:
        missing_both = []

    total_npy_size = sum(info['size'] for info in npy_files.values())
    total_meta_size = sum(info['size'] for info in meta_files.values())

    return jsonify({
        'path': features_prefix,
        'expected_total': expected_total,
        'npy_count': len(npy_files),
        'meta_count': len(meta_files),
        'complete_pairs': len(complete_pairs),
        'missing_npy': missing_npy,
        'missing_meta': missing_meta,
        'missing_both': missing_both,
        'total_npy_size_mb': round(total_npy_size / 1048576, 1),
        'total_meta_size_mb': round(total_meta_size / 1048576, 1),
        'other_files': other_files[:20],
        'valid': len(missing_npy) == 0 and len(missing_meta) == 0 and len(missing_both) == 0 and len(npy_files) > 0,
    })


@app.route('/api/env-defaults', methods=['GET'])
def env_defaults():
    """Return env-based defaults for form fields."""
    return jsonify({
        'r2_account': os.environ.get('R2_ACCOUNT_ID', ''),
        'r2_access': os.environ.get('R2_ACCESS_KEY_ID', ''),
        'r2_secret': os.environ.get('R2_SECRET_ACCESS_KEY', ''),
        'r2_bucket': os.environ.get('R2_BUCKET_NAME', ''),
        'vast_key': os.environ.get('VAST_API_KEY', ''),
    })


@app.route('/api/pending-offers', methods=['POST'])
def pending_offers():
    """Return stored offers that the frontend may have missed due to websocket reconnect."""
    data = request.json
    sid = data.get('session_id', '')
    sess = _get_session(sid)
    # Only return offers if we're actually waiting for the user to select them
    if sess.get('job_state') == 'awaiting_offer_selection':
        offers_data = sess.get('pending_offers')
        if offers_data:
            return jsonify(offers_data)
    return jsonify({'offers': []})


@app.route('/api/dismiss-offers', methods=['POST'])
def dismiss_offers():
    """User dismissed the offer modal without selecting — clear pending state."""
    data = request.json
    sid = data.get('session_id', '')
    sess = _get_session(sid)
    sess['pending_offers'] = None
    if sess.get('job_state') == 'awaiting_offer_selection':
        sess['job_state'] = 'idle'
    return jsonify({'status': 'ok'})


@app.route('/api/job-state', methods=['POST'])
def job_state():
    """Return current job state so frontend can restore UI on reconnect."""
    data = request.json
    sid = data.get('session_id', '')
    sess = _get_session(sid)
    return jsonify({'job_state': sess.get('job_state', 'idle')})


# ── Auto-scrape endpoints ─────────────────────────────────────────────

@app.route('/api/geocode-shapes', methods=['POST'])
def geocode_shapes():
    """Geocode shapes using population-weighted worldcities lookup (batch)."""
    data = request.json
    shapes = data.get('shapes', [])

    # Collect coordinates and indices, skip empties
    coords_list = []
    index_map = []  # maps batch position → original shape
    empty_results = []
    for shape in shapes:
        coords = shape.get('coordinates', [])
        if not coords:
            empty_results.append({'index': shape.get('index'), 'country': 'Unknown',
                                  'state': 'Unknown', 'city': 'Unknown', 'path': 'Unknown/Unknown/Unknown'})
        else:
            index_map.append(shape.get('index'))
            coords_list.append(coords)

    # Batch geocode all shapes at once (two-pass: inside → 100m buffer → fallback)
    geo_results = _geocode_shapes_batch(coords_list) if coords_list else []

    results = list(empty_results)
    for i, (country, state, city) in enumerate(geo_results):
        results.append({
            'index': index_map[i],
            'country': country, 'state': state, 'city': city,
            'path': f"{country}/{state}/{city}",
        })
    return jsonify({'results': results})


@app.route('/api/auto-scrape', methods=['POST'])
def auto_scrape():
    """Start sequential auto-scrape of multiple shapes."""
    if not VPS_AVAILABLE:
        return jsonify({'error': 'VPS modules not available'}), 400

    data = request.json
    sid = data.get('session_id', '')
    sess = _get_session(sid)

    shapes = data.get('shapes', [])
    if not shapes:
        return jsonify({'error': 'No shapes'}), 400

    ctx = {}
    for k in ['r2_account', 'r2_access', 'r2_secret', 'r2_bucket',
              'vast_key', 'docker_image', 'gpu_type', 'geo_filter']:
        ctx[k] = (data.get(k, '') or '').strip()
    ctx.update({
        'num_workers': data.get('num_workers', 10),
        'override_workers': data.get('override_workers', False),
        'share_workers': data.get('share_workers', False),
        'min_vram_gb': data.get('min_vram_gb', 21),
        'max_price': data.get('max_price', 0.5),
        'disk_gb': data.get('disk_gb', 100),
        'total_shapes': len(shapes),
    })

    thread = AutoScrapeThread(sid, shapes, ctx)
    sess['auto_scrape_thread'] = thread
    sess['job_state'] = 'auto_scraping'
    thread.start()
    return jsonify({'status': 'started', 'total': len(shapes)})


@app.route('/api/auto-scrape-confirm-offers', methods=['POST'])
def auto_scrape_confirm_offers():
    """User selected offers for auto-scrape (first shape only)."""
    data = request.json
    sid = data.get('session_id', '')
    sess = _get_session(sid)
    thread = sess.get('auto_scrape_thread')
    if not thread:
        return jsonify({'error': 'No auto-scrape active'}), 400
    thread.on_offers_selected(data.get('selected_offers', []))
    return jsonify({'status': 'ok'})


@app.route('/api/cancel-auto-scrape', methods=['POST'])
def cancel_auto_scrape():
    """Cancel running auto-scrape queue."""
    data = request.json
    sid = data.get('session_id', '')
    sess = _get_session(sid)
    thread = sess.get('auto_scrape_thread')
    if thread:
        thread.cancel()
    sess['job_state'] = 'idle'
    return jsonify({'status': 'cancelled'})


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  VPS Scraper Web GUI")
    print("  Open http://localhost:5050 in your browser")
    print("  Each tab gets its own independent session")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5050, debug=True, allow_unsafe_werkzeug=True)
