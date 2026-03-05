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
    from csv_splitter import split_csv, upload_csv_segments
    from vast_manager import VastManager, ContainerNotFoundError
    from log_monitor_web import R2StatusMonitorThread
    VPS_AVAILABLE = True
except ImportError:
    VPS_AVAILABLE = False

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
    return "".join(c for c in str(s) if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')


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
            try:
                import winloop
                loop = winloop.new_event_loop()
            except ImportError:
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
            socketio.emit('scrape_finished', {
                'total_written': self.scraper.total_written,
                'total_images': self.scraper.total_images,
            }, room=self.session_id)
        except Exception as e:
            import traceback
            traceback.print_exc()
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

        socketio.emit('vps_offers_found', {
            'offers': offers if offers else []
        }, room=self.session_id)


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

            for i, worker_idx in enumerate(self.worker_indices):
                offer = self.selected_offers[i % len(self.selected_offers)]
                env_vars = {
                    'R2_ACCOUNT_ID': self.ctx['r2_account'],
                    'R2_ACCESS_KEY_ID': self.ctx['r2_access'],
                    'R2_SECRET_ACCESS_KEY': self.ctx['r2_secret'],
                    'R2_BUCKET_NAME': self.ctx['r2_bucket'],
                    'WORKER_INDEX': str(worker_idx),
                    'NUM_WORKERS': str(self.total_workers),
                    'CSV_BUCKET_PREFIX': f"CSV/{self.ctx['country']}/{self.ctx['state']}/{self.ctx['city']}",
                    'FEATURES_BUCKET_PREFIX': features_prefix,
                    'CITY_NAME': self.ctx['city'],
                    'VAST_API_KEY': self.ctx['vast_key'],
                }
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
                    try:
                        r2.upload_json(
                            f"Status/{features_prefix}/worker_{worker_idx}_instance.json",
                            {'instance_id': instance_id, 'worker_index': worker_idx},
                        )
                    except Exception:
                        pass
                    socketio.emit('vps_status', {
                        'message': f"Created worker {worker_idx} ({i+1}/{len(self.worker_indices)}) — instance {instance_id}"
                    }, room=self.session_id)

            socketio.emit('vps_instances_created', {
                'instance_worker_map': instance_worker_map
            }, room=self.session_id)

        except Exception as e:
            socketio.emit('vps_error', {'error': str(e)}, room=self.session_id)


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
    """Poll R2 for builder progress."""

    def __init__(self, session_id, r2_client, status_prefix, poll_interval=10.0):
        super().__init__(daemon=True)
        self.session_id = session_id
        self.r2_client = r2_client
        self.status_key = f"Status/{status_prefix}/builder.json"
        self.poll_interval = poll_interval
        self._running = True

    def run(self):
        while self._running:
            try:
                data = self.r2_client.download_json(self.status_key)
                if data:
                    step = data.get('step', '')
                    detail = data.get('detail', '')
                    pct = data.get('pct', 0)
                    status = data.get('status', 'UNKNOWN')
                    socketio.emit('builder_progress', {
                        'step': step, 'detail': detail, 'pct': pct, 'status': status
                    }, room=self.session_id)
                    if status in ("COMPLETED", "FAILED"):
                        socketio.emit('builder_finished', {'status': status}, room=self.session_id)
                        return
            except Exception:
                pass
            time.sleep(self.poll_interval)

    def stop(self):
        self._running = False


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
    """Load GeoJSON file and return parsed data."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    sid = request.form.get('session_id', '')
    max_area = float(request.form.get('max_area', 10000))

    try:
        content = f.read().decode('utf-8')
        geojson = json.loads(content)
        sess = _get_session(sid)
        sess['shapes_data'] = geojson
        return jsonify({'geojson': geojson, 'filename': f.filename})
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

    # Start speed stats emitter
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

    stats_thread = threading.Thread(target=speed_stats_loop, daemon=True)
    stats_thread.start()
    thread.start()

    return jsonify({'status': 'started'})


@app.route('/api/cancel-scrape', methods=['POST'])
def cancel_scrape():
    data = request.json
    sid = data.get('session_id', '')
    sess = _get_session(sid)
    thread = sess.get('scraper_thread')
    if thread and thread.is_alive():
        thread.cancel()
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

    thread = VPSDeployThread(sid, ctx, mode="full")
    thread.start()

    return jsonify({'status': 'deploying'})


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
    status_prefix = f"Features/{ctx['country']}/{ctx['state']}/{ctx['city']}"

    def on_progress(worker_idx, processed, total, eta, speed, status):
        socketio.emit('vps_worker_progress', {
            'worker_idx': worker_idx, 'processed': processed, 'total': total,
            'eta': eta, 'speed': speed, 'status': status
        }, room=sid)

    def on_finished(worker_idx, status):
        socketio.emit('vps_worker_finished', {
            'worker_idx': worker_idx, 'status': status
        }, room=sid)

    def on_log(message):
        socketio.emit('vps_log', {'message': message}, room=sid)

    monitor = R2StatusMonitorThread(
        r2_client=r2, status_prefix=status_prefix,
        num_workers=len(instance_worker_map),
        instance_worker_map=instance_worker_map,
        poll_interval=10.0,
        on_progress=on_progress,
        on_worker_finished=on_finished,
        on_log_message=on_log,
    )
    sess['log_monitor'] = monitor
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

    return jsonify({'status': 'destroyed', 'count': count})


@app.route('/api/build-index', methods=['POST'])
def build_index():
    """Start index builder search."""
    if not VPS_AVAILABLE:
        return jsonify({'error': 'VPS modules not available'}), 400

    data = request.json
    sid = data.get('session_id', '')

    vast_key = data.get('vast_key', '').strip()
    max_price = data.get('max_price', 2.0)
    min_disk = data.get('min_disk', 500)

    if not vast_key:
        return jsonify({'error': 'Missing Vast.ai API key'}), 400

    thread = BuilderSearchThread(sid, vast_key, max_price, min_disk)
    thread.start()

    return jsonify({'status': 'searching'})


@app.route('/api/create-builder', methods=['POST'])
def create_builder():
    """Create builder instance from selected offer."""
    if not VPS_AVAILABLE:
        return jsonify({'error': 'VPS modules not available'}), 400

    data = request.json
    sid = data.get('session_id', '')
    sess = _get_session(sid)
    ctx = sess.get('vps_context', {})

    vm = sess.get('builder_vast_manager')
    if not vm:
        return jsonify({'error': 'No builder manager — search for offers first'}), 400

    offer = data.get('offer', {})
    r2_account = data.get('r2_account', '').strip()
    r2_access = data.get('r2_access', '').strip()
    r2_secret = data.get('r2_secret', '').strip()
    r2_bucket = data.get('r2_bucket', '').strip()
    vast_key = data.get('vast_key', '').strip()

    path_override = data.get('path_override', '').strip().strip('/')
    if path_override:
        features_prefix = f"Features/{path_override}"
        city_name = path_override.split('/')[-1]
    else:
        features_prefix = f"Features/{ctx['country']}/{ctx['state']}/{ctx['city']}"
        city_name = ctx.get('city', 'Unknown')

    env_vars = {
        'R2_ACCOUNT_ID': r2_account,
        'R2_ACCESS_KEY_ID': r2_access,
        'R2_SECRET_ACCESS_KEY': r2_secret,
        'R2_BUCKET_NAME': r2_bucket,
        'FEATURES_BUCKET_PREFIX': features_prefix,
        'CITY_NAME': city_name,
        'VAST_API_KEY': vast_key,
        'INDEX_TYPE': data.get('index_type', 'IVF_PQ'),
        'NLIST': str(data.get('nlist', 4096)),
        'M': str(data.get('m', 64)),
        'NBITS': str(data.get('nbits', 8)),
        'TRAIN_SAMPLES': str(data.get('train_samples', 500000)),
        'NITER': str(data.get('niter', 25)),
    }

    try:
        instance_id = vm.create_instance(
            offer_id=offer['id'],
            docker_image=data.get('builder_image', 'ghcr.io/occultmc/geoaxisbuilder:latest'),
            env_vars=env_vars,
            disk_gb=data.get('disk_gb', 500),
            onstart_cmd="bash /app/entrypoint.sh",
        )
        if not instance_id:
            return jsonify({'error': 'Failed to create builder instance'}), 500

        r2 = R2Client(r2_account, r2_access, r2_secret, r2_bucket)
        r2.upload_json(
            f"Status/{features_prefix}/builder_instance.json",
            {'instance_id': instance_id, 'worker': 'builder'},
        )
        try:
            r2.delete_file(f"Status/{features_prefix}/builder.json")
        except Exception:
            pass

        # Start monitoring
        monitor = BuilderMonitorThread(sid, r2, features_prefix, poll_interval=10.0)
        sess['builder_monitor'] = monitor
        monitor.start()

        return jsonify({'status': 'created', 'instance_id': instance_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/scan-progress', methods=['POST'])
def scan_progress():
    """Scan for completed shapes (local CSV or R2)."""
    data = request.json
    sid = data.get('session_id', '')
    shapes_data = data.get('shapes_data', [])

    if not shapes_data:
        return jsonify({'done_indices': []})

    project_root = Path(__file__).parent.parent
    output_dir = str(project_root / "Output" / "CSV")

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

    # Run scan in background
    def scan_thread():
        try:
            from shapely.geometry import Point, Polygon
            import glob

            polygons = []
            for shape in shapes_data:
                coords = shape.get('coordinates', [])
                if coords:
                    polygons.append(Polygon(coords))
                else:
                    polygons.append(None)

            done_indices = set()

            if r2_client:
                try:
                    files = r2_client.list_files(prefix="CSV/")
                    for file_obj in files:
                        key = file_obj['key']
                        if not key.lower().endswith('.csv'):
                            continue
                        try:
                            resp = r2_client.s3.get_object(
                                Bucket=r2_client.bucket_name, Key=key, Range='bytes=0-1024')
                            chunk = resp['Body'].read().decode('utf-8', errors='ignore')
                            lines = chunk.splitlines()
                            if len(lines) < 2:
                                continue
                            parts = lines[1].split(',')
                            if len(parts) < 3:
                                continue
                            lat, lon = float(parts[1]), float(parts[2])
                            point = Point(lon, lat)
                            for idx, poly in enumerate(polygons):
                                if idx in done_indices:
                                    continue
                                if poly and poly.contains(point):
                                    done_indices.add(idx)
                                    break
                        except Exception:
                            continue
                except Exception:
                    pass
            else:
                import glob as _glob
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
                                if idx in done_indices:
                                    continue
                                if poly and poly.contains(point):
                                    done_indices.add(idx)
                                    break
                    except Exception:
                        continue

            socketio.emit('scan_complete', {'done_indices': list(done_indices)}, room=sid)
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


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  VPS Scraper Web GUI")
    print("  Open http://localhost:5000 in your browser")
    print("  Each tab gets its own independent session")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
