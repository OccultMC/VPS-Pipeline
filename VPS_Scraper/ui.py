"""
Unified Street View Scraper UI + VPS Deployment

PyQt6 interface combining Stage 1 (metadata scraping), Stage 2 (image downloading),
and VPS-based distributed feature extraction via Vast.ai.
"""
import sys
import os
import asyncio
import json
import datetime
import math
import shutil
import logging
import tempfile
import time
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFormLayout, QTabWidget, QLineEdit, QMessageBox,
    QFileDialog, QCheckBox, QSlider, QFrame, QRadioButton, QButtonGroup,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QDialogButtonBox,
    QAbstractItemView, QMenu, QTextEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QTimer, QProcess
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtGui import QPalette, QColor, QPixmap, QPainter

from scraper import UnifiedScraper, UnifiedScraperConfig, _cpu_stitch_and_extract

# VPS modules
try:
    from r2_storage import R2Client
    from csv_splitter import split_csv, upload_csv_segments
    from vast_manager import VastManager, ContainerNotFoundError
    from log_monitor import R2StatusMonitorThread
    VPS_AVAILABLE = True
except ImportError:
    VPS_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


def _sanitize_geo(s):
    """Sanitize a geocoding string for use in file/R2 paths."""
    return "".join(c for c in str(s) if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')


def _reverse_geocode(lat: float, lon: float):
    """
    Reverse geocode coordinates to (country, state, city) using Nominatim.

    Tries multiple zoom levels and an expanded field fallback chain to avoid
    returning 'Unknown' for rural/remote areas.

    Returns:
        (country, state, city) — sanitized strings, never 'Unknown' for city
        unless all strategies fail (falls back to coordinate-based name).
    """
    import urllib.request

    # Nominatim address fields to try for city-level resolution, in priority order.
    # Rural areas (especially Australia) often lack 'city'/'town' but have
    # 'suburb', 'municipality', 'county', 'hamlet', 'locality', etc.
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

    best_country = None
    best_state = None
    best_city = None

    # Try zoom levels: 10 (city), 8 (county), 6 (region)
    for zoom in (10, 8, 6):
        try:
            data = _try_geocode(zoom)
            addr = data.get('address', {})

            # Country — take first valid
            if not best_country:
                cc = addr.get('country_code', '')
                if cc:
                    best_country = cc.upper()

            # State — take first valid
            if not best_state:
                for f in STATE_FIELDS:
                    val = addr.get(f, '').strip()
                    if val:
                        best_state = val
                        break

            # City — take first valid from the priority chain
            if not best_city:
                for f in CITY_FIELDS:
                    val = addr.get(f, '').strip()
                    if val:
                        best_city = val
                        logger.debug(f"Geocode zoom={zoom}: city from '{f}' = '{val}'")
                        break

                # Last resort: parse display_name for a locality
                if not best_city and 'display_name' in data:
                    parts = [p.strip() for p in data['display_name'].split(',')]
                    # First non-numeric, non-country part is usually the locality
                    for part in parts:
                        if part and not part.replace(' ', '').isdigit() and len(part) > 1:
                            best_city = part
                            logger.debug(f"Geocode zoom={zoom}: city from display_name = '{val}'")
                            break

            if best_country and best_state and best_city:
                break  # Got everything, no need to try more zoom levels

            # Rate-limit Nominatim (1 req/sec policy)
            time.sleep(1.1)
        except Exception as e:
            logger.warning(f"Geocode failed at zoom={zoom}: {e}")
            continue

    # Final fallbacks
    country = _sanitize_geo(best_country) if best_country else "Unknown"
    state = _sanitize_geo(best_state) if best_state else "Unknown"

    if best_city:
        city = _sanitize_geo(best_city)
    else:
        # Coordinate-based fallback — never return "Unknown"
        city = f"Region_{lat:.4f}_{lon:.4f}".replace('-', 'S').replace('.', 'd')
        logger.warning(f"Geocoding returned no city for ({lat}, {lon}), using: {city}")

    return country, state, city



class VPSDeployThread(QThread):
    """Thread to run blocking VPS deployment steps off the UI thread."""
    status_update = pyqtSignal(str)
    geocode_done = pyqtSignal(str, str, str)  # country, state, city
    r2_segments_found = pyqtSignal(int)  # count of existing CSV segments
    provision_ready = pyqtSignal(list)  # offers list
    error_occurred = pyqtSignal(str)

    def __init__(self, context, mode="full"):
        super().__init__()
        self.context = context
        self.mode = mode  # "full" = geocode+r2check, "provision" = search offers only

    def run(self):
        try:
            if self.mode == "full":
                self._do_geocode_and_r2_check()
            elif self.mode == "provision":
                self._do_search_offers()
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _do_geocode_and_r2_check(self):
        ctx = self.context

        # Use manual path if provided, otherwise geocode
        if ctx.get('country') and ctx.get('state') and ctx.get('city'):
            country, state, city = ctx['country'], ctx['state'], ctx['city']
            print(f"[DEPLOY] using manual path: country={country!r}, state={state!r}, city={city!r}")
        else:
            coords = ctx['first_coords']
            self.status_update.emit("VPS Deploy: Geocoding region...")
            lats = [c[0] for c in coords]
            lons = [c[1] for c in coords]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)

            print(f"[DEPLOY] center_lat={center_lat:.6f}, center_lon={center_lon:.6f}")
            country, state, city = _reverse_geocode(center_lat, center_lon)
            print(f"[DEPLOY] geocode result: country={country!r}, state={state!r}, city={city!r}")
            logger.info(f"Geocoded ({center_lat:.4f}, {center_lon:.4f}) → {country}/{state}/{city}")

        self.geocode_done.emit(country, state, city)

        # R2 check — CSVs and Features
        self.status_update.emit("VPS Deploy: Checking R2 for existing CSV segments...")
        try:
            r2 = R2Client(ctx['r2_account'], ctx['r2_access'], ctx['r2_secret'], ctx['r2_bucket'])
            prefix = f"CSV/{country}/{state}/{city}/"
            print(f"[DEPLOY] checking R2 CSV prefix: {prefix!r}")
            self.status_update.emit(f"VPS Deploy: Checking R2 path: {prefix}")
            files = r2.list_files(prefix)
            r2_csv_files = [f for f in files if f['key'].endswith(".csv")]
            print(f"[DEPLOY] R2 list_files returned {len(files)} total objects, {len(r2_csv_files)} .csv files")
            for f in r2_csv_files:
                print(f"[DEPLOY]   CSV found: {f['key']}")
            logger.info(f"R2 CSV check: prefix={prefix!r}, total_files={len(files)}, csv_files={len(r2_csv_files)}")

            if r2_csv_files:
                segment_count = len(r2_csv_files)
                print(f"[DEPLOY] segment_count={segment_count}, proceeding to feature scan")
                self.status_update.emit(
                    f"VPS Deploy: Found {segment_count} CSVs at {prefix}. Checking features..."
                )
                completed = self._scan_completed_features(r2, country, state, city, segment_count)
                print(f"[DEPLOY] feature scan complete: completed={completed}")
                ctx['completed_worker_indices'] = completed
                self.r2_segments_found.emit(segment_count)
                return

            print(f"[DEPLOY] no CSVs found at {prefix!r}, will scrape fresh")
            self.status_update.emit(f"VPS Deploy: No CSVs found at {prefix}. Will scrape fresh.")
        except Exception as e:
            print(f"[DEPLOY] R2 check exception: {e}")
            logger.warning(f"R2 Check failed (proceeding with scrape): {e}")

        # No CSVs found — scrape fresh
        ctx['completed_worker_indices'] = []
        self.r2_segments_found.emit(0)

    def _scan_completed_features(self, r2, country, state, city, total_workers):
        """Return sorted list of worker indices that have uploaded both .npy and Metadata_.jsonl."""
        features_prefix = f"Features/{country}/{state}/{city}/"
        print(f"[SCAN] scanning features prefix: {features_prefix!r}, total_workers={total_workers}")
        try:
            files = r2.list_files(features_prefix)
        except Exception as e:
            print(f"[SCAN] list_files exception: {e}")
            logger.warning(f"Features scan failed: {e}")
            return []

        print(f"[SCAN] {len(files)} objects under {features_prefix!r}")
        npy_indices = set()
        meta_indices = set()

        for f in files:
            fname = f['key'].split('/')[-1]
            try:
                if fname.startswith("Metadata_") and fname.endswith(".jsonl"):
                    inner = fname[len("Metadata_"):-len(".jsonl")]
                    base, total_str = inner.rsplit('.', 1)
                    _city, idx_str = base.rsplit('_', 1)
                    match = idx_str.isdigit() and int(total_str) == total_workers
                    print(f"[SCAN]   META {fname!r}: idx={idx_str}, total={total_str}, match={match}")
                    if match:
                        meta_indices.add(int(idx_str))
                elif fname.endswith(".npy"):
                    base, total_str = fname[:-4].rsplit('.', 1)
                    _city, idx_str = base.rsplit('_', 1)
                    match = idx_str.isdigit() and int(total_str) == total_workers
                    print(f"[SCAN]   NPY  {fname!r}: idx={idx_str}, total={total_str}, match={match}")
                    if match:
                        npy_indices.add(int(idx_str))
                else:
                    print(f"[SCAN]   SKIP {fname!r}")
            except (ValueError, IndexError) as e:
                print(f"[SCAN]   PARSE ERROR {fname!r}: {e}")
                continue

        print(f"[SCAN] npy_indices={sorted(npy_indices)}")
        print(f"[SCAN] meta_indices={sorted(meta_indices)}")
        completed = sorted(npy_indices & meta_indices)
        npy_only = sorted(npy_indices - meta_indices)
        meta_only = sorted(meta_indices - npy_indices)
        print(f"[SCAN] completed (both)={completed}")
        print(f"[SCAN] npy only (no meta)={npy_only}")
        print(f"[SCAN] meta only (no npy)={meta_only}")
        logger.info(f"Completed workers ({len(completed)}/{total_workers}): {completed}")
        return completed

    def _do_search_offers(self):
        ctx = self.context
        self.status_update.emit("VPS Deploy: Searching Vast.ai offers...")
        manager = VastManager(api_key=ctx['vast_key'])

        # Build geolocation filter from comma-separated country codes
        geo_filter = ctx.get('geo_filter', '').strip()
        geo_region = None
        if geo_filter:
            # Vast.ai geolocation accepts country codes like "US" or "US CA GB"
            codes = [c.strip().upper() for c in geo_filter.replace(',', ' ').split() if c.strip()]
            if codes:
                geo_region = " ".join(codes)

        offers = manager.search_offers(
            gpu_type=ctx['gpu_type'],
            region=geo_region,
            min_disk_gb=ctx['disk_gb'],
            max_price=ctx['max_price'],
            min_ram_gb=ctx.get('min_vram_gb', 21),
        )
        # Store manager on context so main thread can use it
        ctx['_vast_manager'] = manager
        self.provision_ready.emit(offers if offers else [])


class VPSInstanceCreateThread(QThread):
    """Thread to create Vast.ai instances without freezing the UI."""
    status_update = pyqtSignal(str)
    instances_created = pyqtSignal(dict)  # instance_worker_map
    error_occurred = pyqtSignal(str)

    def __init__(self, vast_manager, selected_offers, worker_indices, total_workers, ctx):
        """
        Args:
            worker_indices: Specific WORKER_INDEX values to deploy (e.g. [5, 10, 11]).
            total_workers: Full NUM_WORKERS value (e.g. 20) used for CSV filename and env var.
        """
        super().__init__()
        self.vast_manager = vast_manager
        self.selected_offers = selected_offers
        self.worker_indices = worker_indices
        self.total_workers = total_workers
        self.ctx = ctx

    def run(self):
        try:
            # R2 client for writing instance ID mappings
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
                    disk_gb=self.ctx['disk_gb'],
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
                    except Exception as e:
                        logger.warning(f"Failed to write instance ID to R2 for worker {worker_idx}: {e}")
                    self.status_update.emit(
                        f"VPS Deploy: Created worker {worker_idx} "
                        f"({i + 1}/{len(self.worker_indices)}) — instance {instance_id}"
                    )
                else:
                    self.status_update.emit(f"VPS Deploy: FAILED to create worker {worker_idx}")

            self.instances_created.emit(instance_worker_map)
        except Exception as e:
            self.error_occurred.emit(str(e))


class LogFetcherThread(QThread):
    """Async thread to fetch logs from Vast.ai to prevent UI freeze."""
    logs_fetched = pyqtSignal(int, str, str)  # worker_idx, instance_id, logs
    error_occurred = pyqtSignal(int, str, str) # worker_idx, instance_id, error_msg

    def __init__(self, vast_manager, instance_id, worker_idx, tail=2000):
        super().__init__()
        self.vast_manager = vast_manager
        self.instance_id = instance_id
        self.worker_idx = worker_idx
        self.tail = tail

    def run(self):
        try:
            logs = self.vast_manager.get_instance_logs(self.instance_id, tail=self.tail)
            self.logs_fetched.emit(self.worker_idx, self.instance_id, logs)
        except Exception as e:
            self.error_occurred.emit(self.worker_idx, self.instance_id, str(e))


class VPSMonitorWindow(QDialog):
    """Floating window showing live status of all VPS workers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VPS Worker Monitor")
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.resize(1000, 520)
        self._parent_win = parent
        self._worker_row_map = {}  # worker_idx -> row
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Overall progress ────────────────────────────────────────────
        top = QHBoxLayout()
        self.overall_bar = QProgressBar()
        self.overall_bar.setFormat("Overall: %v / %m panos  (%p%)")
        self.overall_bar.setMinimumHeight(22)
        top.addWidget(self.overall_bar, stretch=4)
        self.overall_label = QLabel("Waiting for workers...")
        self.overall_label.setStyleSheet("color: #aaaaaa; margin-left: 8px;")
        top.addWidget(self.overall_label, stretch=1)
        layout.addLayout(top)

        # ── Worker table ────────────────────────────────────────────────
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Worker", "Instance ID", "Status", "Progress", "Speed", "ETA", "Logs"]
        )
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)

        # ── Bottom buttons ───────────────────────────────────────────────
        btn_layout = QHBoxLayout()

        self.destroy_all_btn = QPushButton("Destroy All Instances")
        self.destroy_all_btn.setStyleSheet(
            "QPushButton { background-color: #cc0000; color: white; padding: 6px 12px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #ff2222; }"
        )
        self.destroy_all_btn.clicked.connect(self._on_destroy_all)
        btn_layout.addWidget(self.destroy_all_btn)

        self.export_btn = QPushButton("Export All Logs")
        self.export_btn.setStyleSheet(
            "QPushButton { background-color: #0066cc; color: white; padding: 6px 12px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #0088ff; }"
        )
        self.export_btn.clicked.connect(self._on_export_all)
        btn_layout.addWidget(self.export_btn)

        btn_layout.addStretch()

        hide_btn = QPushButton("Hide")
        hide_btn.setStyleSheet("padding: 6px 12px;")
        hide_btn.clicked.connect(self.hide)
        btn_layout.addWidget(hide_btn)

        layout.addLayout(btn_layout)

    def setup_workers(self, instance_worker_map):
        """Initialize table rows for all workers."""
        self._worker_row_map = {}
        sorted_workers = sorted(instance_worker_map.items(), key=lambda x: x[1])
        self.table.setRowCount(len(sorted_workers))
        for row, (iid, widx) in enumerate(sorted_workers):
            self._worker_row_map[widx] = (row, str(iid))
            self.table.setItem(row, 0, QTableWidgetItem(f"#{widx}"))
            self.table.setItem(row, 1, QTableWidgetItem(str(iid)))
            status_item = QTableWidgetItem("Starting...")
            status_item.setForeground(QColor("#aaaaaa"))
            self.table.setItem(row, 2, status_item)
            self.table.setItem(row, 3, QTableWidgetItem("—"))
            self.table.setItem(row, 4, QTableWidgetItem("—"))
            self.table.setItem(row, 5, QTableWidgetItem("—"))

            logs_btn = QPushButton("Logs")
            logs_btn.setStyleSheet(
                "QPushButton { padding: 2px 8px; background: #444; border-radius: 3px; }"
                "QPushButton:hover { background: #666; }"
            )
            logs_btn.clicked.connect(
                lambda checked, w=widx, i=str(iid): self._on_view_logs(w, i)
            )
            self.table.setCellWidget(row, 6, logs_btn)

    def update_worker(self, worker_idx, processed, total, eta, speed, status):
        """Update a single worker row with latest progress."""
        entry = self._worker_row_map.get(worker_idx)
        if entry is None:
            return
        row, _ = entry

        if status == "UPLOADING" or (status.startswith("UPLOADING_") and not status.endswith("_RETRY")):
            if total > 0:
                mb_done = processed / (1024 * 1024)
                mb_total = total / (1024 * 1024)
                pct = int(processed / total * 100)
                progress_str = f"{pct}%  ({mb_done:.0f}/{mb_total:.0f} MB)"
                speed_str = f"{speed:.1f} MB/s"
            else:
                progress_str = "Uploading..."
                speed_str = "—"
            eta_str = f"{int(eta // 60)}m {int(eta % 60)}s" if eta > 0 else "—"
            display_status = "UPLOADING"
        elif status.endswith("_RETRY"):
            progress_str = "Retrying..."
            speed_str = "—"
            eta_str = "—"
            display_status = "UPLOAD RETRY"
        else:
            pct = int(processed / total * 100) if total > 0 else 0
            progress_str = f"{pct}%  ({processed:,} / {total:,})"
            speed_str = f"{speed:.1f}/s" if speed > 0 else "—"
            if eta > 3600:
                eta_str = f"{int(eta // 3600)}h {int((eta % 3600) // 60)}m"
            elif eta > 0:
                eta_str = f"{int(eta // 60)}m {int(eta % 60)}s"
            else:
                eta_str = "—"
            display_status = status

        status_item = QTableWidgetItem(display_status)
        if display_status == "COMPLETED":
            status_item.setForeground(QColor("#00cc00"))
        elif display_status in ("EXTRACTING",):
            status_item.setForeground(QColor("#00aaff"))
        elif display_status.startswith("FAIL") or "DEAD" in display_status or "ERROR" in display_status:
            status_item.setForeground(QColor("#ff4444"))
        elif display_status == "UPLOADING":
            status_item.setForeground(QColor("#ffaa00"))

        self.table.setItem(row, 2, status_item)
        self.table.setItem(row, 3, QTableWidgetItem(progress_str))
        self.table.setItem(row, 4, QTableWidgetItem(speed_str))
        self.table.setItem(row, 5, QTableWidgetItem(eta_str))

    def mark_worker_status(self, worker_idx, status_text, color="#ffffff"):
        """Set a worker's status cell to an arbitrary string with a color."""
        entry = self._worker_row_map.get(worker_idx)
        if entry is None:
            return
        row, _ = entry
        item = QTableWidgetItem(status_text)
        item.setForeground(QColor(color))
        self.table.setItem(row, 2, item)

    def update_overall(self, processed, total, active, finished, total_workers):
        capped_total = min(max(total, 1), 2147483647)
        self.overall_bar.setMaximum(capped_total)
        self.overall_bar.setValue(min(processed, capped_total))
        self.overall_label.setText(
            f"{active} active · {finished}/{total_workers} done"
        )

    def _on_view_logs(self, worker_idx, instance_id):
        if self._parent_win:
            self._parent_win._show_worker_logs(instance_id, worker_idx)

    def _on_destroy_all(self):
        if self._parent_win:
            self._parent_win._destroy_all_instances()

    def _on_export_all(self):
        if self._parent_win:
            self._parent_win._export_all_logs()


class BuilderSearchThread(QThread):
    """Thread to search Vast.ai offers for the builder instance (high RAM/disk)."""
    status_update = pyqtSignal(str)
    offers_found = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, vast_key, max_price, min_disk_gb):
        super().__init__()
        self.vast_key = vast_key
        self.max_price = max_price
        self.min_disk_gb = min_disk_gb

    def run(self):
        try:
            self.status_update.emit("Searching for high-RAM builder instances...")
            manager = VastManager(api_key=self.vast_key)

            # Search: any GPU/CPU, min 128GB RAM, min disk from UI
            query_parts = [
                f"disk_space>={self.min_disk_gb}",
                "cpu_ram>=128",  # GB (vastai uses GB)
                "rentable=true",
            ]
            if self.max_price:
                query_parts.append(f"dph_total<={self.max_price}")

            query = " ".join(query_parts)
            print(f"DEBUG: Builder Vast.ai Query: {query}")

            import json as _json
            try:
                from vast_manager import _run_vastai
                raw = _run_vastai(
                    "search", "offers", query,
                    "--order", "dph_total",
                    "--raw",
                    api_key=self.vast_key,
                )
                offers = _json.loads(raw)
            except Exception as e:
                self.error_occurred.emit(f"Offer search failed: {e}")
                return

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

            self._vast_manager = manager
            self.offers_found.emit(results)

        except Exception as e:
            self.error_occurred.emit(str(e))


class BuilderMonitorThread(QThread):
    """Poll R2 for builder progress updates."""
    progress_update = pyqtSignal(str, str, int, str)  # step, detail, pct, status
    builder_finished = pyqtSignal(str)  # status

    def __init__(self, r2_client, status_prefix, poll_interval=10.0):
        super().__init__()
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

                    self.progress_update.emit(step, detail, pct, status)

                    if status in ("COMPLETED", "FAILED"):
                        self.builder_finished.emit(status)
                        return
            except Exception:
                pass

            time.sleep(self.poll_interval)

    def stop(self):
        self._running = False


class ScraperThread(QThread):

    """Thread for running async scraper with two-phase progress tracking."""
    progress = pyqtSignal(int, int)
    status = pyqtSignal(str)
    phase_status = pyqtSignal(str)  # "metadata" or "images"
    finished_signal = pyqtSignal(list)
    error = pyqtSignal(str)
    # Two-phase point signals
    point_found = pyqtSignal(float, float, str)  # Metadata discovered (blue)
    point_completed = pyqtSignal(str)  # Image downloaded (panoid -> orange)
    
    def __init__(self, scraper, mode, **kwargs):
        super().__init__()
        self.scraper = scraper
        self.mode = mode
        self.kwargs = kwargs
        self._is_cancelled = False
    
    def run(self):
        try:
            # Use winloop for ~5x faster asyncio on Windows
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
            elif self.mode == "random":
                result = loop.run_until_complete(
                    self.scraper.scrape_random_global_two_phase(
                        self.kwargs['target_count'],
                        self.kwargs['radius']
                    )
                )
            else:
                result = []
            
            loop.close()
            self.finished_signal.emit(result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
    
    def cancel(self):
        """Cancel the scraping operation."""
        self._is_cancelled = True
        self.scraper.cancel()



class DeckGLMapView(QWebEngineView):
    """Web view with Deck.gl map for shapes and point visualization."""
    
    MAX_AREA_SQKM = 10000  # Maximum area in square kilometers
    
    def __init__(self):
        super().__init__()
        self.polygons = []
        self.all_shapes_data = []  # Store all loaded shapes with metadata
        self.load_map()
    
    def load_map(self):
        """Load Deck.gl map with polygon and point layers."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://unpkg.com/deck.gl@8.9.33/dist.min.js"></script>
    <style>
        body { margin: 0; padding: 0; background: #1a1a2e; }
        #container { position: absolute; top: 0; bottom: 0; width: 100%; }
        #controls {
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 100;
            background: rgba(30, 30, 30, 0.95);
            padding: 12px;
            border-radius: 8px;
            color: white;
            font-family: Arial, sans-serif;
            font-size: 12px;
        }
        #controls button {
            background: #0066cc;
            color: white;
            border: none;
            padding: 6px 12px;
            margin: 3px;
            border-radius: 4px;
            cursor: pointer;
        }
        #controls button:hover { background: #0088ff; }
        #controls button.active { background: #ff8800; }
        .draw-btn { display: none; }
        .draw-btn.show { display: inline-block; }
        #draw-status {
            color: #ffcc00;
            font-size: 11px;
            margin-top: 4px;
            display: none;
        }
        #selection-info {
            position: absolute;
            bottom: 10px;
            left: 10px;
            z-index: 100;
            background: rgba(0, 150, 0, 0.95);
            padding: 10px 15px;
            border-radius: 8px;
            color: white;
            font-family: Arial, sans-serif;
            font-size: 14px;
            font-weight: bold;
            display: none;
        }
        #tooltip {
            position: absolute;
            z-index: 200;
            pointer-events: none;
            background: rgba(0, 0, 0, 0.85);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-family: Arial, sans-serif;
            font-size: 12px;
            display: none;
        }
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="controls">
        <div id="shape-count">No shapes loaded</div>
        <button onclick="selectAll()">Select All</button>
        <button onclick="clearSelection()">Clear Selection</button>
        <button id="draw-btn" onclick="toggleDrawing()">Draw</button>
        <button class="draw-btn" id="finish-btn" onclick="finishDrawing()">Finish</button>
        <button class="draw-btn" id="cancel-btn" onclick="cancelDrawing()">Cancel</button>
        <button onclick="deleteSelected()">Delete Selected</button>
        <div id="draw-status"></div>
    </div>
    <div id="selection-info">0 shapes selected</div>
    <div id="tooltip"></div>
    
    <script>
        // State
        let allShapes = [];
        let selectedIndices = new Set();
        let scrapedPoints = [];
        let deckgl = null;
        let isDrawing = false;
        let drawingCoords = [];
        let drawnShapeCount = 0;
        let drawRevision = 0;
        let maskShapes = [];
        let maskRevision = 0;
        let mainShapes = [];
        let isIntersectionMode = false;

        // --- Adaptive hex binning ---
        let lastBinZoom = -1;
        let hexBinCache = [];

        function getHexSize(zoom) {
            return 180 / Math.pow(2, zoom + 3);
        }

        function hexVertices(cx, cy, size) {
            const s = size * 0.92;
            const latScale = 1 / Math.cos(cy * Math.PI / 180);
            const v = [];
            for (let i = 0; i < 6; i++) {
                const angle = Math.PI / 3 * i - Math.PI / 6;
                v.push([cx + s * Math.cos(angle) * latScale, cy + s * Math.sin(angle)]);
            }
            return v;
        }

        function rebinHexagons() {
            const zoom = Math.floor(currentViewState.zoom);
            if (zoom === lastBinZoom) return;
            lastBinZoom = zoom;
            if (scrapedPoints.length === 0) { hexBinCache = []; return; }

            const size = getHexSize(zoom);
            const rowH = size * 1.5;
            const colW = size * 2;
            const bins = new Map();

            for (let i = 0; i < scrapedPoints.length; i++) {
                const p = scrapedPoints[i];
                const row = Math.round(p.lat / rowH);
                const isOdd = ((row % 2) + 2) % 2 === 1;
                const col = Math.round((p.lon - (isOdd ? size : 0)) / colW);
                const key = col * 131071 + row;
                let bin = bins.get(key);
                if (!bin) {
                    bin = { cLon: col * colW + (isOdd ? size : 0), cLat: row * rowH, n: 0, c: 0 };
                    bins.set(key, bin);
                }
                bin.n++;
                if (p.completed) bin.c++;
            }

            hexBinCache = [];
            let mx = 1;
            for (const b of bins.values()) { if (b.n > mx) mx = b.n; }
            for (const b of bins.values()) {
                hexBinCache.push({
                    polygon: hexVertices(b.cLon, b.cLat, size),
                    count: b.n, completed: b.c, norm: b.n / mx
                });
            }
        }

        let currentViewState = {
            longitude: -98.5795,
            latitude: 39.8283,
            zoom: 3,
            pitch: 0,
            bearing: 0
        };
        
        // Initialize Deck.gl
        function initDeckGL() {
            deckgl = new deck.DeckGL({
                container: 'container',
                initialViewState: currentViewState,
                controller: true,
                onViewStateChange: ({viewState}) => {
                    const oldZoom = Math.floor(currentViewState.zoom);
                    currentViewState = viewState;
                    if (Math.floor(viewState.zoom) !== oldZoom && scrapedPoints.length > 0) {
                        lastBinZoom = -1;
                        updateLayers();
                    }
                    return viewState;
                },
                layers: [
                    // Base map tile layer
                    new deck.TileLayer({
                        id: 'base-map',
                        data: 'https://c.tile.openstreetmap.org/{z}/{x}/{y}.png',
                        minZoom: 0,
                        maxZoom: 19,
                        tileSize: 256,
                        renderSubLayers: props => {
                            const {bbox: {west, south, east, north}} = props.tile;
                            return new deck.BitmapLayer(props, {
                                data: null,
                                image: props.data,
                                bounds: [west, south, east, north]
                            });
                        }
                    })
                ],
                onClick: handleClick,
                onHover: handleHover
            });
            console.log('Deck.gl initialized');
        }
        
        function handleClick(info, event) {
            // Drawing mode: place vertex at click location
            if (isDrawing) {
                const coord = info.coordinate;
                if (coord) {
                    drawingCoords.push([coord[0], coord[1]]);
                    drawRevision++;
                    const ds = document.getElementById('draw-status');
                    ds.textContent = drawingCoords.length + ' vertices placed' + (drawingCoords.length < 3 ? ' (need at least 3)' : '');
                    updateLayers();
                }
                return;
            }

            if (!info.object || info.layer.id === 'base-map' || info.layer.id === 'main-outlines') return;

            const idx = info.object.index;
            if (idx === undefined) return;

            // Toggle: click selected (green) -> deselect (blue), click unselected (blue) -> select (green)
            if (selectedIndices.has(idx)) {
                selectedIndices.delete(idx);
            } else {
                selectedIndices.add(idx);
            }

            selectionRevision++;
            updateLayers();
            updateSelectionInfo();
        }
        
        function handleHover(info) {
            const tooltip = document.getElementById('tooltip');
            if (info.object && info.layer.id !== 'base-map') {
                const props = info.object.properties || {};
                const label = props.city || props.name || props.NAME || 'Shape';
                const area = props.area_sqkm ? props.area_sqkm.toFixed(1) + ' km²' : '';
                tooltip.innerHTML = '<b>' + label + '</b>' + (area ? '<br>Area: ' + area : '');
                tooltip.style.display = 'block';
                tooltip.style.left = (info.x + 10) + 'px';
                tooltip.style.top = (info.y + 10) + 'px';
            } else {
                tooltip.style.display = 'none';
            }
        }
        
        function updateLayers() {
            if (!deckgl) return;
            
            const layers = [
                // Base map tile layer
                new deck.TileLayer({
                    id: 'base-map',
                    data: 'https://c.tile.openstreetmap.org/{z}/{x}/{y}.png',
                    minZoom: 0,
                    maxZoom: 19,
                    tileSize: 256,
                    renderSubLayers: props => {
                        const {bbox: {west, south, east, north}} = props.tile;
                        return new deck.BitmapLayer(props, {
                            data: null,
                            image: props.data,
                            bounds: [west, south, east, north]
                        });
                    }
                })
            ];
            
            // Selectable shapes (counties or pre-computed intersection shapes)
            if (allShapes.length > 0) {
                layers.push(new deck.PolygonLayer({
                    id: 'shapes',
                    data: allShapes,
                    getPolygon: d => d.coordinates,
                    getFillColor: d => {
                        if (selectedIndices.has(d.index)) return [0, 255, 0, 150]; // Green (Selected)
                        if (d.properties.done) return [255, 0, 0, 150]; // Red (Done)
                        return [0, 100, 255, 100]; // Blue (Default)
                    },
                    getLineColor: d => {
                        if (selectedIndices.has(d.index)) return [0, 255, 0, 255];
                        if (d.properties.done) return [255, 0, 0, 255];
                        return [0, 150, 255, 200];
                    },
                    lineWidthMinPixels: 2,
                    pickable: true,
                    filled: true,
                    stroked: true,
                    extruded: false,
                    wireframe: false,
                    updateTriggers: {
                        getFillColor: [selectionRevision, dataRevision],
                        getLineColor: [selectionRevision, dataRevision]
                    }
                }));
            }

            // County boundary outlines (only in intersection mode)
            if (isIntersectionMode && mainShapes.length > 0) {
                layers.push(new deck.PolygonLayer({
                    id: 'main-outlines',
                    data: mainShapes,
                    getPolygon: d => d.coordinates,
                    getFillColor: [0, 0, 0, 0],
                    getLineColor: [255, 255, 255, 200],
                    lineWidthMinPixels: 3,
                    filled: true,
                    stroked: true,
                    pickable: false
                }));
            }

            // Adaptive hex coverage layer
            // Blue = pending, Orange = completed, alpha = density
            if (scrapedPoints.length > 0) {
                rebinHexagons();
                layers.push(new deck.PolygonLayer({
                    id: 'hex-coverage',
                    data: hexBinCache,
                    getPolygon: d => d.polygon,
                    getFillColor: d => {
                        const alpha = Math.floor(60 + d.norm * 195);
                        if (d.completed > d.count * 0.5) return [255, 140, 0, alpha];
                        return [30, 144, 255, alpha];
                    },
                    getLineColor: [255, 255, 255, 30],
                    lineWidthMinPixels: 0.5,
                    filled: true,
                    stroked: true,
                    pickable: false,
                    updateTriggers: {
                        getFillColor: dataRevision,
                        getPolygon: lastBinZoom
                    }
                }));
            }

            // Drawing-in-progress layers
            if (isDrawing && drawingCoords.length > 0) {
                // Show the polygon outline being drawn
                const pathData = drawingCoords.length >= 2
                    ? [{ path: [...drawingCoords, drawingCoords[0]] }]
                    : [];
                if (pathData.length > 0) {
                    layers.push(new deck.PathLayer({
                        id: 'drawing-line',
                        data: pathData,
                        getPath: d => d.path,
                        getColor: [255, 200, 0, 220],
                        widthMinPixels: 2,
                        getDashArray: [6, 4],
                        dashJustified: true,
                        extensions: [new deck.PathStyleExtension({dash: true})],
                        updateTriggers: { getPath: drawRevision }
                    }));
                }
                // Show placed vertices as dots
                layers.push(new deck.ScatterplotLayer({
                    id: 'drawing-vertices',
                    data: drawingCoords.map((c, i) => ({ pos: c, i })),
                    getPosition: d => d.pos,
                    getFillColor: [255, 200, 0, 255],
                    getRadius: 200,
                    radiusMinPixels: 5,
                    radiusMaxPixels: 8,
                    pickable: false,
                    updateTriggers: { getPosition: drawRevision }
                }));
            }

            deckgl.setProps({ layers });
        }
        
        function updateSelectionInfo() {
            const info = document.getElementById('selection-info');
            const count = selectedIndices.size;
            if (count > 0) {
                info.style.display = 'block';
                info.textContent = count + ' shape(s) selected';
            } else {
                info.style.display = 'none';
            }
        }
        
        // Calculate polygon area in square kilometers (approximate)
        function calculateAreaSqKm(coords) {
            if (!coords || coords.length < 3) return 0;
            
            // Simple spherical excess formula
            const toRad = Math.PI / 180;
            const R = 6371;
            let total = 0;
            
            for (let i = 0; i < coords.length; i++) {
                const j = (i + 1) % coords.length;
                const lng1 = coords[i][0] * toRad;
                const lat1 = coords[i][1] * toRad;
                const lng2 = coords[j][0] * toRad;
                const lat2 = coords[j][1] * toRad;
                
                total += (lng2 - lng1) * (2 + Math.sin(lat1) + Math.sin(lat2));
            }
            
            return Math.abs(total * R * R / 2);
        }
        
        // Load shapes from GeoJSON
        function loadShapes(geojsonData, maxAreaSqKm) {
            try {
                allShapes = [];
                selectedIndices.clear();
                
                if (typeof geojsonData === 'string') {
                    geojsonData = JSON.parse(geojsonData);
                }
                
                let loadedCount = 0;
                let prunedCount = 0;
                
                if (!geojsonData.features) {
                    console.error('No features in GeoJSON');
                    return { loaded: 0, pruned: 0 };
                }
                
                geojsonData.features.forEach(function(feature, featureIdx) {
                    if (!feature.geometry) return;
                    
                    const geomType = feature.geometry.type;
                    let coordSets = [];
                    
                    try {
                        if (geomType === 'MultiPolygon') {
                            feature.geometry.coordinates.forEach(poly => {
                                if (poly && poly[0]) coordSets.push(poly[0]);
                            });
                        } else if (geomType === 'Polygon') {
                            if (feature.geometry.coordinates && feature.geometry.coordinates[0]) {
                                coordSets.push(feature.geometry.coordinates[0]);
                            }
                        }
                    } catch (e) {
                        console.warn('Error parsing geometry for feature', featureIdx, e);
                        return;
                    }
                    
                    coordSets.forEach(function(rawCoords) {
                        if (!rawCoords || rawCoords.length < 3) return;
                        
                        // Validate and clean coordinates [lng, lat]
                        const coords = [];
                        for (let i = 0; i < rawCoords.length; i++) {
                            const c = rawCoords[i];
                            if (c && c.length >= 2) {
                                const lng = parseFloat(c[0]);
                                const lat = parseFloat(c[1]);
                                // Validate ranges
                                if (!isNaN(lng) && !isNaN(lat) && 
                                    lng >= -180 && lng <= 180 && 
                                    lat >= -90 && lat <= 90) {
                                    coords.push([lng, lat]);
                                }
                            }
                        }
                        
                        if (coords.length < 3) return;
                        
                        // Calculate area
                        const areaSqKm = calculateAreaSqKm(coords);
                        const isPruned = areaSqKm > maxAreaSqKm;
                        
                        if (isPruned) {
                            prunedCount++;
                            return; // Skip pruned shapes entirely
                        }
                        loadedCount++;

                        allShapes.push({
                            index: allShapes.length,
                            coordinates: coords,
                            properties: {
                                ...feature.properties,
                                area_sqkm: areaSqKm
                            },
                            pruned: false
                        });
                    });
                });
                
                document.getElementById('shape-count').textContent = 
                    loadedCount + ' shapes' + (prunedCount > 0 ? ' (' + prunedCount + ' pruned)' : '');
                
                console.log('Loaded ' + allShapes.length + ' shapes');
                mainShapes = allShapes.map(s => ({...s}));
                isIntersectionMode = false;

                updateLayers();

                // Fit to bounds
                if (allShapes.length > 0) {
                    fitToBounds();
                }
                
                return { loaded: loadedCount, pruned: prunedCount };
            } catch (e) {
                console.error('Error loading shapes:', e);
                return { loaded: 0, pruned: 0, error: e.message };
            }
        }
        
        function fitToBounds() {
            if (allShapes.length === 0) return;
            
            let minLng = 180, maxLng = -180;
            let minLat = 90, maxLat = -90;
            
            allShapes.forEach(shape => {
                shape.coordinates.forEach(coord => {
                    if (coord[0] < minLng) minLng = coord[0];
                    if (coord[0] > maxLng) maxLng = coord[0];
                    if (coord[1] < minLat) minLat = coord[1];
                    if (coord[1] > maxLat) maxLat = coord[1];
                });
            });
            
            const centerLng = (minLng + maxLng) / 2;
            const centerLat = (minLat + maxLat) / 2;
            
            // Calculate zoom based on extent
            const lngDiff = maxLng - minLng;
            const latDiff = maxLat - minLat;
            const maxDiff = Math.max(lngDiff, latDiff);
            let zoom = 1;
            if (maxDiff < 0.5) zoom = 10;
            else if (maxDiff < 1) zoom = 8;
            else if (maxDiff < 5) zoom = 6;
            else if (maxDiff < 20) zoom = 4;
            else if (maxDiff < 50) zoom = 3;
            else zoom = 2;
            
            currentViewState = {
                longitude: centerLng,
                latitude: centerLat,
                zoom: zoom,
                pitch: 0,
                bearing: 0,
                transitionDuration: 1000
            };
            
            deckgl.setProps({ initialViewState: currentViewState });
        }
        
        function selectAll() {
            selectedIndices.clear();
            allShapes.forEach(shape => {
                selectedIndices.add(shape.index);
            });
            selectionRevision++;
            updateLayers();
            updateSelectionInfo();
        }

        function clearSelection() {
            selectedIndices.clear();
            selectionRevision++;
            updateLayers();
            updateSelectionInfo();
        }
        
        function getSelectedCellsCoords() {
            const coords = [];
            selectedIndices.forEach(idx => {
                const shape = allShapes[idx];
                if (shape) {
                    // Convert to [lat, lng] for scraper
                    coords.push(shape.coordinates.map(c => [c[1], c[0]]));
                }
            });
            return JSON.stringify(coords);
        }

        function getSelectedCellsData() {
            const data = [];
            selectedIndices.forEach(idx => {
                const shape = allShapes[idx];
                if (shape) {
                    data.push(shape.properties);
                }
            });
            return JSON.stringify(data);
        }
        
        function getSelectedCount() {
            return selectedIndices.size;
        }
        
        // Two-phase point management
        // Points are added as pending (blue), then marked completed (orange)
        let pointsByPanoid = {};  // panoid -> index in scrapedPoints
        let dataRevision = 0;    // Bumped on every mutation so deck.gl recalculates attributes
        let selectionRevision = 0; // Bumped on every selection change so deck.gl recalculates colors

        function addPointsBatch(batch) {
            for (let i = 0; i < batch.length; i++) {
                const p = batch[i];
                const idx = scrapedPoints.length;
                scrapedPoints.push({ lat: p[0], lon: p[1], panoid: p[2], completed: false });
                pointsByPanoid[p[2]] = idx;
            }
            dataRevision++;
            lastBinZoom = -1;
            updateLayers();
        }

        function markCompletedBatch(panoids) {
            for (let i = 0; i < panoids.length; i++) {
                const idx = pointsByPanoid[panoids[i]];
                if (idx !== undefined && scrapedPoints[idx]) {
                    scrapedPoints[idx].completed = true;
                }
            }
            dataRevision++;
            lastBinZoom = -1;
            updateLayers();
        }
        
        function getPointStats() {
            const total = scrapedPoints.length;
            const completed = scrapedPoints.filter(p => p.completed).length;
            return JSON.stringify({ total, completed, pending: total - completed });
        }
        
        function clearPoints() {
            scrapedPoints = [];
            pointsByPanoid = {};
            hexBinCache = [];
            lastBinZoom = -1;
            updateLayers();
        }

        // --- Unload shapes ---
        function unloadShapes() {
            allShapes = [];
            mainShapes = [];
            isIntersectionMode = false;
            selectedIndices.clear();
            selectionRevision++;
            document.getElementById('shape-count').textContent = 'No shapes loaded';
            updateLayers();
            updateSelectionInfo();
        }

        // --- Mask shapes ---
        function loadMaskShapes(geojsonData, maxAreaSqKm) {
            try {
                maskShapes = [];
                if (typeof geojsonData === 'string') geojsonData = JSON.parse(geojsonData);
                if (!geojsonData.features) return { loaded: 0 };
                let count = 0;
                geojsonData.features.forEach(function(feature) {
                    if (!feature.geometry) return;
                    const geomType = feature.geometry.type;
                    let coordSets = [];
                    try {
                        if (geomType === 'MultiPolygon') {
                            feature.geometry.coordinates.forEach(poly => { if (poly && poly[0]) coordSets.push(poly[0]); });
                        } else if (geomType === 'Polygon') {
                            if (feature.geometry.coordinates && feature.geometry.coordinates[0])
                                coordSets.push(feature.geometry.coordinates[0]);
                        }
                    } catch(e) { return; }
                    coordSets.forEach(function(rawCoords) {
                        if (!rawCoords || rawCoords.length < 3) return;
                        const coords = [];
                        for (let i = 0; i < rawCoords.length; i++) {
                            const c = rawCoords[i];
                            if (c && c.length >= 2) {
                                const lng = parseFloat(c[0]), lat = parseFloat(c[1]);
                                if (!isNaN(lng) && !isNaN(lat) && lng >= -180 && lng <= 180 && lat >= -90 && lat <= 90)
                                    coords.push([lng, lat]);
                            }
                        }
                        if (coords.length < 3) return;
                        const areaSqKm = calculateAreaSqKm(coords);
                        if (areaSqKm > maxAreaSqKm) return;
                        count++;
                        maskShapes.push({
                            index: maskShapes.length,
                            coordinates: coords,
                            properties: { ...feature.properties, area_sqkm: areaSqKm }
                        });
                    });
                });
                maskRevision++;
                updateLayers();
                return { loaded: count };
            } catch(e) {
                console.error('Error loading mask shapes:', e);
                return { loaded: 0, error: e.message };
            }
        }

        function unloadMaskShapes() {
            maskShapes = [];
            maskRevision++;
            if (isIntersectionMode) {
                allShapes = mainShapes.map((s, i) => ({...s, index: i}));
                isIntersectionMode = false;
                selectedIndices.clear();
                selectionRevision++;
                document.getElementById('shape-count').textContent =
                    allShapes.length + ' shapes';
            }
            updateLayers();
            updateSelectionInfo();
        }

        function getMaskShapesCoords() {
            const coords = [];
            maskShapes.forEach(shape => {
                coords.push(shape.coordinates.map(c => [c[1], c[0]]));
            });
            return JSON.stringify(coords);
        }

        function getMaskShapeCount() {
            return maskShapes.length;
        }

        function loadIntersectionShapes(geojsonData, maxAreaSqKm) {
            try {
                allShapes = [];
                selectedIndices.clear();
                if (typeof geojsonData === 'string') geojsonData = JSON.parse(geojsonData);
                if (!geojsonData.features) return { loaded: 0 };
                let count = 0;
                geojsonData.features.forEach(function(feature) {
                    if (!feature.geometry) return;
                    const geomType = feature.geometry.type;
                    let coordSets = [];
                    try {
                        if (geomType === 'MultiPolygon') {
                            feature.geometry.coordinates.forEach(poly => { if (poly && poly[0]) coordSets.push(poly[0]); });
                        } else if (geomType === 'Polygon') {
                            if (feature.geometry.coordinates && feature.geometry.coordinates[0])
                                coordSets.push(feature.geometry.coordinates[0]);
                        }
                    } catch(e) { return; }
                    coordSets.forEach(function(rawCoords) {
                        if (!rawCoords || rawCoords.length < 3) return;
                        const coords = [];
                        for (let i = 0; i < rawCoords.length; i++) {
                            const c = rawCoords[i];
                            if (c && c.length >= 2) {
                                const lng = parseFloat(c[0]), lat = parseFloat(c[1]);
                                if (!isNaN(lng) && !isNaN(lat) && lng >= -180 && lng <= 180 && lat >= -90 && lat <= 90)
                                    coords.push([lng, lat]);
                            }
                        }
                        if (coords.length < 3) return;
                        const areaSqKm = calculateAreaSqKm(coords);
                        if (areaSqKm > maxAreaSqKm) return;
                        count++;
                        allShapes.push({
                            index: allShapes.length,
                            coordinates: coords,
                            properties: { ...feature.properties, area_sqkm: areaSqKm },
                            pruned: false
                        });
                    });
                });
                isIntersectionMode = true;
                selectionRevision++;
                document.getElementById('shape-count').textContent =
                    count + ' intersection shapes';
                updateLayers();
                if (allShapes.length > 0) fitToBounds();
                return { loaded: count };
            } catch(e) {
                console.error('Error loading intersection shapes:', e);
                return { loaded: 0, error: e.message };
            }
        }

        function setShapesDone(indices) {
            indices.forEach(idx => {
                if (allShapes[idx]) {
                    allShapes[idx].properties.done = true;
                }
            });
            dataRevision++;
            updateLayers();
            const doneCount = allShapes.filter(s => s.properties.done).length;
            console.log(`Marked ${indices.length} shapes as done. Total done: ${doneCount}`);
        }

        // --- Drawing mode ---
        function toggleDrawing() {
            if (isDrawing) {
                cancelDrawing();
                return;
            }
            isDrawing = true;
            drawingCoords = [];
            drawRevision++;
            document.getElementById('draw-btn').classList.add('active');
            document.getElementById('finish-btn').classList.add('show');
            document.getElementById('cancel-btn').classList.add('show');
            const ds = document.getElementById('draw-status');
            ds.style.display = 'block';
            ds.textContent = 'Click on the map to place vertices';
            updateLayers();
        }

        function finishDrawing() {
            if (drawingCoords.length < 3) {
                document.getElementById('draw-status').textContent = 'Need at least 3 vertices!';
                return;
            }
            // Close the polygon
            const coords = [...drawingCoords];
            const areaSqKm = calculateAreaSqKm(coords);
            drawnShapeCount++;

            const newIdx = allShapes.length;
            allShapes.push({
                index: newIdx,
                coordinates: coords,
                properties: {
                    name: 'Drawn Shape ' + drawnShapeCount,
                    area_sqkm: areaSqKm,
                    drawn: true
                },
                pruned: false
            });

            // Auto-select the new shape
            selectedIndices.add(newIdx);
            selectionRevision++;

            // Exit drawing mode
            isDrawing = false;
            drawingCoords = [];
            document.getElementById('draw-btn').classList.remove('active');
            document.getElementById('finish-btn').classList.remove('show');
            document.getElementById('cancel-btn').classList.remove('show');
            document.getElementById('draw-status').style.display = 'none';

            // Update shape count display
            const total = allShapes.length;
            document.getElementById('shape-count').textContent = total + ' shapes';

            updateLayers();
            updateSelectionInfo();
        }

        function cancelDrawing() {
            isDrawing = false;
            drawingCoords = [];
            drawRevision++;
            document.getElementById('draw-btn').classList.remove('active');
            document.getElementById('finish-btn').classList.remove('show');
            document.getElementById('cancel-btn').classList.remove('show');
            document.getElementById('draw-status').style.display = 'none';
            updateLayers();
        }

        function deleteSelected() {
            if (selectedIndices.size === 0) return;
            // Remove selected shapes
            allShapes = allShapes.filter(s => !selectedIndices.has(s.index));
            // Re-index
            allShapes.forEach((s, i) => { s.index = i; });
            selectedIndices.clear();
            selectionRevision++;

            const total = allShapes.length;
            document.getElementById('shape-count').textContent = total > 0 ? total + ' shapes' : 'No shapes loaded';

            updateLayers();
            updateSelectionInfo();
        }

        // Initialize on load
        initDeckGL();
    </script>
</body>
</html>
        """
        self.setHtml(html)
    
    def get_selected_cells_coords(self):
        """Get coordinates of all selected shapes."""
        try:
            from PyQt6.QtCore import QEventLoop
            
            loop = QEventLoop()
            result_container = {'coords': None}
            
            def callback(result):
                result_container['coords'] = result
                loop.quit()
            
            self.page().runJavaScript("getSelectedCellsCoords();", callback)
            loop.exec()
            
            if result_container['coords']:
                return json.loads(result_container['coords'])
            return []
        except Exception:
            return []
    
    def get_selected_cells_data(self):
        """Get metadata of all selected shapes."""
        try:
            from PyQt6.QtCore import QEventLoop
            
            loop = QEventLoop()
            result_container = {'data': None}
            
            def callback(result):
                result_container['data'] = result
                loop.quit()
            
            self.page().runJavaScript("getSelectedCellsData();", callback)
            loop.exec()
            
            if result_container['data']:
                return json.loads(result_container['data'])
            return []
        except Exception:
            return []
    
    def get_selected_count(self):
        """Get number of selected shapes."""
        try:
            from PyQt6.QtCore import QEventLoop
            
            loop = QEventLoop()
            result_container = {'count': 0}
            
            def callback(result):
                result_container['count'] = result
                loop.quit()
            
            self.page().runJavaScript("getSelectedCount();", callback)
            loop.exec()
            
            return result_container['count'] or 0
        except Exception:
            return 0
    
    def __init_point_buffer(self):
        """Initialize point buffering system."""
        if not hasattr(self, '_point_buffer'):
            from PyQt6.QtCore import QTimer
            self._point_buffer = []
            self._completed_buffer = []
            self._flush_timer = QTimer()
            self._flush_timer.setInterval(500)
            self._flush_timer.timeout.connect(self._flush_point_buffer)

    def add_point(self, lat, lon, panoid=None):
        """Buffer a scraped point for batched rendering."""
        self.__init_point_buffer()
        pid = panoid if panoid else f'temp_{lat}_{lon}'
        self._point_buffer.append([lat, lon, pid])
        if not self._flush_timer.isActive():
            self._flush_timer.start()

    def mark_point_completed(self, panoid):
        """Buffer a completed panoid for batched rendering."""
        self.__init_point_buffer()
        self._completed_buffer.append(panoid)
        if not self._flush_timer.isActive():
            self._flush_timer.start()

    def _flush_point_buffer(self):
        """Send buffered points/completions to JS in bulk."""
        if self._point_buffer:
            # Pass as JS array literal directly - no string wrapping needed
            batch = self._point_buffer.copy()
            self._point_buffer.clear()
            self.page().runJavaScript(f"addPointsBatch({json.dumps(batch)});")
        if self._completed_buffer:
            batch = self._completed_buffer.copy()
            self._completed_buffer.clear()
            self.page().runJavaScript(f"markCompletedBatch({json.dumps(batch)});")
        # Stop timer if both buffers are empty
        if not self._point_buffer and not self._completed_buffer:
            self._flush_timer.stop()

    def clear_points(self):
        """Clear all scraped points."""
        self.__init_point_buffer()
        self._point_buffer.clear()
        self._completed_buffer.clear()
        self.page().runJavaScript("clearPoints();")

    def flush_points(self):
        """Force flush any buffered points and update the layer."""
        self.__init_point_buffer()
        self._flush_point_buffer()
        self.page().runJavaScript("updateLayers();")
    
    def unload_shapes(self):
        """Clear all shapes from the map."""
        self.page().runJavaScript("unloadShapes();")


    def load_shapes(self, geojson_path: str, max_area_sqkm: float = 10000):
        """Load shapes from GeoJSON or Shapefile with area filtering."""
        try:
            if geojson_path.lower().endswith('.shp'):
                import geopandas as gpd
                gdf = gpd.read_file(geojson_path)
                geojson_data = json.loads(gdf.to_json())
            else:
                with open(geojson_path, 'r', encoding='utf-8') as f:
                    geojson_data = json.loads(f.read())
            
            geojson_str = json.dumps(geojson_data)
            # Escape for JavaScript
            geojson_str = geojson_str.replace('\\', '\\\\').replace("'", "\\'")
            self.page().runJavaScript(f"loadShapes('{geojson_str}', {max_area_sqkm});")
            return True
        except Exception as e:
            print(f"Error loading shapes: {e}")
            return False

    def set_shapes_done(self, indices):
        """Mark specific shapes as done (Red) by index."""
        if not indices:
            return
        self.page().runJavaScript(f"setShapesDone({json.dumps(indices)});")




class ProgressScannerThread(QThread):
    """
    Background thread to scan for existing CSVs (Local or R2).
    Matches found CSV coordinates against loaded polygons to mark them as 'Done'.
    """
    progress = pyqtSignal(int, int)
    finished_signal = pyqtSignal(list) # List of indices of "done" shapes

    def __init__(self, shapes_data, output_dir, r2_client=None):
        super().__init__()
        self.shapes_data = shapes_data
        self.output_dir = output_dir
        self.r2_client = r2_client

    def run(self):
        try:
            from shapely.geometry import Point, Polygon
            import glob

            if not self.shapes_data:
                self.finished_signal.emit([])
                return

            # Convert shapes to Shapely polygons for fast checking
            polygons = []
            for shape in self.shapes_data:
                coords = shape.get('coordinates', [])
                if coords:
                    polygons.append(Polygon(coords))
                else:
                    polygons.append(None)

            done_indices = set()

            if self.r2_client:
                # ── R2 Scanning Mode ──
                try:
                    # List all CSVs in the bucket
                    files = self.r2_client.list_files(prefix="CSV/")
                    total_files = len(files)
                    
                    for i, file_obj in enumerate(files):
                        key = file_obj['key']
                        if not key.lower().endswith('.csv'):
                            continue

                        try:
                            # Download first 1KB to get the header and first row
                            resp = self.r2_client.s3.get_object(
                                Bucket=self.r2_client.bucket_name,
                                Key=key,
                                Range='bytes=0-1024'
                            )
                            chunk = resp['Body'].read().decode('utf-8', errors='ignore')
                            lines = chunk.splitlines()
                            
                            # Skip header, get first data line
                            if len(lines) < 2: continue
                            
                            first_line = lines[1]
                            parts = first_line.split(',')
                            if len(parts) < 3: continue
                            
                            # CSV: panoid, lat, lon
                            lat = float(parts[1])
                            lon = float(parts[2])
                            point = Point(lon, lat)

                            for idx, poly in enumerate(polygons):
                                if idx in done_indices: continue
                                if poly and poly.contains(point):
                                    done_indices.add(idx)
                                    # Assuming one file belongs to one shape, but multiple files might map to same shape
                                    # We don't break here because other polys might overlap? No, usually not.
                                    break
                                    
                        except Exception:
                            continue
                        
                        if i % 10 == 0:
                            self.progress.emit(i, total_files)
                            
                except Exception as e:
                    print(f"R2 Scan error: {e}")

            else:
                # ── Local Scanning Mode ──
                csv_files = glob.glob(os.path.join(self.output_dir, "**/*.csv"), recursive=True)
                total_files = len(csv_files)
                
                for i, csv_path in enumerate(csv_files):
                    try:
                        with open(csv_path, 'r', encoding='utf-8') as f:
                            f.readline() # Header
                            first_line = f.readline()
                            if not first_line: continue
                            
                            parts = first_line.split(',')
                            if len(parts) < 3: continue
                            
                            lat = float(parts[1])
                            lon = float(parts[2])
                            point = Point(lon, lat)

                            for idx, poly in enumerate(polygons):
                                if idx in done_indices: continue
                                if poly and poly.contains(point):
                                    done_indices.add(idx)
                                    break
                    except Exception:
                        continue
                    
                    if i % 50 == 0:
                        self.progress.emit(i, total_files)

            self.finished_signal.emit(list(done_indices))

        except Exception as e:
            print(f"Scanner error: {e}")
            self.finished_signal.emit([])


class MainWindow(QMainWindow):

    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.scraper = None
        self.scraper_thread = None
        self.all_polygons = []
        self.temp_csv_path = None
        self._last_pano_count = 0
        self._last_image_count = 0
        self._stats_timer = None
        self.init_ui()
        self.apply_dark_theme()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Stage 0 - Unified Street View Scraper")
        self.setGeometry(100, 100, 1500, 950)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Map
        map_container = QVBoxLayout()
        self.map_view = DeckGLMapView()
        map_container.addWidget(self.map_view)
        main_layout.addLayout(map_container, stretch=3)
        
        # Right panel - Controls
        control_panel = QVBoxLayout()
        
        # === REGION SETTINGS ===
        self.region_group = QGroupBox("Region")
        region_outer = QVBoxLayout()
        region_outer.setSpacing(4)

        # Row 1: Main shapes
        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("Shapes:"))
        self.shape_file_edit = QLineEdit()
        self.shape_file_edit.setPlaceholderText("No file loaded")
        self.shape_file_edit.setReadOnly(True)
        self.shape_load_button = QPushButton("Load")
        self.shape_load_button.clicked.connect(self.load_shape_file)
        self.shape_load_button.setMaximumWidth(50)
        self.shape_unload_button = QPushButton("Unload")
        self.shape_unload_button.clicked.connect(self.unload_shape_file)
        self.shape_unload_button.setMaximumWidth(60)
        shape_row.addWidget(self.shape_file_edit)
        shape_row.addWidget(self.shape_load_button)
        shape_row.addWidget(self.shape_unload_button)
        shape_row.addWidget(QLabel("Max km²:"))
        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(100, 100000)
        self.max_area_spin.setValue(10000)
        self.max_area_spin.setSingleStep(1000)
        self.max_area_spin.setMaximumWidth(80)
        shape_row.addWidget(self.max_area_spin)
        region_outer.addLayout(shape_row)


        # Row 2: Action buttons
        action_row = QHBoxLayout()
        self.merge_shapes_button = QPushButton("Merge Shapes")
        self.merge_shapes_button.setToolTip("Select exactly 2 shapes, then click to merge them into one")
        self.merge_shapes_button.clicked.connect(self.merge_selected_shapes)
        action_row.addWidget(self.merge_shapes_button)

        self.scan_btn = QPushButton("Scan Progress")
        self.scan_btn.clicked.connect(self._start_progress_scan)
        self.scan_btn.setToolTip("Scan output folder for existing CSVs and mark shapes as done")
        action_row.addWidget(self.scan_btn)

        action_row.addStretch()
        region_outer.addLayout(action_row)

        self.region_group.setLayout(region_outer)
        control_panel.addWidget(self.region_group)
        
        # === SETTINGS (Tab Widget) ===
        self.settings_tabs = QTabWidget()
        
        # (Performance and Global tabs removed — defaults hardcoded)

        # === VPS DEPLOY TAB ===
        vps_tab = QWidget()
        vps_layout = QFormLayout()

        self.r2_account_edit = QLineEdit()
        self.r2_account_edit.setPlaceholderText("Cloudflare Account ID")
        self.r2_account_edit.setText(os.environ.get("R2_ACCOUNT_ID", ""))
        vps_layout.addRow("R2 Account ID:", self.r2_account_edit)

        self.r2_access_key_edit = QLineEdit()
        self.r2_access_key_edit.setPlaceholderText("Access Key ID")
        self.r2_access_key_edit.setText(os.environ.get("R2_ACCESS_KEY_ID", ""))
        vps_layout.addRow("R2 Access Key:", self.r2_access_key_edit)

        self.r2_secret_key_edit = QLineEdit()
        self.r2_secret_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.r2_secret_key_edit.setPlaceholderText("Secret Access Key")
        self.r2_secret_key_edit.setText(os.environ.get("R2_SECRET_ACCESS_KEY", ""))
        vps_layout.addRow("R2 Secret Key:", self.r2_secret_key_edit)

        self.r2_bucket_edit = QLineEdit()
        self.r2_bucket_edit.setPlaceholderText("Bucket Name")
        self.r2_bucket_edit.setText(os.environ.get("R2_BUCKET_NAME", ""))
        vps_layout.addRow("R2 Bucket:", self.r2_bucket_edit)

        self.vast_api_key_edit = QLineEdit()
        self.vast_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.vast_api_key_edit.setPlaceholderText("Vast.ai API Key")
        self.vast_api_key_edit.setText(os.environ.get("VAST_API_KEY", ""))
        vps_layout.addRow("Vast.ai Key:", self.vast_api_key_edit)

        self.region_path_edit = QLineEdit()
        self.region_path_edit.setPlaceholderText("e.g. US/Florida/Florida  (overrides geocoding)")
        self.region_path_edit.setToolTip(
            "Manual country/state/city path override.\n"
            "If set, geocoding is skipped and this path is used for all R2 dirs:\n"
            "  CSV/{path}/   Features/{path}/   Status/{path}/   Logs/{path}/"
        )
        vps_layout.addRow("Region Path:", self.region_path_edit)

        self.docker_image_edit = QLineEdit()
        self.docker_image_edit.setText("ghcr.io/occultmc/geoaxisimage:latest")
        vps_layout.addRow("Docker Image:", self.docker_image_edit)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 100)
        self.workers_spin.setValue(10)
        self.workers_spin.setToolTip("Number of VPS instances to deploy. The CSV will be split into this many segments.")
        vps_layout.addRow("Instances:", self.workers_spin)

        self.gpu_type_edit = QLineEdit()
        self.gpu_type_edit.setText("")
        self.gpu_type_edit.setPlaceholderText("e.g. RTX 3090 (leave empty for any)")
        vps_layout.addRow("GPU Filter:", self.gpu_type_edit)

        self.min_vram_spin = QSpinBox()
        self.min_vram_spin.setRange(0, 512)
        self.min_vram_spin.setValue(21)
        self.min_vram_spin.setSuffix(" GB")
        self.min_vram_spin.setToolTip("Minimum system RAM")
        vps_layout.addRow("Min RAM:", self.min_vram_spin)

        self.max_price_spin = QDoubleSpinBox()
        self.max_price_spin.setRange(0.01, 50.0)
        self.max_price_spin.setValue(0.20)
        self.max_price_spin.setSingleStep(0.10)
        self.max_price_spin.setPrefix("$")
        vps_layout.addRow("Max $/hr:", self.max_price_spin)

        self.geo_filter_edit = QLineEdit()
        self.geo_filter_edit.setPlaceholderText("e.g. US,CA,GB,AU  (leave empty for all)")
        self.geo_filter_edit.setToolTip(
            "Comma-separated country codes to restrict VPS offers.\n"
            "Uses Vast.ai geolocation filter. Examples:\n"
            "  US         — United States only\n"
            "  US,CA      — US and Canada\n"
            "  US,CA,GB,AU,NZ,SE,NO — Western countries\n"
            "Leave empty to search all regions."
        )
        vps_layout.addRow("Geo Filter:", self.geo_filter_edit)

        self.disk_gb_spin = QSpinBox()
        self.disk_gb_spin.setRange(50, 500)
        self.disk_gb_spin.setValue(100)
        vps_layout.addRow("Disk GB:", self.disk_gb_spin)

        vps_tab.setLayout(vps_layout)
        self.settings_tabs.addTab(vps_tab, "VPS Deploy")

        # === INDEX BUILD TAB ===
        index_tab = QWidget()
        index_layout = QFormLayout()

        self.builder_image_edit = QLineEdit()
        self.builder_image_edit.setText("ghcr.io/occultmc/indbuilder:latest")
        index_layout.addRow("Builder Image:", self.builder_image_edit)

        self.builder_disk_spin = QSpinBox()
        self.builder_disk_spin.setRange(100, 2000)
        self.builder_disk_spin.setValue(700)
        index_layout.addRow("Disk GB:", self.builder_disk_spin)

        self.builder_max_price_spin = QDoubleSpinBox()
        self.builder_max_price_spin.setRange(0.01, 100.0)
        self.builder_max_price_spin.setValue(0.50)
        self.builder_max_price_spin.setSingleStep(0.50)
        self.builder_max_price_spin.setPrefix("$")
        index_layout.addRow("Max $/hr:", self.builder_max_price_spin)

        # --- FAISS Index Settings ---
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("color: #555555;")
        index_layout.addRow(separator)
        index_layout.addRow(QLabel("FAISS Index Settings"))

        from PyQt6.QtWidgets import QComboBox
        self.index_type_combo = QComboBox()
        self.index_type_combo.addItems(["ivfpq", "pq"])
        self.index_type_combo.setCurrentText("pq")
        self.index_type_combo.setToolTip("ivfpq = IVF + Product Quantization (best for large datasets)\npq = Product Quantization only (works with any count)")
        index_layout.addRow("Index Type:", self.index_type_combo)

        self.nlist_spin = QSpinBox()
        self.nlist_spin.setRange(64, 65536)
        self.nlist_spin.setValue(1024)
        self.nlist_spin.setSingleStep(256)
        self.nlist_spin.setToolTip("Number of IVF clusters (only used with ivfpq). Auto-adjusted if too large for dataset.")
        index_layout.addRow("nlist (clusters):", self.nlist_spin)

        self.m_spin = QSpinBox()
        self.m_spin.setRange(8, 256)
        self.m_spin.setValue(256)
        self.m_spin.setSingleStep(8)
        self.m_spin.setToolTip("Number of PQ sub-vectors. Must divide feature dimension (8448). Higher = more accurate but larger index.")
        index_layout.addRow("m (sub-vectors):", self.m_spin)

        self.nbits_spin = QSpinBox()
        self.nbits_spin.setRange(4, 16)
        self.nbits_spin.setValue(8)
        self.nbits_spin.setToolTip("Bits per PQ sub-quantizer. 8 is standard. Lower = smaller index, less accurate.")
        index_layout.addRow("nbits:", self.nbits_spin)

        self.train_samples_spin = QSpinBox()
        self.train_samples_spin.setRange(10000, 10000000)
        self.train_samples_spin.setValue(1000000)
        self.train_samples_spin.setSingleStep(100000)
        self.train_samples_spin.setToolTip("Number of random vectors sampled for training. More = better clustering but slower.")
        index_layout.addRow("Train Samples:", self.train_samples_spin)

        self.niter_spin = QSpinBox()
        self.niter_spin.setRange(10, 500)
        self.niter_spin.setValue(100)
        self.niter_spin.setSingleStep(25)
        self.niter_spin.setToolTip("Clustering iterations for IVF training. Default FAISS is 25, 100 gives better quality.")
        index_layout.addRow("niter (iterations):", self.niter_spin)

        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setStyleSheet("color: #555555;")
        index_layout.addRow(separator2)

        self.auto_build_check = QCheckBox("Auto-build when workers finish")
        self.auto_build_check.setChecked(True)
        self.auto_build_check.setToolTip("Automatically launch index builder when all feature extraction workers complete")
        index_layout.addRow(self.auto_build_check)

        separator3 = QFrame()
        separator3.setFrameShape(QFrame.Shape.HLine)
        separator3.setStyleSheet("color: #555555;")
        index_layout.addRow(separator3)

        self.builder_path_edit = QLineEdit()
        self.builder_path_edit.setPlaceholderText("e.g. US/Florida/Florida")
        self.builder_path_edit.setToolTip(
            "Optional: manually specify the R2 features path as Country/State/City.\n"
            "The builder will pull from Features/<path>/ in your R2 bucket.\n"
            "Leave empty to use the currently selected map region."
        )
        index_layout.addRow("Path Override:", self.builder_path_edit)

        index_tab.setLayout(index_layout)
        self.settings_tabs.addTab(index_tab, "Index Build")

        control_panel.addWidget(self.settings_tabs)
        
        # === PROGRESS ===
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        progress_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.pano_count_label = QLabel("Panoramas: 0 | Images: 0")
        progress_layout.addWidget(self.pano_count_label)

        self.speed_label = QLabel("Speed: -- found/s | -- panos/s | -- views/s")
        self.speed_label.setStyleSheet("color: #aaaaaa;")
        progress_layout.addWidget(self.speed_label)

        progress_group.setLayout(progress_layout)
        control_panel.addWidget(progress_group)
        
        # === BUTTONS ===
        button_layout = QHBoxLayout()
        
        self.scrape_button = QPushButton("Start Scraping")
        self.scrape_button.clicked.connect(self.start_scraping)
        self.scrape_button.setStyleSheet("""
            QPushButton {
                background-color: #00aa00;
                color: white;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #00cc00; }
            QPushButton:disabled { background-color: #555555; }
        """)
        button_layout.addWidget(self.scrape_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_scraping)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #cc0000;
                color: white;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #ff0000; }
            QPushButton:disabled { background-color: #555555; }
        """)
        button_layout.addWidget(self.cancel_button)

        self.deploy_button = QPushButton("Deploy to VPS")
        self.deploy_button.clicked.connect(self.start_vps_deploy)
        self.deploy_button.setStyleSheet("""
            QPushButton {
                background-color: #0066cc;
                color: white;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #0088ff; }
            QPushButton:disabled { background-color: #555555; }
        """)
        button_layout.addWidget(self.deploy_button)

        self.build_index_button = QPushButton("Build Index")
        self.build_index_button.clicked.connect(self.start_index_build)
        self.build_index_button.setStyleSheet("""
            QPushButton {
                background-color: #cc6600;
                color: white;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #ff8800; }
            QPushButton:disabled { background-color: #555555; }
        """)
        button_layout.addWidget(self.build_index_button)

        control_panel.addLayout(button_layout)

        # === VPS PROGRESS PANEL (hidden initially) ===
        self.vps_progress_group = QGroupBox("VPS Workers")
        vps_prog_layout = QVBoxLayout()

        self.vps_overall_progress = QProgressBar()
        self.vps_overall_progress.setFormat("%v/%m panos (%p%)")
        vps_prog_layout.addWidget(self.vps_overall_progress)

        self.vps_status_label = QLabel("No VPS deployment active")
        self.vps_status_label.setWordWrap(True)
        vps_prog_layout.addWidget(self.vps_status_label)

        open_monitor_btn = QPushButton("Open Worker Monitor")
        open_monitor_btn.setStyleSheet(
            "QPushButton { background-color: #0066cc; color: white; padding: 6px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #0088ff; }"
        )
        open_monitor_btn.clicked.connect(self._open_monitor_window)
        vps_prog_layout.addWidget(open_monitor_btn)

        self.vps_progress_group.setLayout(vps_prog_layout)
        self.vps_progress_group.setVisible(False)
        control_panel.addWidget(self.vps_progress_group)

        # === BUILDER PROGRESS PANEL (hidden initially) ===
        self.builder_progress_group = QGroupBox("Index Builder")
        builder_prog_layout = QVBoxLayout()

        self.builder_status_label = QLabel("No builder active")
        self.builder_status_label.setWordWrap(True)
        builder_prog_layout.addWidget(self.builder_status_label)

        self.builder_progress_bar = QProgressBar()
        self.builder_progress_bar.setFormat("%p% — %v")
        builder_prog_layout.addWidget(self.builder_progress_bar)

        self.builder_step_label = QLabel("")
        self.builder_step_label.setStyleSheet("color: #aaaaaa;")
        builder_prog_layout.addWidget(self.builder_step_label)

        self.builder_progress_group.setLayout(builder_prog_layout)
        self.builder_progress_group.setVisible(False)
        control_panel.addWidget(self.builder_progress_group)

        control_panel.addStretch()
        
        main_layout.addLayout(control_panel, stretch=1)
        
        self.statusBar().showMessage("Select a mode and configure settings to begin")
    
    
    def apply_dark_theme(self):
        """Apply dark theme."""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(35, 35, 35))
        
        self.setPalette(palette)
        
        self.setStyleSheet("""
            QGroupBox {
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QProgressBar {
                border: 2px solid #555555;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk { background-color: #00aa00; }
            QTabWidget::pane { border: 2px solid #555555; background: #353535; }
            QTabBar::tab { background: #444444; color: white; padding: 6px 12px; margin: 1px; }
            QTabBar::tab:selected { background: #666666; font-weight: bold; }
            QRadioButton { padding: 5px; }
        """)
    
    def _build_config(self) -> UnifiedScraperConfig:
        """Build scraper config from UI values."""
        return UnifiedScraperConfig(
            concurrency=1000,
            proxy_file=None,
        )
    
    def load_shape_file(self):
        """Load GeoJSON or Shapefile."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Shape File", "",
            "Shape Files (*.geojson *.json *.shp);;All Files (*)"
        )
        if file_path:
            self.shape_file_edit.setText(file_path)
            self._main_shape_path = file_path
            max_area = self.max_area_spin.value()
            if self.map_view.load_shapes(file_path, max_area):
                self.statusBar().showMessage(f"Loaded shapes from {Path(file_path).name}")
                # Trigger scan after loading
                QTimer.singleShot(1000, self._start_progress_scan)
            else:
                QMessageBox.warning(self, "Error", "Failed to load shape file")
    
    def unload_shape_file(self):
        """Unload all shapes from the map."""
        self.shape_file_edit.clear()
        self.map_view.unload_shapes()
        self.statusBar().showMessage("Shapes unloaded")

    def merge_selected_shapes(self):
        """Merge exactly 2 selected shapes into their geometric union."""
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union

        selected_data = self.map_view.get_selected_cells_data()
        if not selected_data or len(selected_data) != 2:
            QMessageBox.warning(self, "Select Two Shapes",
                "Please select exactly 2 shapes to merge.")
            return

        # Build shapely polygons (coords are [lat, lng], Shapely wants [lng, lat])
        coords_list = []
        for shape in selected_data:
            coords = shape.get('coordinates', [])
            if not coords:
                QMessageBox.warning(self, "Error", "Selected shape has no coordinates.")
                return
            coords_list.append(coords)

        poly1 = Polygon([(lng, lat) for lat, lng in coords_list[0]])
        poly2 = Polygon([(lng, lat) for lat, lng in coords_list[1]])

        merged = unary_union([poly1, poly2])

        # Convert back: merged exterior coords [lng, lat] → JS format [lng, lat]
        if merged.geom_type == 'Polygon':
            js_coords = [[lng, lat] for lng, lat in merged.exterior.coords]
        elif merged.geom_type == 'MultiPolygon':
            # Take convex hull to create a single polygon
            hull = merged.convex_hull
            js_coords = [[lng, lat] for lng, lat in hull.exterior.coords]
        else:
            QMessageBox.warning(self, "Error", f"Unexpected geometry type: {merged.geom_type}")
            return

        self.map_view.page().runJavaScript(
            f"mergeSelectedShapes({json.dumps(js_coords)});"
        )
        self.statusBar().showMessage("Merged 2 shapes into 1")

    def _geocode_polygon_centroid(self, coords):
        """Reverse geocode the centroid of polygon coords to get Country/State/City."""
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        return _reverse_geocode(center_lat, center_lon)

    def start_scraping(self):
        """Start the scraping process."""
        config = self._build_config()
        self.scraper = UnifiedScraper(config)
        
        # Set callbacks
        self.scraper.set_progress_callback(self.update_progress_manual)
        self.scraper.set_status_callback(self.update_status_manual)
        self.scraper.set_point_callback(self.on_point_found_manual)
        self.scraper.set_completed_callback(self.on_point_completed_manual)
        
        # Create output directories
        project_root = Path(__file__).parent.parent
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        csv_dir = project_root / "Output" / "CSV"
        images_dir = project_root / "Output" / "Images"
        
        selected_coords = self.map_view.get_selected_cells_coords()

        if not selected_coords or len(selected_coords) == 0:
            QMessageBox.warning(self, "No Selection",
                "Please select at least one shape to scrape!\n\n"
                "Click on shapes in the map to select them.\n"
                "Use Ctrl+Click for multi-select.")
            return

        self.map_view.clear_points()
        self.toggle_controls(False)

        if len(selected_coords) == 1:
            country, state, city = self._geocode_polygon_centroid(selected_coords[0])
            self.statusBar().showMessage(f"Geocoded: {country}/{state}/{city}")
            region_dir = images_dir / country / state / city
            csv_path = csv_dir / country / state / city / f"{timestamp}.csv"

            self.scraper.init_csv(str(csv_path))
            self.scraper.init_images_dir(str(region_dir))

            self.scraper_thread = ScraperThread(
                self.scraper, mode="polygon",
                polygon_coords=selected_coords[0]
            )
        else:
            # Always merge into one CSV
            per_shape_csv_paths = []
            per_shape_images_dirs = []
            geocoded_info = []
            for i, sc in enumerate(selected_coords):
                country, state, city = self._geocode_polygon_centroid(sc)
                self.statusBar().showMessage(f"Geocoding shape {i+1}/{len(selected_coords)}: {country}/{state}/{city}")
                geocoded_info.append((country, state, city))

            first_country, first_state, first_city = geocoded_info[0]
            merged_csv_path = str(csv_dir / first_country / first_state / first_city / f"Merged_{timestamp}.csv")
            self.scraper.init_csv(merged_csv_path)
            self.scraper.init_images_dir(str(images_dir / first_country / first_state / first_city))
            per_shape_csv_paths = [merged_csv_path] * len(selected_coords)
            per_shape_images_dirs = [str(images_dir / c / s / ci) for c, s, ci in geocoded_info]

            self.scraper_thread = ScraperThread(
                self.scraper, mode="multi_polygon",
                polygon_list=selected_coords,
                csv_paths=per_shape_csv_paths,
                images_dirs=per_shape_images_dirs,
                merge_csv=True
            )

        self.statusBar().showMessage(f"Scraping {len(selected_coords)} region(s)...")
        
        self.connect_thread_signals()
        self.scraper_thread.start()
    
    def connect_thread_signals(self):
        self.scraper_thread.progress.connect(self.update_progress)
        self.scraper_thread.status.connect(self.update_status)
        self.scraper_thread.point_found.connect(self.on_point_found)
        self.scraper_thread.point_completed.connect(self.on_point_completed)
        self.scraper_thread.finished_signal.connect(self.scraping_finished)
        self.scraper_thread.error.connect(self.scraping_error)

        # Start speed stats timer
        from PyQt6.QtCore import QTimer
        self._last_found_count = 0
        self._last_image_count = 0
        self._last_pano_dl_count = 0
        self._stats_timer = QTimer()
        self._stats_timer.setInterval(1000)
        self._stats_timer.timeout.connect(self._update_speed_stats)
        self._stats_timer.start()

    def _update_speed_stats(self):
        """Update discovery/s, panos/s, and views/s display."""
        if not self.scraper:
            return
        found = len(self.scraper.found_panos)
        written = self.scraper.total_written
        images = self.scraper.total_images
        tiles = self.scraper.tiles_processed
        total_tiles = self.scraper.total_tiles

        found_speed = found - self._last_found_count
        img_speed = images - self._last_image_count
        # views come in groups per panorama, so derive panos/s from image delta
        num_views = 6
        # num_views = self.scraper.config.get('num_views', 6) # Config is a dataclass, and num_views isn't in it currently
        
        pano_speed = img_speed / num_views if num_views > 0 else 0
        self._last_found_count = found
        self._last_image_count = images

        self.speed_label.setText(
            f"Speed: {found_speed} found/s | {pano_speed:.1f} panos/s | {img_speed} views/s"
        )
        # Phase 2 progress tracking
        p2_done = self.scraper.phase2_completed
        p2_total = self.scraper.phase2_total

        if p2_total > 0:
            # Phase 2 active: show panorama download progress
            self.pano_count_label.setText(
                f"Found: {found} | Downloading: {p2_done}/{p2_total} panos | Views: {images}"
            )
            self.progress_bar.setValue(int((p2_done / p2_total) * 100))
        else:
            self.pano_count_label.setText(
                f"Found: {found} | Written: {written} | Views: {images}"
            )
            # Update progress bar during tile discovery
            if total_tiles > 0:
                self.progress_bar.setValue(int((tiles / total_tiles) * 100))
    
    def update_progress_manual(self, c, t):
        if self.scraper_thread: 
            self.scraper_thread.progress.emit(c, t)
    
    def update_status_manual(self, s):
        if self.scraper_thread: 
            self.scraper_thread.status.emit(s)
    
    def on_point_found_manual(self, lat, lon, panoid):
        if self.scraper_thread: 
            self.scraper_thread.point_found.emit(lat, lon, panoid)

    def on_point_completed_manual(self, panoid):
        if self.scraper_thread:
            self.scraper_thread.point_completed.emit(panoid)
    
    def on_point_found(self, lat, lon, panoid):
        """Called when a panorama is found - add point to map."""
        self.map_view.add_point(lat, lon, panoid)

    def on_point_completed(self, panoid):
        """Called when image download completes - mark point orange."""
        self.map_view.mark_point_completed(panoid)
    
    def _stop_stats_timer(self):
        if self._stats_timer:
            self._stats_timer.stop()
            self._stats_timer = None

    def cancel_scraping(self):
        if self.scraper_thread and self.scraper_thread.isRunning():
            self.scraper_thread.cancel()
            self.scraper_thread.wait()
        self._stop_stats_timer()
        self.map_view.flush_points()
        self.toggle_controls(True)
        self.deploy_button.setEnabled(True)
        self.statusBar().showMessage("Cancelled")
    
    def toggle_controls(self, enabled):
        self.scrape_button.setEnabled(enabled)
        self.cancel_button.setEnabled(not enabled)
        self.region_group.setEnabled(enabled)
        self.settings_tabs.setEnabled(enabled)
    
    def update_progress(self, current, total):
        if total > 0:
            self.progress_bar.setValue(int((current / total) * 100))
    
    def update_status(self, message):
        self.status_label.setText(message)
    
    def _reorganize_csv(self):
        """Move CSV into address-based folder structure: Country/State/City/StreetView_City_timestamp.csv"""
        if not self.scraper or not hasattr(self.scraper, 'csv_file_handle'):
            return
        # Close the CSV file handle so we can move it
        try:
            if self.scraper.csv_file_handle and not self.scraper.csv_file_handle.closed:
                self.scraper.csv_file_handle.flush()
                self.scraper.csv_file_handle.close()
        except Exception:
            pass

        country = getattr(self.scraper, 'first_pano_country', None)
        state = getattr(self.scraper, 'first_pano_state', None)
        city = getattr(self.scraper, 'first_pano_city', None)
        src = Path(self.scraper.csv_filename)

        if not country or not src.exists():
            return

        # Sanitize folder/file name components
        def sanitize(s):
            return "".join(c if c.isalnum() or c in (' ', '_', '-') else '' for c in str(s)).strip().replace(' ', '_')

        country_s = sanitize(country)
        state_s = sanitize(state) if state and state != "Unknown" else "Unknown"
        city_s = sanitize(city) if city and city != "Unknown" else "Unknown"

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_dir = src.parent  # e.g. .../Output/CSV
        organized_dir = csv_dir / country_s / state_s / city_s
        organized_dir.mkdir(parents=True, exist_ok=True)

        new_name = f"StreetView_{city_s}_{timestamp}.csv"
        dest = organized_dir / new_name

        try:
            shutil.move(str(src), str(dest))
            self.scraper.csv_filename = str(dest)
        except Exception as e:
            print(f"Could not reorganize CSV: {e}")

    def scraping_finished(self, result):
        self._stop_stats_timer()
        self.map_view.flush_points()
        # Delayed flush to catch any buffered JS calls
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(500, self.map_view.flush_points)
        self.toggle_controls(True)
        self.speed_label.setText("Speed: -- found/s | -- panos/s | -- views/s")
        # Reorganize CSV into address-based folders
        self._reorganize_csv()
        try:
            csv_path = Path(self.scraper.csv_filename).relative_to(Path(__file__).parent.parent)
        except (ValueError, AttributeError):
            csv_path = Path(self.scraper.csv_filename).name
        self.statusBar().showMessage(f"Complete! {self.scraper.total_written} panoramas, {self.scraper.total_images} views | {csv_path}")
    
    def scraping_error(self, error):
        self._stop_stats_timer()
        QMessageBox.critical(self, "Error", f"Scraping failed: {error}")
        self.statusBar().showMessage("Error occurred")

    def _start_progress_scan(self):
        """Start the scanner thread."""
        self.map_view.page().runJavaScript("JSON.stringify(allShapes)", self._on_all_shapes_received)
        
    def _on_all_shapes_received(self, result):
        if not result: return
        try:
            shapes = json.loads(result)
            project_root = Path(__file__).parent.parent
            output_dir = os.path.join(project_root, "Output", "CSV")
            
            # Check for R2 credentials to enable cloud scanning
            r2_client = None
            if VPS_AVAILABLE:
                try:
                    # Attempt to read from UI fields (populated from .env on init)
                    acc = self.r2_account_edit.text().strip()
                    key = self.r2_access_key_edit.text().strip()
                    secret = self.r2_secret_key_edit.text().strip()
                    bucket = self.r2_bucket_edit.text().strip()
                    
                    if all([acc, key, secret, bucket]):
                        r2_client = R2Client(acc, key, secret, bucket)
                        self.statusBar().showMessage("Connected to R2. Scanning cloud for progress...")
                except Exception:
                    pass

            self.scanner_thread = ProgressScannerThread(shapes, output_dir, r2_client)
            self.scanner_thread.finished_signal.connect(self._on_scan_finished)
            self.scanner_thread.progress.connect(lambda i, t: self.statusBar().showMessage(f"Scanning progress: {i}/{t}"))
            self.scanner_thread.start()
            if not r2_client:
                self.statusBar().showMessage("Scanning local output for progress...")
        except Exception as e:
            print(f"Error starting scan: {e}")

    def _on_scan_finished(self, done_indices):
        self.map_view.set_shapes_done(done_indices)
        self.statusBar().showMessage(f"Scan complete. Found {len(done_indices)} completed regions.")


    # ═══════════════════════════════════════════════════════════════════
    # VPS Deployment
    # ═══════════════════════════════════════════════════════════════════

    def start_vps_deploy(self):
        """Start VPS deployment: scrape CSV → split → upload to R2 → launch instances."""
        if not VPS_AVAILABLE:
            QMessageBox.warning(self, "Missing Dependencies",
                "VPS modules not available. Install: pip install boto3 vastai python-dotenv")
            return

        # Validate R2 + Vast.ai settings
        r2_account = self.r2_account_edit.text().strip()
        r2_access = self.r2_access_key_edit.text().strip()
        r2_secret = self.r2_secret_key_edit.text().strip()
        r2_bucket = self.r2_bucket_edit.text().strip()
        vast_key = self.vast_api_key_edit.text().strip()

        if not all([r2_account, r2_access, r2_secret, r2_bucket]):
            QMessageBox.warning(self, "Missing R2 Config", "Please fill in all R2 fields in the VPS Deploy tab.")
            return
        if not vast_key:
            QMessageBox.warning(self, "Missing Vast.ai Key", "Please enter your Vast.ai API key.")
            return

        selected_coords = self.map_view.get_selected_cells_coords()
        if not selected_coords or len(selected_coords) == 0:
            QMessageBox.warning(self, "No Selection", "Please select at least one shape to scrape.")
            return

        first_coords = selected_coords[0]

        # Parse manual region path override (e.g. "US/Florida/Florida")
        region_path = self.region_path_edit.text().strip().strip('/')
        manual_country, manual_state, manual_city = None, None, None
        if region_path:
            parts = [p.strip() for p in region_path.split('/') if p.strip()]
            if len(parts) == 3:
                manual_country, manual_state, manual_city = parts
            else:
                QMessageBox.warning(self, "Invalid Region Path",
                    f"Region Path must be in format  Country/State/City\n"
                    f"e.g.  US/Florida/Florida\n\nGot: {region_path!r}")
                return

        # Store VPS context (country/state/city filled in by thread after geocoding,
        # or immediately if manual path was supplied)
        self._vps_context = {
            'country': manual_country, 'state': manual_state, 'city': manual_city,
            'r2_account': r2_account, 'r2_access': r2_access,
            'r2_secret': r2_secret, 'r2_bucket': r2_bucket,
            'vast_key': vast_key,
            'num_workers': self.workers_spin.value(),
            'docker_image': self.docker_image_edit.text().strip(),
            'gpu_type': self.gpu_type_edit.text().strip(),
            'min_vram_gb': self.min_vram_spin.value(),
            'max_price': self.max_price_spin.value(),
            'geo_filter': self.geo_filter_edit.text().strip(),
            'disk_gb': self.disk_gb_spin.value(),
            'selected_coords': selected_coords,
            'first_coords': first_coords,
        }

        # Disable controls and run geocode + R2 check in background thread
        self.deploy_button.setEnabled(False)
        if manual_country:
            self.statusBar().showMessage(f"VPS Deploy: Using manual path {manual_country}/{manual_state}/{manual_city}...")
        else:
            self.statusBar().showMessage("VPS Deploy: Geocoding region...")

        self._deploy_thread = VPSDeployThread(self._vps_context, mode="full")
        self._deploy_thread.status_update.connect(lambda msg: self.statusBar().showMessage(msg))
        self._deploy_thread.geocode_done.connect(self._on_deploy_geocode_done)
        self._deploy_thread.r2_segments_found.connect(self._on_deploy_r2_checked)
        self._deploy_thread.error_occurred.connect(self._on_deploy_thread_error)
        self._deploy_thread.start()

    def _on_deploy_geocode_done(self, country, state, city):
        """Handle geocode result from deploy thread."""
        self._vps_context['country'] = country
        self._vps_context['state'] = state
        self._vps_context['city'] = city
        self.statusBar().showMessage(f"VPS Deploy: {country}/{state}/{city}")

    def _on_deploy_r2_checked(self, segment_count):
        """Handle R2 check result from deploy thread."""
        ctx = self._vps_context
        country, state, city = ctx['country'], ctx['state'], ctx['city']
        print(f"[R2CHECK] segment_count={segment_count}, country={country!r}, state={state!r}, city={city!r}")
        print(f"[R2CHECK] completed_worker_indices from ctx: {ctx.get('completed_worker_indices', 'NOT SET')}")
        print(f"[R2CHECK] workers_spin current value: {self.workers_spin.value()}")

        if segment_count > 0:
            # R2 has a complete CSV set — use it unconditionally, ignore spinbox value.
            # total_workers always comes from R2, not from the UI.
            completed = set(ctx.get('completed_worker_indices', []))
            all_indices = list(range(1, segment_count + 1))
            missing = [i for i in all_indices if i not in completed]

            ctx['total_workers'] = segment_count

            logger.info(
                f"R2 scan: {segment_count} CSVs | "
                f"{len(completed)} complete {sorted(completed)} | "
                f"{len(missing)} missing {missing}"
            )
            self.statusBar().showMessage(
                f"R2: {segment_count} CSVs found — "
                f"{len(completed)} workers complete, {len(missing)} missing: {missing}"
            )

            if not missing:
                # Every worker already has features in R2 — nothing left to do.
                self.workers_spin.setValue(segment_count)
                self.statusBar().showMessage(
                    f"All {segment_count} workers already complete in R2. Nothing to deploy."
                )
                self.deploy_button.setEnabled(True)
                self.toggle_controls(True)
                return

            if completed:
                # Some done — show a clear dialog so user knows what's happening
                missing_list = ", ".join(str(i) for i in missing)
                completed_list = ", ".join(str(i) for i in sorted(completed))
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Partial Features Detected")
                msg_box.setText(
                    f"R2 has {segment_count} CSV segments.\n\n"
                    f"✓ Complete  ({len(completed)}): {completed_list}\n"
                    f"✗ Missing   ({len(missing)}):  {missing_list}\n\n"
                    f"Deploy only the {len(missing)} missing instance(s)?"
                )
                missing_btn = msg_box.addButton(
                    f"Deploy missing only  ({len(missing)} instance{'s' if len(missing) != 1 else ''})",
                    QMessageBox.ButtonRole.AcceptRole
                )
                all_btn = msg_box.addButton(
                    f"Re-run all  ({segment_count} instances)",
                    QMessageBox.ButtonRole.DestructiveRole
                )
                msg_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
                msg_box.exec()

                clicked = msg_box.clickedButton()
                if clicked == missing_btn:
                    ctx['actual_workers'] = len(missing)
                    ctx['worker_indices_to_deploy'] = missing
                    self.workers_spin.setValue(len(missing))
                elif clicked == all_btn:
                    ctx['actual_workers'] = segment_count
                    ctx['worker_indices_to_deploy'] = all_indices
                    self.workers_spin.setValue(segment_count)
                else:
                    self.deploy_button.setEnabled(True)
                    self.toggle_controls(True)
                    return
            else:
                # No features yet — deploy all workers.
                logger.info(f"No features found. Deploying all {segment_count} workers.")
                self.statusBar().showMessage(
                    f"Found {segment_count} CSVs in R2. Deploying all {segment_count} workers..."
                )
                ctx['actual_workers'] = segment_count
                ctx['worker_indices_to_deploy'] = all_indices
                self.workers_spin.setValue(segment_count)

            self.toggle_controls(False)
            self._vps_provision_instances()
            return

        # Step 1: Start scraping — no R2 CSVs (or partial), scrape fresh
        self.statusBar().showMessage("VPS Deploy: Scraping CSV...")
        self.toggle_controls(False)

        selected_coords = ctx['selected_coords']
        first_coords = ctx['first_coords']

        config = self._build_config()
        self.scraper = UnifiedScraper(config)
        self.scraper.set_progress_callback(self.update_progress_manual)
        self.scraper.set_status_callback(self.update_status_manual)
        self.scraper.set_point_callback(self.on_point_found_manual)

        project_root = Path(__file__).parent.parent
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_dir = project_root / "Output" / "CSV" / country / state / city

        if len(selected_coords) == 1:
            csv_path = str(csv_dir / f"{timestamp}.csv")
            self.scraper.init_csv(csv_path)
            self.scraper.init_images_dir(str(project_root / "Output" / "Images" / country / state / city))
            self.scraper_thread = ScraperThread(
                self.scraper, mode="polygon",
                polygon_coords=first_coords
            )
        else:
            merge_csv = True  # Always merge for VPS deploy
            merged_csv_path = str(csv_dir / f"Merged_{timestamp}.csv")
            self.scraper.init_csv(merged_csv_path)
            self.scraper.init_images_dir(str(project_root / "Output" / "Images" / country / state / city))
            per_shape_csv_paths = [merged_csv_path] * len(selected_coords)
            per_shape_images_dirs = [str(project_root / "Output" / "Images" / country / state / city)] * len(selected_coords)
            self.scraper_thread = ScraperThread(
                self.scraper, mode="multi_polygon",
                polygon_list=selected_coords,
                csv_paths=per_shape_csv_paths,
                images_dirs=per_shape_images_dirs,
                merge_csv=True
            )

        self.scraper_thread.progress.connect(self.update_progress)
        self.scraper_thread.status.connect(self.update_status)
        self.scraper_thread.point_found.connect(self.on_point_found)
        self.scraper_thread.finished_signal.connect(self._vps_scraping_finished)
        self.scraper_thread.error.connect(self._vps_scraping_error)

        self._last_found_count = 0
        self._last_image_count = 0
        self._last_pano_dl_count = 0
        self._stats_timer = QTimer()
        self._stats_timer.setInterval(1000)
        self._stats_timer.timeout.connect(self._update_speed_stats)
        self._stats_timer.start()

        self.scraper_thread.start()

    def _on_deploy_thread_error(self, error_msg):
        """Handle error from deploy background thread."""
        QMessageBox.critical(self, "VPS Deploy Error", f"Deploy failed: {error_msg}")
        self.deploy_button.setEnabled(True)
        self.toggle_controls(True)

    def _vps_scraping_error(self, error):
        """Handle scraping error during VPS deploy."""
        self._stop_stats_timer()
        QMessageBox.critical(self, "VPS Scraping Error", f"CSV scraping failed: {error}")
        self.deploy_button.setEnabled(True)
        self.toggle_controls(True)

    def _vps_scraping_finished(self, result):
        """CSV scraping done — split, upload, and launch instances."""
        self._stop_stats_timer()
        self.map_view.flush_points()
        self.statusBar().showMessage("VPS Deploy: CSV scraping complete. Splitting...")

        ctx = self._vps_context
        csv_path = self.scraper.csv_filename

        # Close CSV handle
        if self.scraper.csv_file_handle and not self.scraper.csv_file_handle.closed:
            self.scraper.csv_file_handle.flush()
            self.scraper.csv_file_handle.close()

        # Step 2: Split CSV
        try:
            segments = split_csv(
                csv_path, ctx['num_workers'], ctx['city'],
                output_dir=str(Path(csv_path).parent / "segments")
            )
            if not segments:
                QMessageBox.warning(self, "Empty CSV", "No data in CSV to split.")
                self.deploy_button.setEnabled(True)
                self.toggle_controls(True)
                return
        except Exception as e:
            QMessageBox.critical(self, "Split Error", f"CSV splitting failed: {e}")
            self.deploy_button.setEnabled(True)
            self.toggle_controls(True)
            return

        actual_workers = len(segments)
        ctx['actual_workers'] = actual_workers
        ctx['total_workers'] = actual_workers
        ctx['worker_indices_to_deploy'] = list(range(1, actual_workers + 1))
        self.workers_spin.setValue(actual_workers)
        if actual_workers < ctx['num_workers']:
            QMessageBox.warning(
                self, "Worker Count Reduced",
                f"Requested {ctx['num_workers']} instances but the CSV only has enough rows "
                f"for {actual_workers}.\n\nDeploying {actual_workers} instance(s)."
            )
        self.statusBar().showMessage(f"VPS Deploy: Split into {actual_workers} segments. Uploading to R2...")

        # Step 3: Upload to R2
        try:
            r2 = R2Client(
                account_id=ctx['r2_account'],
                access_key_id=ctx['r2_access'],
                secret_access_key=ctx['r2_secret'],
                bucket_name=ctx['r2_bucket'],
            )
            uploaded = upload_csv_segments(
                segments, r2, ctx['country'], ctx['state'], ctx['city']
            )
            if len(uploaded) != actual_workers:
                QMessageBox.warning(self, "Upload Warning",
                    f"Only {len(uploaded)}/{actual_workers} segments uploaded.")
        except Exception as e:
            QMessageBox.critical(self, "R2 Error", f"R2 upload failed: {e}")
            self.deploy_button.setEnabled(True)
            self.toggle_controls(True)
            return

        self.statusBar().showMessage("VPS Deploy: CSV uploaded. Provisioning...")
        self._vps_provision_instances()

    def _vps_provision_instances(self):
        """Step 4-7: Provision VPS instances using context data."""
        ctx = self._vps_context

        self.statusBar().showMessage("VPS Deploy: Searching Vast.ai offers...")

        # Search offers in background thread to avoid UI freeze
        self._provision_thread = VPSDeployThread(ctx, mode="provision")
        self._provision_thread.status_update.connect(lambda msg: self.statusBar().showMessage(msg))
        self._provision_thread.provision_ready.connect(self._on_offers_found)
        self._provision_thread.error_occurred.connect(self._on_deploy_thread_error)
        self._provision_thread.start()

    def _on_offers_found(self, offers):
        """Handle offers search result — show dialog and create instances."""
        ctx = self._vps_context
        worker_indices = ctx.get('worker_indices_to_deploy', list(range(1, ctx.get('actual_workers', 1) + 1)))
        total_workers = ctx.get('total_workers', ctx.get('actual_workers', 1))
        num_to_deploy = len(worker_indices)

        if not offers:
            QMessageBox.warning(self, "No Offers", "No matching Vast.ai offers found.")
            self.deploy_button.setEnabled(True)
            self.toggle_controls(True)
            return

        self._vast_manager = ctx.pop('_vast_manager')

        # Step 5: Show offer selection dialog (num_needed = instances to spawn)
        dialog = OfferSelectionDialog(offers, num_to_deploy, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.statusBar().showMessage("VPS Deploy: Cancelled.")
            self.deploy_button.setEnabled(True)
            self.toggle_controls(True)
            return

        selected_offers = dialog.get_selected_offers()

        # Step 6: Create instances in background thread
        self.statusBar().showMessage(
            f"VPS Deploy: Creating {num_to_deploy} instance(s) "
            f"(workers {worker_indices[0]}–{worker_indices[-1]} of {total_workers})..."
        )
        self._create_thread = VPSInstanceCreateThread(
            self._vast_manager, selected_offers, worker_indices, total_workers, ctx
        )
        self._create_thread.status_update.connect(lambda msg: self.statusBar().showMessage(msg))
        self._create_thread.instances_created.connect(self._on_instances_created)
        self._create_thread.error_occurred.connect(self._on_deploy_thread_error)
        self._create_thread.start()

    def _on_instances_created(self, instance_worker_map):
        """Handle instance creation results."""
        if not instance_worker_map:
            QMessageBox.critical(self, "Creation Failed", "No instances were created.")
            self.deploy_button.setEnabled(True)
            self.toggle_controls(True)
            return

        # Step 7: Start log monitoring
        self._setup_vps_progress(instance_worker_map)
        self.statusBar().showMessage(
            f"VPS Deploy: {len(instance_worker_map)} instances running. Monitoring progress..."
        )

    def _open_monitor_window(self):
        """Show (or bring to front) the VPS monitor window."""
        if hasattr(self, '_monitor_window') and self._monitor_window:
            self._monitor_window.show()
            self._monitor_window.raise_()
            self._monitor_window.activateWindow()

    def _setup_vps_progress(self, instance_worker_map):
        """Set up the VPS progress panel, open the monitor window, and start log monitoring."""
        self.vps_progress_group.setVisible(True)

        num_workers = len(instance_worker_map)
        self.vps_overall_progress.setValue(0)
        self.vps_overall_progress.setMaximum(100)
        self.vps_status_label.setText(f"{num_workers} workers active")

        # Create / reset the monitor window
        if hasattr(self, '_monitor_window') and self._monitor_window:
            self._monitor_window.close()
        self._monitor_window = VPSMonitorWindow(parent=self)
        self._monitor_window.setup_workers(instance_worker_map)
        self._monitor_window.show()

        # Start R2-based status monitor
        r2_client = R2Client(
            account_id=self._vps_context['r2_account'],
            access_key_id=self._vps_context['r2_access'],
            secret_access_key=self._vps_context['r2_secret'],
            bucket_name=self._vps_context['r2_bucket'],
        )
        status_prefix = f"Features/{self._vps_context['country']}/{self._vps_context['state']}/{self._vps_context['city']}"

        self._r2_client = r2_client
        self._status_prefix = status_prefix
        self._log_monitor = R2StatusMonitorThread(
            r2_client=r2_client,
            status_prefix=status_prefix,
            num_workers=num_workers,
            instance_worker_map=instance_worker_map,
            poll_interval=10.0,
            vast_manager=self._vast_manager if hasattr(self, '_vast_manager') else None,
        )
        self._log_monitor.progress_update.connect(self._on_vps_worker_progress)
        self._log_monitor.worker_finished.connect(self._on_vps_worker_finished)
        self._log_monitor.log_message.connect(self._on_vps_log)
        self._log_monitor.instance_needs_replace.connect(self._on_instance_needs_replace)
        self._log_monitor.start()

    def _on_vps_worker_progress(self, worker_idx, processed, total, eta, speed, status):
        """Handle progress update from a VPS worker."""
        if hasattr(self, '_monitor_window') and self._monitor_window:
            self._monitor_window.update_worker(worker_idx, processed, total, eta, speed, status)

        # Update the overall mini-bar in the left panel
        if hasattr(self, '_log_monitor'):
            overall = self._log_monitor.get_overall_progress()
            total_rows = min(max(overall['total_rows'], 1), 2147483647)
            total_processed = min(overall['total_processed'], total_rows)
            self.vps_overall_progress.setMaximum(total_rows)
            self.vps_overall_progress.setValue(total_processed)
            self.vps_status_label.setText(
                f"{overall['active_workers']} active · {overall['finished_workers']} done"
            )
            if hasattr(self, '_monitor_window') and self._monitor_window:
                self._monitor_window.update_overall(
                    overall['total_processed'], overall['total_rows'],
                    overall['active_workers'], overall['finished_workers'],
                    overall['total_workers'],
                )

    def _show_worker_logs(self, instance_id, worker_idx, silent=False):
        """Fetch logs: try R2 first, fall back to vastai. Save to logs/ dir."""
        if not silent:
            self.statusBar().showMessage(f"Fetching logs for worker {worker_idx}...")

        # Try R2 first (works even after instance is destroyed)
        logs = None
        if hasattr(self, '_r2_client') and hasattr(self, '_status_prefix'):
            try:
                log_key = f"Logs/{self._status_prefix}/worker_{worker_idx}.log"
                # Check existence first to avoid 3x retry on 404
                if self._r2_client.file_exists(log_key):
                    log_dir = Path("logs")
                    log_dir.mkdir(exist_ok=True)
                    tmp_path = str(log_dir / f"worker_{worker_idx}_{instance_id}.log")
                    if self._r2_client.download_file(log_key, tmp_path):
                        with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
                            logs = f.read()
                        if logs and logs.strip():
                            logger.info(f"Downloaded logs from R2: {log_key}")
            except Exception as e:
                logger.debug(f"R2 log download failed for worker {worker_idx}: {e}")

        if logs and logs.strip():
            self._save_and_show_logs(worker_idx, instance_id, logs, silent)
            return

        # Fall back to vastai show logs (only works while instance is alive)
        if not hasattr(self, '_vast_manager') or not self._vast_manager:
            if not silent:
                QMessageBox.information(self, "No Logs",
                    f"No log file found for worker {worker_idx}.\n\n"
                    "The worker may not have uploaded logs yet, or the instance died before it could.")
            return

        thread = LogFetcherThread(self._vast_manager, instance_id, worker_idx, tail=2000)

        if not hasattr(self, '_log_fetchers'):
            self._log_fetchers = []
        self._log_fetchers.append(thread)
        self._log_fetchers = [t for t in self._log_fetchers if t.isRunning()]

        def on_logs(w_idx, i_id, fetched_logs):
            self._save_and_show_logs(w_idx, i_id, fetched_logs, silent)

        def on_error(w_idx, i_id, err_msg):
            if "not found" in err_msg.lower() or "No such container" in err_msg:
                if not silent:
                    self.statusBar().showMessage(
                        f"Worker {w_idx}: No logs yet (container loading or destroyed)"
                    )
            elif not silent:
                QMessageBox.warning(self, "Fetch Error", f"Failed to fetch logs: {err_msg}")
                self.statusBar().showMessage(f"Failed to fetch logs for worker {w_idx}")

        thread.logs_fetched.connect(on_logs)
        thread.error_occurred.connect(on_error)
        thread.finished.connect(lambda: self._log_fetchers.remove(thread) if thread in self._log_fetchers else None)
        thread.start()

    def _save_and_show_logs(self, worker_idx, instance_id, logs, silent=False):
        """Save logs to file and optionally show in a dialog."""
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"worker_{worker_idx}_{instance_id}.log"
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(logs if logs else "(No logs found)")

            if not silent:
                logger.info(f"Saved logs to {log_file}")
                dlg = QDialog(self)
                dlg.setWindowTitle(f"Logs - Worker {worker_idx} (Instance {instance_id})")
                dlg.resize(800, 600)
                layout = QVBoxLayout(dlg)

                lbl = QLabel(f"Saved to: {log_file.absolute()}")
                lbl.setStyleSheet("color: #888; font-style: italic;")
                layout.addWidget(lbl)

                text_edit = QTextEdit()
                text_edit.setReadOnly(True)
                text_edit.setPlainText(logs if logs else "(No logs found)")
                text_edit.setStyleSheet("font-family: Consolas, monospace; background: #1e1e1e; color: #d4d4d4;")
                layout.addWidget(text_edit)

                btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
                btn_box.rejected.connect(dlg.reject)
                layout.addWidget(btn_box)

                self.statusBar().showMessage(f"Logs saved to {log_file}")
                dlg.exec()
        except Exception as e:
            logger.error(f"Failed to save/show logs: {e}")

    def _toggle_auto_export(self, checked):
        """Toggle auto-export timer."""
        if checked:
            self.auto_export_timer.start()
            self.statusBar().showMessage("Auto-export enabled (every 60s)")
            # Trigger one immediately
            self._auto_export_tick()
        else:
            self.auto_export_timer.stop()
            self.statusBar().showMessage("Auto-export disabled")

    def _auto_export_tick(self):
        """Run periodic auto-export."""
        if not hasattr(self, '_log_monitor') or not self._log_monitor.instance_worker_map:
            return
        
        self.statusBar().showMessage("Auto-exporting logs...")
        count = 0
        for instance_id, worker_idx in self._log_monitor.instance_worker_map.items():
            # Skip if worker finished? Maybe still want logs.
            self._show_worker_logs(instance_id, worker_idx, silent=True)
            count += 1
        
        # Don't show modal popup for auto-export, just status
        # status bar update will happen when threads finish, but we can update here too

            
    def _export_all_logs(self):
        """Export logs for all active workers."""
        if not hasattr(self, '_log_monitor') or not self._log_monitor.instance_worker_map:
            QMessageBox.warning(self, "No Instances", "No active instances to export logs from.")
            return

        count = 0
        self.statusBar().showMessage("Exporting logs for all instances...")
        try:
            for instance_id, worker_idx in self._log_monitor.instance_worker_map.items():
                self._show_worker_logs(instance_id, worker_idx, silent=True)
                count += 1
                QApplication.processEvents() # Keep UI responsive
            
            QMessageBox.information(self, "Export Complete", f"Exported logs for {count} instances to logs/ folder.")
            self.statusBar().showMessage(f"Exported {count} logs.")
            self.statusBar().showMessage(f"Exported {count} logs.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export all logs: {e}")

        # Update overall progress
        if hasattr(self, '_log_monitor'):
            overall = self._log_monitor.get_overall_progress()
            total_rows = min(max(overall['total_rows'], 1), 2147483647)
            total_processed = min(overall['total_processed'], total_rows)
            self.vps_overall_progress.setMaximum(total_rows)
            self.vps_overall_progress.setValue(total_processed)
            self.vps_status_label.setText(
                f"{overall['active_workers']} active, {overall['finished_workers']} finished"
            )

    def _find_instance_id_for_worker(self, worker_idx):
        """Look up instance ID for a given worker index."""
        if hasattr(self, '_log_monitor') and hasattr(self._log_monitor, 'instance_worker_map'):
            for iid, widx in self._log_monitor.instance_worker_map.items():
                if widx == worker_idx:
                    return iid
        # Fallback: R2 status state
        try:
            state = self._log_monitor._worker_states.get(worker_idx)
            if state and state.instance_id:
                return state.instance_id
        except Exception:
            pass
        return None

    def _on_vps_worker_finished(self, worker_idx, status):
        """Handle worker completion/failure — just update the display, never auto-kill."""
        color = "#00cc00" if status == "COMPLETED" else "#cc0000"
        if hasattr(self, '_monitor_window') and self._monitor_window:
            self._monitor_window.mark_worker_status(worker_idx, status, color)

        self.statusBar().showMessage(f"Worker {worker_idx}: {status}")
        logger.info(f"Worker {worker_idx} finished: {status}")

        # Grab logs silently so they're saved to the logs/ folder
        instance_id = self._find_instance_id_for_worker(worker_idx)
        if instance_id:
            self._show_worker_logs(instance_id, worker_idx, silent=True)

        # Check if all done
        if hasattr(self, '_log_monitor'):
            overall = self._log_monitor.get_overall_progress()
            if overall['active_workers'] == 0:
                self.vps_status_label.setText("All workers finished!")
                self.deploy_button.setEnabled(True)
                self.toggle_controls(True)
                self.statusBar().showMessage("VPS Deploy: All workers complete!")

                if self.auto_build_check.isChecked():
                    QTimer.singleShot(1000, self._auto_launch_builder)

    def _on_instance_needs_replace(self, worker_idx, instance_id, error_category):
        """Log the failure and update the monitor — do NOT auto-destroy or auto-replace."""
        msg = f"Worker {worker_idx} ({instance_id}): {error_category} detected"
        logger.warning(msg)
        self.statusBar().showMessage(msg)

        if hasattr(self, '_monitor_window') and self._monitor_window:
            self._monitor_window.mark_worker_status(
                worker_idx, f"ERROR: {error_category}", "#ff8800"
            )

    def _replace_worker_instance(self, worker_idx):
        """Search for a new offer and create a replacement instance for a failed worker."""
        if not hasattr(self, '_vast_manager') or not hasattr(self, '_vps_context'):
            logger.error(f"Cannot replace worker {worker_idx}: missing manager or context")
            return

        ctx = self._vps_context

        # Re-use existing offer search (quick, single-threaded since we just need one)
        try:
            # Build geo filter from context
            geo_filter = ctx.get('geo_filter', '').strip()
            geo_region = None
            if geo_filter:
                codes = [c.strip().upper() for c in geo_filter.replace(',', ' ').split() if c.strip()]
                if codes:
                    geo_region = " ".join(codes)

            offers = self._vast_manager.search_offers(
                min_disk_gb=ctx.get('disk_gb', 100),
                num_gpus=1,
                region=geo_region,
            )
        except Exception as e:
            logger.error(f"Offer search for replacement failed: {e}")
            self._mark_worker_replace_failed(worker_idx)
            return

        if not offers:
            logger.error(f"No offers available to replace worker {worker_idx}")
            self._mark_worker_replace_failed(worker_idx)
            return

        # Pick the cheapest offer (first in sorted list)
        offer = offers[0]

        features_prefix = f"Features/{ctx['country']}/{ctx['state']}/{ctx['city']}"
        env_vars = {
            'R2_ACCOUNT_ID': ctx['r2_account'],
            'R2_ACCESS_KEY_ID': ctx['r2_access'],
            'R2_SECRET_ACCESS_KEY': ctx['r2_secret'],
            'R2_BUCKET_NAME': ctx['r2_bucket'],
            'WORKER_INDEX': str(worker_idx),
            'NUM_WORKERS': str(ctx.get('total_workers', ctx.get('actual_workers', self._log_monitor.num_workers))),
            'CSV_BUCKET_PREFIX': f"CSV/{ctx['country']}/{ctx['state']}/{ctx['city']}",
            'FEATURES_BUCKET_PREFIX': features_prefix,
            'CITY_NAME': ctx['city'],
            'VAST_API_KEY': ctx['vast_key'],
        }

        try:
            new_id = self._vast_manager.create_instance(
                offer_id=offer['id'],
                docker_image=ctx['docker_image'],
                env_vars=env_vars,
                disk_gb=ctx.get('disk_gb', 100),
                onstart_cmd="bash /app/entrypoint.sh",
                template_hash=VastManager.GEOAXIS_TEMPLATE_HASH,
            )
        except Exception as e:
            logger.error(f"Failed to create replacement instance for worker {worker_idx}: {e}")
            self._mark_worker_replace_failed(worker_idx)
            return

        if not new_id:
            self._mark_worker_replace_failed(worker_idx)
            return

        # Write instance ID to R2 for the worker to find itself
        try:
            self._r2_client.upload_json(
                f"Status/{features_prefix}/worker_{worker_idx}_instance.json",
                {'instance_id': new_id, 'worker_index': worker_idx},
            )
        except Exception as e:
            logger.warning(f"Failed to write replacement instance ID to R2: {e}")

        # Update monitor map and reset health check state
        self._log_monitor.instance_worker_map[new_id] = worker_idx
        self._log_monitor._health_checked.discard(new_id)

        # Reset the worker state so the monitor will re-check it
        state = self._log_monitor._worker_states.get(worker_idx)
        if state:
            state.last_update = 0.0
            state.status = "REPLACING"

        if hasattr(self, '_monitor_window') and self._monitor_window:
            self._monitor_window.mark_worker_status(
                worker_idx, f"REPLACED → {new_id}", "#00aaff"
            )

        self.statusBar().showMessage(
            f"Worker {worker_idx}: replaced with new instance {new_id} "
            f"(offer {offer['id']}, {offer['gpu_name']}, ${offer['price_per_hr']:.3f}/hr)"
        )
        logger.info(f"Worker {worker_idx}: replaced {new_id} on offer {offer['id']}")

    def _mark_worker_replace_failed(self, worker_idx):
        """Mark a worker as failed when replacement also fails."""
        if hasattr(self, '_monitor_window') and self._monitor_window:
            self._monitor_window.mark_worker_status(worker_idx, "REPLACE FAILED", "#cc0000")
        self._log_monitor._finished_workers.add(worker_idx)
        self._log_monitor.worker_finished.emit(worker_idx, "FAILED:replace_failed")

    def _on_vps_log(self, message):
        """Handle general log messages from the monitor."""
        self.statusBar().showMessage(f"VPS: {message}")

    def _destroy_all_instances(self):
        """Export logs for all instances, then destroy them."""
        if not hasattr(self, '_vast_manager'):
            return

        reply = QMessageBox.question(self, "Destroy All",
            "This will export all logs then destroy ALL running instances.\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply != QMessageBox.StandardButton.Yes:
            return

        if not hasattr(self, '_log_monitor') or not self._log_monitor.instance_worker_map:
            # No instances tracked, just destroy
            count = self._vast_manager.destroy_all()
            self.vps_status_label.setText(f"Destroyed {count} instances")
            self.deploy_button.setEnabled(True)
            self.toggle_controls(True)
            return

        self.statusBar().showMessage("Exporting logs before destroying...")

        # Export all logs (tries R2 first, then vastai)
        for instance_id, worker_idx in self._log_monitor.instance_worker_map.items():
            self._show_worker_logs(instance_id, worker_idx, silent=True)

        # Stop monitor and destroy
        self._log_monitor.stop()
        count = self._vast_manager.destroy_all()
        self.vps_status_label.setText(f"Exported logs & destroyed {count} instances")
        self.deploy_button.setEnabled(True)
        self.toggle_controls(True)
        self.statusBar().showMessage(f"Exported all logs and destroyed {count} instances.")

    # ══════════════════════════════════════════════════════════════════════
    # Index Builder Methods
    # ══════════════════════════════════════════════════════════════════════

    def start_index_build(self):
        """Manual trigger: search R2 for features, then launch builder instance."""
        if not VPS_AVAILABLE:
            QMessageBox.warning(self, "Missing Dependencies",
                "VPS modules not available. Install: pip install boto3 vastai python-dotenv")
            return

        r2_account = self.r2_account_edit.text().strip()
        r2_access = self.r2_access_key_edit.text().strip()
        r2_secret = self.r2_secret_key_edit.text().strip()
        r2_bucket = self.r2_bucket_edit.text().strip()
        vast_key = self.vast_api_key_edit.text().strip()

        if not all([r2_account, r2_access, r2_secret, r2_bucket]):
            QMessageBox.warning(self, "Missing R2 Config", "Please fill in all R2 fields in the VPS Deploy tab.")
            return
        if not vast_key:
            QMessageBox.warning(self, "Missing Vast.ai Key", "Please enter your Vast.ai API key.")
            return

        # Check for manual path override first
        path_override = self.builder_path_edit.text().strip().strip('/')
        if path_override:
            parts = [p for p in path_override.split('/') if p]
            if len(parts) < 2:
                QMessageBox.warning(self, "Invalid Path",
                    "Path must be at least Country/City (e.g. US/Florida/Florida).")
                return
            if not hasattr(self, '_vps_context'):
                self._vps_context = {}
            self._vps_context['country'] = parts[0]
            self._vps_context['state'] = parts[1] if len(parts) > 2 else parts[0]
            self._vps_context['city'] = parts[-1]
            self._vps_context['_path_override'] = path_override
            self.statusBar().showMessage(f"Build Index: Using path override → Features/{path_override}")
            self._launch_builder_search()
            return

        if not hasattr(self, '_vps_context') or not self._vps_context.get('country'):
            # No region context yet — try to geocode from selected map shape
            selected_coords = self.map_view.get_selected_cells_coords()
            if not selected_coords or len(selected_coords) == 0:
                QMessageBox.warning(self, "No Region",
                    "No region context. Enter a Path Override (e.g. US/Florida/Florida), "
                    "select a shape on the map, or deploy feature extraction workers first.")
                return

            first_coords = selected_coords[0]
            lats = [c[0] for c in first_coords]
            lons = [c[1] for c in first_coords]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)

            try:
                country, state, city = _reverse_geocode(center_lat, center_lon)
            except Exception as e:
                QMessageBox.warning(self, "Geocode Failed",
                    f"Could not determine region from selected shape:\n{e}")
                return

            self.statusBar().showMessage(f"Build Index: Resolved region → {country}/{state}/{city}")
            logger.info(f"Build Index geocoded ({center_lat:.4f}, {center_lon:.4f}) → {country}/{state}/{city}")

            if not hasattr(self, '_vps_context'):
                self._vps_context = {}
            self._vps_context['country'] = country
            self._vps_context['state'] = state
            self._vps_context['city'] = city
            self._vps_context.pop('_path_override', None)

        self._launch_builder_search()

    def _auto_launch_builder(self):
        """Auto-triggered when all feature extraction workers finish."""
        if not hasattr(self, '_vps_context') or not self._vps_context.get('country'):
            return

        reply = QMessageBox.question(self, "Build Index",
            "All feature extraction workers have finished.\n\n"
            "Launch index builder to merge features and build FAISS index?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            self._launch_builder_search()

    def _launch_builder_search(self):
        """Search for builder-grade VPS offers (240GB+ RAM, 500GB+ disk)."""
        vast_key = self.vast_api_key_edit.text().strip()
        max_price = self.builder_max_price_spin.value()
        min_disk = self.builder_disk_spin.value()

        self.build_index_button.setEnabled(False)
        self.statusBar().showMessage("Searching for builder instances (128GB+ RAM, 500GB+ disk)...")

        self._builder_search_thread = BuilderSearchThread(vast_key, max_price, min_disk)
        self._builder_search_thread.status_update.connect(lambda msg: self.statusBar().showMessage(msg))
        self._builder_search_thread.offers_found.connect(self._on_builder_offers_found)
        self._builder_search_thread.error_occurred.connect(self._on_builder_error)
        self._builder_search_thread.start()

    def _on_builder_error(self, error_msg):
        """Handle builder search error."""
        QMessageBox.critical(self, "Builder Error", f"Builder search failed: {error_msg}")
        self.build_index_button.setEnabled(True)

    def _on_builder_offers_found(self, offers):
        """Show builder offer selection dialog (select 1 instance)."""
        if not offers:
            QMessageBox.warning(self, "No Offers",
                "No Vast.ai offers found with 128GB+ RAM and 500GB+ disk.\n"
                "Try increasing the max $/hr in the Index Build tab.")
            self.build_index_button.setEnabled(True)
            return

        self._builder_vast_manager = self._builder_search_thread._vast_manager

        dialog = OfferSelectionDialog(offers, 1, self)
        dialog.setWindowTitle("Select Builder Instance (128GB+ RAM, 500GB+ storage)")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.statusBar().showMessage("Index build cancelled.")
            self.build_index_button.setEnabled(True)
            return

        selected_offers = dialog.get_selected_offers()
        if not selected_offers:
            self.build_index_button.setEnabled(True)
            return

        self._create_builder_instance(selected_offers[0])

    def _create_builder_instance(self, offer):
        """Create the builder instance with the right env vars."""
        ctx = self._vps_context
        r2_account = self.r2_account_edit.text().strip()
        r2_access = self.r2_access_key_edit.text().strip()
        r2_secret = self.r2_secret_key_edit.text().strip()
        r2_bucket = self.r2_bucket_edit.text().strip()
        vast_key = self.vast_api_key_edit.text().strip()

        if ctx.get('_path_override'):
            features_prefix = f"Features/{ctx['_path_override']}"
        else:
            features_prefix = f"Features/{ctx['country']}/{ctx['state']}/{ctx['city']}"
        city_name = ctx['city']

        env_vars = {
            'R2_ACCOUNT_ID': r2_account,
            'R2_ACCESS_KEY_ID': r2_access,
            'R2_SECRET_ACCESS_KEY': r2_secret,
            'R2_BUCKET_NAME': r2_bucket,
            'FEATURES_BUCKET_PREFIX': features_prefix,
            'CITY_NAME': city_name,
            'VAST_API_KEY': vast_key,
            'INDEX_TYPE': self.index_type_combo.currentText(),
            'NLIST': str(self.nlist_spin.value()),
            'M': str(self.m_spin.value()),
            'NBITS': str(self.nbits_spin.value()),
            'TRAIN_SAMPLES': str(self.train_samples_spin.value()),
            'NITER': str(self.niter_spin.value()),
        }

        self.statusBar().showMessage("Creating builder instance...")

        try:
            instance_id = self._builder_vast_manager.create_instance(
                offer_id=offer['id'],
                docker_image=self.builder_image_edit.text().strip(),
                env_vars=env_vars,
                disk_gb=self.builder_disk_spin.value(),
                onstart_cmd="bash /app/entrypoint.sh",
            )

            if not instance_id:
                QMessageBox.critical(self, "Creation Failed", "Failed to create builder instance.")
                self.build_index_button.setEnabled(True)
                return

            # Write instance ID to R2
            r2 = R2Client(
                account_id=r2_account,
                access_key_id=r2_access,
                secret_access_key=r2_secret,
                bucket_name=r2_bucket,
            )
            r2.upload_json(
                f"Status/{features_prefix}/builder_instance.json",
                {'instance_id': instance_id, 'worker': 'builder'},
            )

            self._builder_instance_id = instance_id
            self._setup_builder_progress(r2, features_prefix)

        except Exception as e:
            QMessageBox.critical(self, "Builder Error", f"Failed to create builder: {e}")
            self.build_index_button.setEnabled(True)

    def _setup_builder_progress(self, r2, features_prefix):
        """Set up builder progress monitoring."""
        # Clear stale status from previous runs so the monitor doesn't
        # immediately pick up an old FAILED/COMPLETED result.
        try:
            r2.delete_file(f"Status/{features_prefix}/builder.json")
        except Exception:
            pass

        self.builder_progress_group.setVisible(True)
        self.builder_status_label.setText("Builder instance starting...")
        self.builder_progress_bar.setValue(0)
        self.builder_progress_bar.setMaximum(100)
        self.builder_step_label.setText("Waiting for first status update...")

        self._builder_monitor = BuilderMonitorThread(
            r2_client=r2,
            status_prefix=features_prefix,
            poll_interval=10.0,
        )
        self._builder_monitor.progress_update.connect(self._on_builder_progress)
        self._builder_monitor.builder_finished.connect(self._on_builder_finished)
        self._builder_monitor.start()

        self.statusBar().showMessage(f"Builder instance {self._builder_instance_id} running. Monitoring...")

    def _on_builder_progress(self, step, detail, pct, status):
        """Handle builder progress update."""
        self.builder_progress_bar.setValue(pct)
        self.builder_status_label.setText(f"{status} — Step: {step}")
        self.builder_step_label.setText(detail)
        self.statusBar().showMessage(f"Builder: {detail} ({pct}%)")

    def _on_builder_finished(self, status):
        """Handle builder completion."""
        if status == "COMPLETED":
            self.builder_status_label.setText("Index built successfully!")
            self.builder_status_label.setStyleSheet("color: #00cc00; font-weight: bold;")
            self.builder_progress_bar.setValue(100)

            ctx = self._vps_context
            index_prefix = f"Index/{ctx['country']}/{ctx['state']}/{ctx['city']}"
            QMessageBox.information(self, "Index Built",
                f"FAISS index has been built and uploaded to R2.\n\n"
                f"Location: {index_prefix}/\n"
                f"Files: megaloc.index, metadata.json, config.json")
        else:
            self.builder_status_label.setText(f"Builder failed: {status}")
            self.builder_status_label.setStyleSheet("color: #cc0000; font-weight: bold;")

        # Try to destroy the builder instance
        if hasattr(self, '_builder_vast_manager') and hasattr(self, '_builder_instance_id'):
            try:
                self._builder_vast_manager.destroy_instance(self._builder_instance_id)
            except Exception:
                pass  # Instance may have self-destructed

        self.build_index_button.setEnabled(True)
        self.statusBar().showMessage(f"Builder: {status}")


class OfferSelectionDialog(QDialog):
    """Dialog for selecting Vast.ai GPU offers."""

    # Region sort priority: North America first, then South America, then rest
    _NORTH_AMERICA = frozenset({'US', 'CA', 'MX'})
    _SOUTH_AMERICA = frozenset({
        'BR', 'AR', 'CL', 'CO', 'PE', 'VE', 'EC', 'BO', 'PY', 'UY',
        'GY', 'SR', 'GF', 'PA', 'CR', 'GT', 'HN', 'SV', 'NI', 'BZ',
    })

    @staticmethod
    def _region_sort_key(offer):
        loc = (offer.get('location') or 'ZZ').upper().strip()
        # Extract country code (handle "US/CA", "US, California", etc.)
        code = loc.split('/')[0].split(',')[0].split('-')[0].strip()[:2]
        if code in OfferSelectionDialog._NORTH_AMERICA:
            region_rank = 0
        elif code in OfferSelectionDialog._SOUTH_AMERICA:
            region_rank = 1
        else:
            region_rank = 2
        # Within each region, sort by price
        return (region_rank, offer.get('price_per_hr', 999))

    def __init__(self, offers, num_needed, parent=None):
        super().__init__(parent)
        self.offers = sorted(offers, key=self._region_sort_key)
        self.num_needed = num_needed
        self.setWindowTitle(f"Select Vast.ai Offers ({num_needed} needed)")
        self.setMinimumSize(800, 500)

        layout = QVBoxLayout(self)

        self.selection_label = QLabel(f"Selected: 0 / {num_needed}")
        self.selection_label.setStyleSheet("font-weight: bold; padding: 5px; font-size: 13px;")
        layout.addWidget(self.selection_label)

        self.table = QTableWidget(len(offers), 11)
        self.table.setHorizontalHeaderLabels([
            "GPU", "VRAM (GB)", "CPU Model", "Cores", "GHz", "RAM (GB)",
            "Disk (GB)", "$/hr", "↓ Mbps", "Reliability", "Region"
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)

        for row, o in enumerate(offers):
            self.table.setItem(row, 0, QTableWidgetItem(f"{o['num_gpus']}x {o['gpu_name']}"))
            self.table.setItem(row, 1, QTableWidgetItem(f"{o['gpu_ram'] / 1024:.0f}"))
            self.table.setItem(row, 2, QTableWidgetItem(o.get('cpu_name', 'Unknown')))
            self.table.setItem(row, 3, QTableWidgetItem(f"{o['cpu_cores']:.0f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{o.get('cpu_ghz', 0):.1f}"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{o['ram'] / 1024:.0f}"))
            self.table.setItem(row, 6, QTableWidgetItem(f"{o['disk']:.0f}"))
            self.table.setItem(row, 7, QTableWidgetItem(f"${o['price_per_hr']:.3f}"))
            self.table.setItem(row, 8, QTableWidgetItem(f"{o['inet_down']:.0f}"))
            self.table.setItem(row, 9, QTableWidgetItem(f"{o['reliability']:.1%}"))
            self.table.setItem(row, 10, QTableWidgetItem(o.get('location', 'Unknown')))

        layout.addWidget(self.table)

        self.ok_button = QPushButton("OK")
        self.ok_button.setEnabled(False)
        self.ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.ok_button)
        btn_layout.addWidget(cancel_button)
        layout.addLayout(btn_layout)

    def _on_selection_changed(self):
        """Enforce exact selection count and update label."""
        selected_rows = set(idx.row() for idx in self.table.selectedIndexes())
        count = len(selected_rows)

        if count > self.num_needed:
            # Block extra selections by undoing the latest ones
            self.table.selectionModel().blockSignals(True)
            self.table.clearSelection()
            # Re-select only the first num_needed rows from the set
            kept = sorted(selected_rows)[:self.num_needed]
            for row in kept:
                self.table.selectRow(row)
            self.table.selectionModel().blockSignals(False)
            count = self.num_needed

        self.selection_label.setText(f"Selected: {count} / {self.num_needed}")

        if count == self.num_needed:
            self.selection_label.setStyleSheet("font-weight: bold; padding: 5px; font-size: 13px; color: #00cc00;")
            self.ok_button.setEnabled(True)
        else:
            self.selection_label.setStyleSheet("font-weight: bold; padding: 5px; font-size: 13px; color: white;")
            self.ok_button.setEnabled(False)

    def get_selected_offers(self):
        """Return list of selected offer dicts."""
        selected_rows = set(idx.row() for idx in self.table.selectedIndexes())
        return [self.offers[r] for r in sorted(selected_rows)]


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
