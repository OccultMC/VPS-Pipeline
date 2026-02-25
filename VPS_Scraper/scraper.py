"""
Unified Street View Scraper

Combines Stage 1 metadata collection with Stage 2 image downloading.
Supports region-based scraping (polygons/shapefiles) and random global mode.
"""
import asyncio
import csv
import os
import sys
import random
import math
from pathlib import Path
from typing import List, Tuple, Set, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from aiohttp import ClientSession, TCPConnector, ClientTimeout

# Try high-performance event loops
try:
    import winloop
    asyncio.set_event_loop_policy(winloop.EventLoopPolicy())
except ImportError:
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass

# Add Stage 1 to path for streetlevel imports (append, not insert(0),
# so that Stage_0/scraper.py is found before Stage_1/scraper.py when
# ProcessPoolExecutor spawns child processes on Windows)
stage1_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Stage_1_Dataset_Collection")
if stage1_dir not in sys.path:
    sys.path.append(stage1_dir)

from streetlevel.streetview import (
    find_panorama_async, 
    find_panorama_by_id_async,
    get_coverage_tile_async,
    StreetViewPanorama
)
from streetlevel.geo import wgs84_to_tile_coord
from shapely.geometry import Point, Polygon

# Handle imports for both package and standalone usage
try:
    from .gsvpd.constants import ZOOM_SIZES, OLD_ZOOM_SIZES, TILE_COUNT_TO_SIZE, TILES_AXIS_COUNT, TILE_SIZE
    from .gsvpd.directional_views import DirectionalViewExtractor, DirectionalViewConfig
    from .gsvpd.augmentations import apply_pixel_augmentations
except ImportError:
    from gsvpd.constants import ZOOM_SIZES, OLD_ZOOM_SIZES, TILE_COUNT_TO_SIZE, TILES_AXIS_COUNT, TILE_SIZE
    from gsvpd.directional_views import DirectionalViewExtractor, DirectionalViewConfig
    from gsvpd.augmentations import apply_pixel_augmentations


def _write_file(filepath: str, data: bytes):
    """Write bytes to file. Runs in ThreadPoolExecutor."""
    with open(filepath, 'wb') as f:
        f.write(data)


def _cpu_stitch_and_extract(
    tile_data_list: list,
    zoom_level: int,
    config: dict,
    panoid: str,
    heading_deg: float = None
) -> dict:
    """
    CPU-intensive work: determine dimensions, stitch tiles, extract perspective views.
    Designed to run in ProcessPoolExecutor.
    """
    from io import BytesIO
    import numpy as np
    import cv2
    from PIL import Image

    result = {
        "success": False,
        "views": [],
        "view_filenames": [],
        "panorama_bytes": None,
        "error": ""
    }

    try:
        # Determine dimensions from tile content
        x_values = {x for x, _, _ in tile_data_list}
        y_values = {y for _, y, _ in tile_data_list}
        x_count, y_count = len(x_values), len(y_values)

        if zoom_level == 0:
            tile_img = Image.open(BytesIO(tile_data_list[0][2]))
            arr = np.array(tile_img)
            tile_img.close()
            black_pixels = np.all(arr <= 10, axis=2)
            black_perc = float(np.sum(black_pixels) / black_pixels.size * 100)
            w, h = OLD_ZOOM_SIZES[zoom_level] if black_perc > 55 else ZOOM_SIZES[zoom_level]
        elif 0 < zoom_level <= 2:
            is_black = False
            if len(tile_data_list) > 1:
                tile_img = Image.open(BytesIO(tile_data_list[1][2]))
                arr = np.array(tile_img)
                tile_img.close()
                is_black = bool(np.all(arr[-5:] <= 10))
            w, h = OLD_ZOOM_SIZES[zoom_level] if is_black else ZOOM_SIZES[zoom_level]
        else:
            w, h = TILE_COUNT_TO_SIZE.get(
                (x_count, y_count),
                ZOOM_SIZES.get(zoom_level, (2048, 1024))
            )

        # Stitch tiles into panorama
        full_img = Image.new("RGB", (w, h))
        for x, y, tile_bytes in tile_data_list:
            tile = Image.open(BytesIO(tile_bytes))
            full_img.paste(tile, (x * TILE_SIZE, y * TILE_SIZE))
            tile.close()

        full_img_cv = cv2.cvtColor(np.array(full_img), cv2.COLOR_RGB2BGR)
        full_img.close()

        # Extract directional views
        if config.get("create_directional_views"):
            view_extractor = DirectionalViewExtractor()
            view_config = DirectionalViewConfig(
                output_resolution=config.get("view_resolution", 512),
                fov_degrees=config.get("view_fov", 90.0),
                num_views=config.get("num_views", 6),
                global_view=config.get("global_view", False),
                augment=config.get("augment", False),
                target_yaw=heading_deg
            )

            view_result = view_extractor.extract_views(full_img_cv, view_config)

            if view_result.success:
                for i, (view, meta) in enumerate(zip(view_result.views, view_result.metadata)):
                    if config.get("augment"):
                        view = apply_pixel_augmentations(view)

                    yaw = meta['yaw']
                    if config.get("global_view") or heading_deg is not None:
                        fname = f"{panoid}_rnd_Y{int(yaw)}.jpg"
                        if config.get("augment"):
                            fname = f"{panoid}_aug_Y{int(yaw)}_P{int(meta['pitch'])}.jpg"
                    else:
                        fname = f"{panoid}_zoom{zoom_level}_view{i:02d}_{yaw:.0f}deg.jpg"

                    _, buffer = cv2.imencode('.jpg', view, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    result["views"].append(buffer.tobytes())
                    result["view_filenames"].append(fname)

        # Encode full panorama if requested
        if config.get("keep_panorama"):
            _, buffer = cv2.imencode('.jpg', full_img_cv, [cv2.IMWRITE_JPEG_QUALITY, 95])
            result["panorama_bytes"] = buffer.tobytes()

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


@dataclass
class UnifiedScraperConfig:
    """Configuration for unified scraper."""
    # Stage 1: Metadata scraping settings
    concurrency: int = 1000
    checkpoint_interval: int = 500
    max_retries: int = 5
    proxy_file: Optional[str] = None
    proxies: Optional[List[str]] = None
    timeout_total: float = 8.0
    timeout_connect: float = 3.0
    session_pools: int = 4
    tile_zoom_level: int = 17
    include_historical: bool = False
    historical_offsets: List[int] = field(default_factory=list)
    ray_workers: int = 8
    output_csv_dir: str = "../Output/CSV"
    output_images_dir: str = "../Output/Images"
    
    # Image processing settings
    create_directional_views: bool = False
    keep_panorama: bool = False
    view_resolution: int = 512
    view_fov: float = 90.0
    num_views: int = 6
    global_view: bool = False
    augment: bool = False


class ProxyManager:
    """Manages proxy rotation for distributed requests."""

    def __init__(self, proxy_file: Optional[str] = None, proxies: Optional[List[str]] = None):
        self.proxies: List[str] = []
        self.index = 0
        self.lock = asyncio.Lock()
        self.failed_proxies: Set[str] = set()
        self.proxy_stats: dict = {}

        if proxy_file:
            self.load_from_file(proxy_file)
        elif proxies:
            self.proxies = [p.strip() for p in proxies if p.strip() and not p.startswith('#')]

        for proxy in self.proxies:
            self.proxy_stats[proxy] = {'success': 0, 'fail': 0}

    def load_from_file(self, filepath: str) -> int:
        """Load proxies from a text file."""
        if not os.path.exists(filepath):
            print(f"Proxy file not found: {filepath}")
            return 0

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('protocol://'):
                    continue
                self.proxies.append(line)
                self.proxy_stats[line] = {'success': 0, 'fail': 0}

        print(f"Loaded {len(self.proxies)} proxies from {filepath}")
        return len(self.proxies)

    def get_proxy(self) -> Optional[str]:
        """Get the next proxy in rotation."""
        if not self.proxies:
            return None
        proxy = self.proxies[self.index % len(self.proxies)]
        self.index += 1
        return proxy

    def get_random_proxy(self) -> Optional[str]:
        """Get a random proxy."""
        if not self.proxies:
            return None
        return random.choice(self.proxies)

    @property
    def count(self) -> int:
        return len(self.proxies)

    @property
    def healthy_count(self) -> int:
        return len([p for p in self.proxies if p not in self.failed_proxies])


class UnifiedScraper:
    """
    Unified scraper combining metadata collection and image downloading.
    """
    
    def __init__(self, config: UnifiedScraperConfig = None):
        self.config = config or UnifiedScraperConfig()
        
        # State
        self.found_panos: Set[str] = set()
        self.total_written = 0
        self.total_images = 0
        self.total_errors = 0
        self.tiles_processed = 0
        self.total_tiles = 0
        self.panos_per_second = 0.0
        self.consecutive_errors = 0
        self.phase2_total = 0
        self.phase2_completed = 0
        
        # Geographic info
        self.first_pano_country = None
        self.first_pano_state = None
        self.first_pano_city = None
        
        # File IO
        self.csv_file_handle = None
        self.csv_writer = None
        self.csv_filename = None
        self.images_output_dir = None
        
        # Callbacks
        self.stats_callback = None
        self.progress_callback = None
        self.status_callback = None
        self.point_callback = None
        self.completed_callback = None
        
        # Control
        self._cancelled = False
        
        # Proxy support
        self.proxy_manager: Optional[ProxyManager] = None
        if self.config.proxy_file or self.config.proxies:
            self.proxy_manager = ProxyManager(
                proxy_file=self.config.proxy_file,
                proxies=self.config.proxies
            )
        
        pass  # All image processing now uses async+ProcessPool pipeline
    
    def _attach_proxy_getter(self, session: ClientSession):
        """Attach proxy rotation to session."""
        if self.proxy_manager and self.proxy_manager.count > 0:
            session._get_proxy = self.proxy_manager.get_proxy
        else:
            session._get_proxy = lambda: None
    
    def set_stats_callback(self, callback):
        """Callback(found_count, pps, active_threads, errors)"""
        self.stats_callback = callback
    
    def set_progress_callback(self, callback):
        """Callback(current, total)"""
        self.progress_callback = callback
        
    def set_status_callback(self, callback):
        """Callback(status_message)"""
        self.status_callback = callback
        
    def set_point_callback(self, callback):
        """Callback(lat, lon, panoid)"""
        self.point_callback = callback
        
    def set_completed_callback(self, callback):
        """Callback(panoid)"""
        self.completed_callback = callback
    
    def cancel(self):
        """Cancel scraping."""
        self._cancelled = True

    def _select_historical_panos(self, full_pano):
        """Select historical panoramas closest to configured year offsets.

        For each offset (e.g. 3, 6), finds the historical pano whose capture
        year is closest to (reference_year - offset). Each pano is only used
        once, so two offsets always resolve to two distinct panoramas.
        """
        if not full_pano or not full_pano.historical or not self.config.historical_offsets:
            return []

        # Reference year from the panorama's own capture date
        ref_year = None
        if full_pano.date and hasattr(full_pano.date, 'year'):
            ref_year = full_pano.date.year

        # Fallback: infer from the most recent historical pano
        if ref_year is None:
            hist_years = [h.date.year for h in full_pano.historical
                          if h.date and hasattr(h.date, 'year') and h.date.year]
            ref_year = max(hist_years) + 1 if hist_years else None
        if ref_year is None:
            return []

        selected = []
        used_ids = {full_pano.id}

        for offset in sorted(self.config.historical_offsets):
            target_year = ref_year - offset
            best = None
            best_diff = float('inf')

            for hist in full_pano.historical:
                if hist.id in used_ids:
                    continue
                if hist.date and hasattr(hist.date, 'year') and hist.date.year:
                    diff = abs(hist.date.year - target_year)
                    if diff < best_diff:
                        best_diff = diff
                        best = hist

            if best:
                used_ids.add(best.id)
                selected.append(best)

        return selected

    def init_csv(self, filename: str):
        """Initialize CSV file."""
        self.csv_filename = filename
        header = ['panoid', 'lat', 'lon']
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        mode = 'a' if os.path.exists(filename) else 'w'
        self.csv_file_handle = open(filename, mode, newline='', encoding='utf-8', buffering=65536)
        self.csv_writer = csv.writer(self.csv_file_handle)
        
        if mode == 'w' or self.csv_file_handle.tell() == 0:
            self.csv_writer.writerow(header)
        
        self.csv_file_handle.flush()
    
    def init_images_dir(self, dirname: str):
        """Initialize images output directory."""
        self.images_output_dir = dirname
        os.makedirs(dirname, exist_ok=True)
        # Removed image-specific directory creation
    
    def close_csv(self):
        if self.csv_file_handle:
            self.csv_file_handle.flush()
            self.csv_file_handle.close()
            self.csv_file_handle = None
    
    def _extract_row_data(self, pano: StreetViewPanorama) -> list:
        """Extract CSV row data from panorama object."""
        return [
            pano.id,
            round(pano.lat, 6),
            round(pano.lon, 6),
        ]
    
    async def _stats_ticker(self):
        """Update stats every second."""
        last_count = 0
        while not self._cancelled:
            current_count = self.total_written
            delta = current_count - last_count
            self.panos_per_second = delta
            last_count = current_count
            
            if self.stats_callback:
                self.stats_callback(
                    current_count, 
                    self.panos_per_second, 
                    self.config.concurrency,
                    self.total_errors
                )
            
            if self.progress_callback:
                self.progress_callback(self.tiles_processed, self.total_tiles)
            
            if self.status_callback:
                tile_progress = f"{self.tiles_processed}/{self.total_tiles}" if self.total_tiles > 0 else "..."
                status_msg = f"Tiles: {tile_progress} | Found: {current_count} | Speed: {self.panos_per_second}/s | Images: {self.total_images}"
                self.status_callback(status_msg)
                
            await asyncio.sleep(1)
    
    async def _writer_task(self, queue: asyncio.Queue):
        """Write panoramas to CSV."""
        batch = []
        batch_size = 100
        
        while not self._cancelled:
            try:
                try:
                    pano = await asyncio.wait_for(queue.get(), timeout=0.5)
                    batch.append(pano)
                    queue.task_done()
                    
                    while len(batch) < batch_size:
                        try:
                            pano = queue.get_nowait()
                            batch.append(pano)
                            queue.task_done()
                        except asyncio.QueueEmpty:
                            break
                    
                except asyncio.TimeoutError:
                    pass
                
                if batch and self.csv_writer:
                    for pano in batch:
                        row = self._extract_row_data(pano)
                        self.csv_writer.writerow(row)
                        self.total_written += 1
                    batch.clear()
                    
                    if self.total_written % self.config.checkpoint_interval == 0:
                        self.csv_file_handle.flush()
                        
            except asyncio.CancelledError:
                if batch and self.csv_writer:
                    for pano in batch:
                        row = self._extract_row_data(pano)
                        self.csv_writer.writerow(row)
                        self.total_written += 1
                    self.csv_file_handle.flush()
                break
            except Exception as e:
                print(f"Write Error: {e}")
                self.total_errors += 1
    
    async def scrape_area(self, polygon_coords: List[Tuple[float, float]]):
        """
        Scrape metadata from a polygon region.
        
        Args:
            polygon_coords: List of (lat, lon) tuples defining the polygon
        """
        self.found_panos.clear()
        self.total_written = 0
        self.total_images = 0
        self.tiles_processed = 0
        self.phase2_total = 0
        self.phase2_completed = 0
        self._cancelled = False
        self.first_pano_country = None
        self.first_pano_state = None
        self.first_pano_city = None

        polygon = Polygon([(lon, lat) for lat, lon in polygon_coords])

        connector = TCPConnector(limit=0, ttl_dns_cache=300, enable_cleanup_closed=True)
        timeout = ClientTimeout(total=30, connect=5)

        queue = asyncio.Queue(maxsize=50000)
        collected_panos = []

        async with ClientSession(connector=connector, timeout=timeout) as session:
            self._attach_proxy_getter(session)
            stats_task = asyncio.create_task(self._stats_ticker())
            writer_task = asyncio.create_task(self._writer_task(queue))
            
            # Calculate tiles
            bounds = polygon.bounds
            min_tile_x, max_tile_y = wgs84_to_tile_coord(bounds[1], bounds[0], self.config.tile_zoom_level)
            max_tile_x, min_tile_y = wgs84_to_tile_coord(bounds[3], bounds[2], self.config.tile_zoom_level)
            
            tiles = [(x, y) for x in range(min_tile_x, max_tile_x + 1) 
                            for y in range(min_tile_y, max_tile_y + 1)]
            self.total_tiles = len(tiles)
            
            # PHASE 1: Tile search for panorama IDs
            basic_panos = []
            semaphore = asyncio.Semaphore(self.config.concurrency)
            
            async def fetch_tile(x, y):
                if self._cancelled: return []
                try:
                    async with semaphore:
                        panos = await get_coverage_tile_async(x, y, session)
                    
                    found = []
                    if panos:
                        for p in panos:
                            if p.id not in self.found_panos:
                                point = Point(p.lon, p.lat)
                                if polygon.contains(point):
                                    self.found_panos.add(p.id)
                                    found.append(p)
                    self.tiles_processed += 1
                    return found
                except Exception:
                    self.total_errors += 1
                    self.tiles_processed += 1
                    return []
            
            # Process tiles in batches
            batch_size = self.config.concurrency
            for i in range(0, len(tiles), batch_size):
                if self._cancelled:
                    break
                    
                batch = tiles[i:i + batch_size]
                tasks = [asyncio.create_task(fetch_tile(x, y)) for x, y in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, list):
                        basic_panos.extend(result)
            
            # PHASE 2: Fetch metadata
            if basic_panos and not self._cancelled:
                if self.status_callback:
                    self.status_callback(f"Fetching metadata for {len(basic_panos)} panoramas...")
                
                async def fetch_metadata(basic_pano):
                    if self._cancelled: return []
                    try:
                        async with semaphore:
                            full_pano = await find_panorama_by_id_async(
                                basic_pano.id, session, download_depth=False
                            )

                        pano_to_write = full_pano if full_pano else basic_pano
                        if self.point_callback:
                            self.point_callback(pano_to_write.lat, pano_to_write.lon, pano_to_write.id)
                        await queue.put(pano_to_write)

                        # Prepare for image processing (removed image-specific data)
                        heading = math.degrees(pano_to_write.heading) if pano_to_write.heading else None
                        results = [{
                            'panoid': pano_to_write.id,
                            'heading_deg': heading,
                            'lat': pano_to_write.lat,
                            'lon': pano_to_write.lon
                        }]

                        # Include historical panoramas (closest to target year offsets)
                        if self.config.include_historical and full_pano:
                            for hist in self._select_historical_panos(full_pano):
                                if hist.id not in self.found_panos:
                                    self.found_panos.add(hist.id)
                                    if not hist.country_code and pano_to_write.country_code:
                                        hist.country_code = pano_to_write.country_code
                                    if not hist.street_names and pano_to_write.street_names:
                                        hist.street_names = pano_to_write.street_names
                                    if not hist.address and pano_to_write.address:
                                        hist.address = pano_to_write.address
                                    if self.point_callback:
                                        self.point_callback(hist.lat, hist.lon, hist.id)
                                    if self.csv_writer:
                                        self._write_csv_row(hist)
                                        self.total_written += 1
                    except Exception:
                        if self.point_callback:
                            self.point_callback(basic_pano.lat, basic_pano.lon, basic_pano.id)
                        if self.csv_writer:
                            self._write_csv_row(basic_pano)
                            self.total_written += 1
                        self.total_errors += 1
                
                for i in range(0, len(basic_panos), batch_size):
                    if self._cancelled:
                        break
                    
                    batch = basic_panos[i:i + batch_size]
                    tasks = [asyncio.create_task(fetch_metadata(p)) for p in batch]
                    await asyncio.gather(*tasks, return_exceptions=True)
            
            # Cleanup
            self.close_csv()
            return []
    
    async def scrape_random_global(self, target_count, radius):
        """Scrape random global locations (metadata only)."""
        self.found_panos = set()
        self.total_written = 0
        self.total_errors = 0
        
        if self.csv_filename:
            # Re-init to be sure, or rely on caller
            pass
            
        concurrency = min(self.config.concurrency, 100) # simpler concurrency for random
        
        # We need a proper loop that respects concurrency but keeps trying until target_count
        # The previous implementation had a complex generator. Let's simplify.
        
        pbar = tqdm(total=target_count, desc="Random Global", unit="loc")
        
        async with aiohttp.ClientSession() as session:
            # We can use a producer-consumer or just a large batch of tasks that respawn
            # Let's use a queue-based approach for cleaner control
            
            queue = asyncio.Queue(maxsize=concurrency * 2)
            
            # Producer: generate random points
            async def producer():
                while self.total_written < target_count and not self._cancelled:
                    if queue.qsize() < concurrency:
                        lat = random.uniform(-85, 85)
                        lon = random.uniform(-180, 180)
                        await queue.put((lat, lon))
                    else:
                        await asyncio.sleep(0.1)
                
                # Signal workers to stop
                for _ in range(concurrency):
                    await queue.put(None)

            # Consumer: check GSV
            async def consumer():
                while True:
                    item = await queue.get()
                    if item is None:
                        queue.task_done()
                        break
                    
                    if self.total_written >= target_count or self._cancelled:
                        queue.task_done()
                        continue

                    lat, lon = item
                    try:
                        pano = await find_panorama_by_location(session, lat, lon, radius=radius)
                        if pano and pano.id not in self.found_panos:
                            self.found_panos.add(pano.id)
                            
                            if self.csv_writer:
                                self._write_csv_row(pano)
                                self.total_written += 1
                                pbar.update(1)
                                
                            if self.point_callback:
                                self.point_callback(pano.lat, pano.lon, pano.id)
                            
                            if self.completed_callback:
                                self.completed_callback(pano.id)
                                
                            if self.status_callback and self.total_written % 10 == 0:
                                self.status_callback(f"Found {self.total_written}/{target_count} locations...")
                    except Exception:
                        pass
                    
                    queue.task_done()

            pro_task = asyncio.create_task(producer())
            workers = [asyncio.create_task(consumer()) for _ in range(concurrency)]
            
            await pro_task
            await asyncio.gather(*workers)
            
        pbar.close()
        self.close_csv()
        if self.status_callback:
            self.status_callback(f"Finished. Found {self.total_written} locations.")
    
    async def scrape_multiple_polygons(self, polygon_list: List[Tuple[float, float]]):
        """
        Scrape multiple polygons sequentially.
        
        Args:
            polygon_list: List of polygons, each being a list of (lat, lon) tuples
        """
        all_results = []
        for i, polygon_coords in enumerate(polygon_list):
            if self._cancelled:
                break
            
            if self.status_callback:
                self.status_callback(f"Processing polygon {i+1}/{len(polygon_list)}...")
            
            await self.scrape_area(polygon_coords)
            
            # Reset state for next polygon (keep accumulated counts)
            self.tiles_processed = 0
            self.total_tiles = 0
            self.phase2_total = 0
            self.phase2_completed = 0
        
        return all_results

    @property
    def needs_images(self):
        return self.config.create_directional_views or self.config.keep_panorama

    async def scrape_area_two_phase(self, polygon_coords: List[Tuple[float, float]]):
        """
        Two-phase scraping: Discover all first (Phase 1), then download (Phase 2).
        When no images are needed, skips the expensive per-pano metadata fetch for speed.
        """
        self.found_panos.clear()
        self.total_written = 0
        self.total_images = 0
        self.tiles_processed = 0
        self.phase2_total = 0
        self.phase2_completed = 0
        self._cancelled = False
        self.first_pano_country = None

        polygon = Polygon([(lon, lat) for lat, lon in polygon_coords])
        timeout = ClientTimeout(total=30, connect=5)

        # Calculate tiles
        bounds = polygon.bounds
        min_tile_x, max_tile_y = wgs84_to_tile_coord(bounds[1], bounds[0], self.config.tile_zoom_level)
        max_tile_x, min_tile_y = wgs84_to_tile_coord(bounds[3], bounds[2], self.config.tile_zoom_level)
        tiles = [(x, y) for x in range(min_tile_x, max_tile_x + 1) for y in range(min_tile_y, max_tile_y + 1)]
        self.total_tiles = len(tiles)

        collected_panos = []

        # Use multiple session pools for better throughput
        num_pools = self.config.session_pools
        connectors = []
        sessions = []
        for _ in range(num_pools):
            conn = TCPConnector(limit=0, ttl_dns_cache=300, enable_cleanup_closed=True)
            connectors.append(conn)
            sess = ClientSession(connector=conn, timeout=timeout)
            self._attach_proxy_getter(sess)
            sessions.append(sess)

        try:
            semaphore = asyncio.Semaphore(self.config.concurrency)

            # --- PHASE 1: TILE DISCOVERY ---
            if self.status_callback:
                self.status_callback("PHASE 1: Discovering panoramas...")

            pool_idx = [0]

            async def fetch_tile(x, y):
                if self._cancelled: return []
                try:
                    sess = sessions[pool_idx[0] % num_pools]
                    pool_idx[0] += 1
                    async with semaphore:
                        panos = await get_coverage_tile_async(x, y, sess)
                    found = []
                    if panos:
                        for p in panos:
                            if p.id not in self.found_panos:
                                point = Point(p.lon, p.lat)
                                if polygon.contains(point):
                                    self.found_panos.add(p.id)
                                    found.append(p)
                                    if not self.needs_images and self.point_callback:
                                        self.point_callback(p.lat, p.lon, p.id)
                    self.tiles_processed += 1
                    if self.progress_callback and self.tiles_processed % 50 == 0:
                        self.progress_callback(self.tiles_processed, self.total_tiles)
                    return found
                except Exception:
                    self.total_errors += 1
                    self.consecutive_errors += 1
                    if self.consecutive_errors > 50:
                        await asyncio.sleep(2.0)
                    self.tiles_processed += 1
                    return []
                else:
                    if self.consecutive_errors > 0:
                        self.consecutive_errors = max(0, self.consecutive_errors - 1)

            basic_panos = []
            batch_size = self.config.concurrency
            for i in range(0, len(tiles), batch_size):
                if self._cancelled: break
                batch = tiles[i:i + batch_size]
                tasks = [asyncio.create_task(fetch_tile(x, y)) for x, y in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for res in results:
                    if isinstance(res, list): basic_panos.extend(res)

            # --- FAST CSV MODE: skip metadata fetch when no images needed ---
            if not self.needs_images:
                if self.status_callback:
                    self.status_callback(f"Writing CSV for {len(basic_panos)} panoramas (fast mode)...")
                for i, bp in enumerate(basic_panos):
                    if self._cancelled: break
                    row = [bp.id, round(bp.lat, 6), round(bp.lon, 6)]
                    if self.csv_writer:
                        self.csv_writer.writerow(row)
                        self.total_written += 1
                    if self.point_callback:
                        self.point_callback(bp.lat, bp.lon, bp.id)
                    if i % 5000 == 0 and self.status_callback:
                        self.status_callback(f"Writing CSV... {i}/{len(basic_panos)}")
                        await asyncio.sleep(0)
                self.csv_file_handle.flush()
                if not getattr(self, '_keep_csv_open', False):
                    self.close_csv()
                return []

            # --- FULL MODE: Metadata Fetch (needed for images/historical) ---
            if basic_panos and not self._cancelled:
                if self.status_callback:
                    self.status_callback(f"PHASE 1: Fetching metadata for {len(basic_panos)} panoramas...")

                async def fetch_metadata(basic_pano):
                    if self._cancelled: return []
                    try:
                        sess = sessions[pool_idx[0] % num_pools]
                        pool_idx[0] += 1
                        async with semaphore:
                            full_pano = await find_panorama_by_id_async(basic_pano.id, sess, download_depth=False)

                        pano = full_pano if full_pano else basic_pano
                        if self.point_callback:
                            self.point_callback(pano.lat, pano.lon, pano.id)

                        heading = math.degrees(pano.heading) if pano.heading else None
                        results = [{
                            'panoid': pano.id, 'heading_deg': heading,
                            'lat': pano.lat, 'lon': pano.lon,
                            'pano_object': pano
                        }]

                        if self.config.include_historical and full_pano:
                            for hist in self._select_historical_panos(full_pano):
                                if hist.id not in self.found_panos:
                                    self.found_panos.add(hist.id)
                                    if not hist.country_code and pano.country_code:
                                        hist.country_code = pano.country_code
                                    if not hist.street_names and pano.street_names:
                                        hist.street_names = pano.street_names
                                    if not hist.address and pano.address:
                                        hist.address = pano.address
                                    if self.point_callback:
                                        self.point_callback(hist.lat, hist.lon, hist.id)
                                    h_heading = math.degrees(hist.heading) if hist.heading else heading
                                    results.append({
                                        'panoid': hist.id, 'heading_deg': h_heading,
                                        'lat': hist.lat, 'lon': hist.lon,
                                        'pano_object': hist
                                    })

                        return results
                    except Exception:
                        self.total_errors += 1
                        if self.point_callback:
                            self.point_callback(basic_pano.lat, basic_pano.lon, basic_pano.id)
                        return [{
                            'panoid': basic_pano.id, 'heading_deg': None,
                            'lat': basic_pano.lat, 'lon': basic_pano.lon,
                            'pano_object': basic_pano
                        }]

                for i in range(0, len(basic_panos), batch_size):
                    if self._cancelled: break
                    batch = basic_panos[i:i + batch_size]
                    tasks = [asyncio.create_task(fetch_metadata(p)) for p in batch]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for res in results:
                        if isinstance(res, list): collected_panos.extend(res)

        finally:
            for sess in sessions:
                await sess.close()
            for conn in connectors:
                await conn.close()

        # --- WRITE CSV between phases ---
        if collected_panos and self.csv_writer and not self._cancelled:
            if self.status_callback:
                self.status_callback(f"Writing CSV for {len(collected_panos)} panoramas...")
            for pano_data in collected_panos:
                pano_obj = pano_data.get('pano_object')
                if pano_obj:
                    row = self._extract_row_data(pano_obj)
                    self.csv_writer.writerow(row)
                    self.total_written += 1
            self.csv_file_handle.flush()

        # --- PHASE 2: IMAGE DOWNLOAD ---
        if collected_panos and self.config.create_directional_views and not self._cancelled:
            if self.status_callback:
                self.status_callback(f"PHASE 2: Downloading images for {len(collected_panos)} panoramas...")

            clean_panos = [{k: v for k, v in p.items() if k != 'pano_object'} for p in collected_panos]
            await self._process_images_streaming(clean_panos)

        if not getattr(self, '_keep_csv_open', False):
            self.close_csv()
        return collected_panos

    def reset_for_new_shape(self):
        """Reset per-shape state between sequential polygon scrapes."""
        self.found_panos.clear()
        self.total_written = 0
        self.total_images = 0
        self.tiles_processed = 0
        self.total_tiles = 0
        self.phase2_total = 0
        self.phase2_completed = 0
        self.first_pano_country = None
        self.first_pano_state = None
        self.first_pano_city = None

    async def scrape_multiple_polygons_two_phase(self, polygon_list, csv_paths=None,
                                                  images_dirs=None, merge_csv=False):
        """Scrape multiple polygons sequentially, optionally merging into one CSV."""
        self._keep_csv_open = merge_csv
        all_results = []
        try:
            for i, coords in enumerate(polygon_list):
                if self._cancelled: break

                self.reset_for_new_shape()

                # Per-shape CSV and images setup (skip CSV switch when merging)
                if not merge_csv and csv_paths and i < len(csv_paths):
                    self.close_csv()
                    self.init_csv(csv_paths[i])
                if images_dirs and i < len(images_dirs):
                    self.init_images_dir(images_dirs[i])

                if self.status_callback:
                    self.status_callback(f"Shape {i+1}/{len(polygon_list)}: Discovering panoramas...")

                res = await self.scrape_area_two_phase(coords)
                all_results.extend(res)
        finally:
            self._keep_csv_open = False
            if merge_csv:
                self.close_csv()
        return all_results

    async def scrape_random_global_two_phase(self, target_count, radius):
        """Random global search in two phases."""
        # This is harder to split because we don't know locations upfront.
        # But we can do: Discovery Phase (find N panoids) -> Download Phase.
        
        self.found_panos.clear()
        self.total_written = 0
        self.total_images = 0
        self._cancelled = False
        
        collected_panos = []
        
        # --- PHASE 1: DISCOVERY ---
        if self.status_callback: self.status_callback("PHASE 1: Discovering random locations...")
        
        timeout = ClientTimeout(total=self.config.timeout_total, connect=self.config.timeout_connect)
        connector = TCPConnector(limit=0, ttl_dns_cache=300)
        async with ClientSession(connector=connector, timeout=timeout) as session:
            self._attach_proxy_getter(session)
            semaphore = asyncio.Semaphore(self.config.concurrency)
            
            async def single_search():
                if self._cancelled: return []
                try:
                    lat, lon = random.uniform(-60, 70), random.uniform(-180, 180)
                    async with semaphore:
                        pano = await find_panorama_async(lat, lon, session, radius=radius)
                    if pano and pano.id not in self.found_panos:
                        self.found_panos.add(pano.id)
                        if self.point_callback:
                            self.point_callback(pano.lat, pano.lon, pano.id)

                        heading = math.degrees(pano.heading) if pano.heading else None
                        results = [{
                            'panoid': pano.id, 'heading_deg': heading,
                            'lat': pano.lat, 'lon': pano.lon,
                            'pano_object': pano
                        }]

                        if self.config.include_historical and pano:
                            for hist in self._select_historical_panos(pano):
                                if hist.id not in self.found_panos:
                                    self.found_panos.add(hist.id)
                                    if not hist.country_code and pano.country_code:
                                        hist.country_code = pano.country_code
                                    if not hist.street_names and pano.street_names:
                                        hist.street_names = pano.street_names
                                    if not hist.address and pano.address:
                                        hist.address = pano.address
                                    if self.point_callback:
                                        self.point_callback(hist.lat, hist.lon, hist.id)
                                    h_heading = math.degrees(hist.heading) if hist.heading else heading
                                    results.append({
                                        'panoid': hist.id, 'heading_deg': h_heading,
                                        'lat': hist.lat, 'lon': hist.lon,
                                        'pano_object': hist
                                    })

                        return results
                except Exception:
                    pass
                return []

            while len(collected_panos) < target_count and not self._cancelled:
                needed = target_count - len(collected_panos)
                batch_size = min(self.config.concurrency, needed + 50) # buffer
                tasks = [asyncio.create_task(single_search()) for _ in range(batch_size)]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for res in results:
                    if res and isinstance(res, list):
                        collected_panos.extend(res)
                        if len(collected_panos) >= target_count: break

                if self.status_callback:
                     self.status_callback(f"Phase 1: Found {len(collected_panos)}/{target_count} panoramas")

        # --- WRITE CSV between phases ---
        if collected_panos and self.csv_writer and not self._cancelled:
            if self.status_callback:
                self.status_callback(f"Writing CSV for {len(collected_panos)} panoramas...")
            for pano_data in collected_panos:
                pano_obj = pano_data.get('pano_object')
                if pano_obj:
                    row = self._extract_row_data(pano_obj)
                    self.csv_writer.writerow(row)
                    self.total_written += 1
            self.csv_file_handle.flush()

        # --- PHASE 2: DOWNLOAD ---
        if collected_panos and self.config.create_directional_views and not self._cancelled:
            if self.status_callback:
                self.status_callback(f"PHASE 2: Downloading images for {len(collected_panos)} panoramas...")

            clean_panos = [{k: v for k, v in p.items() if k != 'pano_object'} for p in collected_panos]
            await self._process_images_streaming(clean_panos)

        self.close_csv()
        return collected_panos
