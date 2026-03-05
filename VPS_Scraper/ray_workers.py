"""
Ray Workers for parallel tile stitching and perspective extraction.

Uses Ray actors to distribute CPU-intensive image processing across multiple workers.
"""
import ray
import asyncio
import aiohttp
from aiohttp import ClientTimeout
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional, Dict, Any, Union
from io import BytesIO
import numpy as np
import cv2
from PIL import Image

# Handle imports for both package and standalone usage
try:
    from .gsvpd.constants import ZOOM_SIZES, OLD_ZOOM_SIZES, TILE_COUNT_TO_SIZE, TILES_AXIS_COUNT, TILE_SIZE
    from .gsvpd.directional_views import DirectionalViewExtractor, DirectionalViewConfig
    from .gsvpd.augmentations import apply_pixel_augmentations
except ImportError:
    from gsvpd.constants import ZOOM_SIZES, OLD_ZOOM_SIZES, TILE_COUNT_TO_SIZE, TILES_AXIS_COUNT, TILE_SIZE
    from gsvpd.directional_views import DirectionalViewExtractor, DirectionalViewConfig
    from gsvpd.augmentations import apply_pixel_augmentations


def _check_black_bottom(tile_data: bytes) -> bool:
    """Check if tile has black bottom (executed in subprocess)."""
    tile = Image.open(BytesIO(tile_data))
    arr = np.array(tile)
    bottom = arr[-5:]
    return bool(np.all(bottom <= 10))


def _check_black_percentage(tile_data: bytes) -> float:
    """Calculate black percentage (executed in subprocess)."""
    tile = Image.open(BytesIO(tile_data))
    img_np = np.array(tile)
    black_pixels = np.all(img_np <= 10, axis=2)
    return float(np.sum(black_pixels) / black_pixels.size * 100)


def _stitch_and_process(
    tile_data_list: List[Tuple[int, int, bytes]],
    width: int,
    height: int,
    config: dict,
    panoid: str,
    zoom_level: int,
    heading_deg: float = None
) -> dict:
    """
    Stitch tiles and extract perspective views.
    This is the CPU-intensive work that Ray distributes.
    """
    result = {
        "success": False,
        "error": "",
        "size": (width, height),
        "tiles": (0, 0),
        "views": [],
        "view_filenames": [],
        "panorama_bytes": None
    }
    
    try:
        # Stitch tiles
        full_img = Image.new("RGB", (width, height))
        x_values = set()
        y_values = set()
        
        for x, y, tile_bytes in tile_data_list:
            tile = Image.open(BytesIO(tile_bytes))
            full_img.paste(tile, (x * 512, y * 512))
            tile.close()
            x_values.add(x)
            y_values.add(y)
        
        result["tiles"] = (len(x_values), len(y_values))
        
        # Convert to OpenCV format
        full_img_cv = cv2.cvtColor(np.array(full_img), cv2.COLOR_RGB2BGR)
        
        # Extract directional views if enabled
        if config.get("create_directional_views"):
            view_extractor = DirectionalViewExtractor()
            
            # Configure view extraction
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
                    # Apply augmentations if enabled
                    if config.get("augment"):
                        view = apply_pixel_augmentations(view)

                    # Determine filename
                    yaw = meta['yaw']
                    if config.get("global_view") or heading_deg is not None:
                        fname = f"{panoid}_rnd_Y{int(yaw)}.jpg"
                        if config.get("augment"):
                            fname = f"{panoid}_aug_Y{int(yaw)}_P{int(meta['pitch'])}.jpg"
                    else:
                        fname = f"{panoid}_zoom{zoom_level}_view{i:02d}_{yaw:.0f}deg.jpg"
                    
                    # Encode to JPEG
                    _, buffer = cv2.imencode('.jpg', view, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    result["views"].append(buffer.tobytes())
                    result["view_filenames"].append(fname)
                    
        # Encode panorama if requested
        if config.get("keep_panorama"):
            _, buffer = cv2.imencode('.jpg', full_img_cv, [cv2.IMWRITE_JPEG_QUALITY, 95])
            result["panorama_bytes"] = buffer.tobytes()
        
        full_img.close()
        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


@ray.remote
class TileStitchingActor:
    """
    Ray actor for downloading tiles, stitching panoramas, and extracting perspective views.
    
    Each actor maintains its own aiohttp session for efficient connection pooling.
    """
    
    def __init__(self, config: dict, actor_id: int = 0):
        self.config = config
        self.actor_id = actor_id
        self.session = None
        self.processed_count = 0
        
    async def _ensure_session(self):
        """Create aiohttp session if not exists."""
        if self.session is None:
            connector = aiohttp.TCPConnector(limit=50, limit_per_host=20)
            timeout = ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    
    async def _close_session(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def fetch_tile(
        self, 
        panoid: str,
        x: int, 
        y: int, 
        zoom_level: int, 
        retries: int = 3
    ) -> Optional[Tuple[int, int, bytes]]:
        """Fetch a single panorama tile with retry support."""
        await self._ensure_session()
        
        url = f"https://cbk0.google.com/cbk?output=tile&panoid={panoid}&zoom={zoom_level}&x={x}&y={y}"
        
        for attempt in range(1, retries + 1):
            try:
                async with self.session.get(url) as response:
                    if response.status != 200:
                        return None
                    
                    BLACK_TILE_BYTE_SIZE = 1184
                    content_length = int(response.headers.get("Content-Length", 0))
                    
                    if content_length == BLACK_TILE_BYTE_SIZE:
                        return None
                    
                    data = await response.read()
                    return (x, y, data)
                    
            except Exception:
                if attempt < retries:
                    await asyncio.sleep(0.2 * (2 ** (attempt - 1)))
        
        return None
    
    async def determine_dimensions(
        self,
        tiles: List[Tuple[int, int, bytes]], 
        zoom_level: int, 
        x_tiles_count: int, 
        y_tiles_count: int
    ) -> Tuple[int, int]:
        """Determine panorama dimensions based on tile analysis."""
        if zoom_level == 0:
            black_perc = _check_black_percentage(tiles[0][2])
            return OLD_ZOOM_SIZES[zoom_level] if black_perc > 55 else ZOOM_SIZES[zoom_level]
        elif 0 < zoom_level <= 2:
            black = _check_black_bottom(tiles[1][2]) if len(tiles) > 1 else False
            return OLD_ZOOM_SIZES[zoom_level] if black else ZOOM_SIZES[zoom_level]
        return TILE_COUNT_TO_SIZE.get((x_tiles_count, y_tiles_count), ZOOM_SIZES.get(zoom_level, (2048, 1024)))
    
    async def process_panoid(self, panoid_data: Union[str, dict]) -> dict:
        """
        Download, stitch, and extract perspectives for a single panorama.
        
        Args:
            panoid_data: Either a panoid string or dict with 'panoid' and optional 'heading_deg'
            
        Returns:
            Dict with success, views, panorama_bytes, error, etc.
        """
        # Extract panoid and heading
        if isinstance(panoid_data, dict):
            panoid = panoid_data.get('panoid')
            heading_deg = panoid_data.get('heading_deg')
        else:
            panoid = panoid_data
            heading_deg = None
        
        result = {
            "panoid": panoid,
            "success": False,
            "error": "",
            "views": [],
            "view_filenames": [],
            "panorama_bytes": None,
            "actor_id": self.actor_id
        }
        
        try:
            await self._ensure_session()
            
            zoom_level = self.config.get("zoom_level", 2)
            tiles_x, tiles_y = TILES_AXIS_COUNT[zoom_level]
            
            # Fetch all tiles concurrently
            tasks = [
                self.fetch_tile(panoid, x, y, zoom_level)
                for x in range(tiles_x + 1)
                for y in range(tiles_y + 1)
            ]
            tiles = [t for t in await asyncio.gather(*tasks) if t is not None]
            
            if not tiles:
                result["error"] = "No tiles fetched"
                return result
            
            # Count tiles
            x_values = {x for x, _, _ in tiles}
            y_values = {y for _, y, _ in tiles}
            x_tiles_count, y_tiles_count = len(x_values), len(y_values)
            
            # Determine dimensions
            w, h = await self.determine_dimensions(tiles, zoom_level, x_tiles_count, y_tiles_count)
            
            # Stitch and process (CPU-intensive)
            process_result = _stitch_and_process(
                tiles, w, h, self.config, panoid, zoom_level, heading_deg
            )
            
            if process_result["success"]:
                result["success"] = True
                result["views"] = process_result["views"]
                result["view_filenames"] = process_result["view_filenames"]
                result["panorama_bytes"] = process_result["panorama_bytes"]
                result["size"] = process_result["size"]
                result["tiles"] = process_result["tiles"]
            else:
                result["error"] = process_result["error"]
            
            self.processed_count += 1
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    async def process_batch(self, panoid_list: List[Union[str, dict]]) -> List[dict]:
        """Process multiple panoramas sequentially within this actor."""
        results = []
        for panoid_data in panoid_list:
            result = await self.process_panoid(panoid_data)
            results.append(result)
        return results
    
    def get_stats(self) -> dict:
        """Get actor statistics."""
        return {
            "actor_id": self.actor_id,
            "processed_count": self.processed_count
        }
    
    async def shutdown(self):
        """Clean shutdown of the actor."""
        await self._close_session()


@ray.remote
class OptimizedTileFetcher:
    """
    Optimized tile fetcher for global single-image mode.
    
    Only downloads tiles needed for a specific perspective view,
    then composites them onto a black 2:1 canvas.
    """
    
    def __init__(self, config: dict, actor_id: int = 0):
        self.config = config
        self.actor_id = actor_id
        self.session = None
        self.processed_count = 0
    
    async def _ensure_session(self):
        if self.session is None:
            connector = aiohttp.TCPConnector(limit=50, limit_per_host=20)
            timeout = ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    
    @staticmethod
    def calculate_required_tiles(
        yaw_deg: float,
        pitch_deg: float,
        fov_deg: float,
        output_size: int,
        zoom_level: int
    ) -> List[Tuple[int, int]]:
        """
        Calculate which tiles are needed to render a perspective view.
        
        Args:
            yaw_deg: Horizontal look direction (0-360)
            pitch_deg: Vertical look direction (-90 to 90)
            fov_deg: Field of view in degrees
            output_size: Output image size in pixels
            zoom_level: Panorama zoom level
            
        Returns:
            List of (x, y) tile coordinates needed
        """
        # Get panorama dimensions for this zoom level
        if zoom_level in ZOOM_SIZES:
            pano_width, pano_height = ZOOM_SIZES[zoom_level]
        else:
            pano_width, pano_height = TILE_COUNT_TO_SIZE.get(
                (TILES_AXIS_COUNT[zoom_level][0] + 1, TILES_AXIS_COUNT[zoom_level][1] + 1),
                (2048, 1024)
            )
        
        tiles_x, tiles_y = TILES_AXIS_COUNT[zoom_level]
        tiles_x += 1
        tiles_y += 1
        
        # Convert yaw/pitch to normalized panorama coordinates
        # Yaw 0 = center, 180 = left edge, -180/360 = right edge
        center_u = (yaw_deg / 360.0) % 1.0
        center_v = 0.5 - (pitch_deg / 180.0)
        
        # Calculate FOV coverage in UV space
        # Horizontal FOV spans some fraction of the panorama width
        fov_u = (fov_deg / 360.0) * 1.5  # Add margin
        fov_v = (fov_deg / 180.0) * 1.5  # Add margin
        
        # Calculate UV bounds
        u_min = center_u - fov_u / 2
        u_max = center_u + fov_u / 2
        v_min = max(0, center_v - fov_v / 2)
        v_max = min(1, center_v + fov_v / 2)
        
        # Convert to tile coordinates
        required_tiles = set()
        
        # Handle wrapping around the panorama
        for u in [u_min, u_max, center_u]:
            u_normalized = u % 1.0
            if u_normalized < 0:
                u_normalized += 1.0
            
            tile_x = int(u_normalized * tiles_x) % tiles_x
            
            for v in [v_min, v_max, center_v]:
                tile_y = int(v * tiles_y)
                tile_y = max(0, min(tiles_y - 1, tile_y))
                required_tiles.add((tile_x, tile_y))
        
        # Fill in the rectangle of tiles
        tile_x_min = int((u_min % 1.0) * tiles_x) % tiles_x
        tile_x_max = int((u_max % 1.0) * tiles_x) % tiles_x
        tile_y_min = int(v_min * tiles_y)
        tile_y_max = int(v_max * tiles_y)
        
        tile_y_min = max(0, tile_y_min)
        tile_y_max = min(tiles_y - 1, tile_y_max)
        
        # Handle wraparound for x tiles
        if tile_x_max >= tile_x_min:
            for x in range(tile_x_min, tile_x_max + 1):
                for y in range(tile_y_min, tile_y_max + 1):
                    required_tiles.add((x % tiles_x, y))
        else:
            # Wraps around
            for x in list(range(tile_x_min, tiles_x)) + list(range(0, tile_x_max + 1)):
                for y in range(tile_y_min, tile_y_max + 1):
                    required_tiles.add((x, y))
        
        return list(required_tiles)
    
    async def fetch_tile(
        self, 
        panoid: str,
        x: int, 
        y: int, 
        zoom_level: int
    ) -> Optional[Tuple[int, int, bytes]]:
        """Fetch a single tile."""
        await self._ensure_session()
        
        url = f"https://cbk0.google.com/cbk?output=tile&panoid={panoid}&zoom={zoom_level}&x={x}&y={y}"
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None
                
                BLACK_TILE_BYTE_SIZE = 1184
                content_length = int(response.headers.get("Content-Length", 0))
                
                if content_length == BLACK_TILE_BYTE_SIZE:
                    return None
                
                data = await response.read()
                return (x, y, data)
                
        except Exception:
            return None
    
    async def process_panoid_optimized(self, panoid_data: Union[str, dict]) -> dict:
        """
        Process a panorama using optimized partial tile download.
        
        Only downloads tiles needed for the perspective view, then composites
        them onto a black 2:1 canvas before extraction.
        """
        import random
        
        # Extract panoid and heading
        if isinstance(panoid_data, dict):
            panoid = panoid_data.get('panoid')
            heading_deg = panoid_data.get('heading_deg')
        else:
            panoid = panoid_data
            heading_deg = None
        
        result = {
            "panoid": panoid,
            "success": False,
            "error": "",
            "views": [],
            "view_filenames": [],
            "panorama_bytes": None,
            "tiles_downloaded": 0,
            "tiles_total_possible": 0
        }
        
        try:
            await self._ensure_session()
            
            zoom_level = self.config.get("zoom_level", 2)
            view_resolution = self.config.get("view_resolution", 512)
            fov_deg = self.config.get("view_fov", 90.0)
            
            # Calculate random yaw if no heading provided
            if heading_deg is not None:
                yaw_deg = heading_deg
            else:
                yaw_deg = random.uniform(0, 360)
            
            pitch_deg = 0  # Keep level
            if self.config.get("augment"):
                pitch_deg = random.uniform(-5, 5)
            
            # Calculate required tiles
            required_tiles = self.calculate_required_tiles(
                yaw_deg, pitch_deg, fov_deg, view_resolution, zoom_level
            )
            
            tiles_x_count, tiles_y_count = TILES_AXIS_COUNT[zoom_level]
            result["tiles_total_possible"] = (tiles_x_count + 1) * (tiles_y_count + 1)
            result["tiles_downloaded"] = len(required_tiles)
            
            # Fetch only required tiles
            tasks = [
                self.fetch_tile(panoid, x, y, zoom_level)
                for x, y in required_tiles
            ]
            tiles = [t for t in await asyncio.gather(*tasks) if t is not None]
            
            if not tiles:
                result["error"] = "No tiles fetched"
                return result
            
            # Get full panorama dimensions
            if zoom_level in ZOOM_SIZES:
                pano_width, pano_height = ZOOM_SIZES[zoom_level]
            else:
                pano_width, pano_height = (2048, 1024)
            
            # Create black canvas and composite tiles
            full_img = Image.new("RGB", (pano_width, pano_height), (0, 0, 0))
            
            for x, y, tile_bytes in tiles:
                tile = Image.open(BytesIO(tile_bytes))
                full_img.paste(tile, (x * TILE_SIZE, y * TILE_SIZE))
                tile.close()
            
            # Convert to OpenCV and extract perspective
            full_img_cv = cv2.cvtColor(np.array(full_img), cv2.COLOR_RGB2BGR)
            
            view_extractor = DirectionalViewExtractor()
            view_config = DirectionalViewConfig(
                output_resolution=view_resolution,
                fov_degrees=fov_deg,
                num_views=1,
                global_view=True,
                augment=self.config.get("augment", False),
                target_yaw=yaw_deg
            )
            
            view_result = view_extractor.extract_views(full_img_cv, view_config)
            
            if view_result.success and view_result.views:
                view = view_result.views[0]
                meta = view_result.metadata[0]
                
                if self.config.get("augment"):
                    view = apply_pixel_augmentations(view)
                
                yaw = meta['yaw']
                fname = f"{panoid}_rnd_Y{int(yaw)}.jpg"
                if self.config.get("augment"):
                    fname = f"{panoid}_aug_Y{int(yaw)}_P{int(meta.get('pitch', 0))}.jpg"
                
                _, buffer = cv2.imencode('.jpg', view, [cv2.IMWRITE_JPEG_QUALITY, 95])
                result["views"].append(buffer.tobytes())
                result["view_filenames"].append(fname)
                result["success"] = True
            else:
                result["error"] = "Failed to extract view"
            
            full_img.close()
            self.processed_count += 1
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    async def shutdown(self):
        """Clean shutdown."""
        if self.session:
            await self.session.close()
            self.session = None


class RayWorkerPool:
    """
    Manages a pool of Ray actors for parallel panorama processing.
    """
    
    def __init__(self, config: dict, num_workers: int = 8, use_optimized: bool = False):
        """
        Initialize the worker pool.
        
        Args:
            config: Processing configuration
            num_workers: Number of parallel Ray actors
            use_optimized: Use optimized partial tile fetcher for global single-image mode
        """
        self.config = config
        self.num_workers = num_workers
        self.use_optimized = use_optimized
        self.actors = []
        self.initialized = False
    
    def initialize(self):
        """Initialize Ray and create actors."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        ActorClass = OptimizedTileFetcher if self.use_optimized else TileStitchingActor
        
        self.actors = [
            ActorClass.remote(self.config, i)
            for i in range(self.num_workers)
        ]
        self.initialized = True
        print(f"[Ray] Initialized {self.num_workers} {'optimized' if self.use_optimized else 'standard'} workers")
    
    async def process_all(
        self, 
        panoid_list: List[Union[str, dict]],
        progress_callback=None
    ) -> List[dict]:
        """
        Process all panoramas using the worker pool.
        
        Distributes work across actors using round-robin assignment.
        """
        if not self.initialized:
            self.initialize()
        
        # Distribute panoramas to actors
        actor_tasks = [[] for _ in range(self.num_workers)]
        for i, panoid in enumerate(panoid_list):
            actor_tasks[i % self.num_workers].append(panoid)
        
        # Submit individual tasks to actors (round-robin)
        futures = []
        for actor, tasks in zip(self.actors, actor_tasks):
            if tasks:
                for task in tasks:
                    if self.use_optimized:
                        futures.append(actor.process_panoid_optimized.remote(task))
                    else:
                        futures.append(actor.process_panoid.remote(task))

        # Collect results as they complete
        all_results = []
        completed = 0

        for future in futures:
            result = await asyncio.wrap_future(future.future())
            all_results.append(result)
            completed += 1
            if progress_callback:
                progress_callback(completed, len(panoid_list))

        return all_results
    
    def shutdown(self):
        """Shutdown all actors and Ray."""
        for actor in self.actors:
            ray.get(actor.shutdown.remote())
        self.actors = []
        self.initialized = False
