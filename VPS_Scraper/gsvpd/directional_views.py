"""
Directional view extraction from equirectangular panoramas.

Converts equirectangular (360°) panoramas into multiple rectilinear perspective views
at different directions (e.g., front, back, left, right, up, down).
"""
import numpy as np
import cv2
import random
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class DirectionalViewConfig:
    """Configuration for directional view extraction."""
    output_resolution: int = 512
    fov_degrees: float = 90.0
    num_views: int = 6
    global_view: bool = False  # Extract only one random view
    augment: bool = False      # Apply random FOV and geometric tilts
    target_yaw: float = None   # Specific yaw to extract (overrides random/multi)


@dataclass
class DirectionalViewResult:
    """Result of directional view extraction."""
    views: List[np.ndarray]
    directions: List[float]    # Yaw angles
    metadata: List[dict]       # Extra info (pitch, roll, fov)
    success: bool = False
    error: str = ""


class DirectionalViewExtractor:
    """Extract directional views from equirectangular panoramas."""
    
    def __init__(self):
        pass
    
    def extract_views(
        self,
        panorama: np.ndarray,
        config: DirectionalViewConfig
    ) -> DirectionalViewResult:
        """
        Extract directional views from equirectangular panorama.
        
        Args:
            panorama: Input equirectangular panorama image (numpy array)
            config: Configuration for view extraction
            
        Returns:
            DirectionalViewResult containing extracted views and metadata
        """
        result = DirectionalViewResult(views=[], directions=[], metadata=[])
        
        if panorama is None or panorama.size == 0:
            result.error = "Input panorama is empty"
            return result
        
        try:
            views_to_generate = []
            
            if config.target_yaw is not None and (config.global_view or config.num_views == 1):
                # SPECIFIC YAW MODE: Extract one view at specific yaw
                # This overrides both global random and standard multi-view
                yaw = config.target_yaw
                
                if config.augment:
                    # Augment: Random FOV (70-100), Random Pitch/Roll (+/- 5)
                    fov = random.uniform(60.0, 100.0)
                    pitch = random.uniform(-5.0, 5.0)
                    roll = random.uniform(-5.0, 5.0)
                else:
                    # Standard: Fixed FOV, level horizon
                    fov = config.fov_degrees
                    pitch = 0.0
                    roll = 0.0
                    
                views_to_generate.append({
                    'yaw': yaw,
                    'pitch': pitch,
                    'roll': roll,
                    'fov': fov
                })

            elif config.global_view:
                # GLOBAL MODE: One single random view
                yaw = random.uniform(0, 360)
                
                if config.augment:
                    # Augment: Random FOV (70-100), Random Pitch/Roll (+/- 5)
                    fov = random.uniform(60.0, 100.0)
                    pitch = random.uniform(-5.0, 5.0)
                    roll = random.uniform(-5.0, 5.0)
                else:
                    # Standard Global: Fixed FOV, level horizon
                    fov = config.fov_degrees
                    pitch = 0.0
                    roll = 0.0
                    
                views_to_generate.append({
                    'yaw': yaw,
                    'pitch': pitch,
                    'roll': roll,
                    'fov': fov
                })
                
            else:
                # STANDARD MODE: Fixed number of views around the horizon
                angle_step = 360.0 / config.num_views
                for i in range(config.num_views):
                    views_to_generate.append({
                        'yaw': i * angle_step,
                        'pitch': 0.0,
                        'roll': 0.0,
                        'fov': config.fov_degrees
                    })

            # Process all planned views
            for params in views_to_generate:
                yaw_rad = np.radians(params['yaw'])
                pitch_rad = np.radians(params['pitch'])
                roll_rad = np.radians(params['roll'])
                fov_rad = np.radians(params['fov'])
                
                view = self._extract_single_view(
                    panorama,
                    yaw_rad,
                    pitch_rad,
                    roll_rad,
                    fov_rad,
                    config.output_resolution
                )
                
                if view is not None and view.size > 0:
                    result.views.append(view)
                    result.directions.append(params['yaw'])
                    result.metadata.append(params)
            
            result.success = len(result.views) > 0
            if not result.success:
                result.error = "Failed to extract any views"
                
        except Exception as e:
            result.success = False
            result.error = f"Extraction failed: {str(e)}"
        
        return result
    
    def _extract_single_view(
        self,
        panorama: np.ndarray,
        yaw_rad: float,
        pitch_rad: float,
        roll_rad: float,
        fov_radians: float,
        output_size: int
    ) -> np.ndarray:
        """
        Extract a single directional view from panorama with 3D rotation.
        """
        pano_height, pano_width = panorama.shape[:2]
        
        # Create remap matrices with 3D rotation support
        map_x, map_y = self._create_remap_matrices(
            pano_width,
            pano_height,
            output_size,
            yaw_rad,
            pitch_rad,
            roll_rad,
            fov_radians
        )
        
        # Apply remapping
        output = cv2.remap(
            panorama,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP
        )
        
        return output
    
    def _create_remap_matrices(
        self,
        pano_width: int,
        pano_height: int,
        output_size: int,
        yaw: float,
        pitch: float,
        roll: float,
        fov_radians: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create remap matrices for equirectangular to rectilinear projection.
        Supports Yaw, Pitch, and Roll rotations.
        """
        # Horizontal FOV (assuming aspect ratio 1:1)
        hfov_rad = 2.0 * np.arctan(np.tan(fov_radians / 2.0))
        
        # Create pixel coordinate grids (vectorized)
        x_coords, y_coords = np.meshgrid(
            np.arange(output_size, dtype=np.float32),
            np.arange(output_size, dtype=np.float32)
        )
        
        # Convert output pixels to normalized coordinates [-1, 1]
        nx = (2.0 * x_coords / output_size) - 1.0
        ny = 1.0 - (2.0 * y_coords / output_size)
        
        # Calculate 3D ray directions (Camera space)
        # Z is forward
        tan_hfov_half = np.tan(hfov_rad / 2.0)
        tan_fov_half = np.tan(fov_radians / 2.0)
        
        ray_x = tan_hfov_half * nx
        ray_y = tan_fov_half * ny
        ray_z = np.ones_like(ray_x)
        
        # Normalize rays
        ray_len = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2)
        ray_x /= ray_len
        ray_y /= ray_len
        ray_z /= ray_len
        
        # --- 3D Rotation Matrices Application ---
        # We rotate the rays from Camera Space to World Space
        
        # 1. Roll (Rotation around Z axis)
        if roll != 0:
            c, s = np.cos(roll), np.sin(roll)
            rx = ray_x * c - ray_y * s
            ry = ray_x * s + ray_y * c
            ray_x, ray_y = rx, ry

        # 2. Pitch (Rotation around X axis)
        if pitch != 0:
            c, s = np.cos(pitch), np.sin(pitch)
            ry = ray_y * c - ray_z * s
            rz = ray_y * s + ray_z * c
            ray_y, ray_z = ry, rz
            
        # 3. Yaw (Rotation around Y axis)
        # Standard compass: 0 is North (Z+), 90 East (X+)
        c, s = np.cos(yaw), np.sin(yaw)
        rx = ray_x * c + ray_z * s
        rz = -ray_x * s + ray_z * c
        ray_x, ray_z = rx, rz
        
        # --- Convert to Spherical Coordinates ---
        # theta = longitude, phi = latitude
        theta = np.arctan2(ray_x, ray_z)
        phi = np.arcsin(np.clip(ray_y, -1.0, 1.0))
        
        # Convert to panorama pixel coordinates
        # U = (theta / 2pi) + 0.5
        # V = 0.5 - (phi / pi)
        map_x = (theta / (2.0 * np.pi) + 0.5) * pano_width
        map_y = (0.5 - phi / np.pi) * pano_height
        
        return map_x.astype(np.float32), map_y.astype(np.float32)
    
    @staticmethod
    def save_views(
        result: DirectionalViewResult,
        output_dir: str,
        panoid: str,
        zoom_level: int,
        prefix: str = "view"
    ) -> bool:
        """
        Save directional views to disk.
        """
        if not result.success or not result.views:
            return False
        
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            for i, (view, meta) in enumerate(zip(result.views, result.metadata)):
                yaw = meta['yaw']
                # If global/augment mode, filename might include specific details
                if 'pitch' in meta and meta['pitch'] != 0:
                     filename = f"{panoid}_zoom{zoom_level}_{prefix}_rnd_Y{int(yaw)}_P{int(meta['pitch'])}.jpg"
                else:
                    filename = f"{panoid}_zoom{zoom_level}_{prefix}{i:02d}_{yaw:.0f}deg.jpg"
                    
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, view, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            return True
            
        except Exception as e:
            print(f"Error saving views: {e}")
            return False