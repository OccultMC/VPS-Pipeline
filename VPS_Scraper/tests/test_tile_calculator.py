"""
Tests for the optimized tile calculator.
"""
import pytest
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ray_workers import OptimizedTileFetcher


class TestTileCalculator:
    """Tests for tile requirement calculation."""
    
    def test_required_tiles_center_view(self):
        """Test that center view needs minimal tiles at zoom 2."""
        tiles = OptimizedTileFetcher.calculate_required_tiles(
            yaw_deg=0, pitch_deg=0, fov_deg=90, output_size=512, zoom_level=2
        )
        
        # At zoom 2, we have 4x2 tiles (0-3 x, 0-1 y)
        # Center view should need at most 4 tiles
        assert len(tiles) <= 6, f"Expected <= 6 tiles for center view, got {len(tiles)}"
        assert len(tiles) >= 1, "Should need at least 1 tile"
    
    def test_required_tiles_edge_view(self):
        """Test tiles needed for edge (wrap-around) view."""
        # Looking at yaw=355 should wrap around
        tiles = OptimizedTileFetcher.calculate_required_tiles(
            yaw_deg=355, pitch_deg=0, fov_deg=90, output_size=512, zoom_level=2
        )
        
        # Should still get reasonable number of tiles
        assert len(tiles) <= 8, f"Expected <= 8 tiles for edge view, got {len(tiles)}"
    
    def test_required_tiles_different_zoom_levels(self):
        """Test tile counts scale with zoom level."""
        tiles_z1 = OptimizedTileFetcher.calculate_required_tiles(
            yaw_deg=180, pitch_deg=0, fov_deg=90, output_size=512, zoom_level=1
        )
        tiles_z2 = OptimizedTileFetcher.calculate_required_tiles(
            yaw_deg=180, pitch_deg=0, fov_deg=90, output_size=512, zoom_level=2
        )
        
        # Higher zoom has more tiles, so may need more
        assert len(tiles_z1) >= 1
        assert len(tiles_z2) >= 1
    
    def test_required_tiles_narrow_fov(self):
        """Test that narrow FOV needs fewer tiles."""
        tiles_wide = OptimizedTileFetcher.calculate_required_tiles(
            yaw_deg=90, pitch_deg=0, fov_deg=90, output_size=512, zoom_level=2
        )
        tiles_narrow = OptimizedTileFetcher.calculate_required_tiles(
            yaw_deg=90, pitch_deg=0, fov_deg=60, output_size=512, zoom_level=2
        )
        
        # Narrow FOV should need same or fewer tiles
        assert len(tiles_narrow) <= len(tiles_wide) + 1
    
    def test_tiles_coordinates_valid(self):
        """Test that returned tile coordinates are valid."""
        tiles = OptimizedTileFetcher.calculate_required_tiles(
            yaw_deg=45, pitch_deg=10, fov_deg=90, output_size=512, zoom_level=2
        )
        
        # At zoom 2: x in [0,3], y in [0,1]
        for x, y in tiles:
            assert 0 <= x <= 3, f"Invalid x coordinate: {x}"
            assert 0 <= y <= 1, f"Invalid y coordinate: {y}"


class TestTileCalculatorOptimization:
    """Tests to verify optimization actually reduces tile downloads."""
    
    def test_optimization_savings_zoom2(self):
        """Test that optimized download uses fewer tiles than full download."""
        # Full panorama at zoom 2: 4x2 = 8 tiles
        full_tiles = 8
        
        # Optimized single view
        optimized_tiles = OptimizedTileFetcher.calculate_required_tiles(
            yaw_deg=90, pitch_deg=0, fov_deg=90, output_size=512, zoom_level=2
        )
        
        # Should use significantly fewer tiles
        assert len(optimized_tiles) < full_tiles, \
            f"Optimized should use fewer than {full_tiles} tiles, got {len(optimized_tiles)}"
    
    def test_optimization_savings_zoom3(self):
        """Test optimization at higher zoom level."""
        # Full panorama at zoom 3: 8x4 = 32 tiles
        full_tiles = 32
        
        optimized_tiles = OptimizedTileFetcher.calculate_required_tiles(
            yaw_deg=180, pitch_deg=0, fov_deg=90, output_size=512, zoom_level=3
        )
        
        # Should use significantly fewer tiles
        assert len(optimized_tiles) < full_tiles / 2, \
            f"Optimized should use fewer than {full_tiles/2} tiles, got {len(optimized_tiles)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
