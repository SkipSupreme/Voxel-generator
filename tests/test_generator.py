"""
Unit tests for the Voxel Generator.
"""

import sys
from pathlib import Path
import numpy as np
import unittest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voxel_generator import VoxelGenerator
from voxel_generator.voxelizer import VoxelGrid
from voxel_generator.depth import DepthEstimator, DepthMode
from voxel_generator.greedy_mesh import GreedyMesher, NaiveMesher
from voxel_generator.color import srgb_to_linear, linear_to_srgb
from voxel_generator.projection import ProjectionMatrix, IsometricProjection


class TestVoxelGrid(unittest.TestCase):
    """Tests for VoxelGrid class."""

    def test_create_grid(self):
        """Test grid creation."""
        grid = VoxelGrid(16, 16, 16)
        assert grid.shape == (16, 16, 16)
        assert grid.count_voxels() == 0

    def test_set_get_voxel(self):
        """Test setting and getting voxels."""
        grid = VoxelGrid(8, 8, 8)
        grid.set_voxel(1, 2, 3, 255, 128, 64, 255)

        voxel = grid.get_voxel(1, 2, 3)
        assert voxel is not None
        assert list(voxel) == [255, 128, 64, 255]

    def test_out_of_bounds(self):
        """Test out-of-bounds access."""
        grid = VoxelGrid(8, 8, 8)
        grid.set_voxel(100, 0, 0, 255, 0, 0, 255)  # Should not crash
        assert grid.get_voxel(100, 0, 0) is None

    def test_occupancy(self):
        """Test occupancy mask."""
        grid = VoxelGrid(4, 4, 4)
        grid.set_voxel(1, 1, 1, 255, 0, 0, 255)
        grid.set_voxel(2, 2, 2, 0, 255, 0, 255)

        assert grid.count_voxels() == 2
        assert grid.is_solid(1, 1, 1)
        assert grid.is_solid(2, 2, 2)
        assert not grid.is_solid(0, 0, 0)

    def test_sparse_conversion(self):
        """Test sparse representation."""
        grid = VoxelGrid(8, 8, 8)
        grid.set_voxel(0, 0, 0, 255, 0, 0, 255)
        grid.set_voxel(7, 7, 7, 0, 0, 255, 255)

        coords, colors = grid.to_sparse()
        assert len(coords) == 2
        assert len(colors) == 2


class TestDepthEstimator(unittest.TestCase):
    """Tests for depth estimation."""

    def test_flat_depth(self):
        """Test flat depth mode."""
        mask = np.ones((8, 8), dtype=bool)
        estimator = DepthEstimator(DepthMode.FLAT, max_depth=16)
        depth = estimator.estimate(mask)

        assert depth.shape == (8, 8)
        assert np.all(depth[mask] == 8)  # Half of max_depth

    def test_distance_transform(self):
        """Test EDT depth mode."""
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:12, 4:12] = True

        estimator = DepthEstimator(DepthMode.DISTANCE_TRANSFORM, max_depth=16)
        depth = estimator.estimate(mask)

        # Center should be deepest
        center_depth = depth[8, 8]
        edge_depth = depth[4, 4]
        assert center_depth > edge_depth

    def test_explicit_depth(self):
        """Test explicit depth map mode."""
        mask = np.ones((8, 8), dtype=bool)
        depth_map = np.full((8, 8), 128, dtype=np.uint8)

        estimator = DepthEstimator(DepthMode.EXPLICIT, max_depth=16)
        estimator.set_explicit_depth(depth_map)
        depth = estimator.estimate(mask)

        expected = int(128 / 255 * 16)
        assert np.allclose(depth[mask], expected, atol=1)


class TestGreedyMesher(unittest.TestCase):
    """Tests for greedy meshing."""

    def test_empty_grid(self):
        """Test meshing empty grid."""
        grid = VoxelGrid(4, 4, 4)
        mesher = GreedyMesher()
        mesh = mesher.mesh(grid.data)

        assert len(mesh.vertices) == 0
        assert len(mesh.indices) == 0

    def test_single_voxel(self):
        """Test meshing single voxel."""
        grid = VoxelGrid(4, 4, 4)
        grid.set_voxel(1, 1, 1, 255, 0, 0, 255)

        mesher = GreedyMesher()
        mesh = mesher.mesh(grid.data)

        # Single voxel = 6 faces = 24 vertices = 12 triangles
        assert len(mesh.vertices) == 24
        assert len(mesh.indices) == 36  # 12 triangles * 3

    def test_greedy_reduction(self):
        """Test that greedy meshing reduces vertex count."""
        # Create 2x2x2 solid block
        grid = VoxelGrid(4, 4, 4)
        for x in range(1, 3):
            for y in range(1, 3):
                for z in range(1, 3):
                    grid.set_voxel(x, y, z, 255, 0, 0, 255)

        greedy = GreedyMesher()
        naive = NaiveMesher()

        greedy_mesh = greedy.mesh(grid.data)
        naive_mesh = naive.mesh(grid.data)

        # Greedy should produce fewer vertices
        assert len(greedy_mesh.vertices) < len(naive_mesh.vertices)


class TestColorConversion(unittest.TestCase):
    """Tests for color space conversion."""

    def test_srgb_linear_roundtrip(self):
        """Test sRGB <-> Linear roundtrip."""
        # Test with typical colors
        original = np.array([[128, 64, 192, 255]], dtype=np.uint8)

        linear = srgb_to_linear(original)
        back = linear_to_srgb(linear)

        # Should be close to original (some precision loss expected)
        assert np.allclose(original, back, atol=2)

    def test_linear_zero_one(self):
        """Test black and white conversion."""
        colors = np.array([[0, 0, 0, 255], [255, 255, 255, 255]], dtype=np.uint8)
        linear = srgb_to_linear(colors)

        # Black should stay 0
        assert np.allclose(linear[0, :3], 0, atol=0.01)
        # White should stay 1
        assert np.allclose(linear[1, :3], 1, atol=0.01)


class TestProjection(unittest.TestCase):
    """Tests for projection mathematics."""

    def test_world_to_screen(self):
        """Test world to screen projection."""
        proj = ProjectionMatrix(tile_width=32, tile_height=16)

        # Origin should project to offset
        u, v = proj.world_to_screen(0, 0, 0, offset_u=100, offset_v=100)
        assert u == 100
        assert v == 100

    def test_screen_to_world_roundtrip(self):
        """Test screen <-> world roundtrip."""
        proj = ProjectionMatrix(tile_width=32, tile_height=16)

        # Test point
        x, y, z = 5, 3, 2

        # Forward project
        u, v = proj.world_to_screen(x, y, z)

        # Inverse project with known z
        x2, y2, z2 = proj.screen_to_world(u, v, z)

        assert np.isclose(x, x2, atol=0.01)
        assert np.isclose(y, y2, atol=0.01)
        assert z == z2


class TestVoxelGenerator(unittest.TestCase):
    """Integration tests for VoxelGenerator."""

    def test_basic_pipeline(self):
        """Test basic voxelization pipeline."""
        # Create test image
        rgba = np.zeros((16, 16, 4), dtype=np.uint8)
        rgba[4:12, 4:12, :3] = [255, 128, 64]
        rgba[4:12, 4:12, 3] = 255

        generator = VoxelGenerator(max_depth=8)
        generator.load_array(rgba)
        generator.set_depth_mode("flat")
        generator.voxelize()
        generator.generate_mesh()

        assert generator.voxel_count > 0
        assert generator.vertex_count > 0
        assert generator.triangle_count > 0

    def test_billboard_mode(self):
        """Test billboard voxelization."""
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[2:6, 2:6, :] = 255

        generator = VoxelGenerator()
        generator.load_array(rgba)
        generator.voxelize_billboard(thickness=2)

        assert generator.voxel_count > 0


if __name__ == "__main__":
    unittest.main(verbosity=2)
