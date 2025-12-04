"""
Voxel Data Structures and Voxelization Engine

This module provides:
- VoxelGrid: Dense 3D array for storing voxel data
- Voxelizer: Engine for converting 2D images to 3D voxel grids

Memory consideration: A 256³ grid with RGBA = 256³ × 4 bytes ≈ 64 MB
For typical pixel art (< 128³), memory is ~8 MB - trivial for modern systems.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Iterator
import numpy as np

from .projection import IsometricProjection
from .depth import DepthEstimator, DepthMode


@dataclass
class VoxelGrid:
    """
    Dense 3D voxel grid with RGBA color support.

    The grid uses a dense numpy array for O(1) access and efficient
    iteration. This is optimal for single-object voxel art where
    bounds are limited (< 256³).

    Coordinate system: X-right, Y-back, Z-up (standard mathematical)
    """

    size_x: int
    size_y: int
    size_z: int
    _data: np.ndarray = field(init=False, repr=False)
    _bounds_min: np.ndarray = field(init=False, repr=False)
    _bounds_max: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the voxel grid data array."""
        # RGBA data, uint8 for memory efficiency
        self._data = np.zeros(
            (self.size_x, self.size_y, self.size_z, 4),
            dtype=np.uint8
        )
        # Track occupied bounds for optimization
        self._bounds_min = np.array([self.size_x, self.size_y, self.size_z])
        self._bounds_max = np.array([0, 0, 0])

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get grid dimensions (x, y, z)."""
        return (self.size_x, self.size_y, self.size_z)

    @property
    def data(self) -> np.ndarray:
        """Get the raw RGBA data array."""
        return self._data

    @property
    def occupancy(self) -> np.ndarray:
        """Get binary occupancy mask (True where voxel exists)."""
        return self._data[:, :, :, 3] > 0

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the occupied region bounds (min_xyz, max_xyz)."""
        return (self._bounds_min.copy(), self._bounds_max.copy())

    @property
    def occupied_bounds(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Get tight bounds around occupied voxels."""
        occupied = np.argwhere(self.occupancy)
        if len(occupied) == 0:
            return ((0, 0, 0), (0, 0, 0))
        min_coords = occupied.min(axis=0)
        max_coords = occupied.max(axis=0) + 1  # exclusive upper bound
        return (tuple(min_coords), tuple(max_coords))

    def set_voxel(self, x: int, y: int, z: int, r: int, g: int, b: int, a: int = 255):
        """
        Set a voxel at the given coordinates.

        Args:
            x, y, z: Voxel coordinates
            r, g, b: Color values (0-255)
            a: Alpha value (0-255), default 255 (opaque)
        """
        if not self._in_bounds(x, y, z):
            return  # Silently ignore out-of-bounds

        self._data[x, y, z] = [r, g, b, a]
        self._update_bounds(x, y, z)

    def set_voxel_rgba(self, x: int, y: int, z: int, rgba: np.ndarray):
        """
        Set a voxel using an RGBA array.

        Args:
            x, y, z: Voxel coordinates
            rgba: Array of [r, g, b, a] values
        """
        if not self._in_bounds(x, y, z):
            return

        self._data[x, y, z] = rgba
        self._update_bounds(x, y, z)

    def get_voxel(self, x: int, y: int, z: int) -> Optional[np.ndarray]:
        """
        Get voxel color at coordinates.

        Args:
            x, y, z: Voxel coordinates

        Returns:
            RGBA array or None if empty/out of bounds
        """
        if not self._in_bounds(x, y, z):
            return None

        voxel = self._data[x, y, z]
        if voxel[3] == 0:  # Transparent = no voxel
            return None
        return voxel.copy()

    def is_solid(self, x: int, y: int, z: int) -> bool:
        """Check if a voxel exists at the given coordinates."""
        if not self._in_bounds(x, y, z):
            return False
        return self._data[x, y, z, 3] > 0

    def clear_voxel(self, x: int, y: int, z: int):
        """Remove a voxel at the given coordinates."""
        if self._in_bounds(x, y, z):
            self._data[x, y, z] = [0, 0, 0, 0]

    def clear(self):
        """Clear all voxels from the grid."""
        self._data.fill(0)
        self._bounds_min = np.array([self.size_x, self.size_y, self.size_z])
        self._bounds_max = np.array([0, 0, 0])

    def _in_bounds(self, x: int, y: int, z: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return (
            0 <= x < self.size_x and
            0 <= y < self.size_y and
            0 <= z < self.size_z
        )

    def _update_bounds(self, x: int, y: int, z: int):
        """Update the occupied bounds tracking."""
        self._bounds_min = np.minimum(self._bounds_min, [x, y, z])
        self._bounds_max = np.maximum(self._bounds_max, [x + 1, y + 1, z + 1])

    def count_voxels(self) -> int:
        """Count the number of solid voxels."""
        return int(np.sum(self.occupancy))

    def get_unique_colors(self) -> np.ndarray:
        """
        Get all unique colors in the grid.

        Returns:
            Array of shape (N, 4) with unique RGBA colors
        """
        occupied = self.occupancy
        colors = self._data[occupied]
        return np.unique(colors, axis=0)

    def iterate_voxels(self) -> Iterator[Tuple[int, int, int, np.ndarray]]:
        """
        Iterate over all solid voxels.

        Yields:
            Tuples of (x, y, z, rgba)
        """
        indices = np.argwhere(self.occupancy)
        for x, y, z in indices:
            yield (int(x), int(y), int(z), self._data[x, y, z].copy())

    def to_sparse(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to sparse representation.

        Returns:
            Tuple of (coordinates, colors) where:
            - coordinates: Array of shape (N, 3) with xyz indices
            - colors: Array of shape (N, 4) with RGBA values
        """
        occupied = self.occupancy
        coords = np.argwhere(occupied)
        colors = self._data[occupied]
        return coords, colors

    def from_sparse(self, coords: np.ndarray, colors: np.ndarray):
        """
        Load voxels from sparse representation.

        Args:
            coords: Array of shape (N, 3) with xyz indices
            colors: Array of shape (N, 4) with RGBA values
        """
        self.clear()
        for i in range(len(coords)):
            x, y, z = coords[i]
            self._data[x, y, z] = colors[i]
            self._update_bounds(x, y, z)

    def crop_to_bounds(self) -> "VoxelGrid":
        """
        Create a new grid cropped to the occupied region.

        Returns:
            New VoxelGrid containing only the occupied region
        """
        (min_x, min_y, min_z), (max_x, max_y, max_z) = self.occupied_bounds

        new_size_x = max_x - min_x
        new_size_y = max_y - min_y
        new_size_z = max_z - min_z

        if new_size_x <= 0 or new_size_y <= 0 or new_size_z <= 0:
            return VoxelGrid(1, 1, 1)

        new_grid = VoxelGrid(new_size_x, new_size_y, new_size_z)
        new_grid._data = self._data[
            min_x:max_x,
            min_y:max_y,
            min_z:max_z
        ].copy()
        new_grid._bounds_min = np.array([0, 0, 0])
        new_grid._bounds_max = np.array([new_size_x, new_size_y, new_size_z])

        return new_grid


class Voxelizer:
    """
    Engine for converting 2D pixel art to 3D voxel grids.

    The voxelizer handles:
    - Isometric projection mathematics
    - Depth estimation
    - Voxel grid population with colors
    """

    def __init__(
        self,
        tile_width: float = 2.0,
        tile_height: float = 1.0,
        max_depth: int = 32,
        depth_mode: DepthMode = DepthMode.DISTANCE_TRANSFORM
    ):
        """
        Initialize the voxelizer.

        Args:
            tile_width: Pixel width of a single voxel projection
            tile_height: Pixel height of a single voxel projection
            max_depth: Maximum depth (z) dimension
            depth_mode: Depth estimation strategy
        """
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.max_depth = max_depth
        self.depth_mode = depth_mode

        self.projection = IsometricProjection(tile_width, tile_height)
        self.depth_estimator = DepthEstimator(depth_mode, max_depth)

        self._grid: Optional[VoxelGrid] = None

    def set_depth_mode(
        self,
        mode: DepthMode,
        scale: float = 1.0,
        invert: bool = False
    ):
        """
        Configure depth estimation.

        Args:
            mode: Depth estimation strategy
            scale: Depth scaling factor
            invert: Whether to invert depth values
        """
        self.depth_mode = mode
        self.depth_estimator = DepthEstimator(mode, self.max_depth, scale, invert)

    def set_explicit_depth(self, depth_map: np.ndarray):
        """
        Set explicit depth map for EXPLICIT mode.

        Args:
            depth_map: Grayscale depth image
        """
        self.depth_estimator.set_explicit_depth(depth_map)

    def voxelize(
        self,
        color_image: np.ndarray,
        alpha_mask: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        fill_below: bool = True,
        extrusion_mode: str = "column"
    ) -> VoxelGrid:
        """
        Convert a 2D image to a 3D voxel grid.

        Args:
            color_image: RGBA image array of shape (H, W, 4)
            alpha_mask: Binary mask of opaque pixels
            depth_map: Optional explicit depth map (overrides depth_mode)
            fill_below: If True, fill voxels from z=0 up to the depth value
            extrusion_mode: "column" (fill below), "surface" (single layer),
                          or "shell" (hollow with thickness)

        Returns:
            Populated VoxelGrid
        """
        h, w = alpha_mask.shape

        # Calculate depth for each pixel
        if depth_map is not None:
            self.depth_estimator.set_explicit_depth(depth_map)
            self.depth_estimator.mode = DepthMode.EXPLICIT

        depth = self.depth_estimator.estimate(alpha_mask, color_image)

        # Calculate grid size based on image dimensions and depth
        # The isometric projection means X and Y dimensions are derived from image
        grid_x = w + h  # Rough estimate for isometric coverage
        grid_y = w + h
        grid_z = self.max_depth + 1

        # Adjust for reasonable bounds
        grid_x = min(max(grid_x, 32), 256)
        grid_y = min(max(grid_y, 32), 256)

        # Set projection origin at image center-bottom
        self.projection.set_origin_from_image(w, h)

        # Create the voxel grid
        self._grid = VoxelGrid(grid_x, grid_y, grid_z)

        # Offset to center the model in the grid
        offset_x = grid_x // 2
        offset_y = grid_y // 2

        # Process each opaque pixel
        opaque_pixels = np.argwhere(alpha_mask)

        for py, px in opaque_pixels:
            pixel_depth = int(depth[py, px])
            if pixel_depth == 0:
                continue

            # Get pixel color (RGB only, alpha already confirmed)
            color = color_image[py, px, :3]

            # Calculate base voxel position
            # For simplicity, we use a direct mapping approach
            # In true isometric, we'd inverse-project, but for pixel art
            # a direct X-Y mapping with depth as Z works well

            vx = px + offset_x - w // 2
            vy = py + offset_y - h // 2

            if extrusion_mode == "column" or fill_below:
                # Fill from z=0 up to depth
                for vz in range(pixel_depth + 1):
                    self._grid.set_voxel(vx, vy, vz, color[0], color[1], color[2])
            elif extrusion_mode == "surface":
                # Single voxel layer at depth
                self._grid.set_voxel(vx, vy, pixel_depth, color[0], color[1], color[2])
            elif extrusion_mode == "shell":
                # Top and bottom surfaces only
                self._grid.set_voxel(vx, vy, pixel_depth, color[0], color[1], color[2])
                self._grid.set_voxel(vx, vy, 0, color[0], color[1], color[2])

        return self._grid

    def voxelize_layered(
        self,
        color_image: np.ndarray,
        alpha_mask: np.ndarray,
        depth_map: np.ndarray,
        layer_colors: bool = False
    ) -> VoxelGrid:
        """
        Voxelize with distinct layers based on depth map.

        Unlike column filling, this creates only voxels at the
        specific depth indicated by the depth map.

        Args:
            color_image: RGBA image array
            alpha_mask: Binary mask
            depth_map: Depth values for each pixel
            layer_colors: If True, color varies by depth layer

        Returns:
            VoxelGrid with layered structure
        """
        h, w = alpha_mask.shape

        # Grid dimensions
        grid_x = w
        grid_y = h
        grid_z = self.max_depth + 1

        self._grid = VoxelGrid(grid_x, grid_y, grid_z)

        opaque_pixels = np.argwhere(alpha_mask)

        for py, px in opaque_pixels:
            pixel_depth = int(depth_map[py, px])
            color = color_image[py, px, :3]

            if layer_colors:
                # Tint color based on depth for visual clarity
                depth_factor = pixel_depth / self.max_depth
                tinted = (color * (0.5 + 0.5 * depth_factor)).astype(np.uint8)
                self._grid.set_voxel(px, py, pixel_depth, tinted[0], tinted[1], tinted[2])
            else:
                self._grid.set_voxel(px, py, pixel_depth, color[0], color[1], color[2])

        return self._grid

    def voxelize_billboard(
        self,
        color_image: np.ndarray,
        alpha_mask: np.ndarray,
        thickness: int = 1
    ) -> VoxelGrid:
        """
        Create a flat billboard-style voxel model.

        This is the simplest mode: just extrude the sprite
        by a fixed number of voxels.

        Args:
            color_image: RGBA image array
            alpha_mask: Binary mask
            thickness: Number of voxel layers

        Returns:
            VoxelGrid with flat billboard geometry
        """
        h, w = alpha_mask.shape

        self._grid = VoxelGrid(w, h, thickness)

        opaque_pixels = np.argwhere(alpha_mask)

        for py, px in opaque_pixels:
            color = color_image[py, px, :3]
            for vz in range(thickness):
                self._grid.set_voxel(px, py, vz, color[0], color[1], color[2])

        return self._grid

    @property
    def grid(self) -> Optional[VoxelGrid]:
        """Get the current voxel grid."""
        return self._grid


def merge_grids(
    grids: List[VoxelGrid],
    offsets: List[Tuple[int, int, int]]
) -> VoxelGrid:
    """
    Merge multiple voxel grids into one.

    Args:
        grids: List of VoxelGrid objects
        offsets: List of (x, y, z) offsets for each grid

    Returns:
        Merged VoxelGrid
    """
    if not grids:
        raise ValueError("At least one grid required")

    # Calculate required size
    max_x = max_y = max_z = 0

    for grid, (ox, oy, oz) in zip(grids, offsets):
        max_x = max(max_x, grid.size_x + ox)
        max_y = max(max_y, grid.size_y + oy)
        max_z = max(max_z, grid.size_z + oz)

    merged = VoxelGrid(max_x, max_y, max_z)

    for grid, (ox, oy, oz) in zip(grids, offsets):
        for x, y, z, rgba in grid.iterate_voxels():
            merged.set_voxel_rgba(x + ox, y + oy, z + oz, rgba)

    return merged
