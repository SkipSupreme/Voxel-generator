"""
Projection Mathematics for Isometric/Dimetric Pixel Art

This module implements the 2:1 dimetric projection commonly used in pixel art,
along with inverse projection for 3D reconstruction.

The "2:1" projection uses an angle of arctan(0.5) ≈ 26.565° rather than true
isometric (30°), as it produces cleaner pixel stepping patterns.

Coordinate Systems:
- Internal: Right-handed, Z-up (+X Right, +Y Back, +Z Up)
- Godot: Right-handed, Y-up (+X Right, +Y Up, +Z Back)
- Blender: Right-handed, Z-up (same as internal)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional
import math
import numpy as np


class CoordinateSystem(Enum):
    """Target coordinate system for export."""
    INTERNAL = "internal"      # Z-up, right-handed (mathematical standard)
    GODOT = "godot"           # Y-up, right-handed (OpenGL)
    BLENDER = "blender"       # Z-up, right-handed (same as internal)
    MAGICAVOXEL = "magicavoxel"  # Z-up, specific axis ordering


@dataclass
class ProjectionMatrix:
    """
    2:1 Dimetric Projection Matrix for pixel art.

    The projection transforms from World Space (x, y, z) to Screen Space (u, v).
    In standard 2:1 pixel art, a tile is typically 2W x W pixels (e.g., 32x16).

    The projection matrix is:
        | W/2   -W/2    0   |
        | H/4    H/4   -H   |

    Where:
        - W = tile width in pixels
        - H = tile height in pixels (typically W/2)
        - u = screen x coordinate
        - v = screen y coordinate (y-down in image space)
    """

    tile_width: float = 32.0
    tile_height: float = 16.0

    def __post_init__(self):
        """Precompute projection constants."""
        self.w = self.tile_width
        self.h = self.tile_height
        # Build the 2x3 projection matrix
        self._build_matrix()

    def _build_matrix(self):
        """Construct the projection matrix."""
        # Forward projection: World -> Screen
        # u = (W/2)*x - (W/2)*y + offset_u
        # v = (H/4)*x + (H/4)*y - H*z + offset_v
        self.proj_matrix = np.array([
            [self.w / 2, -self.w / 2, 0],
            [self.h / 4, self.h / 4, -self.h]
        ], dtype=np.float64)

        # For inverse projection, we need the 2x2 submatrix (ignoring z)
        # [u]   [W/2  -W/2] [x]
        # [v] = [H/4   H/4] [y]  + z_contribution
        self.proj_xy = np.array([
            [self.w / 2, -self.w / 2],
            [self.h / 4, self.h / 4]
        ], dtype=np.float64)

        # Inverse of the xy submatrix
        self.inv_proj_xy = np.linalg.inv(self.proj_xy)

    def world_to_screen(
        self,
        x: float, y: float, z: float,
        offset_u: float = 0, offset_v: float = 0
    ) -> Tuple[float, float]:
        """
        Project a world coordinate to screen space.

        Args:
            x, y, z: World coordinates
            offset_u, offset_v: Screen space offset (e.g., image center)

        Returns:
            (u, v): Screen coordinates
        """
        world = np.array([x, y, z])
        screen = self.proj_matrix @ world
        return (screen[0] + offset_u, screen[1] + offset_v)

    def screen_to_world(
        self,
        u: float, v: float, z: float,
        offset_u: float = 0, offset_v: float = 0
    ) -> Tuple[float, float, float]:
        """
        Inverse project a screen coordinate to world space given depth z.

        The inverse projection requires a known z value to resolve the
        underdetermined system (2 equations, 3 unknowns).

        Args:
            u, v: Screen coordinates
            z: Known depth value
            offset_u, offset_v: Screen space offset (e.g., image center)

        Returns:
            (x, y, z): World coordinates
        """
        # Remove offset
        u_rel = u - offset_u
        v_rel = v - offset_v

        # Remove z contribution from screen coords
        # v = (H/4)*x + (H/4)*y - H*z
        # v + H*z = (H/4)*x + (H/4)*y
        v_adjusted = v_rel + self.h * z

        # Now solve the 2x2 system
        screen_adjusted = np.array([u_rel, v_adjusted])
        xy = self.inv_proj_xy @ screen_adjusted

        return (xy[0], xy[1], z)

    def screen_to_world_batch(
        self,
        uv: np.ndarray,
        z: np.ndarray,
        offset_u: float = 0, offset_v: float = 0
    ) -> np.ndarray:
        """
        Batch inverse projection for multiple points.

        Args:
            uv: Array of shape (N, 2) with screen coordinates
            z: Array of shape (N,) with depth values
            offset_u, offset_v: Screen space offset

        Returns:
            xyz: Array of shape (N, 3) with world coordinates
        """
        n = uv.shape[0]
        u_rel = uv[:, 0] - offset_u
        v_rel = uv[:, 1] - offset_v

        v_adjusted = v_rel + self.h * z

        # Apply inverse matrix
        screen_adjusted = np.column_stack([u_rel, v_adjusted])
        xy = (self.inv_proj_xy @ screen_adjusted.T).T

        return np.column_stack([xy, z])


class IsometricProjection:
    """
    Complete isometric projection system for voxel reconstruction.

    This class handles the full pipeline of converting screen coordinates
    to voxel grid indices, accounting for:
    - Image coordinate system (y-down)
    - Voxel grid quantization
    - Multiple viewing angles
    """

    def __init__(
        self,
        tile_width: float = 2.0,
        tile_height: float = 1.0,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
        angle: str = "standard"
    ):
        """
        Initialize the isometric projection.

        Args:
            tile_width: Width of a single tile in pixels
            tile_height: Height of a single tile in pixels (typically tile_width/2)
            origin_x: X offset of the projection origin in screen space
            origin_y: Y offset of the projection origin in screen space
            angle: Projection angle ("standard" = 2:1, "true_iso" = 30°)
        """
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.angle = angle

        self.proj = ProjectionMatrix(tile_width, tile_height)

    def set_origin_from_image(self, image_width: int, image_height: int):
        """
        Set the projection origin to the center-bottom of an image.

        This is the typical origin for isometric sprites where the
        "foot" of the object is at the image center-bottom.

        Args:
            image_width: Width of the source image
            image_height: Height of the source image
        """
        self.origin_x = image_width / 2.0
        self.origin_y = image_height  # Bottom of image (y increases downward)

    def pixel_to_voxel(
        self,
        px: int, py: int, depth: int
    ) -> Tuple[int, int, int]:
        """
        Convert a pixel coordinate to voxel grid indices.

        Args:
            px, py: Pixel coordinates in image space (y-down)
            depth: Depth value (z-level) for this pixel

        Returns:
            (vx, vy, vz): Voxel grid indices
        """
        # Convert from image space (y-down) to math space (y-up)
        u = px
        v = self.origin_y - py  # Flip y

        # Inverse project
        wx, wy, wz = self.proj.screen_to_world(
            u, v, float(depth),
            self.origin_x, 0
        )

        # Quantize to voxel grid
        vx = int(round(wx))
        vy = int(round(wy))
        vz = int(depth)

        return (vx, vy, vz)

    def voxel_to_pixel(
        self,
        vx: int, vy: int, vz: int
    ) -> Tuple[float, float]:
        """
        Convert voxel grid indices to pixel coordinates.

        Args:
            vx, vy, vz: Voxel grid indices

        Returns:
            (px, py): Pixel coordinates in image space (y-down)
        """
        u, v = self.proj.world_to_screen(
            float(vx), float(vy), float(vz),
            self.origin_x, 0
        )

        # Convert back to image space (y-down)
        px = u
        py = self.origin_y - v

        return (px, py)


def get_coordinate_transform(
    source: CoordinateSystem,
    target: CoordinateSystem
) -> np.ndarray:
    """
    Get the 3x3 transformation matrix between coordinate systems.

    Args:
        source: Source coordinate system
        target: Target coordinate system

    Returns:
        3x3 transformation matrix
    """
    # Identity for same systems
    if source == target:
        return np.eye(3, dtype=np.float64)

    # Internal (Z-up) to Godot (Y-up)
    # Godot: +X Right, +Y Up, +Z Back
    # Internal: +X Right, +Y Back, +Z Up
    # Transform: x' = x, y' = z, z' = y
    internal_to_godot = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ], dtype=np.float64)

    # Internal to Blender (same system)
    internal_to_blender = np.eye(3, dtype=np.float64)

    # Internal to MagicaVoxel (may need axis swap depending on version)
    # MagicaVoxel uses Z-up but with specific orientation
    internal_to_magicavoxel = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    # Build transformation lookup
    transforms = {
        (CoordinateSystem.INTERNAL, CoordinateSystem.GODOT): internal_to_godot,
        (CoordinateSystem.INTERNAL, CoordinateSystem.BLENDER): internal_to_blender,
        (CoordinateSystem.INTERNAL, CoordinateSystem.MAGICAVOXEL): internal_to_magicavoxel,
    }

    # Direct lookup
    if (source, target) in transforms:
        return transforms[(source, target)]

    # Inverse lookup
    if (target, source) in transforms:
        return np.linalg.inv(transforms[(target, source)])

    # Chain through internal
    if source != CoordinateSystem.INTERNAL and target != CoordinateSystem.INTERNAL:
        to_internal = get_coordinate_transform(source, CoordinateSystem.INTERNAL)
        from_internal = get_coordinate_transform(CoordinateSystem.INTERNAL, target)
        return from_internal @ to_internal

    raise ValueError(f"No transform defined from {source} to {target}")


def transform_vertices(
    vertices: np.ndarray,
    source: CoordinateSystem,
    target: CoordinateSystem
) -> np.ndarray:
    """
    Transform an array of vertices between coordinate systems.

    Args:
        vertices: Array of shape (N, 3) containing vertex positions
        source: Source coordinate system
        target: Target coordinate system

    Returns:
        Transformed vertices array of shape (N, 3)
    """
    matrix = get_coordinate_transform(source, target)
    return (matrix @ vertices.T).T


def calculate_isometric_angle() -> dict:
    """
    Calculate and return key angles for isometric/dimetric projections.

    Returns:
        Dictionary with projection angle information
    """
    true_iso_angle = math.degrees(math.asin(1 / math.sqrt(3)))  # ~35.264°
    dimetric_21_angle = math.degrees(math.atan(0.5))  # ~26.565°

    return {
        "true_isometric": {
            "horizontal_rotation": 45.0,
            "vertical_rotation": true_iso_angle,
            "description": "Engineering isometric, equal foreshortening"
        },
        "pixel_art_2_1": {
            "horizontal_rotation": 45.0,
            "vertical_rotation": dimetric_21_angle,
            "pixel_ratio": "2:1",
            "description": "Standard pixel art dimetric for clean pixel stepping"
        }
    }
