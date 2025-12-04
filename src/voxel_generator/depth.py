"""
Depth Estimation Heuristics Module

This module provides multiple strategies for inferring depth from 2D images:
1. Distance Transform (EDT) - Creates dome/pillow shapes from silhouettes
2. Luminosity-Based - Assumes brightness correlates with depth
3. Explicit Depth Maps - Uses user-provided grayscale height maps
4. Flat Extrusion - Constant depth for all opaque pixels
5. Gradient Extrusion - Linear depth gradient across the image

These heuristics resolve the underdetermined inverse projection problem
where depth (z) cannot be uniquely determined from 2D coordinates alone.
"""

from enum import Enum
from typing import Optional, Tuple, Callable
import numpy as np
from scipy import ndimage


class DepthMode(Enum):
    """Available depth estimation strategies."""
    FLAT = "flat"                       # Constant depth
    DISTANCE_TRANSFORM = "distance_transform"  # EDT for organic shapes
    LUMINOSITY = "luminosity"           # Brightness-based
    EXPLICIT = "explicit"               # User-provided depth map
    GRADIENT_X = "gradient_x"           # Linear gradient along X
    GRADIENT_Y = "gradient_y"           # Linear gradient along Y
    SYMMETRY = "symmetry"               # Symmetric extrusion from center


class DepthEstimator:
    """
    Depth estimation engine for 2D to 3D voxel conversion.

    The estimator takes a binary mask (silhouette) and produces a depth map
    where each opaque pixel is assigned a z-value.
    """

    def __init__(
        self,
        mode: DepthMode = DepthMode.DISTANCE_TRANSFORM,
        max_depth: int = 16,
        scale: float = 1.0,
        invert: bool = False
    ):
        """
        Initialize the depth estimator.

        Args:
            mode: Depth estimation strategy
            max_depth: Maximum depth value (number of voxel layers)
            scale: Scaling factor for depth values
            invert: If True, invert the depth values (near becomes far)
        """
        self.mode = mode
        self.max_depth = max_depth
        self.scale = scale
        self.invert = invert
        self._explicit_depth: Optional[np.ndarray] = None

    def set_explicit_depth(self, depth_map: np.ndarray):
        """
        Set an explicit user-provided depth map.

        Args:
            depth_map: Grayscale depth map (0-255)
        """
        self._explicit_depth = depth_map.astype(np.float32)

    def estimate(
        self,
        mask: np.ndarray,
        color_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Estimate depth for all pixels in the mask.

        Args:
            mask: Binary mask where True = opaque pixel
            color_image: Optional RGBA image for luminosity mode

        Returns:
            Depth map array with integer depth values (0 to max_depth)
        """
        if self.mode == DepthMode.FLAT:
            depth = self._flat_depth(mask)
        elif self.mode == DepthMode.DISTANCE_TRANSFORM:
            depth = self._distance_transform_depth(mask)
        elif self.mode == DepthMode.LUMINOSITY:
            if color_image is None:
                raise ValueError("Luminosity mode requires color_image")
            depth = self._luminosity_depth(mask, color_image)
        elif self.mode == DepthMode.EXPLICIT:
            if self._explicit_depth is None:
                raise ValueError("Explicit mode requires depth map (call set_explicit_depth)")
            depth = self._explicit_depth_map(mask)
        elif self.mode == DepthMode.GRADIENT_X:
            depth = self._gradient_depth(mask, axis=1)
        elif self.mode == DepthMode.GRADIENT_Y:
            depth = self._gradient_depth(mask, axis=0)
        elif self.mode == DepthMode.SYMMETRY:
            depth = self._symmetry_depth(mask)
        else:
            raise ValueError(f"Unknown depth mode: {self.mode}")

        # Apply scaling and inversion
        depth = depth * self.scale

        if self.invert:
            depth = self.max_depth - depth

        # Clamp to valid range
        depth = np.clip(depth, 0, self.max_depth)

        # Apply mask - only opaque pixels have depth
        depth = depth * mask.astype(np.float32)

        return depth.astype(np.uint8)

    def _flat_depth(self, mask: np.ndarray) -> np.ndarray:
        """
        Flat extrusion - constant depth for all pixels.

        Args:
            mask: Binary opacity mask

        Returns:
            Depth map with constant value
        """
        return np.where(mask, self.max_depth // 2, 0).astype(np.float32)

    def _distance_transform_depth(self, mask: np.ndarray) -> np.ndarray:
        """
        Euclidean Distance Transform for organic shapes.

        Creates a "pillow" or "dome" effect where the center of the
        silhouette becomes the deepest point.

        Args:
            mask: Binary opacity mask

        Returns:
            Depth map based on distance to edge
        """
        if not np.any(mask):
            return np.zeros_like(mask, dtype=np.float32)

        # Calculate distance to nearest background pixel
        distance = ndimage.distance_transform_edt(mask)

        # Normalize to max_depth range
        max_dist = distance.max()
        if max_dist > 0:
            normalized = (distance / max_dist) * self.max_depth
        else:
            normalized = np.zeros_like(distance)

        return normalized.astype(np.float32)

    def _luminosity_depth(
        self,
        mask: np.ndarray,
        color_image: np.ndarray
    ) -> np.ndarray:
        """
        Luminosity-based depth estimation.

        Assumes lighter pixels are closer to the viewer (higher z).
        Formula: luminosity = 0.299*R + 0.587*G + 0.114*B

        WARNING: This heuristic can fail for objects with dark textures.

        Args:
            mask: Binary opacity mask
            color_image: RGBA image array

        Returns:
            Depth map based on pixel brightness
        """
        # Calculate luminosity (standard BT.601 weights)
        r = color_image[:, :, 0].astype(np.float32)
        g = color_image[:, :, 1].astype(np.float32)
        b = color_image[:, :, 2].astype(np.float32)

        luminosity = 0.299 * r + 0.587 * g + 0.114 * b

        # Normalize to max_depth range
        normalized = (luminosity / 255.0) * self.max_depth

        return normalized * mask.astype(np.float32)

    def _explicit_depth_map(self, mask: np.ndarray) -> np.ndarray:
        """
        Use explicit user-provided depth map.

        Args:
            mask: Binary opacity mask (used to filter results)

        Returns:
            Scaled depth values from explicit map
        """
        # Scale 0-255 grayscale to 0-max_depth
        depth = (self._explicit_depth / 255.0) * self.max_depth
        return depth * mask.astype(np.float32)

    def _gradient_depth(
        self,
        mask: np.ndarray,
        axis: int = 1
    ) -> np.ndarray:
        """
        Linear gradient depth across one axis.

        Args:
            mask: Binary opacity mask
            axis: 0 for vertical gradient, 1 for horizontal

        Returns:
            Depth map with linear gradient
        """
        h, w = mask.shape

        if axis == 0:
            # Vertical gradient (top to bottom)
            gradient = np.linspace(0, self.max_depth, h)
            depth = np.tile(gradient[:, np.newaxis], (1, w))
        else:
            # Horizontal gradient (left to right)
            gradient = np.linspace(0, self.max_depth, w)
            depth = np.tile(gradient[np.newaxis, :], (h, 1))

        return depth.astype(np.float32) * mask.astype(np.float32)

    def _symmetry_depth(self, mask: np.ndarray) -> np.ndarray:
        """
        Symmetric extrusion from the horizontal center.

        Creates depth that increases toward the center line,
        useful for objects with bilateral symmetry.

        Args:
            mask: Binary opacity mask

        Returns:
            Depth map with symmetric profile
        """
        h, w = mask.shape
        center_x = w / 2.0

        # Calculate distance from center line
        x_coords = np.arange(w)
        dist_from_center = np.abs(x_coords - center_x)

        # Invert so center is deepest
        max_dist = center_x
        if max_dist > 0:
            depth_profile = (1.0 - dist_from_center / max_dist) * self.max_depth
        else:
            depth_profile = np.zeros(w)

        # Broadcast to full image
        depth = np.tile(depth_profile[np.newaxis, :], (h, 1))

        return depth.astype(np.float32) * mask.astype(np.float32)


class MultiViewDepthEstimator:
    """
    Advanced depth estimation using multiple views or layered sprites.

    For complex models, artists may provide:
    - Front/Back views
    - Top-down views
    - Multiple isometric angles

    This class combines information from multiple views to produce
    more accurate depth estimates.
    """

    def __init__(self, max_depth: int = 32):
        """
        Initialize multi-view estimator.

        Args:
            max_depth: Maximum depth value
        """
        self.max_depth = max_depth
        self._views: dict = {}

    def add_view(
        self,
        name: str,
        mask: np.ndarray,
        depth_estimator: DepthEstimator,
        weight: float = 1.0
    ):
        """
        Add a view for multi-view reconstruction.

        Args:
            name: View identifier (e.g., "front", "top", "side")
            mask: Binary mask for this view
            depth_estimator: Estimator to use for this view
            weight: Blending weight for this view
        """
        self._views[name] = {
            "mask": mask,
            "estimator": depth_estimator,
            "weight": weight
        }

    def estimate_combined(self) -> np.ndarray:
        """
        Combine depth estimates from all views.

        Returns:
            Combined depth map
        """
        if not self._views:
            raise RuntimeError("No views added")

        # Get depth from each view
        depths = []
        weights = []

        for view_data in self._views.values():
            depth = view_data["estimator"].estimate(view_data["mask"])
            depths.append(depth)
            weights.append(view_data["weight"])

        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()

        combined = np.zeros_like(depths[0], dtype=np.float32)
        for depth, weight in zip(depths, weights):
            combined += depth.astype(np.float32) * weight

        return combined.astype(np.uint8)


def create_depth_from_layers(
    layers: list,
    spacing: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create depth map from layered pixel art.

    Some artists create sprites as stacked layers (like onion skin).
    This function assigns depth based on layer order.

    Args:
        layers: List of RGBA layer arrays (front to back)
        spacing: Depth units between layers

    Returns:
        Tuple of (combined_color, depth_map)
    """
    if not layers:
        raise ValueError("At least one layer required")

    h, w = layers[0].shape[:2]
    combined_color = np.zeros((h, w, 4), dtype=np.uint8)
    depth_map = np.zeros((h, w), dtype=np.uint8)

    # Process layers from back to front
    for i, layer in enumerate(reversed(layers)):
        layer_depth = (len(layers) - i - 1) * spacing
        layer_mask = layer[:, :, 3] > 127

        # Where this layer is opaque, set its color and depth
        combined_color[layer_mask] = layer[layer_mask]
        depth_map[layer_mask] = layer_depth

    return combined_color, depth_map


def estimate_depth_from_normals(
    normal_map: np.ndarray,
    mask: np.ndarray,
    max_depth: int = 16
) -> np.ndarray:
    """
    Estimate depth from a normal map using integration.

    Normal maps encode surface orientation. By integrating the
    normal field, we can recover depth (up to a constant).

    Args:
        normal_map: RGB normal map (range 0-255, where 128 = neutral)
        mask: Binary opacity mask
        max_depth: Maximum depth value

    Returns:
        Estimated depth map
    """
    # Decode normals from RGB (128 = 0, 0 = -1, 255 = 1)
    nx = (normal_map[:, :, 0].astype(np.float32) - 128) / 127.0
    ny = (normal_map[:, :, 1].astype(np.float32) - 128) / 127.0
    nz = (normal_map[:, :, 2].astype(np.float32) - 128) / 127.0

    # Avoid division by zero
    nz = np.maximum(nz, 0.01)

    # Surface gradient from normals
    dz_dx = -nx / nz
    dz_dy = -ny / nz

    # Simple integration (Poisson-like)
    # This is a simplified approach - full Poisson reconstruction is complex
    h, w = mask.shape

    # Cumulative sum along x and y
    depth_x = np.cumsum(dz_dx, axis=1)
    depth_y = np.cumsum(dz_dy, axis=0)

    # Average both integrations
    depth = (depth_x + depth_y) / 2.0

    # Normalize to range
    depth = depth - depth.min()
    if depth.max() > 0:
        depth = (depth / depth.max()) * max_depth

    return (depth * mask.astype(np.float32)).astype(np.uint8)
