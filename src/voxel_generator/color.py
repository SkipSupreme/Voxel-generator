"""
Color Management Module

Handles:
- sRGB to Linear color space conversion (for glTF export)
- Linear to sRGB conversion (for display)
- Color quantization for palette-limited formats (.vox)
- Palette generation using K-Means or Octree methods

Color Space Background:
- PNG images are in sRGB (perceptual) space
- glTF expects Linear (physical) vertex colors
- Failure to convert causes "washed out" colors in engines
"""

from typing import Tuple, Optional
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _srgb_to_linear_component(c: float) -> float:
    """
    Convert a single sRGB component to Linear.

    The sRGB standard uses a piecewise function:
    - Linear below threshold (0.04045)
    - Gamma curve above threshold

    Args:
        c: sRGB value normalized to [0, 1]

    Returns:
        Linear value
    """
    if c <= 0.04045:
        return c / 12.92
    else:
        return ((c + 0.055) / 1.055) ** 2.4


@njit(cache=True)
def _linear_to_srgb_component(c: float) -> float:
    """
    Convert a single Linear component to sRGB.

    Args:
        c: Linear value normalized to [0, 1]

    Returns:
        sRGB value
    """
    if c <= 0.0031308:
        return c * 12.92
    else:
        return 1.055 * (c ** (1.0 / 2.4)) - 0.055


@njit(cache=True, parallel=True)
def srgb_to_linear(colors: np.ndarray) -> np.ndarray:
    """
    Convert sRGB colors to Linear color space.

    Args:
        colors: Array of shape (N, 3) or (N, 4) with uint8 sRGB values

    Returns:
        Array of same shape with float32 Linear values [0, 1]
    """
    n = colors.shape[0]
    channels = colors.shape[1]
    result = np.empty((n, channels), dtype=np.float32)

    for i in prange(n):
        for c in range(min(channels, 3)):  # Only convert RGB, not alpha
            normalized = colors[i, c] / 255.0
            result[i, c] = _srgb_to_linear_component(normalized)

        if channels == 4:
            # Alpha stays as-is (just normalize)
            result[i, 3] = colors[i, 3] / 255.0

    return result


@njit(cache=True, parallel=True)
def linear_to_srgb(colors: np.ndarray) -> np.ndarray:
    """
    Convert Linear colors to sRGB color space.

    Args:
        colors: Array of shape (N, 3) or (N, 4) with float32 Linear values [0, 1]

    Returns:
        Array of same shape with uint8 sRGB values
    """
    n = colors.shape[0]
    channels = colors.shape[1]
    result = np.empty((n, channels), dtype=np.uint8)

    for i in prange(n):
        for c in range(min(channels, 3)):
            linear_val = max(0.0, min(1.0, colors[i, c]))
            srgb_val = _linear_to_srgb_component(linear_val)
            result[i, c] = np.uint8(srgb_val * 255.0 + 0.5)

        if channels == 4:
            result[i, 3] = np.uint8(max(0.0, min(1.0, colors[i, 3])) * 255.0 + 0.5)

    return result


def srgb_to_linear_simple(colors: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """
    Simple gamma approximation for sRGB to Linear.

    This is faster but less accurate than the full sRGB formula.
    Useful when precision isn't critical.

    Args:
        colors: uint8 color array
        gamma: Gamma value (default 2.2)

    Returns:
        float32 linear color array
    """
    normalized = colors.astype(np.float32) / 255.0
    return np.power(normalized, gamma)


def linear_to_srgb_simple(colors: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """
    Simple gamma approximation for Linear to sRGB.

    Args:
        colors: float32 linear color array [0, 1]
        gamma: Gamma value (default 2.2)

    Returns:
        uint8 sRGB color array
    """
    clamped = np.clip(colors, 0.0, 1.0)
    srgb = np.power(clamped, 1.0 / gamma)
    return (srgb * 255.0 + 0.5).astype(np.uint8)


class ColorQuantizer:
    """
    Color quantization for palette-limited formats.

    The .vox format is limited to 256 colors. This class reduces
    the color count while minimizing perceptual difference.
    """

    def __init__(self, max_colors: int = 256, method: str = "kmeans"):
        """
        Initialize the quantizer.

        Args:
            max_colors: Maximum number of colors in output palette
            method: Quantization method ("kmeans", "octree", "median_cut")
        """
        self.max_colors = max_colors
        self.method = method

    def quantize(
        self,
        colors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize colors to a limited palette.

        Args:
            colors: Array of shape (N, 3) or (N, 4) with RGB(A) values

        Returns:
            Tuple of (palette, indices) where:
            - palette: Array of shape (M, 3/4) with unique colors
            - indices: Array of shape (N,) mapping each input to palette index
        """
        # Strip alpha if present
        has_alpha = colors.shape[1] == 4
        rgb = colors[:, :3] if has_alpha else colors

        # Get unique colors first
        unique, inverse = np.unique(rgb, axis=0, return_inverse=True)

        if len(unique) <= self.max_colors:
            # No quantization needed
            if has_alpha:
                # Add alpha back to palette
                palette = np.column_stack([
                    unique,
                    np.full(len(unique), 255, dtype=np.uint8)
                ])
            else:
                palette = unique
            return palette, inverse

        # Apply quantization method
        if self.method == "kmeans":
            palette, indices = self._kmeans_quantize(rgb, unique)
        elif self.method == "octree":
            palette, indices = self._octree_quantize(rgb)
        elif self.method == "median_cut":
            palette, indices = self._median_cut_quantize(rgb)
        else:
            raise ValueError(f"Unknown quantization method: {self.method}")

        if has_alpha:
            palette = np.column_stack([
                palette,
                np.full(len(palette), 255, dtype=np.uint8)
            ])

        return palette, indices

    def _kmeans_quantize(
        self,
        colors: np.ndarray,
        unique_colors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        K-Means color quantization.

        Args:
            colors: All colors (may have duplicates)
            unique_colors: Unique colors for centroid initialization

        Returns:
            (palette, indices)
        """
        from scipy.cluster.vq import kmeans2

        # Use unique colors as potential initial centroids
        k = min(self.max_colors, len(unique_colors))

        # Run K-Means
        colors_float = colors.astype(np.float32)
        centroids, labels = kmeans2(
            colors_float,
            k,
            minit='++',  # K-Means++ initialization
            iter=20
        )

        palette = np.clip(centroids, 0, 255).astype(np.uint8)
        return palette, labels

    def _octree_quantize(
        self,
        colors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Octree color quantization.

        This method builds an octree of colors and merges leaves
        until the desired palette size is reached.

        Args:
            colors: RGB colors

        Returns:
            (palette, indices)
        """
        # Simplified octree quantization
        # Build histogram in 5-bit color space (32^3 = 32768 bins)
        bins = 32
        scale = 256 // bins

        # Quantize to bins
        binned = (colors // scale).astype(np.int32)
        bin_indices = binned[:, 0] * bins * bins + binned[:, 1] * bins + binned[:, 2]

        # Count occurrences
        unique_bins, inverse, counts = np.unique(
            bin_indices, return_inverse=True, return_counts=True
        )

        # Get average color for each bin
        bin_colors = np.zeros((len(unique_bins), 3), dtype=np.float32)
        for i, idx in enumerate(range(len(colors))):
            bin_idx = inverse[idx]
            bin_colors[bin_idx] += colors[idx]

        for i in range(len(unique_bins)):
            bin_colors[i] /= counts[i]

        # If still too many colors, merge smallest bins
        if len(unique_bins) > self.max_colors:
            # Sort by count, keep most frequent
            sorted_indices = np.argsort(-counts)[:self.max_colors]
            palette = bin_colors[sorted_indices].astype(np.uint8)

            # Remap indices
            new_indices = np.zeros(len(colors), dtype=np.int32)
            for i, color in enumerate(colors):
                # Find nearest palette color
                dists = np.sum((palette.astype(np.float32) - color) ** 2, axis=1)
                new_indices[i] = np.argmin(dists)

            return palette, new_indices
        else:
            return bin_colors.astype(np.uint8), inverse

    def _median_cut_quantize(
        self,
        colors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Median Cut color quantization.

        Recursively splits the color space along the axis
        with the largest range.

        Args:
            colors: RGB colors

        Returns:
            (palette, indices)
        """
        def median_cut_recursive(color_list, depth):
            if depth == 0 or len(color_list) == 0:
                if len(color_list) == 0:
                    return [[0, 0, 0]]
                return [np.mean(color_list, axis=0).astype(np.uint8).tolist()]

            # Find axis with largest range
            ranges = np.ptp(color_list, axis=0)
            axis = np.argmax(ranges)

            # Sort by that axis
            sorted_colors = color_list[color_list[:, axis].argsort()]

            # Split at median
            mid = len(sorted_colors) // 2

            return (
                median_cut_recursive(sorted_colors[:mid], depth - 1) +
                median_cut_recursive(sorted_colors[mid:], depth - 1)
            )

        # Calculate depth needed for desired palette size
        depth = int(np.ceil(np.log2(self.max_colors)))
        palette_list = median_cut_recursive(colors, depth)

        palette = np.array(palette_list[:self.max_colors], dtype=np.uint8)

        # Map colors to nearest palette entry
        indices = np.zeros(len(colors), dtype=np.int32)
        for i, color in enumerate(colors):
            dists = np.sum((palette.astype(np.float32) - color) ** 2, axis=1)
            indices[i] = np.argmin(dists)

        return palette, indices


class ColorManager:
    """
    Central color management for the voxel generator.

    Handles all color space conversions and palette management.
    """

    def __init__(self, quantizer: Optional[ColorQuantizer] = None):
        """
        Initialize the color manager.

        Args:
            quantizer: Optional pre-configured quantizer
        """
        self.quantizer = quantizer or ColorQuantizer()
        self._palette: Optional[np.ndarray] = None
        self._palette_map: dict = {}

    def set_palette(self, palette: np.ndarray):
        """
        Set an explicit palette.

        Args:
            palette: Array of shape (N, 3) or (N, 4) with palette colors
        """
        self._palette = palette.astype(np.uint8)
        self._build_palette_map()

    def _build_palette_map(self):
        """Build a lookup table from colors to palette indices."""
        self._palette_map = {}
        for i, color in enumerate(self._palette):
            key = tuple(color[:3])
            self._palette_map[key] = i

    def get_palette_index(self, color: np.ndarray) -> int:
        """
        Get palette index for a color.

        If color not in palette, returns nearest match.

        Args:
            color: RGB or RGBA color

        Returns:
            Palette index
        """
        key = tuple(color[:3])
        if key in self._palette_map:
            return self._palette_map[key]

        # Find nearest
        dists = np.sum((self._palette[:, :3].astype(np.float32) - color[:3]) ** 2, axis=1)
        return int(np.argmin(dists))

    def quantize_colors(self, colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize colors and return palette + indices.

        Args:
            colors: RGB(A) colors

        Returns:
            (palette, indices)
        """
        palette, indices = self.quantizer.quantize(colors)
        self.set_palette(palette)
        return palette, indices

    @property
    def palette(self) -> Optional[np.ndarray]:
        """Get the current palette."""
        return self._palette

    @staticmethod
    def blend_colors(
        color1: np.ndarray,
        color2: np.ndarray,
        factor: float
    ) -> np.ndarray:
        """
        Blend two colors.

        Args:
            color1: First RGB(A) color
            color2: Second RGB(A) color
            factor: Blend factor (0 = color1, 1 = color2)

        Returns:
            Blended color
        """
        return (color1 * (1 - factor) + color2 * factor).astype(np.uint8)

    @staticmethod
    def adjust_brightness(
        colors: np.ndarray,
        factor: float
    ) -> np.ndarray:
        """
        Adjust brightness of colors.

        Args:
            colors: RGB(A) colors
            factor: Brightness factor (>1 = brighter, <1 = darker)

        Returns:
            Adjusted colors
        """
        adjusted = colors.astype(np.float32) * factor
        return np.clip(adjusted, 0, 255).astype(np.uint8)


def apply_ambient_occlusion(
    colors: np.ndarray,
    ao_values: np.ndarray,
    strength: float = 0.5
) -> np.ndarray:
    """
    Apply ambient occlusion darkening to colors.

    Args:
        colors: RGB(A) colors of shape (N, 3/4)
        ao_values: AO values (0 = fully occluded, 1 = fully lit)
        strength: AO effect strength

    Returns:
        Colors with AO applied
    """
    ao_factor = 1.0 - (1.0 - ao_values[:, np.newaxis]) * strength
    result = colors.copy().astype(np.float32)
    result[:, :3] *= ao_factor
    return np.clip(result, 0, 255).astype(np.uint8)
