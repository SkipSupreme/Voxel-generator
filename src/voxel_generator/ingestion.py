"""
Image Ingestion and Preprocessing Module

This module handles:
- Loading pixel art images with strict nearest-neighbor interpolation
- Alpha thresholding for binary opacity (voxels exist or don't)
- Palette extraction for .vox format compatibility
- Depth map loading and validation
"""

from pathlib import Path
from typing import Optional, Tuple, List, Union
import numpy as np
from PIL import Image


class ImageLoader:
    """
    Pixel art image loader with voxelization-specific preprocessing.

    Key features:
    - Strict nearest-neighbor interpolation (no color blurring)
    - Binary alpha thresholding
    - Optional depth map loading
    """

    def __init__(self, alpha_threshold: int = 127):
        """
        Initialize the image loader.

        Args:
            alpha_threshold: Pixels with alpha > threshold become solid (0-255)
        """
        self.alpha_threshold = alpha_threshold
        self._color_image: Optional[np.ndarray] = None
        self._alpha_mask: Optional[np.ndarray] = None
        self._depth_map: Optional[np.ndarray] = None
        self._original_size: Optional[Tuple[int, int]] = None

    def load(
        self,
        image_path: Union[str, Path],
        depth_path: Optional[Union[str, Path]] = None
    ) -> "ImageLoader":
        """
        Load a pixel art image and optional depth map.

        Args:
            image_path: Path to the main color image (PNG recommended)
            depth_path: Optional path to grayscale depth map

        Returns:
            self for method chaining
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load with PIL, convert to RGBA
        img = Image.open(image_path)
        self._original_size = img.size  # (width, height)

        # Ensure RGBA format
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        # Convert to numpy array
        self._color_image = np.array(img, dtype=np.uint8)

        # Extract and threshold alpha channel
        alpha = self._color_image[:, :, 3]
        self._alpha_mask = alpha > self.alpha_threshold

        # Load depth map if provided
        if depth_path is not None:
            self._load_depth_map(depth_path)

        return self

    def _load_depth_map(self, depth_path: Union[str, Path]):
        """
        Load and validate a depth map.

        Args:
            depth_path: Path to grayscale depth map image
        """
        depth_path = Path(depth_path)
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth map not found: {depth_path}")

        # Load depth map
        depth_img = Image.open(depth_path)

        # Convert to grayscale if needed
        if depth_img.mode != "L":
            depth_img = depth_img.convert("L")

        # Validate dimensions match
        if depth_img.size != self._original_size:
            raise ValueError(
                f"Depth map dimensions {depth_img.size} don't match "
                f"color image dimensions {self._original_size}"
            )

        self._depth_map = np.array(depth_img, dtype=np.uint8)

    def load_from_array(
        self,
        rgba_array: np.ndarray,
        depth_array: Optional[np.ndarray] = None
    ) -> "ImageLoader":
        """
        Load from numpy arrays instead of files.

        Args:
            rgba_array: RGBA image array of shape (H, W, 4)
            depth_array: Optional grayscale depth array of shape (H, W)

        Returns:
            self for method chaining
        """
        if rgba_array.ndim != 3 or rgba_array.shape[2] != 4:
            raise ValueError("Color array must have shape (H, W, 4)")

        self._color_image = rgba_array.astype(np.uint8)
        self._original_size = (rgba_array.shape[1], rgba_array.shape[0])

        # Extract and threshold alpha
        alpha = self._color_image[:, :, 3]
        self._alpha_mask = alpha > self.alpha_threshold

        if depth_array is not None:
            if depth_array.shape != rgba_array.shape[:2]:
                raise ValueError("Depth array dimensions must match color array")
            self._depth_map = depth_array.astype(np.uint8)

        return self

    def resize(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        scale: Optional[float] = None
    ) -> "ImageLoader":
        """
        Resize the image using NEAREST NEIGHBOR interpolation.

        WARNING: Only use nearest neighbor for pixel art to preserve colors!

        Args:
            width: Target width (if height not specified, maintains aspect)
            height: Target height (if width not specified, maintains aspect)
            scale: Scale factor (e.g., 2.0 = double size)

        Returns:
            self for method chaining
        """
        if self._color_image is None:
            raise RuntimeError("No image loaded")

        orig_h, orig_w = self._color_image.shape[:2]

        if scale is not None:
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
        elif width is not None and height is not None:
            new_w, new_h = width, height
        elif width is not None:
            new_w = width
            new_h = int(orig_h * width / orig_w)
        elif height is not None:
            new_h = height
            new_w = int(orig_w * height / orig_h)
        else:
            return self

        # Convert back to PIL for resize
        img = Image.fromarray(self._color_image)
        img = img.resize((new_w, new_h), Image.Resampling.NEAREST)
        self._color_image = np.array(img, dtype=np.uint8)

        # Update alpha mask
        alpha = self._color_image[:, :, 3]
        self._alpha_mask = alpha > self.alpha_threshold

        # Resize depth map if present
        if self._depth_map is not None:
            depth_img = Image.fromarray(self._depth_map)
            depth_img = depth_img.resize((new_w, new_h), Image.Resampling.NEAREST)
            self._depth_map = np.array(depth_img, dtype=np.uint8)

        self._original_size = (new_w, new_h)
        return self

    @property
    def color_image(self) -> np.ndarray:
        """Get the RGBA color image array."""
        if self._color_image is None:
            raise RuntimeError("No image loaded")
        return self._color_image

    @property
    def alpha_mask(self) -> np.ndarray:
        """Get the binary alpha mask (True = solid, False = air)."""
        if self._alpha_mask is None:
            raise RuntimeError("No image loaded")
        return self._alpha_mask

    @property
    def depth_map(self) -> Optional[np.ndarray]:
        """Get the depth map if loaded."""
        return self._depth_map

    @property
    def has_depth_map(self) -> bool:
        """Check if a depth map is loaded."""
        return self._depth_map is not None

    @property
    def size(self) -> Tuple[int, int]:
        """Get image size as (width, height)."""
        if self._original_size is None:
            raise RuntimeError("No image loaded")
        return self._original_size

    @property
    def width(self) -> int:
        """Get image width."""
        return self.size[0]

    @property
    def height(self) -> int:
        """Get image height."""
        return self.size[1]

    def get_opaque_pixels(self) -> np.ndarray:
        """
        Get coordinates of all opaque pixels.

        Returns:
            Array of shape (N, 2) with (row, col) indices of opaque pixels
        """
        return np.argwhere(self._alpha_mask)

    def get_pixel_colors(self) -> np.ndarray:
        """
        Get RGB colors of all opaque pixels.

        Returns:
            Array of shape (N, 3) with RGB values (0-255)
        """
        rows, cols = np.where(self._alpha_mask)
        return self._color_image[rows, cols, :3]

    def extract_palette(self, max_colors: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract unique colors from the image.

        Args:
            max_colors: Maximum number of colors to extract

        Returns:
            Tuple of (palette, indices) where:
            - palette: Array of shape (M, 3) with unique RGB colors
            - indices: Array mapping each opaque pixel to a palette index
        """
        colors = self.get_pixel_colors()

        # Get unique colors
        unique_colors, inverse_indices = np.unique(
            colors, axis=0, return_inverse=True
        )

        if len(unique_colors) <= max_colors:
            return unique_colors, inverse_indices

        # Need to quantize - use simple k-means style approach
        from .color import ColorQuantizer
        quantizer = ColorQuantizer(max_colors)
        palette, indices = quantizer.quantize(colors)
        return palette, indices


class SpriteSheetLoader(ImageLoader):
    """
    Extended loader for sprite sheets with multiple frames.

    Supports both regular grids and custom frame definitions.
    """

    def __init__(self, alpha_threshold: int = 127):
        super().__init__(alpha_threshold)
        self._frames: List[np.ndarray] = []
        self._frame_depth_maps: List[Optional[np.ndarray]] = []

    def split_grid(
        self,
        frame_width: int,
        frame_height: int,
        columns: Optional[int] = None,
        rows: Optional[int] = None
    ) -> "SpriteSheetLoader":
        """
        Split the sprite sheet into a grid of frames.

        Args:
            frame_width: Width of each frame in pixels
            frame_height: Height of each frame in pixels
            columns: Number of columns (auto-detected if None)
            rows: Number of rows (auto-detected if None)

        Returns:
            self for method chaining
        """
        if self._color_image is None:
            raise RuntimeError("No image loaded")

        img_h, img_w = self._color_image.shape[:2]

        if columns is None:
            columns = img_w // frame_width
        if rows is None:
            rows = img_h // frame_height

        self._frames = []
        self._frame_depth_maps = []

        for row in range(rows):
            for col in range(columns):
                x = col * frame_width
                y = row * frame_height

                # Extract frame
                frame = self._color_image[y:y + frame_height, x:x + frame_width]

                # Check if frame has any opaque pixels
                if np.any(frame[:, :, 3] > self.alpha_threshold):
                    self._frames.append(frame)

                    # Extract corresponding depth region if available
                    if self._depth_map is not None:
                        depth_frame = self._depth_map[y:y + frame_height, x:x + frame_width]
                        self._frame_depth_maps.append(depth_frame)
                    else:
                        self._frame_depth_maps.append(None)

        return self

    @property
    def frame_count(self) -> int:
        """Get the number of frames extracted."""
        return len(self._frames)

    def get_frame(self, index: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get a specific frame and its depth map.

        Args:
            index: Frame index

        Returns:
            Tuple of (frame_rgba, frame_depth) arrays
        """
        if not self._frames:
            raise RuntimeError("No frames extracted. Call split_grid() first.")
        if index < 0 or index >= len(self._frames):
            raise IndexError(f"Frame index {index} out of range [0, {len(self._frames)})")

        return self._frames[index], self._frame_depth_maps[index]

    def create_frame_loader(self, index: int) -> ImageLoader:
        """
        Create a new ImageLoader for a specific frame.

        Args:
            index: Frame index

        Returns:
            New ImageLoader instance with the frame data
        """
        frame, depth = self.get_frame(index)
        loader = ImageLoader(self.alpha_threshold)
        loader.load_from_array(frame, depth)
        return loader
