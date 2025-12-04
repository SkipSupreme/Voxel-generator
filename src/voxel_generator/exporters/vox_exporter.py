"""
MagicaVoxel .vox Format Exporter

The .vox format is a RIFF-style chunk-based binary format used by MagicaVoxel.
It stores voxels as sparse data with a 256-color palette.

File Structure:
- Header: "VOX " (4 bytes) + version (4 bytes, int32)
- MAIN chunk (container)
  - SIZE chunk: dimensions (x, y, z)
  - XYZI chunk: voxel data (x, y, z, color_index per voxel)
  - RGBA chunk: 256-color palette

Limitations:
- Maximum 256 colors
- Maximum 256x256x256 dimensions per model
- Coordinates are uint8
"""

from pathlib import Path
from typing import Union, Optional, Tuple
import struct
import numpy as np

from ..color import ColorQuantizer


# VOX format constants
VOX_MAGIC = b'VOX '
VOX_VERSION = 150  # Current version


class VoxChunk:
    """Base class for VOX chunks."""

    def __init__(self, chunk_id: bytes):
        self.chunk_id = chunk_id
        self.content = b''
        self.children = b''

    def pack(self) -> bytes:
        """Pack the chunk into bytes."""
        content_size = len(self.content)
        children_size = len(self.children)

        return (
            self.chunk_id +
            struct.pack('<II', content_size, children_size) +
            self.content +
            self.children
        )


class SizeChunk(VoxChunk):
    """SIZE chunk containing model dimensions."""

    def __init__(self, size_x: int, size_y: int, size_z: int):
        super().__init__(b'SIZE')
        # Note: VOX uses x, y, z where z is up
        self.content = struct.pack('<III', size_x, size_y, size_z)


class XYZIChunk(VoxChunk):
    """XYZI chunk containing voxel positions and color indices."""

    def __init__(self):
        super().__init__(b'XYZI')
        self._voxels = []

    def add_voxel(self, x: int, y: int, z: int, color_index: int):
        """
        Add a voxel to the chunk.

        Args:
            x, y, z: Voxel coordinates (0-255)
            color_index: Palette index (1-255, 0 is reserved)
        """
        # Clamp to valid range
        x = max(0, min(255, x))
        y = max(0, min(255, y))
        z = max(0, min(255, z))
        color_index = max(1, min(255, color_index))  # 0 is air

        self._voxels.append((x, y, z, color_index))

    def finalize(self):
        """Build the content bytes from added voxels."""
        num_voxels = len(self._voxels)
        self.content = struct.pack('<I', num_voxels)

        for x, y, z, ci in self._voxels:
            self.content += struct.pack('<BBBB', x, y, z, ci)


class RGBAChunk(VoxChunk):
    """RGBA chunk containing the 256-color palette."""

    def __init__(self):
        super().__init__(b'RGBA')
        # Initialize with default MagicaVoxel palette (or zeros)
        self._palette = np.zeros((256, 4), dtype=np.uint8)
        self._palette[:, 3] = 255  # Default to fully opaque

    def set_color(self, index: int, r: int, g: int, b: int, a: int = 255):
        """
        Set a palette color.

        Args:
            index: Palette index (0-255)
            r, g, b, a: Color components (0-255)
        """
        if 0 <= index < 256:
            self._palette[index] = [r, g, b, a]

    def set_palette(self, palette: np.ndarray):
        """
        Set the entire palette.

        Args:
            palette: Array of shape (N, 3) or (N, 4) with colors
        """
        n = min(len(palette), 256)
        for i in range(n):
            if palette.shape[1] >= 3:
                self._palette[i, :3] = palette[i, :3]
            if palette.shape[1] >= 4:
                self._palette[i, 3] = palette[i, 3]

    def finalize(self):
        """Build the content bytes from the palette."""
        # VOX palette format: 256 * RGBA (1024 bytes)
        # Note: palette index 0 is unused (represents air)
        self.content = self._palette.tobytes()


class MainChunk(VoxChunk):
    """MAIN container chunk."""

    def __init__(self):
        super().__init__(b'MAIN')

    def add_child(self, chunk: VoxChunk):
        """Add a child chunk."""
        self.children += chunk.pack()


class VoxExporter:
    """
    Export voxel data to MagicaVoxel .vox format.

    Usage:
        exporter = VoxExporter()
        exporter.export(voxel_grid, "output.vox")
    """

    def __init__(self, quantize_colors: bool = True):
        """
        Initialize the exporter.

        Args:
            quantize_colors: If True, quantize colors to 255 max
        """
        self.quantize_colors = quantize_colors
        self._quantizer = ColorQuantizer(max_colors=255)

    def export(
        self,
        grid,  # VoxelGrid
        output_path: Union[str, Path],
        palette: Optional[np.ndarray] = None
    ):
        """
        Export a VoxelGrid to .vox format.

        Args:
            grid: VoxelGrid instance
            output_path: Output file path
            palette: Optional pre-defined palette (will auto-generate if None)
        """
        output_path = Path(output_path)

        # Get sparse voxel data
        coords, colors = grid.to_sparse()

        if len(coords) == 0:
            raise ValueError("Cannot export empty voxel grid")

        # Crop to occupied bounds
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        size = max_coords - min_coords + 1

        # Validate dimensions
        if any(s > 256 for s in size):
            raise ValueError(
                f"VOX format limited to 256x256x256. Grid size: {size}"
            )

        # Offset coordinates to start at 0
        coords = coords - min_coords

        # Generate or quantize palette
        if palette is None:
            rgb_colors = colors[:, :3]
            if self.quantize_colors:
                palette, color_indices = self._quantizer.quantize(rgb_colors)
            else:
                # Direct mapping (if under 255 unique colors)
                unique, color_indices = np.unique(
                    rgb_colors, axis=0, return_inverse=True
                )
                palette = unique

            # Shift indices by 1 (index 0 is air in VOX)
            color_indices = color_indices + 1
        else:
            # Map colors to provided palette
            color_indices = np.zeros(len(colors), dtype=np.int32)
            for i, color in enumerate(colors[:, :3]):
                dists = np.sum((palette[:, :3].astype(np.float32) - color) ** 2, axis=1)
                color_indices[i] = np.argmin(dists) + 1  # +1 for VOX indexing

        # Build chunks
        size_chunk = SizeChunk(int(size[0]), int(size[1]), int(size[2]))

        xyzi_chunk = XYZIChunk()
        for i in range(len(coords)):
            x, y, z = coords[i]
            ci = color_indices[i]
            xyzi_chunk.add_voxel(int(x), int(y), int(z), int(ci))
        xyzi_chunk.finalize()

        rgba_chunk = RGBAChunk()
        rgba_chunk.set_palette(palette)
        rgba_chunk.finalize()

        # Assemble MAIN chunk
        main_chunk = MainChunk()
        main_chunk.add_child(size_chunk)
        main_chunk.add_child(xyzi_chunk)
        main_chunk.add_child(rgba_chunk)

        # Write file
        with open(output_path, 'wb') as f:
            # Header
            f.write(VOX_MAGIC)
            f.write(struct.pack('<I', VOX_VERSION))

            # Main chunk
            f.write(main_chunk.pack())

    def export_multi_model(
        self,
        grids: list,
        output_path: Union[str, Path],
        offsets: Optional[list] = None
    ):
        """
        Export multiple models to a single .vox file.

        Args:
            grids: List of VoxelGrid instances
            output_path: Output file path
            offsets: Optional list of (x, y, z) offsets for each model
        """
        # For multi-model support, we'd need to implement nTRN, nGRP, nSHP chunks
        # For simplicity, merge all grids into one
        from ..voxelizer import merge_grids

        if offsets is None:
            offsets = [(0, 0, 0) for _ in grids]

        merged = merge_grids(grids, offsets)
        self.export(merged, output_path)


def load_vox(file_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a .vox file.

    Args:
        file_path: Path to .vox file

    Returns:
        Tuple of (dimensions, voxels, palette) where:
        - dimensions: (x, y, z) size
        - voxels: Array of shape (N, 4) with (x, y, z, color_index)
        - palette: Array of shape (256, 4) with RGBA colors
    """
    file_path = Path(file_path)

    with open(file_path, 'rb') as f:
        # Read header
        magic = f.read(4)
        if magic != VOX_MAGIC:
            raise ValueError(f"Invalid VOX file: bad magic {magic}")

        version = struct.unpack('<I', f.read(4))[0]

        dimensions = None
        voxels = None
        palette = np.zeros((256, 4), dtype=np.uint8)
        palette[:, 3] = 255  # Default opaque

        def read_chunk():
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                return None, None, None

            content_size, children_size = struct.unpack('<II', f.read(8))
            content = f.read(content_size)
            # Children are read recursively
            return chunk_id, content, children_size

        # Read MAIN chunk
        main_id, main_content, main_children_size = read_chunk()
        if main_id != b'MAIN':
            raise ValueError("Expected MAIN chunk")

        # Read child chunks
        bytes_read = 0
        while bytes_read < main_children_size:
            chunk_id, content, children_size = read_chunk()
            if chunk_id is None:
                break

            chunk_total = 12 + len(content) + children_size
            bytes_read += chunk_total

            if chunk_id == b'SIZE':
                dimensions = struct.unpack('<III', content[:12])

            elif chunk_id == b'XYZI':
                num_voxels = struct.unpack('<I', content[:4])[0]
                voxels = np.zeros((num_voxels, 4), dtype=np.uint8)
                for i in range(num_voxels):
                    offset = 4 + i * 4
                    voxels[i] = struct.unpack('<BBBB', content[offset:offset+4])

            elif chunk_id == b'RGBA':
                # 256 colors * 4 bytes
                for i in range(256):
                    offset = i * 4
                    palette[i] = struct.unpack('<BBBB', content[offset:offset+4])

            # Skip children bytes if any
            if children_size > 0:
                f.read(children_size)

    return dimensions, voxels, palette
