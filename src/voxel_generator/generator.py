"""
Main VoxelGenerator Class

This is the primary interface for the voxel generation pipeline.
It orchestrates:
1. Image loading and preprocessing
2. Depth estimation
3. Voxelization
4. Mesh generation (Greedy Meshing)
5. Export to various formats

Example Usage:
    generator = VoxelGenerator()
    generator.load_image("sprite.png")
    generator.set_depth_mode("distance_transform", max_depth=16)
    generator.voxelize()
    generator.export_glb("output.glb")
"""

from pathlib import Path
from typing import Union, Optional, Tuple
import numpy as np

from .ingestion import ImageLoader, SpriteSheetLoader
from .depth import DepthEstimator, DepthMode
from .voxelizer import VoxelGrid, Voxelizer
from .greedy_mesh import GreedyMesher, NaiveMesher, MeshData, compare_mesh_stats
from .color import ColorManager, srgb_to_linear
from .projection import CoordinateSystem
from .exporters import VoxExporter, GLTFExporter, OBJExporter


class VoxelGenerator:
    """
    High-level interface for deterministic voxel reconstruction.

    This class provides a streamlined workflow for converting 2D pixel art
    into optimized 3D voxel models.

    Attributes:
        image_loader: The loaded image data
        voxelizer: The voxelization engine
        mesher: The mesh generation engine
        grid: The current voxel grid
        mesh: The current mesh data
    """

    def __init__(
        self,
        alpha_threshold: int = 127,
        max_depth: int = 32,
        voxel_scale: float = 1.0
    ):
        """
        Initialize the VoxelGenerator.

        Args:
            alpha_threshold: Pixels with alpha > threshold become solid
            max_depth: Maximum depth (z) dimension for voxelization
            voxel_scale: Scale factor for voxel size in output
        """
        self.alpha_threshold = alpha_threshold
        self.max_depth = max_depth
        self.voxel_scale = voxel_scale

        self._image_loader: Optional[ImageLoader] = None
        self._voxelizer: Optional[Voxelizer] = None
        self._mesher: Optional[GreedyMesher] = None
        self._grid: Optional[VoxelGrid] = None
        self._mesh: Optional[MeshData] = None
        self._depth_mode: DepthMode = DepthMode.DISTANCE_TRANSFORM
        self._depth_scale: float = 1.0
        self._depth_invert: bool = False

    def load_image(
        self,
        image_path: Union[str, Path],
        depth_path: Optional[Union[str, Path]] = None
    ) -> "VoxelGenerator":
        """
        Load a pixel art image for voxelization.

        Args:
            image_path: Path to the sprite image (PNG recommended)
            depth_path: Optional path to grayscale depth map

        Returns:
            self for method chaining
        """
        self._image_loader = ImageLoader(self.alpha_threshold)
        self._image_loader.load(image_path, depth_path)

        # Set depth mode to EXPLICIT if depth map provided
        if depth_path is not None:
            self._depth_mode = DepthMode.EXPLICIT

        return self

    def load_array(
        self,
        rgba_array: np.ndarray,
        depth_array: Optional[np.ndarray] = None
    ) -> "VoxelGenerator":
        """
        Load image data from numpy arrays.

        Args:
            rgba_array: RGBA image array of shape (H, W, 4)
            depth_array: Optional grayscale depth array of shape (H, W)

        Returns:
            self for method chaining
        """
        self._image_loader = ImageLoader(self.alpha_threshold)
        self._image_loader.load_from_array(rgba_array, depth_array)

        if depth_array is not None:
            self._depth_mode = DepthMode.EXPLICIT

        return self

    def set_depth_mode(
        self,
        mode: Union[str, DepthMode],
        max_depth: Optional[int] = None,
        scale: float = 1.0,
        invert: bool = False
    ) -> "VoxelGenerator":
        """
        Configure the depth estimation strategy.

        Args:
            mode: Depth mode name or DepthMode enum
                - "flat": Constant depth
                - "distance_transform": EDT for organic shapes
                - "luminosity": Brightness-based
                - "explicit": Use provided depth map
                - "gradient_x": Horizontal gradient
                - "gradient_y": Vertical gradient
                - "symmetry": Symmetric from center
            max_depth: Maximum depth value (overrides constructor value)
            scale: Depth scaling factor
            invert: If True, invert depth values

        Returns:
            self for method chaining
        """
        if isinstance(mode, str):
            mode = DepthMode(mode)

        self._depth_mode = mode
        self._depth_scale = scale
        self._depth_invert = invert

        if max_depth is not None:
            self.max_depth = max_depth

        return self

    def voxelize(
        self,
        extrusion_mode: str = "surface"
    ) -> "VoxelGenerator":
        """
        Convert the loaded image to a voxel grid.

        Args:
            extrusion_mode: How to fill voxels
                - "column": Fill from z=0 to depth (solid pillars)
                - "surface": Single layer at depth (hollow, can see behind)
                - "shell": Top and bottom surfaces only (hollow box)

        Returns:
            self for method chaining
        """
        if self._image_loader is None:
            raise RuntimeError("No image loaded. Call load_image() first.")

        # Initialize voxelizer
        self._voxelizer = Voxelizer(
            tile_width=2.0,
            tile_height=1.0,
            max_depth=self.max_depth,
            depth_mode=self._depth_mode
        )

        # Configure depth estimator
        self._voxelizer.depth_estimator.scale = self._depth_scale
        self._voxelizer.depth_estimator.invert = self._depth_invert

        # Set explicit depth if available
        if self._image_loader.has_depth_map:
            self._voxelizer.set_explicit_depth(self._image_loader.depth_map)

        # Perform voxelization
        self._grid = self._voxelizer.voxelize(
            color_image=self._image_loader.color_image,
            alpha_mask=self._image_loader.alpha_mask,
            extrusion_mode=extrusion_mode
        )

        return self

    def voxelize_billboard(self, thickness: int = 1) -> "VoxelGenerator":
        """
        Create a simple flat billboard model.

        This is the fastest mode - just extrudes the sprite by a
        fixed number of layers with no depth estimation.

        Args:
            thickness: Number of voxel layers

        Returns:
            self for method chaining
        """
        if self._image_loader is None:
            raise RuntimeError("No image loaded. Call load_image() first.")

        self._voxelizer = Voxelizer(max_depth=thickness)
        self._grid = self._voxelizer.voxelize_billboard(
            color_image=self._image_loader.color_image,
            alpha_mask=self._image_loader.alpha_mask,
            thickness=thickness
        )

        return self

    def generate_mesh(
        self,
        greedy: bool = True,
        center: bool = True
    ) -> "VoxelGenerator":
        """
        Generate optimized mesh from voxel grid.

        Args:
            greedy: If True, use Greedy Meshing for optimization
            center: If True, center the mesh at origin

        Returns:
            self for method chaining
        """
        if self._grid is None:
            raise RuntimeError("No voxel grid. Call voxelize() first.")

        if greedy:
            self._mesher = GreedyMesher(scale=self.voxel_scale, center=center)
        else:
            self._mesher = NaiveMesher(scale=self.voxel_scale, center=center)

        self._mesh = self._mesher.mesh(self._grid.data)

        return self

    def export_vox(
        self,
        output_path: Union[str, Path],
        quantize: bool = True
    ):
        """
        Export to MagicaVoxel .vox format.

        Args:
            output_path: Output file path
            quantize: Whether to quantize colors to 255
        """
        if self._grid is None:
            raise RuntimeError("No voxel grid. Call voxelize() first.")

        exporter = VoxExporter(quantize_colors=quantize)
        exporter.export(self._grid, output_path)

    def export_glb(
        self,
        output_path: Union[str, Path],
        convert_colors: bool = True,
        coordinate_system: CoordinateSystem = CoordinateSystem.GODOT
    ):
        """
        Export to glTF 2.0 binary format (.glb).

        Args:
            output_path: Output file path
            convert_colors: Convert sRGB to Linear
            coordinate_system: Target coordinate system
        """
        if self._mesh is None:
            self.generate_mesh()

        exporter = GLTFExporter(
            convert_colors=convert_colors,
            coordinate_system=coordinate_system,
            scale=1.0  # Mesh is already scaled by the mesher
        )
        exporter.export(self._mesh, output_path)

    def export_gltf(self, output_path: Union[str, Path], **kwargs):
        """Alias for export_glb."""
        self.export_glb(output_path, **kwargs)

    def export_obj(
        self,
        output_path: Union[str, Path],
        include_colors: bool = True,
        coordinate_system: CoordinateSystem = CoordinateSystem.BLENDER
    ):
        """
        Export to Wavefront OBJ format.

        Args:
            output_path: Output file path
            include_colors: Include vertex colors (extended format)
            coordinate_system: Target coordinate system
        """
        if self._mesh is None:
            self.generate_mesh()

        exporter = OBJExporter(
            coordinate_system=coordinate_system,
            scale=1.0,  # Mesh is already scaled by the mesher
            vertex_colors_mode="extended" if include_colors else "none"
        )
        exporter.export(self._mesh, output_path)

    def export_all(
        self,
        base_path: Union[str, Path],
        formats: Optional[list] = None
    ):
        """
        Export to multiple formats at once.

        Args:
            base_path: Base file path (without extension)
            formats: List of formats to export (default: all)
        """
        base_path = Path(base_path)
        formats = formats or ["vox", "glb", "obj"]

        if "vox" in formats:
            self.export_vox(base_path.with_suffix(".vox"))

        if "glb" in formats or "gltf" in formats:
            self.export_glb(base_path.with_suffix(".glb"))

        if "obj" in formats:
            self.export_obj(base_path.with_suffix(".obj"))

    @property
    def grid(self) -> Optional[VoxelGrid]:
        """Get the current voxel grid."""
        return self._grid

    @property
    def mesh(self) -> Optional[MeshData]:
        """Get the current mesh data."""
        return self._mesh

    @property
    def voxel_count(self) -> int:
        """Get the number of solid voxels."""
        if self._grid is None:
            return 0
        return self._grid.count_voxels()

    @property
    def vertex_count(self) -> int:
        """Get the number of mesh vertices."""
        if self._mesh is None:
            return 0
        return len(self._mesh.vertices)

    @property
    def triangle_count(self) -> int:
        """Get the number of mesh triangles."""
        if self._mesh is None:
            return 0
        return len(self._mesh.indices) // 3

    def get_mesh_stats(self) -> dict:
        """
        Get mesh statistics including greedy meshing effectiveness.

        Returns:
            Dictionary with mesh statistics
        """
        if self._grid is None:
            return {"error": "No voxel grid"}

        # Generate both greedy and naive meshes for comparison
        greedy = GreedyMesher(scale=self.voxel_scale)
        naive = NaiveMesher(scale=self.voxel_scale)

        greedy_mesh = greedy.mesh(self._grid.data)
        naive_mesh = naive.mesh(self._grid.data)

        stats = compare_mesh_stats(greedy_mesh, naive_mesh)
        stats["voxel_count"] = self._grid.count_voxels()
        stats["grid_size"] = self._grid.shape

        return stats

    def preview(self) -> dict:
        """
        Get a preview of the current state.

        Returns:
            Dictionary with current state information
        """
        info = {
            "image_loaded": self._image_loader is not None,
            "voxelized": self._grid is not None,
            "meshed": self._mesh is not None,
        }

        if self._image_loader:
            info["image_size"] = self._image_loader.size
            info["has_depth_map"] = self._image_loader.has_depth_map

        if self._grid:
            info["grid_size"] = self._grid.shape
            info["voxel_count"] = self._grid.count_voxels()

        if self._mesh:
            info["vertex_count"] = len(self._mesh.vertices)
            info["triangle_count"] = len(self._mesh.indices) // 3

        return info


class BatchProcessor:
    """
    Batch processing for multiple sprites/frames.

    Use this for processing sprite sheets or multiple images
    with consistent settings.
    """

    def __init__(self, **generator_kwargs):
        """
        Initialize the batch processor.

        Args:
            **generator_kwargs: Arguments passed to VoxelGenerator
        """
        self.generator_kwargs = generator_kwargs
        self._results = []

    def process_sprite_sheet(
        self,
        image_path: Union[str, Path],
        frame_width: int,
        frame_height: int,
        output_dir: Union[str, Path],
        depth_mode: str = "distance_transform",
        formats: list = None
    ) -> list:
        """
        Process a sprite sheet and export each frame.

        Args:
            image_path: Path to sprite sheet
            frame_width: Width of each frame
            frame_height: Height of each frame
            output_dir: Output directory
            depth_mode: Depth estimation mode
            formats: Export formats

        Returns:
            List of output file paths
        """
        formats = formats or ["glb"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load sprite sheet
        loader = SpriteSheetLoader()
        loader.load(image_path)
        loader.split_grid(frame_width, frame_height)

        outputs = []

        for i in range(loader.frame_count):
            frame_loader = loader.create_frame_loader(i)

            generator = VoxelGenerator(**self.generator_kwargs)
            generator._image_loader = frame_loader
            generator.set_depth_mode(depth_mode)
            generator.voxelize()
            generator.generate_mesh()

            base_path = output_dir / f"frame_{i:04d}"
            generator.export_all(base_path, formats)

            outputs.append(str(base_path))

        return outputs

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        pattern: str = "*.png",
        **kwargs
    ) -> list:
        """
        Process all images in a directory.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            pattern: Glob pattern for input files
            **kwargs: Arguments passed to process_image

        Returns:
            List of output file paths
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = []

        for image_path in input_dir.glob(pattern):
            generator = VoxelGenerator(**self.generator_kwargs)
            generator.load_image(image_path)
            generator.set_depth_mode(kwargs.get("depth_mode", "distance_transform"))
            generator.voxelize()
            generator.generate_mesh()

            base_name = image_path.stem
            base_path = output_dir / base_name
            generator.export_all(base_path, kwargs.get("formats", ["glb"]))

            outputs.append(str(base_path))

        return outputs
