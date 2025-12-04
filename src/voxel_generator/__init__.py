"""
Deterministic Voxel Generator
=============================

A high-fidelity pipeline for isometric pixel art conversion to 3D voxel models.

This package provides deterministic, rule-based conversion of 2D isometric pixel art
into optimized 3D voxel models, outputting industry-standard formats (.gltf, .obj, .vox).

Key Features:
- Pixel-perfect 2:1 dimetric projection mathematics
- Multiple depth estimation heuristics (EDT, luminosity, explicit depth maps)
- High-performance Greedy Meshing with Numba JIT compilation
- Clean, manifold geometry suitable for game engines and 3D modeling
- Export to MagicaVoxel (.vox), glTF 2.0 (.glb), and Wavefront (.obj)

Example Usage:
    from voxel_generator import VoxelGenerator

    generator = VoxelGenerator()
    generator.load_image("sprite.png")
    generator.set_depth_mode("distance_transform", max_depth=16)
    generator.voxelize()
    generator.export_glb("output.glb")
"""

__version__ = "1.0.0"
__author__ = "Voxel Generator Team"

from .generator import VoxelGenerator
from .projection import ProjectionMatrix, IsometricProjection
from .voxelizer import VoxelGrid, Voxelizer
from .greedy_mesh import GreedyMesher
from .color import ColorManager, srgb_to_linear, linear_to_srgb

__all__ = [
    "VoxelGenerator",
    "ProjectionMatrix",
    "IsometricProjection",
    "VoxelGrid",
    "Voxelizer",
    "GreedyMesher",
    "ColorManager",
    "srgb_to_linear",
    "linear_to_srgb",
]
