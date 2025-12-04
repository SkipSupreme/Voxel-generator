#!/usr/bin/env python3
"""
Voxel Generator Demo Script

This script demonstrates the full voxel generation pipeline by:
1. Creating synthetic test sprites (no external images needed)
2. Running the voxelization pipeline
3. Exporting to all supported formats
4. Printing statistics and comparisons

Run with: python examples/demo.py
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voxel_generator import VoxelGenerator
from voxel_generator.depth import DepthMode
from voxel_generator.greedy_mesh import GreedyMesher, NaiveMesher, compare_mesh_stats
from voxel_generator.voxelizer import VoxelGrid


def create_test_sprite_circle(size: int = 32) -> tuple:
    """
    Create a simple circular test sprite.

    Returns:
        Tuple of (rgba_array, depth_array)
    """
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    depth = np.zeros((size, size), dtype=np.uint8)

    center = size // 2
    radius = size // 2 - 2

    for y in range(size):
        for x in range(size):
            dx = x - center
            dy = y - center
            dist = np.sqrt(dx * dx + dy * dy)

            if dist < radius:
                # Inside circle
                rgba[y, x] = [100, 150, 200, 255]  # Blue-ish
                # Depth based on distance from center (sphere shape)
                depth[y, x] = int((1 - dist / radius) * 255)

    return rgba, depth


def create_test_sprite_square(size: int = 32) -> tuple:
    """
    Create a simple square test sprite with gradient.

    Returns:
        Tuple of (rgba_array, depth_array)
    """
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    depth = np.zeros((size, size), dtype=np.uint8)

    margin = 4
    for y in range(margin, size - margin):
        for x in range(margin, size - margin):
            # Create a color gradient
            r = int(255 * x / size)
            g = int(255 * y / size)
            b = 128
            rgba[y, x] = [r, g, b, 255]

            # Flat depth
            depth[y, x] = 128

    return rgba, depth


def create_test_sprite_character(size: int = 32) -> tuple:
    """
    Create a simple character-like test sprite.

    Returns:
        Tuple of (rgba_array, depth_array)
    """
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    depth = np.zeros((size, size), dtype=np.uint8)

    cx = size // 2

    # Body (rectangle)
    body_top = size // 4
    body_bottom = size - size // 6
    body_left = cx - size // 6
    body_right = cx + size // 6

    for y in range(body_top, body_bottom):
        for x in range(body_left, body_right):
            rgba[y, x] = [80, 120, 180, 255]  # Blue body
            depth[y, x] = 200

    # Head (circle)
    head_cy = size // 6
    head_radius = size // 8

    for y in range(size):
        for x in range(size):
            dx = x - cx
            dy = y - head_cy
            dist = np.sqrt(dx * dx + dy * dy)

            if dist < head_radius:
                rgba[y, x] = [220, 180, 150, 255]  # Skin tone
                depth[y, x] = int((1 - dist / head_radius) * 200) + 50

    return rgba, depth


def create_test_sprite_tree(size: int = 32) -> tuple:
    """
    Create a simple tree test sprite.

    Returns:
        Tuple of (rgba_array, depth_array)
    """
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    depth = np.zeros((size, size), dtype=np.uint8)

    cx = size // 2

    # Trunk
    trunk_width = size // 8
    trunk_top = size // 2
    trunk_bottom = size - 2

    for y in range(trunk_top, trunk_bottom):
        for x in range(cx - trunk_width, cx + trunk_width):
            rgba[y, x] = [101, 67, 33, 255]  # Brown
            depth[y, x] = 100

    # Foliage (triangle/cone)
    foliage_top = 2
    foliage_bottom = size // 2 + size // 8

    for y in range(foliage_top, foliage_bottom):
        # Width increases with y
        progress = (y - foliage_top) / (foliage_bottom - foliage_top)
        half_width = int(progress * size // 3) + 2

        for x in range(cx - half_width, cx + half_width):
            if 0 <= x < size:
                rgba[y, x] = [34, 139, 34, 255]  # Forest green
                # Depth based on distance from center
                dist_from_center = abs(x - cx)
                depth[y, x] = int((1 - dist_from_center / half_width) * 200) + 50

    return rgba, depth


def run_demo():
    """Run the demonstration."""
    print("=" * 60)
    print("Deterministic Voxel Generator - Demo")
    print("=" * 60)
    print()

    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Test sprites
    test_sprites = [
        ("circle", create_test_sprite_circle(32)),
        ("square", create_test_sprite_square(32)),
        ("character", create_test_sprite_character(64)),
        ("tree", create_test_sprite_tree(48)),
    ]

    total_start = time.time()

    for name, (rgba, depth) in test_sprites:
        print(f"\n--- Processing: {name} ---")
        print(f"Input size: {rgba.shape[1]}x{rgba.shape[0]} pixels")

        sprite_start = time.time()

        # Create generator
        generator = VoxelGenerator(
            alpha_threshold=127,
            max_depth=24,
            voxel_scale=0.1
        )

        # Load from array
        generator.load_array(rgba, depth)

        # Test different depth modes
        print("\nTesting depth modes:")

        for mode in [DepthMode.DISTANCE_TRANSFORM, DepthMode.EXPLICIT]:
            mode_name = mode.value

            # Set depth mode
            generator.set_depth_mode(mode)

            # Voxelize
            vox_start = time.time()
            generator.voxelize(extrusion_mode="column")
            vox_time = time.time() - vox_start

            print(f"  {mode_name}:")
            print(f"    Voxelization: {vox_time*1000:.1f}ms")
            print(f"    Voxel count: {generator.voxel_count}")

            # Generate mesh
            mesh_start = time.time()
            generator.generate_mesh(greedy=True)
            mesh_time = time.time() - mesh_start

            print(f"    Mesh generation: {mesh_time*1000:.1f}ms")
            print(f"    Vertices: {generator.vertex_count}")
            print(f"    Triangles: {generator.triangle_count}")

        # Get mesh stats (comparing greedy vs naive)
        stats = generator.get_mesh_stats()
        print(f"\n  Greedy Meshing Effectiveness:")
        print(f"    Vertex reduction: {stats['vertex_reduction_percent']:.1f}%")
        print(f"    Triangle reduction: {stats['triangle_reduction_percent']:.1f}%")

        # Export to all formats
        print(f"\n  Exporting...")
        base_path = output_dir / name

        export_start = time.time()

        # Use explicit depth for final export
        generator.set_depth_mode(DepthMode.EXPLICIT)
        generator.voxelize()
        generator.generate_mesh()

        try:
            generator.export_vox(base_path.with_suffix(".vox"))
            print(f"    Saved: {base_path.with_suffix('.vox')}")
        except Exception as e:
            print(f"    VOX export failed: {e}")

        try:
            generator.export_glb(base_path.with_suffix(".glb"))
            print(f"    Saved: {base_path.with_suffix('.glb')}")
        except Exception as e:
            print(f"    GLB export failed: {e}")

        try:
            generator.export_obj(base_path.with_suffix(".obj"))
            print(f"    Saved: {base_path.with_suffix('.obj')}")
        except Exception as e:
            print(f"    OBJ export failed: {e}")

        export_time = time.time() - export_start
        sprite_time = time.time() - sprite_start

        print(f"    Export time: {export_time*1000:.1f}ms")
        print(f"    Total time: {sprite_time*1000:.1f}ms")

    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print(f"Demo complete! Total time: {total_time:.2f}s")
    print(f"Output files in: {output_dir}")
    print("=" * 60)

    return 0


def benchmark_greedy_meshing():
    """Benchmark greedy meshing performance."""
    print("\n--- Greedy Meshing Benchmark ---\n")

    sizes = [16, 32, 64, 128]

    for size in sizes:
        # Create a solid cube
        grid = VoxelGrid(size, size, size)
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    grid.set_voxel(x, y, z, 100, 150, 200, 255)

        # Benchmark greedy
        greedy = GreedyMesher()
        start = time.time()
        greedy_mesh = greedy.mesh(grid.data)
        greedy_time = time.time() - start

        # Benchmark naive
        naive = NaiveMesher()
        start = time.time()
        naive_mesh = naive.mesh(grid.data)
        naive_time = time.time() - start

        # Compare
        stats = compare_mesh_stats(greedy_mesh, naive_mesh)

        print(f"Grid size: {size}x{size}x{size}")
        print(f"  Greedy: {greedy_time*1000:.1f}ms, {stats['greedy_vertices']} verts")
        print(f"  Naive:  {naive_time*1000:.1f}ms, {stats['naive_vertices']} verts")
        print(f"  Reduction: {stats['vertex_reduction_percent']:.1f}%")
        print()


if __name__ == "__main__":
    run_demo()

    # Uncomment to run benchmark
    # benchmark_greedy_meshing()
