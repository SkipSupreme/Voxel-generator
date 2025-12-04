"""
Command-Line Interface for Voxel Generator

Usage:
    voxgen input.png -o output.glb
    voxgen input.png --depth distance_transform --max-depth 16 -o output
    voxgen input.png --depth-map depth.png -o output --format vox glb obj

"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import time

from .generator import VoxelGenerator, BatchProcessor
from .depth import DepthMode
from .projection import CoordinateSystem


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="voxgen",
        description="Deterministic Voxel Generator - Convert 2D pixel art to 3D voxel models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  voxgen sprite.png -o model.glb
      Convert sprite.png to glTF format

  voxgen sprite.png --depth distance_transform --max-depth 24 -o model
      Use distance transform depth with 24 layers

  voxgen sprite.png --depth-map depth.png -o model --format vox glb
      Use explicit depth map, export to VOX and GLB

  voxgen --batch sprites/ --output-dir models/ --format glb
      Batch process all PNGs in sprites directory

Depth Modes:
  flat              - Constant depth for all pixels
  distance_transform - EDT for organic/rounded shapes (default)
  luminosity        - Brightness-based depth
  explicit          - Use provided depth map
  gradient_x        - Horizontal depth gradient
  gradient_y        - Vertical depth gradient
  symmetry          - Symmetric from center
        """
    )

    # Input
    parser.add_argument(
        "input",
        nargs="?",
        help="Input image file (PNG recommended)"
    )

    # Output
    parser.add_argument(
        "-o", "--output",
        help="Output file path (extension determines format, or use --format)"
    )

    # Depth settings
    parser.add_argument(
        "-d", "--depth",
        choices=["flat", "distance_transform", "luminosity", "explicit",
                 "gradient_x", "gradient_y", "symmetry"],
        default="distance_transform",
        help="Depth estimation mode (default: distance_transform)"
    )

    parser.add_argument(
        "--depth-map",
        help="Path to grayscale depth map image"
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=16,
        help="Maximum depth in voxel layers (default: 16)"
    )

    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Depth scaling factor (default: 1.0)"
    )

    parser.add_argument(
        "--invert-depth",
        action="store_true",
        help="Invert depth values"
    )

    # Voxelization settings
    parser.add_argument(
        "--extrusion",
        choices=["column", "surface", "shell", "billboard"],
        default="column",
        help="Voxel extrusion mode (default: column)"
    )

    parser.add_argument(
        "--billboard-thickness",
        type=int,
        default=1,
        help="Thickness for billboard mode (default: 1)"
    )

    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=127,
        help="Alpha threshold for opacity (0-255, default: 127)"
    )

    # Output settings
    parser.add_argument(
        "-f", "--format",
        nargs="+",
        choices=["vox", "glb", "gltf", "obj"],
        default=["glb"],
        help="Output format(s) (default: glb)"
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Output scale factor (default: 1.0)"
    )

    parser.add_argument(
        "--no-center",
        action="store_true",
        help="Don't center the mesh at origin"
    )

    parser.add_argument(
        "--coordinate-system",
        choices=["godot", "blender", "internal"],
        default="godot",
        help="Target coordinate system (default: godot)"
    )

    parser.add_argument(
        "--no-color-convert",
        action="store_true",
        help="Don't convert sRGB to Linear for glTF"
    )

    # Meshing settings
    parser.add_argument(
        "--naive-mesh",
        action="store_true",
        help="Use naive meshing (no optimization, for debugging)"
    )

    # Batch processing
    parser.add_argument(
        "--batch",
        help="Batch process directory of images"
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory for batch processing"
    )

    parser.add_argument(
        "--pattern",
        default="*.png",
        help="File pattern for batch processing (default: *.png)"
    )

    # Sprite sheet
    parser.add_argument(
        "--sprite-sheet",
        action="store_true",
        help="Treat input as sprite sheet"
    )

    parser.add_argument(
        "--frame-width",
        type=int,
        help="Frame width for sprite sheet"
    )

    parser.add_argument(
        "--frame-height",
        type=int,
        help="Frame height for sprite sheet"
    )

    # Misc
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output with statistics"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print mesh statistics"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )

    return parser


def get_coordinate_system(name: str) -> CoordinateSystem:
    """Convert string to CoordinateSystem enum."""
    return {
        "godot": CoordinateSystem.GODOT,
        "blender": CoordinateSystem.BLENDER,
        "internal": CoordinateSystem.INTERNAL,
    }[name]


def process_single(args) -> int:
    """Process a single image file."""
    if not args.input:
        print("Error: No input file specified", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Determine output path
    if args.output:
        output_base = Path(args.output)
    else:
        output_base = input_path.with_suffix("")

    start_time = time.time()

    try:
        # Create generator
        generator = VoxelGenerator(
            alpha_threshold=args.alpha_threshold,
            max_depth=args.max_depth,
            voxel_scale=args.scale
        )

        # Load image
        if args.verbose:
            print(f"Loading: {input_path}")

        generator.load_image(input_path, args.depth_map)

        # Configure depth
        depth_mode = args.depth
        if args.depth_map:
            depth_mode = "explicit"

        generator.set_depth_mode(
            depth_mode,
            scale=args.depth_scale,
            invert=args.invert_depth
        )

        # Voxelize
        if args.verbose:
            print(f"Voxelizing with mode: {args.extrusion}")

        if args.extrusion == "billboard":
            generator.voxelize_billboard(args.billboard_thickness)
        else:
            generator.voxelize(
                extrusion_mode=args.extrusion,
                fill_below=(args.extrusion == "column")
            )

        # Generate mesh
        if args.verbose:
            print("Generating mesh...")

        generator.generate_mesh(
            greedy=not args.naive_mesh,
            center=not args.no_center
        )

        # Print stats if requested
        if args.stats or args.verbose:
            stats = generator.get_mesh_stats()
            print("\nMesh Statistics:")
            print(f"  Voxels: {stats['voxel_count']}")
            print(f"  Grid size: {stats['grid_size']}")
            print(f"  Greedy vertices: {stats['greedy_vertices']}")
            print(f"  Naive vertices: {stats['naive_vertices']}")
            print(f"  Vertex reduction: {stats['vertex_reduction_percent']:.1f}%")
            print(f"  Triangle reduction: {stats['triangle_reduction_percent']:.1f}%")

        # Export
        coord_sys = get_coordinate_system(args.coordinate_system)

        for fmt in args.format:
            if fmt in ("glb", "gltf"):
                output_path = output_base.with_suffix(".glb")
                generator.export_glb(
                    output_path,
                    convert_colors=not args.no_color_convert,
                    coordinate_system=coord_sys
                )
                if args.verbose:
                    print(f"Exported: {output_path}")

            elif fmt == "vox":
                output_path = output_base.with_suffix(".vox")
                generator.export_vox(output_path)
                if args.verbose:
                    print(f"Exported: {output_path}")

            elif fmt == "obj":
                output_path = output_base.with_suffix(".obj")
                generator.export_obj(output_path, coordinate_system=coord_sys)
                if args.verbose:
                    print(f"Exported: {output_path}")

        elapsed = time.time() - start_time
        if args.verbose:
            print(f"\nCompleted in {elapsed:.2f}s")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def process_batch(args) -> int:
    """Process a batch of images."""
    if not args.batch:
        print("Error: No batch directory specified", file=sys.stderr)
        return 1

    batch_dir = Path(args.batch)
    if not batch_dir.is_dir():
        print(f"Error: Batch directory not found: {batch_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else batch_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        processor = BatchProcessor(
            alpha_threshold=args.alpha_threshold,
            max_depth=args.max_depth,
            voxel_scale=args.scale
        )

        outputs = processor.process_directory(
            batch_dir,
            output_dir,
            pattern=args.pattern,
            depth_mode=args.depth,
            formats=args.format
        )

        elapsed = time.time() - start_time
        print(f"Processed {len(outputs)} files in {elapsed:.2f}s")
        print(f"Output directory: {output_dir}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def process_sprite_sheet(args) -> int:
    """Process a sprite sheet."""
    if not args.input:
        print("Error: No input file specified", file=sys.stderr)
        return 1

    if not args.frame_width or not args.frame_height:
        print("Error: --frame-width and --frame-height required for sprite sheets",
              file=sys.stderr)
        return 1

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "frames"

    start_time = time.time()

    try:
        processor = BatchProcessor(
            alpha_threshold=args.alpha_threshold,
            max_depth=args.max_depth,
            voxel_scale=args.scale
        )

        outputs = processor.process_sprite_sheet(
            input_path,
            args.frame_width,
            args.frame_height,
            output_dir,
            depth_mode=args.depth,
            formats=args.format
        )

        elapsed = time.time() - start_time
        print(f"Processed {len(outputs)} frames in {elapsed:.2f}s")
        print(f"Output directory: {output_dir}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Determine mode
    if args.batch:
        return process_batch(args)
    elif args.sprite_sheet:
        return process_sprite_sheet(args)
    else:
        return process_single(args)


if __name__ == "__main__":
    sys.exit(main())
