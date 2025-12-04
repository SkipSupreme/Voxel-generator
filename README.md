# Deterministic Voxel Generator

A high-fidelity pipeline for converting isometric pixel art to 3D voxel models.

## Overview

This tool provides deterministic, rule-based conversion of 2D isometric pixel art into optimized 3D voxel models, outputting industry-standard formats (`.glb`, `.obj`, `.vox`).

Unlike AI-based tools that introduce noise and artifacts, this generator uses mathematical projection and explicit depth heuristics to produce "pixel-perfect" results suitable for game engines like Godot and 3D modeling software like Blender.

## Key Features

- **Pixel-Perfect Accuracy**: 2:1 dimetric projection mathematics ensures voxels align exactly with source pixels
- **Multiple Depth Modes**: Distance Transform, luminosity-based, explicit depth maps, and more
- **High-Performance Meshing**: Greedy Meshing algorithm with Numba JIT compilation (80-95% vertex reduction)
- **Clean Geometry**: Manifold, optimized topology with no artifacts
- **Multiple Export Formats**: MagicaVoxel (`.vox`), glTF 2.0 (`.glb`), Wavefront (`.obj`)
- **Game Engine Ready**: Proper coordinate system transformations for Godot, Blender, etc.

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install numpy numba Pillow scipy pygltflib
```

## Quick Start

### Command Line

```bash
# Basic conversion
voxgen sprite.png -o model.glb

# With custom depth settings
voxgen sprite.png --depth distance_transform --max-depth 24 -o model

# Using explicit depth map
voxgen sprite.png --depth-map depth.png -o model --format vox glb obj

# Batch processing
voxgen --batch sprites/ --output-dir models/ --format glb
```

### Python API

```python
from voxel_generator import VoxelGenerator

# Create generator
generator = VoxelGenerator(max_depth=16)

# Load image
generator.load_image("sprite.png")

# Configure depth estimation
generator.set_depth_mode("distance_transform", scale=1.0)

# Voxelize
generator.voxelize(extrusion_mode="column")

# Generate optimized mesh
generator.generate_mesh(greedy=True)

# Export
generator.export_glb("output.glb")
generator.export_vox("output.vox")
generator.export_obj("output.obj")
```

## Depth Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `flat` | Constant depth | Simple sprites, icons |
| `distance_transform` | EDT-based dome shape | Organic forms, characters |
| `luminosity` | Brightness-based | Clay renders, untextured |
| `explicit` | User-provided depth map | Maximum control |
| `gradient_x/y` | Linear gradient | Architectural elements |
| `symmetry` | Center-symmetric | Bilateral shapes |

## Output Formats

| Format | Extension | Best For |
|--------|-----------|----------|
| glTF 2.0 | `.glb` | Game engines (Godot, Unity) |
| MagicaVoxel | `.vox` | Voxel editing |
| Wavefront | `.obj` | Universal compatibility |

## Architecture

```
src/voxel_generator/
├── projection.py      # 2:1 dimetric math
├── ingestion.py       # Image loading
├── depth.py           # Depth estimation
├── voxelizer.py       # Voxel grid
├── greedy_mesh.py     # Optimized meshing (Numba)
├── color.py           # sRGB/Linear, quantization
├── generator.py       # Main pipeline
├── cli.py             # Command-line interface
└── exporters/
    ├── vox_exporter.py
    ├── gltf_exporter.py
    └── obj_exporter.py
```

## Performance

The Greedy Meshing algorithm, accelerated by Numba JIT compilation, provides:
- **~100x speedup** over pure Python
- **80-95% vertex reduction** for typical models
- Sub-second processing for sprites up to 128x128

| Grid Size | Voxelize | Mesh | Greedy Reduction |
|-----------|----------|------|------------------|
| 32³ | ~10ms | ~20ms | 90-95% |
| 64³ | ~30ms | ~80ms | 85-92% |
| 128³ | ~100ms | ~300ms | 80-90% |

## Comparison with AI Tools

| Feature | This Tool | AI (Meshy/Tripo) |
|---------|-----------|------------------|
| Accuracy | Pixel-perfect | Approximate |
| Geometry | Clean, manifold | Noisy, artifacts |
| Speed | Real-time | 10-60s (cloud) |
| Offline | Yes | No |
| Control | Deterministic | Stochastic |
| Style | Preserves input | May alter |

## Examples

Run the demo script:

```bash
python examples/demo.py
```

This generates test sprites and demonstrates all features.

## License

GPL-3.0 - See LICENSE file.

## Contributing

Contributions welcome! Please ensure code follows existing patterns and includes tests.
