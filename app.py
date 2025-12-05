#!/usr/bin/env python3
"""
Voxel Generator Web Interface

A simple Gradio-based web UI for converting 2D pixel art to 3D voxel models.

Run with: python app.py
Then open http://localhost:7860 in your browser
"""

import sys
from pathlib import Path
import tempfile
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
from voxel_generator import VoxelGenerator


def process_image(
    image,
    depth_mode: str,
    max_depth: int,
    extrusion_mode: str,
    voxel_scale: float,
    alpha_threshold: int,
    export_glb: bool,
    export_vox: bool,
    export_obj: bool
):
    """
    Process an uploaded image and generate voxel models.

    Returns preview path, stats text, and file paths for downloads.
    """
    if image is None:
        return None, "Please upload an image first.", None, None, None

    # Convert to numpy array with RGBA
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            # Grayscale - convert to RGBA
            rgba = np.stack([image, image, image, np.full_like(image, 255)], axis=-1)
        elif image.shape[2] == 3:
            # RGB - add alpha
            rgba = np.concatenate([image, np.full((*image.shape[:2], 1), 255, dtype=np.uint8)], axis=-1)
        else:
            rgba = image.astype(np.uint8)
    else:
        return None, "Invalid image format.", None, None, None

    # Create generator
    generator = VoxelGenerator(
        alpha_threshold=alpha_threshold,
        max_depth=max_depth,
        voxel_scale=voxel_scale
    )

    # Load the image
    generator.load_array(rgba)

    # Set depth mode
    mode_map = {
        "Distance Transform": "distance_transform",
        "Flat": "flat",
        "Luminosity": "luminosity",
        "Gradient X": "gradient_x",
        "Gradient Y": "gradient_y",
        "Symmetry": "symmetry"
    }
    generator.set_depth_mode(mode_map.get(depth_mode, "distance_transform"))

    # Voxelize
    extrusion_map = {"Surface": "surface", "Column": "column", "Shell": "shell"}
    generator.voxelize(extrusion_mode=extrusion_map.get(extrusion_mode, "surface"))

    # Generate mesh
    generator.generate_mesh(greedy=True, center=True)

    # Get stats
    stats = generator.get_mesh_stats()

    stats_text = f"""## Voxelization Complete!

| Metric | Value |
|--------|-------|
| Input Size | {rgba.shape[1]} x {rgba.shape[0]} pixels |
| Voxel Count | {stats['voxel_count']:,} |
| Grid Size | {stats['grid_size']} |
| Vertices | {stats['greedy_vertices']:,} |
| Triangles | {stats['greedy_triangles']:,} |
| Vertex Reduction | {stats['vertex_reduction_percent']:.1f}% |

**Settings:** {depth_mode}, Depth={max_depth}, {extrusion_mode}, Scale={voxel_scale}
"""

    # Create temp directory for exports
    export_dir = tempfile.mkdtemp(prefix="voxel_")

    # Always create GLB for preview
    preview_path = str(Path(export_dir) / "preview.glb")
    generator.export_glb(preview_path)

    glb_path = None
    vox_path = None
    obj_path = None

    # Export requested formats for download
    if export_glb:
        glb_path = str(Path(export_dir) / "model.glb")
        generator.export_glb(glb_path)

    if export_vox:
        vox_path = str(Path(export_dir) / "model.vox")
        generator.export_vox(vox_path)

    if export_obj:
        obj_path = str(Path(export_dir) / "model.obj")
        generator.export_obj(obj_path)

    return preview_path, stats_text, glb_path, vox_path, obj_path


def create_demo_image(style: str):
    """Create a demo image for testing."""
    if not style:
        return None

    size = 64
    rgba = np.zeros((size, size, 4), dtype=np.uint8)

    if style == "Circle":
        center = size // 2
        radius = size // 2 - 4
        for y in range(size):
            for x in range(size):
                dist = np.sqrt((x - center)**2 + (y - center)**2)
                if dist < radius:
                    rgba[y, x] = [100, 150, 220, 255]

    elif style == "Square":
        margin = 8
        for y in range(margin, size - margin):
            for x in range(margin, size - margin):
                r = int(255 * x / size)
                g = int(255 * y / size)
                rgba[y, x] = [r, g, 128, 255]

    elif style == "Character":
        cx = size // 2
        # Body
        for y in range(size//4, size - size//6):
            for x in range(cx - size//6, cx + size//6):
                rgba[y, x] = [80, 120, 180, 255]
        # Head
        head_cy = size // 6
        head_r = size // 8
        for y in range(size):
            for x in range(size):
                if np.sqrt((x-cx)**2 + (y-head_cy)**2) < head_r:
                    rgba[y, x] = [220, 180, 150, 255]

    elif style == "Tree":
        cx = size // 2
        # Trunk
        for y in range(size//2, size - 4):
            for x in range(cx - size//10, cx + size//10):
                rgba[y, x] = [101, 67, 33, 255]
        # Foliage
        for y in range(4, size//2 + size//8):
            progress = (y - 4) / (size//2 + size//8 - 4)
            half_w = int(progress * size//3) + 4
            for x in range(cx - half_w, cx + half_w):
                if 0 <= x < size:
                    rgba[y, x] = [34, 139, 34, 255]

    return rgba


# Build the Gradio interface
with gr.Blocks(title="Voxel Generator") as app:

    gr.Markdown("""
    # Voxel Generator
    ### Convert 2D Pixel Art to 3D Voxel Models

    Upload a PNG image or try a demo, adjust the settings, and download your 3D model!
    """)

    with gr.Row():
        # Left column - Input
        with gr.Column(scale=1):
            gr.Markdown("### Input Image")

            image_input = gr.Image(
                label="Upload Image (PNG recommended)",
                type="numpy",
                image_mode="RGBA"
            )

            with gr.Row():
                demo_dropdown = gr.Dropdown(
                    choices=["Circle", "Square", "Character", "Tree"],
                    label="Or try a demo"
                )
                demo_btn = gr.Button("Load Demo")

            gr.Markdown("### Settings")

            depth_mode = gr.Dropdown(
                choices=[
                    "Distance Transform",
                    "Flat",
                    "Luminosity",
                    "Gradient X",
                    "Gradient Y",
                    "Symmetry"
                ],
                value="Distance Transform",
                label="Depth Mode"
            )

            max_depth = gr.Slider(
                minimum=4,
                maximum=64,
                value=16,
                step=4,
                label="Max Depth (voxel layers)"
            )

            extrusion_mode = gr.Dropdown(
                choices=["Surface", "Column", "Shell"],
                value="Surface",
                label="Extrusion Mode"
            )

            voxel_scale = gr.Slider(
                minimum=0.01,
                maximum=1.0,
                value=0.1,
                step=0.01,
                label="Voxel Scale"
            )

            alpha_threshold = gr.Slider(
                minimum=0,
                maximum=255,
                value=127,
                step=1,
                label="Alpha Threshold"
            )

            gr.Markdown("### Export Formats")
            with gr.Row():
                export_glb = gr.Checkbox(value=True, label="GLB")
                export_vox = gr.Checkbox(value=True, label="VOX")
                export_obj = gr.Checkbox(value=False, label="OBJ")

            generate_btn = gr.Button("Generate Voxel Model", variant="primary")

        # Middle column - 3D Preview
        with gr.Column(scale=2):
            gr.Markdown("### 3D Preview")
            gr.Markdown("*Click and drag to rotate, scroll to zoom*")

            model_preview = gr.Model3D(
                label="3D Model Preview",
                clear_color=[0.1, 0.1, 0.1, 1.0]
            )

            stats_output = gr.Markdown(
                value="Upload an image and click 'Generate' to see results."
            )

        # Right column - Downloads
        with gr.Column(scale=1):
            gr.Markdown("### Downloads")

            glb_output = gr.File(label="GLB (Godot/Blender)")
            vox_output = gr.File(label="VOX (MagicaVoxel)")
            obj_output = gr.File(label="OBJ (Universal)")

            gr.Markdown("""
            ---
            **Tips:**
            - **Surface** = hollow (edit-friendly)
            - **Column** = solid pillars
            - **Distance Transform** = organic
            - **Flat** = hard edges
            """)

    # Wire up events
    demo_btn.click(
        fn=create_demo_image,
        inputs=[demo_dropdown],
        outputs=[image_input]
    )

    generate_btn.click(
        fn=process_image,
        inputs=[
            image_input,
            depth_mode,
            max_depth,
            extrusion_mode,
            voxel_scale,
            alpha_threshold,
            export_glb,
            export_vox,
            export_obj
        ],
        outputs=[model_preview, stats_output, glb_output, vox_output, obj_output]
    )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Voxel Generator Web Interface")
    print("="*60)
    print("\nStarting server...")
    print("Open http://localhost:7860 in your browser\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
