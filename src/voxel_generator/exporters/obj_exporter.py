"""
Wavefront OBJ Format Exporter

OBJ is a universal text-based format supported by virtually all 3D software.
While it doesn't natively support vertex colors, we provide options for:
- Geometry-only export
- MTL file with per-face materials (approximation)
- Extended format with vertex colors (v x y z r g b)

Limitations:
- Text format = larger file sizes
- No native vertex color support
- Requires MTL file for materials
"""

from pathlib import Path
from typing import Union, Optional, List, Tuple
import numpy as np

from ..greedy_mesh import MeshData
from ..projection import CoordinateSystem, transform_vertices


class OBJExporter:
    """
    Export mesh data to Wavefront OBJ format.

    Supports:
    - Standard OBJ with MTL materials
    - Extended OBJ with vertex colors (v x y z r g b)
    - Coordinate system transformation
    """

    def __init__(
        self,
        coordinate_system: CoordinateSystem = CoordinateSystem.BLENDER,
        scale: float = 1.0,
        include_normals: bool = True,
        vertex_colors_mode: str = "extended"
    ):
        """
        Initialize the exporter.

        Args:
            coordinate_system: Target coordinate system
            scale: Scale factor for vertex positions
            include_normals: Whether to include vertex normals
            vertex_colors_mode: How to handle vertex colors
                - "none": No colors
                - "extended": v x y z r g b format
                - "mtl": Generate MTL file with materials
        """
        self.coordinate_system = coordinate_system
        self.scale = scale
        self.include_normals = include_normals
        self.vertex_colors_mode = vertex_colors_mode

    def export(
        self,
        mesh: MeshData,
        output_path: Union[str, Path],
        model_name: str = "voxel_model"
    ):
        """
        Export mesh to OBJ file.

        Args:
            mesh: MeshData from GreedyMesher
            output_path: Output file path (.obj)
            model_name: Name for the model/object
        """
        output_path = Path(output_path)

        if len(mesh.vertices) == 0:
            raise ValueError("Cannot export empty mesh")

        # Prepare vertex data
        vertices = mesh.vertices.copy() * self.scale
        normals = mesh.normals.copy()
        colors = mesh.colors.copy()
        indices = mesh.indices.copy()

        # Transform coordinate system
        if self.coordinate_system != CoordinateSystem.INTERNAL:
            vertices = transform_vertices(
                vertices, CoordinateSystem.INTERNAL, self.coordinate_system
            )
            normals = transform_vertices(
                normals, CoordinateSystem.INTERNAL, self.coordinate_system
            )

        # Generate OBJ content
        lines = []
        lines.append(f"# Voxel Generator OBJ Export")
        lines.append(f"# Vertices: {len(vertices)}")
        lines.append(f"# Triangles: {len(indices) // 3}")
        lines.append("")

        # MTL reference if using materials
        if self.vertex_colors_mode == "mtl":
            mtl_path = output_path.with_suffix('.mtl')
            lines.append(f"mtllib {mtl_path.name}")
            lines.append("")

        lines.append(f"o {model_name}")
        lines.append("")

        # Vertices
        if self.vertex_colors_mode == "extended":
            # Extended format with colors
            for i, (v, c) in enumerate(zip(vertices, colors)):
                r, g, b = c[0] / 255.0, c[1] / 255.0, c[2] / 255.0
                lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r:.4f} {g:.4f} {b:.4f}")
        else:
            # Standard format
            for v in vertices:
                lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

        lines.append("")

        # Normals
        if self.include_normals:
            # Get unique normals
            unique_normals, normal_indices = np.unique(
                normals, axis=0, return_inverse=True
            )
            for n in unique_normals:
                lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
            lines.append("")

        # Faces
        if self.vertex_colors_mode == "mtl":
            # Group faces by color for material assignment
            self._export_with_materials(
                lines, indices, colors, normals if self.include_normals else None
            )
        else:
            # Simple face export
            for i in range(0, len(indices), 3):
                i0, i1, i2 = indices[i] + 1, indices[i+1] + 1, indices[i+2] + 1

                if self.include_normals:
                    # Get normal index for first vertex of face
                    ni = normal_indices[indices[i]] + 1
                    lines.append(f"f {i0}//{ni} {i1}//{ni} {i2}//{ni}")
                else:
                    lines.append(f"f {i0} {i1} {i2}")

        # Write OBJ file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        # Write MTL file if needed
        if self.vertex_colors_mode == "mtl":
            self._write_mtl(mesh, output_path.with_suffix('.mtl'))

    def _export_with_materials(
        self,
        lines: List[str],
        indices: np.ndarray,
        colors: np.ndarray,
        normals: Optional[np.ndarray]
    ):
        """Export faces grouped by material."""
        # Get unique face colors
        face_colors = []
        for i in range(0, len(indices), 3):
            # Use first vertex color for the face
            color = tuple(colors[indices[i]][:3])
            face_colors.append(color)

        unique_colors = list(set(face_colors))
        color_to_material = {c: f"material_{i}" for i, c in enumerate(unique_colors)}

        # Group faces by color
        color_faces = {c: [] for c in unique_colors}
        for face_idx, color in enumerate(face_colors):
            color_faces[color].append(face_idx)

        # Export face groups
        for color, face_indices in color_faces.items():
            material_name = color_to_material[color]
            lines.append(f"usemtl {material_name}")

            for face_idx in face_indices:
                i = face_idx * 3
                i0, i1, i2 = indices[i] + 1, indices[i+1] + 1, indices[i+2] + 1

                if normals is not None:
                    unique_normals, normal_indices = np.unique(
                        normals, axis=0, return_inverse=True
                    )
                    ni = normal_indices[indices[i]] + 1
                    lines.append(f"f {i0}//{ni} {i1}//{ni} {i2}//{ni}")
                else:
                    lines.append(f"f {i0} {i1} {i2}")

            lines.append("")

    def _write_mtl(self, mesh: MeshData, mtl_path: Path):
        """Write MTL material file."""
        # Get unique colors
        unique_colors = np.unique(mesh.colors[:, :3], axis=0)

        lines = []
        lines.append("# Voxel Generator MTL Export")
        lines.append("")

        for i, color in enumerate(unique_colors):
            r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
            name = f"material_{i}"

            lines.append(f"newmtl {name}")
            lines.append(f"Kd {r:.4f} {g:.4f} {b:.4f}")  # Diffuse color
            lines.append(f"Ka {r*0.1:.4f} {g*0.1:.4f} {b*0.1:.4f}")  # Ambient
            lines.append("Ks 0.0 0.0 0.0")  # Specular (none for voxels)
            lines.append("Ns 0")  # Specular exponent
            lines.append("d 1.0")  # Opacity
            lines.append("illum 1")  # Illumination model
            lines.append("")

        with open(mtl_path, 'w') as f:
            f.write('\n'.join(lines))

    def export_geometry_only(
        self,
        mesh: MeshData,
        output_path: Union[str, Path]
    ):
        """
        Export geometry without any color information.

        Args:
            mesh: MeshData
            output_path: Output file path
        """
        old_mode = self.vertex_colors_mode
        self.vertex_colors_mode = "none"
        self.export(mesh, output_path)
        self.vertex_colors_mode = old_mode


class PLYExporter:
    """
    Export to PLY format (Stanford Polygon File Format).

    PLY supports vertex colors natively and is well-supported
    by Blender and other 3D software.
    """

    def __init__(
        self,
        binary: bool = True,
        coordinate_system: CoordinateSystem = CoordinateSystem.BLENDER,
        scale: float = 1.0
    ):
        """
        Initialize the PLY exporter.

        Args:
            binary: If True, use binary format (smaller files)
            coordinate_system: Target coordinate system
            scale: Scale factor
        """
        self.binary = binary
        self.coordinate_system = coordinate_system
        self.scale = scale

    def export(
        self,
        mesh: MeshData,
        output_path: Union[str, Path]
    ):
        """
        Export mesh to PLY file.

        Args:
            mesh: MeshData
            output_path: Output file path (.ply)
        """
        output_path = Path(output_path)

        vertices = mesh.vertices.copy() * self.scale
        colors = mesh.colors.copy()
        indices = mesh.indices.copy()

        # Transform coordinates
        if self.coordinate_system != CoordinateSystem.INTERNAL:
            vertices = transform_vertices(
                vertices, CoordinateSystem.INTERNAL, self.coordinate_system
            )

        num_vertices = len(vertices)
        num_faces = len(indices) // 3

        # Build header
        header_lines = [
            "ply",
            "format binary_little_endian 1.0" if self.binary else "format ascii 1.0",
            f"element vertex {num_vertices}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property uchar alpha",
            f"element face {num_faces}",
            "property list uchar int vertex_indices",
            "end_header"
        ]
        header = '\n'.join(header_lines) + '\n'

        if self.binary:
            self._write_binary(output_path, header, vertices, colors, indices)
        else:
            self._write_ascii(output_path, header, vertices, colors, indices)

    def _write_binary(
        self,
        path: Path,
        header: str,
        vertices: np.ndarray,
        colors: np.ndarray,
        indices: np.ndarray
    ):
        """Write binary PLY file."""
        import struct

        with open(path, 'wb') as f:
            f.write(header.encode('ascii'))

            # Write vertices with colors
            for i in range(len(vertices)):
                v = vertices[i]
                c = colors[i]
                f.write(struct.pack('<fff', v[0], v[1], v[2]))
                f.write(struct.pack('<BBBB', c[0], c[1], c[2], c[3]))

            # Write faces
            for i in range(0, len(indices), 3):
                f.write(struct.pack('<B', 3))  # 3 vertices per face
                f.write(struct.pack('<iii', indices[i], indices[i+1], indices[i+2]))

    def _write_ascii(
        self,
        path: Path,
        header: str,
        vertices: np.ndarray,
        colors: np.ndarray,
        indices: np.ndarray
    ):
        """Write ASCII PLY file."""
        with open(path, 'w') as f:
            f.write(header)

            # Write vertices with colors
            for i in range(len(vertices)):
                v = vertices[i]
                c = colors[i]
                f.write(f"{v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]} {c[3]}\n")

            # Write faces
            for i in range(0, len(indices), 3):
                f.write(f"3 {indices[i]} {indices[i+1]} {indices[i+2]}\n")
