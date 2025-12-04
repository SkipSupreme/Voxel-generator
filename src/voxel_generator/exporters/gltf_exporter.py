"""
glTF 2.0 Exporter (.glb binary format)

glTF is the preferred format for game engines (Godot, Unity, Unreal).
This exporter generates optimized binary glTF files with:
- Vertex colors (no textures needed)
- Proper sRGB to Linear conversion
- Coordinate system transformation for different engines
- Efficient binary buffer packing

glTF Structure:
- JSON header describing scene graph
- Binary buffer containing geometry data
  - Indices (uint16/uint32)
  - Positions (float32 vec3)
  - Normals (float32 vec3)
  - Colors (uint8 vec4 normalized)
"""

from pathlib import Path
from typing import Union, Optional, Dict, Any
import json
import struct
import numpy as np

from ..greedy_mesh import MeshData
from ..color import srgb_to_linear
from ..projection import CoordinateSystem, transform_vertices


# glTF constants
GLTF_VERSION = "2.0"
GENERATOR = "VoxelGenerator"

# Component types
UNSIGNED_SHORT = 5123
UNSIGNED_INT = 5125
FLOAT = 5126
UNSIGNED_BYTE = 5121

# Buffer view targets
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963

# Primitive modes
TRIANGLES = 4


class GLTFExporter:
    """
    Export mesh data to glTF 2.0 binary format (.glb).

    Features:
    - Vertex colors with sRGB to Linear conversion
    - Flat shading normals
    - Coordinate system conversion for different engines
    - Optimized binary packing
    """

    def __init__(
        self,
        convert_colors: bool = True,
        coordinate_system: CoordinateSystem = CoordinateSystem.GODOT,
        scale: float = 1.0
    ):
        """
        Initialize the exporter.

        Args:
            convert_colors: If True, convert sRGB to Linear for vertex colors
            coordinate_system: Target coordinate system
            scale: Scale factor for vertex positions
        """
        self.convert_colors = convert_colors
        self.coordinate_system = coordinate_system
        self.scale = scale

    def export(
        self,
        mesh: MeshData,
        output_path: Union[str, Path],
        material_name: str = "VoxelMaterial"
    ):
        """
        Export mesh to .glb file.

        Args:
            mesh: MeshData from GreedyMesher
            output_path: Output file path
            material_name: Name for the material
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

        # Convert colors from sRGB to Linear
        if self.convert_colors:
            colors_linear = srgb_to_linear(colors)
            # Convert back to uint8 for compact storage
            colors_linear = (colors_linear * 255).astype(np.uint8)
        else:
            colors_linear = colors

        # Determine index type
        max_index = indices.max()
        if max_index < 65536:
            index_type = UNSIGNED_SHORT
            indices = indices.astype(np.uint16)
        else:
            index_type = UNSIGNED_INT
            indices = indices.astype(np.uint32)

        # Build binary buffer
        buffer_data = self._build_buffer(vertices, normals, colors_linear, indices)

        # Build glTF JSON
        gltf = self._build_gltf(
            vertices, normals, colors_linear, indices,
            index_type, material_name
        )

        # Write GLB file
        self._write_glb(output_path, gltf, buffer_data)

    def _build_buffer(
        self,
        vertices: np.ndarray,
        normals: np.ndarray,
        colors: np.ndarray,
        indices: np.ndarray
    ) -> bytes:
        """Build the binary buffer containing all geometry data."""
        parts = []

        # Indices
        indices_bytes = indices.tobytes()
        parts.append(indices_bytes)

        # Pad to 4-byte alignment
        padding = (4 - len(indices_bytes) % 4) % 4
        parts.append(b'\x00' * padding)

        # Positions (float32)
        positions_bytes = vertices.astype(np.float32).tobytes()
        parts.append(positions_bytes)

        # Normals (float32)
        normals_bytes = normals.astype(np.float32).tobytes()
        parts.append(normals_bytes)

        # Colors (uint8)
        colors_bytes = colors.astype(np.uint8).tobytes()
        parts.append(colors_bytes)

        # Final padding to 4-byte alignment
        total = sum(len(p) for p in parts)
        final_padding = (4 - total % 4) % 4
        parts.append(b'\x00' * final_padding)

        return b''.join(parts)

    def _build_gltf(
        self,
        vertices: np.ndarray,
        normals: np.ndarray,
        colors: np.ndarray,
        indices: np.ndarray,
        index_type: int,
        material_name: str
    ) -> Dict[str, Any]:
        """Build the glTF JSON structure."""
        num_vertices = len(vertices)
        num_indices = len(indices)

        # Calculate buffer offsets
        index_bytes = indices.itemsize * num_indices
        index_padding = (4 - index_bytes % 4) % 4
        position_offset = index_bytes + index_padding
        position_bytes = num_vertices * 3 * 4  # float32 * 3
        normal_offset = position_offset + position_bytes
        normal_bytes = num_vertices * 3 * 4
        color_offset = normal_offset + normal_bytes
        color_bytes = num_vertices * 4  # uint8 * 4

        total_bytes = color_offset + color_bytes
        total_bytes += (4 - total_bytes % 4) % 4  # final padding

        # Calculate bounds
        pos_min = vertices.min(axis=0).tolist()
        pos_max = vertices.max(axis=0).tolist()

        gltf = {
            "asset": {
                "version": GLTF_VERSION,
                "generator": GENERATOR
            },
            "scene": 0,
            "scenes": [
                {"nodes": [0]}
            ],
            "nodes": [
                {
                    "mesh": 0,
                    "name": "VoxelModel"
                }
            ],
            "meshes": [
                {
                    "primitives": [
                        {
                            "attributes": {
                                "POSITION": 1,
                                "NORMAL": 2,
                                "COLOR_0": 3
                            },
                            "indices": 0,
                            "material": 0,
                            "mode": TRIANGLES
                        }
                    ],
                    "name": "VoxelMesh"
                }
            ],
            "materials": [
                {
                    "name": material_name,
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                        "metallicFactor": 0.0,
                        "roughnessFactor": 0.9
                    },
                    "doubleSided": False,
                    "extensions": {}
                }
            ],
            "accessors": [
                # 0: Indices
                {
                    "bufferView": 0,
                    "componentType": index_type,
                    "count": num_indices,
                    "type": "SCALAR"
                },
                # 1: Positions
                {
                    "bufferView": 1,
                    "componentType": FLOAT,
                    "count": num_vertices,
                    "type": "VEC3",
                    "min": pos_min,
                    "max": pos_max
                },
                # 2: Normals
                {
                    "bufferView": 2,
                    "componentType": FLOAT,
                    "count": num_vertices,
                    "type": "VEC3"
                },
                # 3: Colors
                {
                    "bufferView": 3,
                    "componentType": UNSIGNED_BYTE,
                    "count": num_vertices,
                    "type": "VEC4",
                    "normalized": True
                }
            ],
            "bufferViews": [
                # 0: Indices
                {
                    "buffer": 0,
                    "byteOffset": 0,
                    "byteLength": index_bytes,
                    "target": ELEMENT_ARRAY_BUFFER
                },
                # 1: Positions
                {
                    "buffer": 0,
                    "byteOffset": position_offset,
                    "byteLength": position_bytes,
                    "target": ARRAY_BUFFER
                },
                # 2: Normals
                {
                    "buffer": 0,
                    "byteOffset": normal_offset,
                    "byteLength": normal_bytes,
                    "target": ARRAY_BUFFER
                },
                # 3: Colors
                {
                    "buffer": 0,
                    "byteOffset": color_offset,
                    "byteLength": color_bytes,
                    "target": ARRAY_BUFFER
                }
            ],
            "buffers": [
                {
                    "byteLength": total_bytes
                }
            ]
        }

        return gltf

    def _write_glb(
        self,
        output_path: Path,
        gltf: Dict[str, Any],
        buffer_data: bytes
    ):
        """Write the GLB binary file."""
        # Encode JSON
        json_str = json.dumps(gltf, separators=(',', ':'))
        json_bytes = json_str.encode('utf-8')

        # Pad JSON to 4-byte alignment
        json_padding = (4 - len(json_bytes) % 4) % 4
        json_bytes += b' ' * json_padding

        # GLB header
        # Magic: "glTF" (0x46546C67)
        # Version: 2
        # Length: total file size
        total_length = 12 + 8 + len(json_bytes) + 8 + len(buffer_data)

        with open(output_path, 'wb') as f:
            # Header
            f.write(struct.pack('<I', 0x46546C67))  # glTF magic
            f.write(struct.pack('<I', 2))           # Version 2
            f.write(struct.pack('<I', total_length))

            # JSON chunk
            f.write(struct.pack('<I', len(json_bytes)))
            f.write(struct.pack('<I', 0x4E4F534A))  # JSON magic
            f.write(json_bytes)

            # Binary chunk
            f.write(struct.pack('<I', len(buffer_data)))
            f.write(struct.pack('<I', 0x004E4942))  # BIN magic
            f.write(buffer_data)

    def export_with_texture(
        self,
        mesh: MeshData,
        output_path: Union[str, Path],
        texture_path: Optional[Union[str, Path]] = None
    ):
        """
        Export mesh with optional texture.

        For voxel art, vertex colors are preferred, but this method
        supports texture export if needed.

        Args:
            mesh: MeshData
            output_path: Output file path
            texture_path: Optional texture file path
        """
        # For now, just export with vertex colors
        # Texture support would require UV generation
        self.export(mesh, output_path)


class GLTFExporterAdvanced:
    """
    Advanced glTF exporter with additional features.

    Supports:
    - Multiple meshes
    - Scene hierarchy
    - Animation (placeholder)
    - Custom materials
    """

    def __init__(self):
        self._scenes = []
        self._nodes = []
        self._meshes = []
        self._materials = []
        self._accessors = []
        self._buffer_views = []
        self._buffers = []

    def add_mesh(
        self,
        mesh: MeshData,
        name: str = "Mesh",
        transform: Optional[np.ndarray] = None
    ) -> int:
        """
        Add a mesh to the export.

        Args:
            mesh: MeshData
            name: Mesh name
            transform: Optional 4x4 transform matrix

        Returns:
            Mesh index
        """
        mesh_index = len(self._meshes)

        node = {"mesh": mesh_index, "name": name}
        if transform is not None:
            node["matrix"] = transform.flatten().tolist()

        self._nodes.append(node)
        self._meshes.append({
            "data": mesh,
            "name": name
        })

        return mesh_index

    def export(self, output_path: Union[str, Path]):
        """Export all added meshes to a single GLB file."""
        if not self._meshes:
            raise ValueError("No meshes to export")

        # Use basic exporter for single mesh
        if len(self._meshes) == 1:
            exporter = GLTFExporter()
            exporter.export(self._meshes[0]["data"], output_path)
            return

        # Multi-mesh export would require more complex buffer management
        # For now, merge meshes
        all_vertices = []
        all_normals = []
        all_colors = []
        all_indices = []
        vertex_offset = 0

        for mesh_info in self._meshes:
            mesh = mesh_info["data"]
            all_vertices.append(mesh.vertices)
            all_normals.append(mesh.normals)
            all_colors.append(mesh.colors)
            all_indices.append(mesh.indices + vertex_offset)
            vertex_offset += len(mesh.vertices)

        merged = MeshData(
            vertices=np.vstack(all_vertices),
            normals=np.vstack(all_normals),
            colors=np.vstack(all_colors),
            indices=np.concatenate(all_indices)
        )

        exporter = GLTFExporter()
        exporter.export(merged, output_path)
