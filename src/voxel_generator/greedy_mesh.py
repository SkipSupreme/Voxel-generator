"""
Greedy Meshing Algorithm with Numba JIT Compilation

This module implements the Greedy Meshing algorithm for optimal polygon
reduction of voxel geometry. The algorithm merges adjacent faces with
identical properties (normal, color) into larger rectangular quads.

Performance: Numba JIT provides ~100x speedup over pure Python.
Reduction: Typically 80-95% vertex reduction for solid objects.

Algorithm Overview:
1. Face Culling: Only generate faces between solid and air voxels
2. Greedy Sweep: For each 2D slice, merge adjacent faces into quads
3. Emit Geometry: Generate optimized vertex and index data
"""

from typing import List, Tuple, Optional, NamedTuple
from enum import IntEnum
import numpy as np
from numba import njit, prange
from numba.typed import List as TypedList


class FaceDirection(IntEnum):
    """Face normal directions."""
    WEST = 0   # -X
    EAST = 1   # +X
    SOUTH = 2  # -Y
    NORTH = 3  # +Y
    BOTTOM = 4 # -Z
    TOP = 5    # +Z


# Normal vectors for each face direction
FACE_NORMALS = np.array([
    [-1, 0, 0],  # WEST
    [1, 0, 0],   # EAST
    [0, -1, 0],  # SOUTH
    [0, 1, 0],   # NORTH
    [0, 0, -1],  # BOTTOM
    [0, 0, 1],   # TOP
], dtype=np.float32)


class MeshData(NamedTuple):
    """Container for mesh geometry data."""
    vertices: np.ndarray     # (N, 3) float32 positions
    normals: np.ndarray      # (N, 3) float32 normals
    colors: np.ndarray       # (N, 4) uint8 RGBA colors
    indices: np.ndarray      # (M,) uint32 triangle indices


@njit(cache=True)
def _is_solid(voxels: np.ndarray, x: int, y: int, z: int) -> bool:
    """Check if a voxel is solid (alpha > 0)."""
    sx, sy, sz = voxels.shape[:3]
    if x < 0 or x >= sx or y < 0 or y >= sy or z < 0 or z >= sz:
        return False
    return voxels[x, y, z, 3] > 0


@njit(cache=True)
def _colors_equal(c1: np.ndarray, c2: np.ndarray) -> bool:
    """Check if two RGBA colors are equal."""
    return (c1[0] == c2[0] and c1[1] == c2[1] and
            c1[2] == c2[2] and c1[3] == c2[3])


@njit(cache=True)
def _face_visible(
    voxels: np.ndarray,
    x: int, y: int, z: int,
    direction: int
) -> bool:
    """
    Check if a face is visible (exposed to air).

    A face is visible if the voxel on one side is solid
    and the voxel on the other side is air.
    """
    if not _is_solid(voxels, x, y, z):
        return False

    # Check neighbor in the face direction
    if direction == 0:  # WEST (-X)
        return not _is_solid(voxels, x - 1, y, z)
    elif direction == 1:  # EAST (+X)
        return not _is_solid(voxels, x + 1, y, z)
    elif direction == 2:  # SOUTH (-Y)
        return not _is_solid(voxels, x, y - 1, z)
    elif direction == 3:  # NORTH (+Y)
        return not _is_solid(voxels, x, y + 1, z)
    elif direction == 4:  # BOTTOM (-Z)
        return not _is_solid(voxels, x, y, z - 1)
    elif direction == 5:  # TOP (+Z)
        return not _is_solid(voxels, x, y, z + 1)
    return False


@njit(cache=True)
def _greedy_mesh_slice(
    voxels: np.ndarray,
    direction: int,
    slice_pos: int,
    dim1_size: int,
    dim2_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform greedy meshing on a single 2D slice.

    Args:
        voxels: The voxel grid (X, Y, Z, 4)
        direction: Face direction (0-5)
        slice_pos: Position along the slice axis
        dim1_size: Size along first dimension of slice
        dim2_size: Size along second dimension of slice

    Returns:
        Tuple of (quads, colors, counts) arrays
    """
    # Allocate mask for tracking processed faces
    mask = np.zeros((dim1_size, dim2_size), dtype=np.int32)

    # Output storage - preallocate for worst case
    max_quads = dim1_size * dim2_size
    quads = np.zeros((max_quads, 6), dtype=np.int32)  # x, y, z, w, h, direction
    colors = np.zeros((max_quads, 4), dtype=np.uint8)
    quad_count = 0

    # Coordinate mapping based on direction
    # For each direction, we need to map (d1, d2, slice) to (x, y, z)
    for d1 in range(dim1_size):
        d2 = 0
        while d2 < dim2_size:
            # Get 3D coordinates for this slice position
            if direction == 0 or direction == 1:  # X faces
                x, y, z = slice_pos, d1, d2
            elif direction == 2 or direction == 3:  # Y faces
                x, y, z = d1, slice_pos, d2
            else:  # Z faces
                x, y, z = d1, d2, slice_pos

            # Skip if already processed or no face here
            if mask[d1, d2] != 0 or not _face_visible(voxels, x, y, z, direction):
                d2 += 1
                continue

            # Get the color of this face
            color = voxels[x, y, z].copy()

            # Expand width (along d2)
            width = 1
            while d2 + width < dim2_size:
                # Get coordinates for width expansion
                if direction == 0 or direction == 1:
                    nx, ny, nz = slice_pos, d1, d2 + width
                elif direction == 2 or direction == 3:
                    nx, ny, nz = d1, slice_pos, d2 + width
                else:
                    nx, ny, nz = d1, d2 + width, slice_pos

                if (mask[d1, d2 + width] != 0 or
                    not _face_visible(voxels, nx, ny, nz, direction) or
                    not _colors_equal(voxels[nx, ny, nz], color)):
                    break
                width += 1

            # Expand height (along d1)
            height = 1
            done = False
            while d1 + height < dim1_size and not done:
                # Check entire row at this height
                for w in range(width):
                    if direction == 0 or direction == 1:
                        nx, ny, nz = slice_pos, d1 + height, d2 + w
                    elif direction == 2 or direction == 3:
                        nx, ny, nz = d1 + height, slice_pos, d2 + w
                    else:
                        nx, ny, nz = d1 + height, d2 + w, slice_pos

                    if (mask[d1 + height, d2 + w] != 0 or
                        not _face_visible(voxels, nx, ny, nz, direction) or
                        not _colors_equal(voxels[nx, ny, nz], color)):
                        done = True
                        break

                if not done:
                    height += 1

            # Store the quad
            if direction == 0 or direction == 1:
                quads[quad_count] = [slice_pos, d1, d2, height, width, direction]
            elif direction == 2 or direction == 3:
                quads[quad_count] = [d1, slice_pos, d2, height, width, direction]
            else:
                quads[quad_count] = [d1, d2, slice_pos, width, height, direction]

            colors[quad_count] = color
            quad_count += 1

            # Mark the region as processed
            for h in range(height):
                for w in range(width):
                    mask[d1 + h, d2 + w] = 1

            d2 += width

    return quads[:quad_count], colors[:quad_count], np.array([quad_count])


@njit(cache=True, parallel=False)
def _greedy_mesh_direction(
    voxels: np.ndarray,
    direction: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy mesh all slices for a single direction.

    Args:
        voxels: The voxel grid
        direction: Face direction (0-5)

    Returns:
        Tuple of (all_quads, all_colors)
    """
    sx, sy, sz = voxels.shape[:3]

    # Determine slice axis and dimensions
    if direction == 0 or direction == 1:  # X faces
        num_slices = sx
        dim1_size = sy
        dim2_size = sz
    elif direction == 2 or direction == 3:  # Y faces
        num_slices = sy
        dim1_size = sx
        dim2_size = sz
    else:  # Z faces
        num_slices = sz
        dim1_size = sx
        dim2_size = sy

    # Collect all quads
    all_quads = []
    all_colors = []

    for slice_idx in range(num_slices):
        quads, colors, counts = _greedy_mesh_slice(
            voxels, direction, slice_idx, dim1_size, dim2_size
        )
        for i in range(len(quads)):
            all_quads.append(quads[i])
            all_colors.append(colors[i])

    # Convert to arrays
    if len(all_quads) == 0:
        return np.zeros((0, 6), dtype=np.int32), np.zeros((0, 4), dtype=np.uint8)

    result_quads = np.zeros((len(all_quads), 6), dtype=np.int32)
    result_colors = np.zeros((len(all_quads), 4), dtype=np.uint8)

    for i in range(len(all_quads)):
        result_quads[i] = all_quads[i]
        result_colors[i] = all_colors[i]

    return result_quads, result_colors


def _quad_to_vertices(
    quad: np.ndarray,
    color: np.ndarray,
    direction: int,
    scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a quad to vertices, normals, and colors.

    Args:
        quad: [x, y, z, dim1, dim2, direction]
        color: RGBA color
        direction: Face direction
        scale: Vertex scale factor

    Returns:
        (vertices, normals, colors) for 4 vertices
    """
    x, y, z = quad[0], quad[1], quad[2]
    d1, d2 = quad[3], quad[4]

    # Generate 4 corners based on direction
    if direction == 0:  # WEST (-X)
        v0 = [x, y, z]
        v1 = [x, y + d1, z]
        v2 = [x, y + d1, z + d2]
        v3 = [x, y, z + d2]
    elif direction == 1:  # EAST (+X)
        v0 = [x + 1, y, z]
        v1 = [x + 1, y, z + d2]
        v2 = [x + 1, y + d1, z + d2]
        v3 = [x + 1, y + d1, z]
    elif direction == 2:  # SOUTH (-Y)
        v0 = [x, y, z]
        v1 = [x, y, z + d2]
        v2 = [x + d1, y, z + d2]
        v3 = [x + d1, y, z]
    elif direction == 3:  # NORTH (+Y)
        v0 = [x, y + 1, z]
        v1 = [x + d1, y + 1, z]
        v2 = [x + d1, y + 1, z + d2]
        v3 = [x, y + 1, z + d2]
    elif direction == 4:  # BOTTOM (-Z)
        v0 = [x, y, z]
        v1 = [x + d1, y, z]
        v2 = [x + d1, y + d2, z]
        v3 = [x, y + d2, z]
    else:  # TOP (+Z)
        v0 = [x, y, z + 1]
        v1 = [x, y + d2, z + 1]
        v2 = [x + d1, y + d2, z + 1]
        v3 = [x + d1, y, z + 1]

    vertices = np.array([v0, v1, v2, v3], dtype=np.float32) * scale
    normal = FACE_NORMALS[direction]
    normals = np.tile(normal, (4, 1))
    colors = np.tile(color, (4, 1))

    return vertices, normals, colors


class GreedyMesher:
    """
    High-performance greedy meshing for voxel grids.

    This class wraps the Numba-accelerated meshing kernels and
    provides a clean interface for mesh generation.
    """

    def __init__(self, scale: float = 1.0, center: bool = True):
        """
        Initialize the mesher.

        Args:
            scale: Vertex position scale factor (default 1.0 = 1 unit per voxel)
            center: If True, center the mesh at origin
        """
        self.scale = scale
        self.center = center

    def mesh(self, voxels: np.ndarray) -> MeshData:
        """
        Generate optimized mesh from voxel grid.

        Args:
            voxels: Voxel grid array of shape (X, Y, Z, 4)

        Returns:
            MeshData containing vertices, normals, colors, and indices
        """
        if voxels.ndim != 4 or voxels.shape[3] != 4:
            raise ValueError("Voxels must have shape (X, Y, Z, 4)")

        all_vertices = []
        all_normals = []
        all_colors = []
        all_indices = []
        vertex_offset = 0

        # Process each face direction
        for direction in range(6):
            quads, colors = _greedy_mesh_direction(voxels, direction)

            for i in range(len(quads)):
                verts, norms, cols = _quad_to_vertices(
                    quads[i], colors[i], direction, self.scale
                )

                all_vertices.append(verts)
                all_normals.append(norms)
                all_colors.append(cols)

                # Generate two triangles for the quad
                # Triangle 1: 0, 1, 2
                # Triangle 2: 0, 2, 3
                indices = np.array([
                    vertex_offset + 0,
                    vertex_offset + 1,
                    vertex_offset + 2,
                    vertex_offset + 0,
                    vertex_offset + 2,
                    vertex_offset + 3,
                ], dtype=np.uint32)
                all_indices.append(indices)
                vertex_offset += 4

        if not all_vertices:
            # Return empty mesh
            return MeshData(
                vertices=np.zeros((0, 3), dtype=np.float32),
                normals=np.zeros((0, 3), dtype=np.float32),
                colors=np.zeros((0, 4), dtype=np.uint8),
                indices=np.zeros((0,), dtype=np.uint32)
            )

        vertices = np.vstack(all_vertices)
        normals = np.vstack(all_normals)
        colors = np.vstack(all_colors).astype(np.uint8)
        indices = np.concatenate(all_indices)

        # Center the mesh if requested
        if self.center and len(vertices) > 0:
            center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
            vertices = vertices - center

        return MeshData(
            vertices=vertices,
            normals=normals,
            colors=colors,
            indices=indices
        )

    def mesh_from_grid(self, grid) -> MeshData:
        """
        Generate mesh from a VoxelGrid object.

        Args:
            grid: VoxelGrid instance

        Returns:
            MeshData
        """
        return self.mesh(grid.data)


class NaiveMesher:
    """
    Naive meshing for comparison/debugging.

    Generates one cube (12 triangles) per solid voxel.
    Use this only for small grids or debugging.
    """

    def __init__(self, scale: float = 1.0, center: bool = True):
        self.scale = scale
        self.center = center

    def mesh(self, voxels: np.ndarray) -> MeshData:
        """Generate naive mesh (one cube per voxel)."""
        all_vertices = []
        all_normals = []
        all_colors = []
        all_indices = []
        vertex_offset = 0

        sx, sy, sz = voxels.shape[:3]

        for x in range(sx):
            for y in range(sy):
                for z in range(sz):
                    if voxels[x, y, z, 3] == 0:
                        continue

                    color = voxels[x, y, z]

                    # Generate all 6 faces for this voxel
                    for direction in range(6):
                        # Check if face is visible
                        if direction == 0 and x > 0 and voxels[x-1, y, z, 3] > 0:
                            continue
                        if direction == 1 and x < sx-1 and voxels[x+1, y, z, 3] > 0:
                            continue
                        if direction == 2 and y > 0 and voxels[x, y-1, z, 3] > 0:
                            continue
                        if direction == 3 and y < sy-1 and voxels[x, y+1, z, 3] > 0:
                            continue
                        if direction == 4 and z > 0 and voxels[x, y, z-1, 3] > 0:
                            continue
                        if direction == 5 and z < sz-1 and voxels[x, y, z+1, 3] > 0:
                            continue

                        quad = np.array([x, y, z, 1, 1, direction], dtype=np.int32)
                        verts, norms, cols = _quad_to_vertices(
                            quad, color, direction, self.scale
                        )

                        all_vertices.append(verts)
                        all_normals.append(norms)
                        all_colors.append(cols)

                        indices = np.array([
                            vertex_offset + 0,
                            vertex_offset + 1,
                            vertex_offset + 2,
                            vertex_offset + 0,
                            vertex_offset + 2,
                            vertex_offset + 3,
                        ], dtype=np.uint32)
                        all_indices.append(indices)
                        vertex_offset += 4

        if not all_vertices:
            return MeshData(
                vertices=np.zeros((0, 3), dtype=np.float32),
                normals=np.zeros((0, 3), dtype=np.float32),
                colors=np.zeros((0, 4), dtype=np.uint8),
                indices=np.zeros((0,), dtype=np.uint32)
            )

        vertices = np.vstack(all_vertices)
        normals = np.vstack(all_normals)
        colors = np.vstack(all_colors).astype(np.uint8)
        indices = np.concatenate(all_indices)

        if self.center and len(vertices) > 0:
            center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
            vertices = vertices - center

        return MeshData(
            vertices=vertices,
            normals=normals,
            colors=colors,
            indices=indices
        )


def compare_mesh_stats(greedy_mesh: MeshData, naive_mesh: MeshData) -> dict:
    """
    Compare statistics between greedy and naive meshing.

    Args:
        greedy_mesh: MeshData from GreedyMesher
        naive_mesh: MeshData from NaiveMesher

    Returns:
        Dictionary with comparison statistics
    """
    greedy_verts = len(greedy_mesh.vertices)
    naive_verts = len(naive_mesh.vertices)
    greedy_tris = len(greedy_mesh.indices) // 3
    naive_tris = len(naive_mesh.indices) // 3

    reduction_verts = (1 - greedy_verts / naive_verts) * 100 if naive_verts > 0 else 0
    reduction_tris = (1 - greedy_tris / naive_tris) * 100 if naive_tris > 0 else 0

    return {
        "greedy_vertices": greedy_verts,
        "naive_vertices": naive_verts,
        "greedy_triangles": greedy_tris,
        "naive_triangles": naive_tris,
        "vertex_reduction_percent": reduction_verts,
        "triangle_reduction_percent": reduction_tris,
    }
