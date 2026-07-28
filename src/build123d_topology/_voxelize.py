# SPDX-License-Identifier: GPL-3.0-or-later
"""Trimesh-based voxelisation (replaces Blender's BVH path)."""

import numpy as np
import trimesh


def voxelize(mesh: trimesh.Trimesh, resolution: int,
             bbox_min=None, bbox_max=None):
    """Turn a closed mesh into a regular voxel grid.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    resolution : int
        Voxels along the longest axis.
    bbox_min, bbox_max : (3,) float or None
        Override the bounding box used for grid placement.  If *None*,
        ``mesh.bounds`` is used.

    Returns
    -------
    dims : (int, int, int)
    origin : (3,) float
    vsize : float
    active : (nx*ny*nz,) bool
        True for voxels whose centre lies inside the mesh.
    """
    if bbox_min is None:
        bbox_min = mesh.bounds[0]
    if bbox_max is None:
        bbox_max = mesh.bounds[1]

    bmin = np.asarray(bbox_min, dtype=float)
    bmax = np.asarray(bbox_max, dtype=float)
    ext = bmax - bmin
    longest = float(np.max(ext))
    vsize = longest / resolution

    nx = max(1, int(np.ceil(ext[0] / vsize)))
    ny = max(1, int(np.ceil(ext[1] / vsize)))
    nz = max(1, int(np.ceil(ext[2] / vsize)))

    # Center the grid on the AABB
    total_ext = np.array([nx, ny, nz], dtype=float) * vsize
    origin = bmin - 0.5 * (total_ext - ext)

    # Voxel centres
    xs = (np.arange(nx) + 0.5) * vsize + origin[0]
    ys = (np.arange(ny) + 0.5) * vsize + origin[1]
    zs = (np.arange(nz) + 0.5) * vsize + origin[2]
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    centres = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    active = mesh.contains(centres)
    return (nx, ny, nz), origin, vsize, active
