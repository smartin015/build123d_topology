# SPDX-License-Identifier: GPL-3.0-or-later
"""Result container from a topology optimisation run."""

import os
import tempfile

import numpy as np
import trimesh


class Result:
    """Holds the output of a topology-optimisation run.

    Attributes
    ----------
    mesh : trimesh.Trimesh
        The extracted optimised shape (smoothed surface nets).
    density : np.ndarray
        Raw density field, shape ``(nx, ny, nz)``, values in [0, 1].
    dims : (int, int, int)
        Voxel grid dimensions.
    origin : (float, float, float)
        World-space min corner of the grid.
    vsize : float
        Voxel edge length.
    """

    def __init__(self, verts, faces, density_flat, dims, origin, vsize):
        self._verts = np.asarray(verts, dtype=np.float64)
        self._faces = np.asarray(faces, dtype=np.int64)
        self.density = np.asarray(density_flat).reshape(dims, order='F')
        self.dims = tuple(dims)
        self.origin = np.asarray(origin, dtype=np.float64)
        self.vsize = float(vsize)

        self._mesh = None

    @property
    def mesh(self) -> trimesh.Trimesh:
        """The extracted surface as a ``trimesh.Trimesh`` (lazy, cached)."""
        if self._mesh is None:
            m = trimesh.Trimesh(vertices=self._verts, faces=self._faces,
                                process=False)
            m.update_faces(m.nondegenerate_faces())
            m.remove_unreferenced_vertices()
            m.fix_normals()
            self._mesh = m
        return self._mesh

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def export_stl(self, path: str) -> str:
        """Write the optimised shape as an STL file.  Returns *path*."""
        self.mesh.export(path)
        return path

    def to_build123d_solid(self):
        """Import the mesh into build123d as a ``Solid``.

        Requires build123d to be installed.  Uses a temp-file round-trip
        through STL (build123d can import STL directly).
        """
        import build123d as bd
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tf:
            self.mesh.export(tf.name)
            solid = bd.import_stl(tf.name)
        os.unlink(tf.name)
        return solid

    def __repr__(self):
        v = len(self._verts)
        f = len(self._faces)
        nx, ny, nz = self.dims
        return (f"Result(vertices={v:,}, faces={f:,}, "
                f"grid={nx}×{ny}×{nz}, vsize={self.vsize:.3f})")
