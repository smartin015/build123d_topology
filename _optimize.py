# SPDX-License-Identifier: GPL-3.0-or-later
"""High-level optimisation runner."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import trimesh

from .core.simp import Problem as _SIMProblem
from .core.extract import surface_nets
from ._voxelize import voxelize
from ._result import Result


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

class _BC:
    """Internal normalised boundary condition."""
    __slots__ = ("nodes", "fix_x", "fix_y", "fix_z", "force")

    def __init__(self, nodes, fix_x=False, fix_y=False, fix_z=False,
                 force=None):
        self.nodes = np.asarray(nodes, dtype=np.int64)
        self.fix_x = fix_x
        self.fix_y = fix_y
        self.fix_z = fix_z
        self.force = np.asarray(force, dtype=float) if force is not None else None


def _resolve_region(mesh: trimesh.Trimesh, node_coords, region: dict):
    """Turn a user-friendly region dict into a node mask.

    Supported keys
    ---------------
    center : (3,) float
        World-space centre of the region.
    normal : (3,) float
        Direction: nodes on faces whose normal points in this direction
        are selected.  Also used as the force direction for loads.
    radius : float
        Max distance from *center* (mm).  Nodes beyond this are excluded.
    angle : float
        Max angle (degrees) between a node-to-center vector and the
        *opposite* of *normal*.  Default 90 (half-space).
    """
    center = np.asarray(region["center"], dtype=float)
    normal = np.asarray(region.get("normal", (1, 0, 0)), dtype=float)
    normal = normal / np.linalg.norm(normal)
    radius = float(region.get("radius", np.inf))
    angle = np.radians(float(region.get("angle", 90)))

    vec = node_coords - center
    dist = np.linalg.norm(vec, axis=1)

    # Cosine similarity with the *reverse* of normal (points "into" the face)
    dot = np.dot(vec, -normal) / np.maximum(dist, 1e-9)

    mask = (dist <= radius) & (dot >= np.cos(angle))
    return np.where(mask)[0]


def _build_boundary_conditions(mesh, node_coords, vsize,
                                fixed: Sequence[dict],
                                loads: Sequence[dict]):
    """Convert user BC specs into fixed_dofs array and force vector.

    Returns
    -------
    fixed_dofs : (F,) int64
    force : (ndof,) float64
    """
    ndof = 3 * len(node_coords)
    fixed_dofs = []
    force = np.zeros(ndof)

    # Bookmarks let the user write ``fixed=["left", "right", …]``.
    BOOKMARKS = _face_bookmarks(mesh.bounds)

    for spec in fixed:
        if isinstance(spec, str):
            spec = BOOKMARKS.get(spec, {})
            if not spec:
                raise ValueError(
                    f"Unknown bookmark '{spec}'. "
                    f"Known: {list(BOOKMARKS)}")
        nodes = _resolve_region(mesh, node_coords, spec)
        fix_x = bool(spec.get("fix_x", True))
        fix_y = bool(spec.get("fix_y", True))
        fix_z = bool(spec.get("fix_z", True))
        for nid in nodes:
            if fix_x:
                fixed_dofs.append(3 * nid)
            if fix_y:
                fixed_dofs.append(3 * nid + 1)
            if fix_z:
                fixed_dofs.append(3 * nid + 2)

    for spec in loads:
        if isinstance(spec, str):
            spec = BOOKMARKS.get(spec, {})
            if not spec:
                raise ValueError(
                    f"Unknown bookmark '{spec}'. "
                    f"Known: {list(BOOKMARKS)}")
        nodes = _resolve_region(mesh, node_coords, spec)
        fvec = np.asarray(spec.get("force", (0, 0, -1)), dtype=float)
        if len(nodes):
            f_per_node = fvec / len(nodes)
            for nid in nodes:
                force[3 * nid:3 * nid + 3] += f_per_node

    fixed_dofs = (np.unique(np.asarray(fixed_dofs, dtype=np.int64))
                  if fixed_dofs else np.zeros(0, dtype=np.int64))
    return fixed_dofs, force


def _face_bookmarks(bounds):
    """Return a dict of convenience bookmarks for the six AABB faces.

    Each value is a ``dict(center=…, normal=…, radius=…, angle=…)``
    that selects a thin slab of nodes at that face.
    """
    bmin, bmax = np.asarray(bounds[0]), np.asarray(bounds[1])
    ext = bmax - bmin
    eps = 1e-3 * float(np.max(ext))
    cx, cy, cz = 0.5 * (bmin + bmax)
    return {
        "left":   dict(center=(bmin[0] + eps, cy, cz),
                       normal=(-1, 0, 0), radius=ext[0] * 0.55, angle=90),
        "right":  dict(center=(bmax[0] - eps, cy, cz),
                       normal=(1, 0, 0), radius=ext[0] * 0.55, angle=90),
        "front":  dict(center=(cx, bmin[1] + eps, cz),
                       normal=(0, -1, 0), radius=ext[1] * 0.55, angle=90),
        "back":   dict(center=(cx, bmax[1] - eps, cz),
                       normal=(0, 1, 0), radius=ext[1] * 0.55, angle=90),
        "bottom": dict(center=(cx, cy, bmin[2] + eps),
                       normal=(0, 0, -1), radius=ext[2] * 0.55, angle=90),
        "top":    dict(center=(cx, cy, bmax[2] - eps),
                       normal=(0, 0, 1), radius=ext[2] * 0.55, angle=90),
    }


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

class Optimizer:
    """Thin stateful wrapper so you can tweak parameters between runs."""

    def __init__(self, build_mesh: trimesh.Trimesh, **kwargs):
        self.build_mesh = build_mesh
        self.kwargs = kwargs
        self._last_result: Result | None = None

    def run(self, **overrides) -> Result:
        opts = {**self.kwargs, **overrides}
        self._last_result = optimize(self.build_mesh, **opts)
        return self._last_result

    @property
    def result(self) -> Result | None:
        return self._last_result


def optimize(
    build_mesh: trimesh.Trimesh,
    *,
    fixed: Sequence[dict | str] = (),
    loads: Sequence[dict | str] = (),
    exclude: Sequence[trimesh.Trimesh] = (),
    resolution: int = 60,
    volfrac: float = 0.3,
    max_iter: int = 40,
    penalty: float = 3.0,
    rmin: float | None = None,
    style: str = "SMOOTH",
    iso: float = 0.5,
    verbose: bool = True,
) -> Result:
    """Run SIMP topology optimisation on *build_mesh*.

    Parameters
    ----------
    build_mesh : trimesh.Trimesh
        Closed mesh defining the design space.
    fixed : list of dict or str
        Where the part is clamped.  Each entry is either a region dict
        (``center``, ``normal``, ``radius``, ``angle``) or a bookmark
        string (``"left"``, ``"right"``, ``"front"``, ``"back"``,
        ``"top"``, ``"bottom"``).  All DOFs at selected nodes are fixed.
    loads : list of dict or str
        Where external forces act.  Region dicts also need a ``force``
        key ``(fx, fy, fz)``.  The total force is spread evenly across
        all selected nodes.
    exclude : list of trimesh.Trimesh
        Keep-out regions; voxels inside these are forced empty.
    resolution : int
        Voxels along the longest axis of *build_mesh*.
    volfrac : float
        Target volume fraction (0–1).
    max_iter : int
        Maximum SIMP iterations.
    penalty : float
        SIMP penalty exponent (≥ 1, usually 3).
    rmin : float or None
        Density-filter radius in voxels.  ``None`` picks a sensible default.
    style : str
        ``"SMOOTH"`` (surface nets) or ``"BLOCKY"`` (voxel faces).
    iso : float
        Isovalue for the extracted surface.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    Result
    """
    # ---- voxelise --------------------------------------------------------
    if verbose:
        print(f"Voxelising build space (resolution={resolution}) …")

    dims, origin, vsize, active = voxelize(build_mesh, resolution)

    # Exclusions
    if exclude:
        for ex in exclude:
            _, _, _, ex_active = voxelize(ex, resolution,
                                          bbox_min=build_mesh.bounds[0],
                                          bbox_max=build_mesh.bounds[1])
            active = active & ~ex_active

    nx, ny, nz = dims
    active_count = active.sum()
    if verbose:
        print(f"  Grid: {nx}×{ny}×{nz} = {nx*ny*nz:,} voxels "
              f"(vsize={vsize:.3f} mm)")
        print(f"  Active: {active_count:,} "
              f"({100 * active_count / len(active):.1f}%)")

    if active_count == 0:
        raise RuntimeError("No active voxels — check build mesh / exclusions.")

    # ---- node coords -----------------------------------------------------
    nxp, nyp, nzp = nx + 1, ny + 1, nz + 1
    n_nodes = nxp * nyp * nzp
    n_ids = np.arange(n_nodes)
    ix = n_ids % nxp
    iy = (n_ids // nxp) % nyp
    iz = n_ids // (nxp * nyp)
    node_coords = np.stack([ix, iy, iz], axis=1).astype(float) * vsize + origin

    # ---- boundary conditions ---------------------------------------------
    fixed_dofs, force = _build_boundary_conditions(
        build_mesh, node_coords, vsize, fixed, loads)

    if verbose:
        print(f"  Fixed DOFs: {len(fixed_dofs):,}")
        f_nodes = len(np.where(np.abs(force) > 1e-12)[0]) // 3
        print(f"  Loaded nodes: {f_nodes:,}")

    # ---- filter radius ---------------------------------------------------
    if rmin is None:
        rmin = max(1.5, min(nx, ny, nz) / 12.0)
    if verbose:
        print(f"  Filter radius: {rmin:.2f} voxels")

    # ---- run SIMP --------------------------------------------------------
    if verbose:
        print(f"\n{'='*55}")
        print(f"  SIMP | volfrac={volfrac}  penalty={penalty}  "
              f"max_iter={max_iter}")
        print(f"{'='*55}")
        print(f"  {'Iter':>4s}  {'compliance':>12s}  {'change':>8s}")
        print(f"  {'-'*4}  {'-'*12}  {'-'*8}")

    prob = _SIMProblem(
        nx, ny, nz,
        active=active,
        fixed_dofs=fixed_dofs,
        force=force,
        volfrac=volfrac,
        penalty=penalty,
        rmin=rmin,
        nu=0.3,
        use_multigrid=True,
        compute_mode="AUTO",
        verbose=False,
    )

    best_rho = None
    try:
        for it, compliance, change, rho in prob.optimize(max_iter=max_iter,
                                                          tol=0.01):
            if verbose:
                print(f"  {it:4d}  {compliance:12.2f}  {change:8.4f}")
            best_rho = rho.copy()
    finally:
        prob.close()

    if best_rho is None:
        raise RuntimeError("Optimisation produced no result.")

    # ---- extract surface -------------------------------------------------
    if verbose:
        print(f"\n{'='*55}")
        print("Extracting surface ...")

    rho3d = best_rho.reshape(nx, ny, nz)

    if style.upper() == "BLOCKY":
        from .core.extract import cubes_from_density
        verts, faces = cubes_from_density(rho3d, iso, origin, vsize)
    else:
        verts, faces = surface_nets(rho3d, iso, origin, vsize)

    if verbose:
        print(f"  {len(verts):,} vertices, {len(faces):,} faces")

    return Result(verts, faces, best_rho, dims, origin, vsize)
