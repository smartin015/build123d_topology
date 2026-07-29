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


def _to_trimesh(obj) -> trimesh.Trimesh:
    """Convert a build123d shape (or trimesh) to a trimesh.Trimesh.

    Accepts anything with a ``tessellate()`` method (build123d shapes,
    Compounds, etc.) or an existing trimesh.Trimesh passed through as-is.
    """
    if isinstance(obj, trimesh.Trimesh):
        return obj

    tess = getattr(obj, "tessellate", None)
    if tess is None:
        raise TypeError(
            f"Expected a trimesh.Trimesh or an object with a tessellate() "
            f"method (e.g. a build123d Solid), got {type(obj).__name__}")

    verts_raw, faces_raw = tess(0.5)
    try:
        verts = np.array([[float(v.X), float(v.Y), float(v.Z)]
                          for v in verts_raw], dtype=np.float64)
    except (TypeError, AttributeError):
        verts = np.asarray(verts_raw, dtype=np.float64)
    faces = np.asarray(faces_raw, dtype=np.int64)

    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


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


def _resolve_region(mesh, node_coords, region, vsize=1.0):
    """Turn a region spec into a node-index array.

    *region* can be:

    - ``dict`` — ``center``, ``normal``, ``radius``, ``angle`` keys.
    - a build123d ``Face`` — nodes near that face plane are selected.
    - a build123d ``Solid`` / ``Compound`` — nodes whose coordinates fall
      inside the tessellated shape are selected (volumetric region).
    - a ``trimesh.Trimesh`` — same as Solid: nodes inside the mesh.
    """
    # -- dict path ------------------------------------------------------
    if isinstance(region, dict):
        center = np.asarray(region["center"], dtype=float)
        normal = np.asarray(region.get("normal", (1, 0, 0)), dtype=float)
        normal = normal / np.linalg.norm(normal)
        radius = float(region.get("radius", np.inf))
        angle = np.radians(float(region.get("angle", 90)))

        vec = node_coords - center
        dist = np.linalg.norm(vec, axis=1)
        dot = np.dot(vec, -normal) / np.maximum(dist, 1e-9)
        mask = (dist <= radius) & (dot >= np.cos(angle))
        return np.where(mask)[0]

    # -- build123d Face / Solid / trimesh --------------------------------
    if isinstance(region, str):
        raise TypeError(f"String regions must be resolved before "
                        f"_resolve_region (got {region!r})")

    reg_mesh = _to_trimesh(region)

    # Heuristic: a Face tessellates to a flat-ish patch (few vertices,
    # nearly coplanar -> small bounding-box thickness vs extent).
    bbox = reg_mesh.bounds
    ext = bbox[1] - bbox[0]
    is_flat = (np.min(ext) < 0.05 * np.max(ext)) and len(reg_mesh.vertices) < 200

    if is_flat:
        center = reg_mesh.vertices.mean(axis=0)
        normal = reg_mesh.face_normals[0]
        radius = float(np.max(np.linalg.norm(
            reg_mesh.vertices - center, axis=1)))
        radius += vsize * 0.75

        vec = node_coords - center
        dist = np.linalg.norm(vec, axis=1)
        dot = np.dot(vec, -normal) / np.maximum(dist, 1e-9)
        mask = (dist <= radius) & (dot >= 0.0)
        return np.where(mask)[0]

    # Volumetric region: nodes inside the closed mesh.
    inside = reg_mesh.contains(node_coords)
    return np.where(inside)[0]


def _compute_passive_solid(nx, ny, nz, active, fixed_specs, loads_specs,
                           bounds):
    """Compute passive-solid element mask for BC anchor regions.

    Returns a bool array of length nx*ny*nz where True elements are forced
    to density 1.0 (non-design).

    For bookmark strings, we directly mark the outermost layer of elements
    on the corresponding AABB face.  For Face/Solid/trimesh objects we
    defer to BC-node-based marking, which is precise for those types.
    """
    passive = np.zeros(nx * ny * nz, dtype=bool)

    # Bookmarks map to exact element face layers
    BOOKMARK_FACES = {
        "left":   ("ex", 0),
        "right":  ("ex", nx - 1),
        "front":  ("ey", 0),
        "back":   ("ey", ny - 1),
        "bottom": ("ez", 0),
        "top":    ("ez", nz - 1),
    }

    # Collect node coords for Face/Solid-based passive marking
    nxp, nyp, nzp = nx + 1, ny + 1, nz + 1
    n_nodes = nxp * nyp * nzp
    n_ids = np.arange(n_nodes)
    ix_arr = n_ids % nxp
    iy_arr = (n_ids // nxp) % nyp
    iz_arr = n_ids // (nxp * nyp)

    bmin = np.asarray(bounds[0])
    bmax = np.asarray(bounds[1])
    ext = bmax - bmin
    vsize = float(np.max(ext)) / max(nx, ny, nz, 1)  # approximate
    origin = bmin
    node_coords = np.stack([ix_arr, iy_arr, iz_arr], axis=1).astype(float) * vsize + origin

    bc_node_set = set()

    all_specs = [(s, False) for s in fixed_specs] + [(s, True) for s in loads_specs]

    for raw_spec, _is_load in all_specs:
        if isinstance(raw_spec, tuple) and len(raw_spec) == 2:
            spec, force_vec = raw_spec
        else:
            spec = raw_spec

        if isinstance(spec, str):
            # Bookmark: mark the exact face layer
            if spec in BOOKMARK_FACES:
                axis, layer = BOOKMARK_FACES[spec]
                if axis == "ex":
                    passive.reshape(nx, ny, nz, order='F')[layer, :, :] = True
                elif axis == "ey":
                    passive.reshape(nx, ny, nz, order='F')[:, layer, :] = True
                else:
                    passive.reshape(nx, ny, nz, order='F')[:, :, layer] = True
            continue

        # For non-bookmark regions (Face, Solid, trimesh, dict), use
        # the node-based approach: mark elements adjacent to BC nodes
        # that are on the boundary of the active region.
        nodes = _resolve_region(None, node_coords, spec, vsize)
        bc_node_set.update(nodes.tolist())

    # Node-based passive marking for non-bookmark regions
    if bc_node_set:
        _mark_elements_from_nodes(passive, bc_node_set, nx, ny, nz, active)

    # Only mark active elements
    passive = passive & active.astype(bool)
    return passive


def _mark_elements_from_nodes(passive, bc_nodes, nx, ny, nz, active):
    """Mark elements adjacent to bc_nodes, restricted to the boundary of
    the active region.  Modifies *passive* in place."""
    nxp = nx + 1
    nyp = ny + 1
    area = nxp * nyp
    active3d = active.reshape(nx, ny, nz, order='F')

    # Precompute boundary mask: element is on boundary if any 6-neighbour
    # is outside the active region.
    boundary = np.zeros((nx, ny, nz), dtype=bool)
    for axis in range(3):
        nbr_lo = np.roll(active3d, 1, axis=axis)
        nbr_hi = np.roll(active3d, -1, axis=axis)
        if axis == 0:
            nbr_lo[0, :, :] = False
            nbr_hi[-1, :, :] = False
        elif axis == 1:
            nbr_lo[:, 0, :] = False
            nbr_hi[:, -1, :] = False
        else:
            nbr_lo[:, :, 0] = False
            nbr_hi[:, :, -1] = False
        boundary |= active3d & (~nbr_lo | ~nbr_hi)

    for nid in bc_nodes:
        iz = nid // area
        rem = nid % area
        iy = rem // nxp
        ix = rem % nxp

        for ez in (iz - 1, iz):
            if ez < 0 or ez >= nz:
                continue
            for ey in (iy - 1, iy):
                if ey < 0 or ey >= ny:
                    continue
                for ex in (ix - 1, ix):
                    if ex < 0 or ex >= nx:
                        continue
                    if boundary[ex, ey, ez]:
                        passive[ex + nx * (ey + ny * ez)] = True


def _build_boundary_conditions(mesh, node_coords, vsize,
                                fixed: Sequence,
                                loads: Sequence):
    """Convert user BC specs into fixed_dofs array and force vector.

    Returns (fixed_dofs, force).  For passive-solid marking, use
    _compute_passive_solid() separately with the raw specs.
    """
    ndof = 3 * len(node_coords)
    fixed_dofs = []
    force = np.zeros(ndof)

    BOOKMARKS = _face_bookmarks(mesh.bounds)

    def _normalise(spec):
        if isinstance(spec, str):
            d = BOOKMARKS.get(spec)
            if d is None:
                raise ValueError(
                    f"Unknown bookmark '{spec}'. "
                    f"Known: {list(BOOKMARKS)}")
            return d
        return spec

    for spec in fixed:
        spec = _normalise(spec)
        fix_x = fix_y = fix_z = True
        if isinstance(spec, dict):
            fix_x = bool(spec.get("fix_x", True))
            fix_y = bool(spec.get("fix_y", True))
            fix_z = bool(spec.get("fix_z", True))
        nodes = _resolve_region(mesh, node_coords, spec, vsize)
        for nid in nodes:
            if fix_x:
                fixed_dofs.append(3 * nid)
            if fix_y:
                fixed_dofs.append(3 * nid + 1)
            if fix_z:
                fixed_dofs.append(3 * nid + 2)

    for item in loads:
        if isinstance(item, tuple) and len(item) == 2:
            spec, fvec = item
            spec = _normalise(spec)
        else:
            spec = _normalise(item)
            fvec = spec.get("force", (0, 0, -1)) if isinstance(spec, dict) else (0, 0, -1)

        nodes = _resolve_region(mesh, node_coords, spec, vsize)
        fvec = np.asarray(fvec, dtype=float)
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

    def __init__(self, build_mesh, **kwargs):
        self.build_mesh = _to_trimesh(build_mesh)
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
    build_mesh,
    *,
    fixed: Sequence[dict | str] = (),
    loads: Sequence[dict | str] = (),
    exclude: Sequence = (),
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
    build_mesh : trimesh.Trimesh or build123d Shape
        The design space.
    fixed : list of dict or str
        Where the part is clamped.  Bookmarks, dicts, build123d Face/Solid.
    loads : list of dict, str, or (region, force) tuples.
    exclude : list
        Keep-out regions.
    resolution : int
        Voxels along the longest axis.
    volfrac : float
        Target volume fraction (0–1).
    max_iter : int
        Maximum SIMP iterations.
    penalty : float
        SIMP penalty exponent.
    rmin : float or None
        Density-filter radius in voxels.
    style : str
        ``"SMOOTH"`` or ``"BLOCKY"``.
    iso : float
        Isovalue for surface extraction.
    verbose : bool
        Print progress.

    Returns
    -------
    Result
    """
    # ---- normalise input ------------------------------------------------
    build_mesh = _to_trimesh(build_mesh)
    exclude = [_to_trimesh(e) for e in exclude]

    # ---- voxelise --------------------------------------------------------
    if verbose:
        print(f"Voxelising build space (resolution={resolution}) …")

    dims, origin, vsize, active = voxelize(build_mesh, resolution)

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

    # ---- passive-solid elements (BC anchors) ----------------------------
    passive_solid = _compute_passive_solid(
        nx, ny, nz, active, fixed, loads, build_mesh.bounds)

    if verbose:
        n_passive = passive_solid.sum()
        print(f"  Fixed DOFs: {len(fixed_dofs):,}")
        f_nodes = len(np.where(np.abs(force) > 1e-12)[0]) // 3
        print(f"  Loaded nodes: {f_nodes:,}")
        if n_passive:
            print(f"  Passive-solid elements (BC anchors): {n_passive:,}")

    # ---- filter radius ---------------------------------------------------
    if rmin is None:
        # ~8% of the shortest dimension, at least 2.5 elements.
        # Smaller values produce checkerboarded/gray designs that don't
        # binarize; larger values produce simpler (but possibly over-smoothed)
        # topologies.
        rmin = max(2.5, min(nx, ny, nz) / 4.0)
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
        passive_solid=passive_solid,
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

    rho3d = best_rho.reshape(nx, ny, nz, order='F')

    if style.upper() == "BLOCKY":
        from .core.extract import cubes_from_density
        verts, faces = cubes_from_density(rho3d, iso, origin, vsize)
    else:
        verts, faces = surface_nets(rho3d, iso, origin, vsize)

    if verbose:
        print(f"  {len(verts):,} vertices, {len(faces):,} faces")

    return Result(verts, faces, best_rho, dims, origin, vsize)
