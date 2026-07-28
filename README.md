# build123d_topology

SIMP topology optimisation for [build123d](https://github.com/gumyr/build123d)
— run a cantilever (or any closed-mesh design space) through a matrix-free
finite-element solver and get back a structurally efficient shape.

Pure-numpy core extracted from the
[blendtopo](https://extensions.blender.org/add-ons/blendtopo/) Blender add-on;
the original Blender BVH voxeliser is replaced with
[trimesh](https://github.com/mikedh/trimesh) so the whole pipeline runs
headless — no Blender required.

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-org/build123d_topology.git
cd build123d_topology
```

### 2. Install dependencies

```bash
pip install numpy trimesh rtree threadpoolctl
```

Optionally, install build123d itself if you want to use
`result.to_build123d_solid()`:

```bash
pip install build123d
```

### 3. Install the package

```bash
pip install -e .
```

That's it — `pip install -e .` registers the package and installs its
dependencies.  Verify with:

```bash
python -c "from build123d_topology import optimize; print('OK')"
```

## Quick start

```python
import build123d as bd
from build123d_topology import optimize

# Build space: a 60×20×10 mm beam
build = bd.Box(60, 20, 10)

result = optimize(
    build,
    fixed=["left"],                              # clamp the left face
    loads=[dict(center=(30, 0, -5),              # load at bottom-right
                normal=(0, 0, -1),
                force=(0, 0, -10),
                radius=3)],
    resolution=60,                                # voxels along longest axis
    volfrac=0.3,                                  # keep 30 % of material
    max_iter=50,
)

# Inspect
print(result)                   # Result(vertices=…, faces=…, grid=…)
result.export_stl("beam.stl")   # write STL

# Bring into build123d
solid = result.to_build123d_solid()
# … use solid in a build123d sketch / Part / assembly …
```

## API

### `optimize(build_mesh, *, fixed, loads, exclude, …)`

Main entry point.  Returns a `Result`.

| Parameter    | Type                  | Default  | Notes |
|-------------|-----------------------|----------|-------|
| `build_mesh`| `Solid \| Compound \| trimesh.Trimesh` | — | Design space (anything with ``tessellate()``). |
| `fixed`     | `list[dict | str]`    | `()`     | Where the part is clamped (see below). |
| `loads`     | `list[dict | str]`    | `()`     | Where external forces act. |
| `exclude`   | `list[Solid | trimesh.Trimesh]` |`()`  | Keep-out regions; voxels inside are forced empty. |
| `resolution`| `int`                 | `60`     | Voxels along the longest axis of the build mesh. |
| `volfrac`   | `float`               | `0.3`    | Target volume fraction (0–1). |
| `max_iter`  | `int`                 | `40`     | Maximum SIMP iterations. |
| `penalty`   | `float`               | `3.0`    | SIMP penalty exponent. |
| `rmin`      | `float | None`        | `None`   | Filter radius in voxels; `None` picks a sensible default. |
| `style`     | `str`                 | `"SMOOTH"`| `"SMOOTH"` (surface nets) or `"BLOCKY"` (voxel faces). |
| `iso`       | `float`               | `0.5`    | Isovalue for surface extraction. |
| `verbose`   | `bool`                | `True`   | Print progress. |

### Region specs

Each element of `fixed` and `loads` can be either a **dict** or a
**bookmark string**.

#### Bookmarks

`"left"`, `"right"`, `"front"`, `"back"`, `"top"`, `"bottom"` —
select a thin slab of nodes at the corresponding face of the AABB.

#### Dict keys

| Key      | Type        | Default        | Notes |
|----------|-------------|----------------|-------|
| `center` | `(3,) float`| —              | World-space centre of the region. |
| `normal` | `(3,) float`| `(1, 0, 0)`    | Direction; nodes on faces pointing this way are selected. |
| `radius` | `float`     | `inf`          | Max distance from centre (mm). |
| `angle`  | `float`     | `90`           | Max angle (degrees) between node→center and `-normal`. |
| `force`  | `(3,) float`| `(0, 0, -1)`   | **Loads only.** Force vector; spread evenly across selected nodes. |
| `fix_x`  | `bool`      | `True`         | **Fixed only.** Clamp X displacement. |
| `fix_y`  | `bool`      | `True`         | **Fixed only.** Clamp Y displacement. |
| `fix_z`  | `bool`      | `True`         | **Fixed only.** Clamp Z displacement. |

### `class Optimizer`

Stateful wrapper that remembers the build mesh and defaults so you can
re-run with different parameters.

```python
from build123d_topology import Optimizer

opt = Optimizer(build, resolution=40, volfrac=0.25)
r1 = opt.run(max_iter=20)   # quick coarse pass
r2 = opt.run(max_iter=60)   # finer pass with same setup
```

### `class Result`

| Attribute / method          | Description |
|-----------------------------|-------------|
| `result.mesh`               | Extracted surface as `trimesh.Trimesh` (lazy, cached). |
| `result.density`            | Raw density field `(nx, ny, nz)` float64, values 0–1. |
| `result.dims`               | Voxel grid `(nx, ny, nz)`. |
| `result.origin`             | World min-corner of the grid. |
| `result.vsize`              | Voxel edge length (mm). |
| `result.export_stl(path)`   | Write an STL file. |
| `result.to_build123d_solid()`| Import into build123d as a `Solid`. |

## Requirements

- Python ≥ 3.10
- numpy, trimesh, rtree, threadpoolctl
- build123d (optional, for `to_build123d_solid()`)

## License

GPL-3.0-or-later (inherited from blendtopo).
