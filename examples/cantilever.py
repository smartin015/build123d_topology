#!/usr/bin/env python3
"""
Cantilever beam topology optimisation — example for build123d_topology.

Build space:  60 × 20 × 10 mm box
Supports:     left face fully clamped
Load:         downward force on the bottom-right edge
Target:       30 % volume fraction

Run from the repo root::

    pip install -e .
    python examples/cantilever.py
"""

import trimesh
from build123d_topology import optimize


def main():
    LENGTH, HEIGHT, WIDTH = 60, 20, 10  # mm
    RESOLUTION = 60
    VOLFRAC = 0.3
    MAX_ITER = 50

    print("=" * 60)
    print("Cantilever beam topology optimisation")
    print(f"Build space: {LENGTH}×{HEIGHT}×{WIDTH} mm")
    print(f"Resolution: {RESOLUTION}, volume fraction: {VOLFRAC}")
    print("=" * 60)

    build = trimesh.creation.box((LENGTH, HEIGHT, WIDTH))

    result = optimize(
        build,
        fixed=["left"],
        loads=[dict(
            center=(LENGTH / 2, 0, -WIDTH / 2),
            normal=(0, 0, -1),
            force=(0, 0, -10),
            radius=LENGTH * 0.55,
        )],
        resolution=RESOLUTION,
        volfrac=VOLFRAC,
        max_iter=MAX_ITER,
    )

    print("\n" + "=" * 60)
    print(result)

    result.export_stl("cantilever_optimised.stl")
    print("Wrote cantilever_optimised.stl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
