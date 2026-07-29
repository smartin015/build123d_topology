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

The result is also pushed to the OCP CAD Viewer in VSCode via ``show()``
(requires ``ocp-vscode`` — part of the standard build123d toolchain).
"""

from build123d import *
from build123d_topology import optimize
from ocp_vscode import show

LENGTH, HEIGHT, WIDTH = 60, 20, 10  # mm
RESOLUTION = 60
VOLFRAC = 0.3
MAX_ITER = 10

print("=" * 60)
print("Cantilever beam topology optimisation")
print(f"Build space: {LENGTH}×{HEIGHT}×{WIDTH} mm")
print(f"Resolution: {RESOLUTION}, volume fraction: {VOLFRAC}")
print("=" * 60)

build = Box(LENGTH, HEIGHT, WIDTH)

# Select faces for boundary conditions via build123d's selector API
left_face = build.faces().sort_by(Axis.X).first
right_face = build.faces().sort_by(Axis.X).last

result = optimize(
    build,
    fixed=[left_face],
    loads=[(right_face, (0, 0, -10))],
    resolution=RESOLUTION,
    volfrac=VOLFRAC,
    max_iter=MAX_ITER,
)

print("\n" + "=" * 60)
print(result)

result.export_stl("cantilever_optimised.stl")
print("Wrote cantilever_optimised.stl")

show(build, result.to_build123d_solid(), left_face, right_face)
print("Sent to OCP CAD Viewer")