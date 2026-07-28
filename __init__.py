# SPDX-License-Identifier: GPL-3.0-or-later
"""
topology_gen — SIMP topology optimization for build123d.

Thin wrapper around the blendtopo pure-numpy core that replaces Blender's
BVH voxelizer with trimesh, so the optimizer runs headless in any Python
process (no bpy needed).

Typical usage::

    import trimesh
    from topology_gen import optimize

    build = trimesh.creation.box((60, 20, 10))
    result = optimize(
        build,
        fixed=[dict(center=(-30, 0, 0), normal=(-1, 0, 0), radius=3)],
        loads=[dict(center=(30, 0, -5), normal=(0, 0, -1), force=(0, 0, -10), radius=2)],
        resolution=60,
        volfrac=0.3,
        max_iter=40,
    )
    result.export_stl("optimized.stl")
    # result.mesh is a trimesh.Trimesh ready for build123d
"""

import sys
import os

# Ensure the core package is importable
_core_dir = os.path.join(os.path.dirname(__file__), "core")
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from ._result import Result
from ._optimize import optimize, Optimizer

__all__ = ["optimize", "Optimizer", "Result"]
