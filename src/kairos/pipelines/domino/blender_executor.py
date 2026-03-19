"""Blender executor for the domino pipeline.

Thin re-export of the shared blender_executor from the marble pipeline.
The domino scripts live in the same blend/scripts/ directory.
"""

from kairos.pipelines.marble.blender_executor import (
    find_blender,
    run_blender_script,
)

__all__ = ["find_blender", "run_blender_script"]
