"""One-shot import rewriter for the Phase 1 structural refactor.

Run from the project root:
    python scripts/rewrite_imports.py

This script rewrites Python import paths to match the new directory structure.
It is idempotent — running it twice produces the same result.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"

# ── Import path mappings (old -> new) ────────────────────────────────────────
# Order matters: longer/more-specific patterns first to avoid partial matches.
IMPORT_REWRITES: list[tuple[str, str]] = [
    # ── models -> schemas ──────────────────────────────────────────────────
    ("kairos.models.contracts", "kairos.schemas.contracts"),
    ("kairos.models.idea", "kairos.schemas.idea"),
    ("kairos.models.simulation", "kairos.schemas.simulation"),
    ("kairos.models.video_editor", "kairos.schemas.video_editor"),
    ("kairos.models", "kairos.schemas"),

    # ── agents.base -> pipelines.contracts ─────────────────────────────────
    # Also rewrite class names: BaseXxxAgent -> XxxAgent
    ("kairos.agents.base", "kairos.pipelines.contracts"),

    # ── pipeline (singular) -> orchestrator ────────────────────────────────
    ("kairos.pipeline.graph", "kairos.orchestrator.graph"),
    ("kairos.pipeline.registry", "kairos.orchestrator.registry"),
    ("kairos.pipeline", "kairos.orchestrator"),

    # ── services.llm_routing -> ai.llm.routing ─────────────────────────────
    ("kairos.services.llm_routing", "kairos.ai.llm.routing"),
    ("kairos.services.llm_config", "kairos.ai.llm.config"),
    ("kairos.services.model_capabilities", "kairos.ai.llm.capabilities"),
    ("kairos.services.response_cache", "kairos.ai.llm.cache"),

    # ── services.prompt_registry -> ai.prompts.registry ────────────────────
    ("kairos.services.prompt_registry", "kairos.ai.prompts.registry"),

    # ── services.video_review -> ai.review.video_review_agent ──────────────
    ("kairos.services.video_review", "kairos.ai.review.video_review_agent"),
    ("kairos.services.audio_review", "kairos.ai.review.audio_review_agent"),

    # ── services.monitoring -> ai.tracing.sinks.langfuse_sink ──────────────
    ("kairos.services.monitoring", "kairos.ai.tracing.sinks.langfuse_sink"),

    # ── services.learning_loop -> ai.learning.learning_loop ────────────────
    ("kairos.services.learning_loop", "kairos.ai.learning.learning_loop"),

    # ── eval -> ai.eval ────────────────────────────────────────────────────
    ("kairos.eval", "kairos.ai.eval"),


    # ── services.environment -> engines.blender.environment / services.audio
    ("kairos.services.environment.orchestrator", "kairos.engines.blender.environment.orchestrator"),
    ("kairos.services.environment.theme_catalogue", "kairos.engines.blender.environment.theme_catalogue"),
    ("kairos.services.environment.poly_haven", "kairos.engines.blender.environment.poly_haven"),
    ("kairos.services.environment.mix_audio", "kairos.services.audio.mix_audio"),
    ("kairos.services.environment.sfx_pool", "kairos.services.audio.sfx_pool"),
    ("kairos.services.environment.freesound_sfx", "kairos.services.audio.freesound_sfx"),
    ("kairos.services.environment.synthetic_sfx", "kairos.services.audio.synthetic_sfx"),
    ("kairos.services.environment", "kairos.engines.blender.environment"),

    # ── blender_executor consolidation ─────────────────────────────────────
    ("kairos.pipelines.marble.blender_executor", "kairos.engines.blender.executor"),
    ("kairos.pipelines.domino.blender_executor", "kairos.engines.blender.executor"),

    # ── adapter moves ──────────────────────────────────────────────────────
    ("kairos.pipelines.physics.adapter", "kairos.pipelines.adapters.physics_adapter"),
    ("kairos.pipelines.domino.adapter", "kairos.pipelines.adapters.domino_adapter"),
    ("kairos.pipelines.marble.adapter", "kairos.pipelines.adapters.marble_adapter"),

    # ── physics prompts -> ai.prompts.physics ──────────────────────────────
    ("kairos.pipelines.physics.prompts.builder", "kairos.ai.prompts.physics.builder"),
    ("kairos.pipelines.physics.prompts", "kairos.ai.prompts.physics"),

    # ── services.session_logging (deleted, but fix any remaining refs) ─────
    ("kairos.services.session_logging", "kairos.ai.tracing.tracer"),
    ("kairos.services.step_artifacts", "kairos.ai.tracing.tracer"),

    # ── cli.py -> cli.main ─────────────────────────────────────────────────
    ("kairos.cli", "kairos.cli.main"),
]

# ── Class name renames (Base* -> *) ──────────────────────────────────────────
CLASS_RENAMES: list[tuple[str, str]] = [
    ("BaseIdeaAgent", "IdeaAgent"),
    ("BaseSimulationAgent", "SimulationAgent"),
    ("BaseVideoEditorAgent", "VideoEditorAgent"),
    ("BaseVideoReviewAgent", "VideoReviewAgent"),
    ("BaseAudioReviewAgent", "AudioReviewAgent"),
    ("BasePipelineAdapter", "PipelineAdapter"),
]


def rewrite_file(path: Path) -> int:
    """Rewrite imports in a single file. Returns count of changes."""
    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return 0

    original = content
    changes = 0

    # Apply import path rewrites
    for old_path, new_path in IMPORT_REWRITES:
        # Match: from old_path import ... OR import old_path
        # Be careful not to match partial paths (e.g. kairos.pipeline should not
        # match kairos.pipelines). Use word boundary after the old path.
        pattern = re.compile(
            r'(?<![.\w])' + re.escape(old_path) + r'(?![.\w])'
        )
        content, n = pattern.subn(new_path, content)
        changes += n

    # Apply class name renames (only in import lines and type annotations)
    for old_name, new_name in CLASS_RENAMES:
        pattern = re.compile(r'\b' + re.escape(old_name) + r'\b')
        content, n = pattern.subn(new_name, content)
        changes += n

    if content != original:
        path.write_text(content, encoding="utf-8")
        return changes
    return 0


def main() -> None:
    """Rewrite all Python files under src/ and tests/."""
    total_changes = 0
    total_files = 0

    search_dirs = [SRC_DIR, TESTS_DIR]
    # Also check pyproject.toml for entry points
    extra_files = [PROJECT_ROOT / "pyproject.toml"]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for py_file in search_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            n = rewrite_file(py_file)
            if n > 0:
                total_files += 1
                total_changes += n
                print(f"  {py_file.relative_to(PROJECT_ROOT)}: {n} changes")

    for f in extra_files:
        if f.exists():
            n = rewrite_file(f)
            if n > 0:
                total_files += 1
                total_changes += n
                print(f"  {f.relative_to(PROJECT_ROOT)}: {n} changes")

    print(f"\nTotal: {total_changes} import rewrites across {total_files} files")


if __name__ == "__main__":
    main()
