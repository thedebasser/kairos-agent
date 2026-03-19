"""Kairos Agent — Response Cache.

Two-layer cache:

1. **Global cache** (``runs/_cache/``):
   LLM responses and sandbox outputs keyed by content hash.
   Shared across ALL runs — a new run automatically reuses any
   identical LLM call or sandbox execution from any prior run.

2. **Per-run step cache** (``runs/<run_id>/cache/``):
   Complete step outputs (the full state-update dict a graph node returns).
   Only valid within the same run_id.

Cache keys are deterministic:
  - LLM calls: hash(step_name + model + messages)
  - Sandbox: hash(simulation_code)

Flow per step:
  1. Check step cache (per-run) -> if HIT, skip entire step
  2. For each LLM/sandbox call inside the step:
     a. Check global cache -> if HIT, skip call
     b. Execute call
     c. Verify output
     d. Store in global cache
  3. Store step output in per-run step cache

LRU eviction (Finding 7.5):
  When the global cache exceeds ``max_cache_size_mb``, the oldest files
  (by modification time) are deleted until the size drops below 90% of
  the threshold.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RUNS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "runs"
GLOBAL_CACHE_DIR = RUNS_DIR / "_cache"


def _hash_key(*parts: str) -> str:
    """Create a short deterministic hash from string parts."""
    combined = "\n---\n".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


class ResponseCache:
    """Two-layer disk-backed cache.

    - LLM and sandbox caches are GLOBAL (shared across all runs).
    - Step caches are per-run (only valid within the same run_id).
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        # Per-run step cache
        self.step_cache_dir = RUNS_DIR / run_id / "cache"
        self.step_cache_dir.mkdir(parents=True, exist_ok=True)
        # Global LLM/sandbox cache (shared across runs)
        self.global_cache_dir = GLOBAL_CACHE_DIR
        self.global_cache_dir.mkdir(parents=True, exist_ok=True)

    # ── LLM Response Cache ──────────────────────────────────────────────

    def _llm_key(self, step: str, model: str, messages: list[dict[str, str]]) -> str:
        msg_str = json.dumps(messages, sort_keys=True, default=str)
        return _hash_key(step, model, msg_str)

    def get_llm(
        self,
        step: str,
        model: str,
        messages: list[dict[str, str]],
    ) -> dict[str, Any] | None:
        """Check cache for a previous LLM response.

        Returns:
            Cached response dict (model_dump), or None if miss.
        """
        key = self._llm_key(step, model, messages)
        path = self.global_cache_dir / f"llm_{step}_{key}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                logger.info("[cache] HIT llm/%s/%s (global) — skipping LLM call", step, key[:8])
                return data
            except Exception:
                logger.warning("[cache] Corrupt cache file %s, ignoring", path)
        return None

    def put_llm(
        self,
        step: str,
        model: str,
        messages: list[dict[str, str]],
        response: Any,
    ) -> Path:
        """Save an LLM response to cache.

        Args:
            step: Pipeline step name (e.g. 'idea_agent', 'caption_writer').
            model: LLM model used.
            messages: Input messages (for cache key).
            response: Pydantic model or dict to cache.

        Returns:
            Path to the cache file.
        """
        key = self._llm_key(step, model, messages)
        path = self.global_cache_dir / f"llm_{step}_{key}.json"

        if hasattr(response, "model_dump"):
            data = response.model_dump(mode="json")
        elif isinstance(response, dict):
            data = response
        else:
            data = {"raw": str(response)}

        # Also store metadata for debugging
        envelope = {
            "step": step,
            "model": model,
            "cache_key": key,
            "response": data,
        }
        path.write_text(json.dumps(envelope, indent=2, default=str), encoding="utf-8")
        logger.info("[cache] STORED llm/%s/%s", step, key[:8])
        return path

    # ── Sandbox Output Cache ────────────────────────────────────────────

    def _sandbox_key(self, code: str) -> str:
        return _hash_key(code)

    def get_sandbox(self, code: str) -> dict[str, Any] | None:
        """Check cache for a previous sandbox execution.

        Returns:
            Dict with 'returncode', 'stdout', 'stderr', 'output_files',
            or None if miss. Note: output_files must still exist on disk.
        """
        key = self._sandbox_key(code)
        path = self.global_cache_dir / f"sandbox_{key}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                # Verify the output files still exist
                files = data.get("response", {}).get("output_files", [])
                if files and all(Path(f).exists() for f in files):
                    logger.info("[cache] HIT sandbox/%s (global) -- skipping sandbox execution", key[:8])
                    return data["response"]
                else:
                    logger.info("[cache] sandbox/%s files missing, re-executing", key[:8])
                    return None
            except Exception:
                logger.warning("[cache] Corrupt cache file %s, ignoring", path)
        return None

    def put_sandbox(
        self,
        code: str,
        result: dict[str, Any],
    ) -> Path:
        """Save sandbox execution result to cache.

        Args:
            code: Simulation source code (for cache key).
            result: Dict with returncode, stdout, stderr, output_files.

        Returns:
            Path to cache file.
        """
        key = self._sandbox_key(code)
        path = self.global_cache_dir / f"sandbox_{key}.json"

        # Copy output files to a persistent location in the global cache
        persisted_files = []
        output_dir = GLOBAL_CACHE_DIR / "sandbox_output"
        output_dir.mkdir(exist_ok=True)
        for f in result.get("output_files", []):
            src = Path(f).resolve()
            if src.exists():
                # Include cache key in filename to prevent collisions (Finding 7.5)
                dst = output_dir / f"{key}_{src.name}"
                # Skip copy if source is already in the target directory
                if dst.resolve() != src:
                    shutil.copy2(src, dst)
                persisted_files.append(str(dst))

        envelope = {
            "cache_key": key,
            "response": {
                "returncode": result.get("returncode", 0),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "output_files": persisted_files,
            },
        }
        path.write_text(json.dumps(envelope, indent=2, default=str), encoding="utf-8")
        logger.info("[cache] STORED sandbox/%s (%d files)", key[:8], len(persisted_files))
        return path

    # ── Step-level Output Cache ─────────────────────────────────────────

    def get_step(self, step_name: str, input_hash: str = "") -> dict[str, Any] | None:
        """Check cache for a complete step output (the full state update).

        This is the highest-level cache — if a step's output is cached,
        the entire step (all its LLM calls, sandbox runs, etc.) is skipped.

        Args:
            step_name: Graph node name (e.g. ``"idea_node"``).
            input_hash: Optional hash of the step's relevant input fields.
                        When provided, the cache key incorporates the hash so
                        different inputs (e.g. a new concept) produce different
                        cache entries (Finding 2.3).

        Returns:
            The state update dict that the graph node would return, or None.
        """
        suffix = f"_{input_hash}" if input_hash else ""
        path = self.step_cache_dir / f"step_{step_name}{suffix}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                logger.info("[cache] HIT step/%s%s (run-local) -- skipping entire step", step_name, suffix)
                return data
            except Exception:
                logger.warning("[cache] Corrupt step cache %s, ignoring", path)
        return None

    def put_step(self, step_name: str, state_update: dict[str, Any], input_hash: str = "") -> Path:
        """Save a complete step output to cache.

        Called after a graph node completes successfully.
        On retry, the entire node is skipped if this cache exists.

        Args:
            step_name: Graph node name.
            state_update: The dict the graph node returns.
            input_hash: Must match the value passed to ``get_step`` for the
                        same step to ensure correct cache key alignment.
        """
        suffix = f"_{input_hash}" if input_hash else ""
        path = self.step_cache_dir / f"step_{step_name}{suffix}.json"
        path.write_text(json.dumps(state_update, indent=2, default=str), encoding="utf-8")
        logger.info("[cache] STORED step/%s%s", step_name, suffix)
        return path

    # ── LRU Eviction (Finding 7.5) ─────────────────────────────────────

    def evict_if_needed(self, max_size_mb: int | None = None) -> int:
        """Delete oldest global cache files until size < 90% of threshold.

        Args:
            max_size_mb: Override the ``max_cache_size_mb`` config setting.

        Returns:
            Number of files deleted.
        """
        if max_size_mb is None:
            try:
                from kairos.config import get_settings
                max_size_mb = get_settings().max_cache_size_mb
            except Exception:
                max_size_mb = 2048

        max_bytes = max_size_mb * 1024 * 1024
        target_bytes = int(max_bytes * 0.9)  # Evict down to 90%

        # Collect all files in global cache (recursive)
        all_files: list[tuple[float, int, Path]] = []
        for p in self.global_cache_dir.rglob("*"):
            if p.is_file():
                stat = p.stat()
                all_files.append((stat.st_mtime, stat.st_size, p))

        total_size = sum(size for _, size, _ in all_files)
        if total_size <= max_bytes:
            return 0

        # Sort oldest first (ascending mtime)
        all_files.sort(key=lambda x: x[0])

        deleted = 0
        for _mtime, size, path in all_files:
            if total_size <= target_bytes:
                break
            try:
                path.unlink()
                total_size -= size
                deleted += 1
            except OSError:
                pass

        logger.info(
            "[cache] Evicted %d files, cache now %.1f MB (limit %d MB)",
            deleted,
            total_size / (1024 * 1024),
            max_size_mb,
        )
        return deleted

    @staticmethod
    def cache_stats() -> dict[str, Any]:
        """Return size and file-count stats for the global cache.

        Returns:
            Dict with ``total_size_mb``, ``file_count``,
            ``llm_count``, ``sandbox_count``, ``sandbox_output_count``.
        """
        total_size = 0
        file_count = 0
        llm_count = 0
        sandbox_json_count = 0
        sandbox_output_count = 0

        if not GLOBAL_CACHE_DIR.exists():
            return {
                "total_size_mb": 0.0,
                "file_count": 0,
                "llm_count": 0,
                "sandbox_count": 0,
                "sandbox_output_count": 0,
            }

        for p in GLOBAL_CACHE_DIR.rglob("*"):
            if p.is_file():
                total_size += p.stat().st_size
                file_count += 1
                name = p.name
                if name.startswith("llm_"):
                    llm_count += 1
                elif name.startswith("sandbox_") and name.endswith(".json"):
                    sandbox_json_count += 1
                elif p.parent.name == "sandbox_output":
                    sandbox_output_count += 1

        return {
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_count": file_count,
            "llm_count": llm_count,
            "sandbox_count": sandbox_json_count,
            "sandbox_output_count": sandbox_output_count,
        }


# ── Module-level singleton (set per pipeline run) ───────────────────────

_current_cache: ResponseCache | None = None


def init_cache(run_id: str) -> ResponseCache:
    """Initialise the response cache for a pipeline run."""
    global _current_cache  # noqa: PLW0603
    _current_cache = ResponseCache(run_id)
    return _current_cache


def get_cache() -> ResponseCache | None:
    """Get the current run's cache (or None)."""
    return _current_cache
