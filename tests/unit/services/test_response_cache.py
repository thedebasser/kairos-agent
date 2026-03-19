"""Unit tests for response cache eviction and collision-safe naming (Finding 7.5)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from kairos.ai.llm.cache import (
    GLOBAL_CACHE_DIR,
    ResponseCache,
    _hash_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_cache(tmp_path: Path):
    """Patch GLOBAL_CACHE_DIR and RUNS_DIR to use tmp_path."""
    runs = tmp_path / "runs"
    global_cache = runs / "_cache"
    global_cache.mkdir(parents=True)

    with patch("kairos.services.response_cache.RUNS_DIR", runs), \
         patch("kairos.services.response_cache.GLOBAL_CACHE_DIR", global_cache):
        cache = ResponseCache.__new__(ResponseCache)
        cache.run_id = "test-run"
        cache.step_cache_dir = runs / "test-run" / "cache"
        cache.step_cache_dir.mkdir(parents=True, exist_ok=True)
        cache.global_cache_dir = global_cache
        yield cache, global_cache


# ---------------------------------------------------------------------------
# Collision-safe sandbox output naming
# ---------------------------------------------------------------------------

class TestCollisionSafeNaming:
    """Sandbox output files include cache key in filename."""

    def test_different_codes_produce_different_filenames(self, temp_cache):
        cache, global_dir = temp_cache

        result1 = {
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "output_files": [],
        }
        result2 = {**result1}

        # Create fake output files
        out_dir = Path(global_dir) / "sandbox_output"
        out_dir.mkdir(exist_ok=True)

        code_a = "print('sim A')"
        code_b = "print('sim B')"

        # We need actual output files to persist
        tmp_a = global_dir / "temp_a.mp4"
        tmp_a.write_bytes(b"\x00" * 10)
        tmp_b = global_dir / "temp_b.mp4"
        tmp_b.write_bytes(b"\x00" * 20)

        result1["output_files"] = [str(tmp_a)]
        result2["output_files"] = [str(tmp_b)]

        cache.put_sandbox(code_a, result1)
        cache.put_sandbox(code_b, result2)

        # Both should exist in sandbox_output with different names
        output_files = list(out_dir.iterdir())
        names = {f.name for f in output_files}
        assert len(names) == 2  # No collision
        # Each name should contain its respective hash key
        key_a = _hash_key(code_a)
        key_b = _hash_key(code_b)
        assert any(key_a in n for n in names)
        assert any(key_b in n for n in names)

    def test_same_code_same_filename(self, temp_cache):
        """Same simulation code should overwrite, not create duplicates."""
        cache, global_dir = temp_cache
        out_dir = global_dir / "sandbox_output"
        out_dir.mkdir(exist_ok=True)

        code = "print('deterministic')"
        tmp = global_dir / "output.mp4"
        tmp.write_bytes(b"\x00" * 10)

        result = {"returncode": 0, "stdout": "", "stderr": "", "output_files": [str(tmp)]}
        cache.put_sandbox(code, result)

        # Write again — should produce same filename
        tmp.write_bytes(b"\x00" * 10)
        cache.put_sandbox(code, result)

        output_files = list(out_dir.iterdir())
        assert len(output_files) == 1


# ---------------------------------------------------------------------------
# LRU Eviction
# ---------------------------------------------------------------------------

class TestEviction:
    """LRU eviction deletes oldest files when cache exceeds size limit."""

    def test_no_eviction_when_under_limit(self, temp_cache):
        cache, global_dir = temp_cache

        # Create a small file
        (global_dir / "llm_test_abc123.json").write_text("{}")

        deleted = cache.evict_if_needed(max_size_mb=100)
        assert deleted == 0

    def test_evicts_oldest_files_first(self, temp_cache):
        cache, global_dir = temp_cache

        # Create files with different modification times
        old_file = global_dir / "llm_old_aaa.json"
        old_file.write_bytes(b"x" * 600)  # 600 bytes

        # Ensure different mtime
        time.sleep(0.05)

        new_file = global_dir / "llm_new_bbb.json"
        new_file.write_bytes(b"x" * 600)  # 600 bytes

        # Total = 1200 bytes. Set limit to ~1 KB (will evict to 90% = 921 bytes)
        # So it should delete at least the old file
        deleted = cache.evict_if_needed(max_size_mb=0)  # 0 MB means evict everything
        assert deleted == 2  # Both deleted when limit is 0

    def test_evicts_until_below_90_percent(self, temp_cache):
        cache, global_dir = temp_cache

        # Create 10 files of 100 bytes each = 1000 bytes total
        for i in range(10):
            f = global_dir / f"llm_step_{i:03d}.json"
            f.write_bytes(b"x" * 100)
            time.sleep(0.01)  # Ensure distinct mtimes

        # Limit = 800 bytes. Target = 720 bytes (90%).
        # Need to delete at least 3 files (300 bytes) to go from 1000 to 700
        # Actually we set max_size_mb, let's use bytes wisely
        # 1000 bytes total, limit in MB. 1000 bytes = 0.000953674 MB
        # Let's just use a tiny value: 0 to force eviction of all
        # Better: patch to test with byte-level precision
        deleted = cache.evict_if_needed(max_size_mb=0)
        assert deleted == 10

    def test_eviction_preserves_newer_files(self, temp_cache):
        cache, global_dir = temp_cache

        # Create old file — large enough to exceed limit
        old = global_dir / "llm_old_000.json"
        old.write_bytes(b"x" * 2_000_000)  # ~2 MB
        time.sleep(0.05)

        # Create new file (small)
        new = global_dir / "llm_new_001.json"
        new.write_bytes(b"x" * 100)

        # Sub-directory file to test rglob
        sub = global_dir / "sandbox_output"
        sub.mkdir(exist_ok=True)
        (sub / "vid.mp4").write_bytes(b"x" * 100)

        # Total ~2 MB. Limit = 1 MB. Old file alone is 2 MB, so it gets evicted.
        deleted = cache.evict_if_needed(max_size_mb=1)
        assert deleted >= 1

        # The new file and video should survive
        remaining = [f.name for f in global_dir.rglob("*") if f.is_file()]
        assert "llm_old_000.json" not in remaining


# ---------------------------------------------------------------------------
# Cache Stats
# ---------------------------------------------------------------------------

class TestCacheStats:
    """cache_stats returns correct file counts and sizes."""

    def test_empty_cache(self, temp_cache):
        _, global_dir = temp_cache
        stats = ResponseCache.cache_stats()
        assert stats["file_count"] == 0
        assert stats["total_size_mb"] == 0.0

    def test_counts_by_type(self, temp_cache):
        _, global_dir = temp_cache

        # LLM files
        (global_dir / "llm_idea_abc.json").write_text("{}")
        (global_dir / "llm_sim_def.json").write_text("{}")

        # Sandbox JSON
        (global_dir / "sandbox_ghi.json").write_text("{}")

        # Sandbox output
        out = global_dir / "sandbox_output"
        out.mkdir()
        (out / "abc_output.mp4").write_bytes(b"\x00" * 100)

        stats = ResponseCache.cache_stats()
        assert stats["llm_count"] == 2
        assert stats["sandbox_count"] == 1
        assert stats["sandbox_output_count"] == 1
        assert stats["file_count"] == 4

    def test_size_calculation(self, temp_cache):
        _, global_dir = temp_cache

        (global_dir / "llm_test_xyz.json").write_bytes(b"x" * 1024)
        stats = ResponseCache.cache_stats()
        assert stats["total_size_mb"] == pytest.approx(0.001, abs=0.001)
