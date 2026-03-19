"""Tests for P3.28 — Standardise UTC datetimes.

Verifies that all datetime.now() calls across the codebase produce
timezone-aware UTC datetimes (not naive).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "src"


def _collect_python_files() -> list[Path]:
    """Collect all .py files under src/."""
    return sorted(SRC_ROOT.rglob("*.py"))


class TestNoNaiveDatetimeNow:
    """Ensure no naive datetime.now() calls exist in source code."""

    def test_no_naive_datetime_now_in_source(self) -> None:
        """Scan all source files for datetime.now() without timezone arg."""
        pattern = re.compile(r"datetime\.now\(\s*\)")
        violations: list[str] = []

        for py_file in _collect_python_files():
            try:
                source = py_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            for i, line in enumerate(source.splitlines(), start=1):
                # Skip comments
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                if pattern.search(line):
                    rel = py_file.relative_to(SRC_ROOT.parent)
                    violations.append(f"{rel}:{i}: {stripped.strip()}")

        assert violations == [], (
            "Found naive datetime.now() calls (should be "
            "datetime.now(timezone.utc)):\n" + "\n".join(violations)
        )

    def test_no_utcnow_in_source(self) -> None:
        """datetime.utcnow() is deprecated in 3.12+; ensure it's not used."""
        pattern = re.compile(r"datetime\.utcnow\(\)")
        violations: list[str] = []

        for py_file in _collect_python_files():
            try:
                source = py_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            for i, line in enumerate(source.splitlines(), start=1):
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                if pattern.search(line):
                    rel = py_file.relative_to(SRC_ROOT.parent)
                    violations.append(f"{rel}:{i}: {stripped.strip()}")

        assert violations == [], (
            "Found deprecated datetime.utcnow() calls:\n"
            + "\n".join(violations)
        )


class TestTimezoneImports:
    """Verify that files using datetime.now(timezone.utc) import timezone."""

    def test_timezone_imported_where_used(self) -> None:
        """Every file calling datetime.now(timezone...) must import timezone."""
        aware_pattern = re.compile(r"datetime\.now\(timezone")
        violations: list[str] = []

        for py_file in _collect_python_files():
            try:
                source = py_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            if not aware_pattern.search(source):
                continue

            # Check for timezone import
            has_import = (
                "from datetime import" in source
                and "timezone" in source
            ) or "import datetime" in source

            if not has_import:
                rel = py_file.relative_to(SRC_ROOT.parent)
                violations.append(str(rel))

        assert violations == [], (
            "Files using timezone.utc but missing timezone import:\n"
            + "\n".join(violations)
        )
