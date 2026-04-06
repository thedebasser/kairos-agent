"""Session storage — persists creative pipeline attempt artefacts.

Writes each agent attempt to:
  sessions/{session_id}/agents/{agent_name}/attempt_{n}/
    ├── output.yaml
    ├── validation.yaml
    └── summary.md

Also writes top-level session manifest and final result.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from kairos.pipelines.domino.creative.models import (
    AgentRole,
    ConnectorOutput,
    IterationHistory,
    PathOutput,
    SceneManifest,
    StepValidationResult,
)

logger = logging.getLogger(__name__)


def _dump_yaml(data: dict | BaseModel, path: Path) -> None:
    """Write data as YAML, converting Pydantic models first."""
    if isinstance(data, BaseModel):
        raw = data.model_dump(mode="json")
    else:
        raw = data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, default_flow_style=False, sort_keys=False), encoding="utf-8")


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class SessionStorage:
    """Manages file-based persistence for creative pipeline sessions."""

    def __init__(self, base_dir: Path, session_id: str) -> None:
        self._root = base_dir / "sessions" / session_id
        self._root.mkdir(parents=True, exist_ok=True)
        logger.debug("[session] Storage root: %s", self._root)

    @property
    def root(self) -> Path:
        return self._root

    # ─── Agent attempt writes ────────────────────────────────────────

    def save_attempt(
        self,
        agent: AgentRole,
        attempt: int,
        output: BaseModel,
        validation: StepValidationResult,
    ) -> Path:
        """Persist one agent attempt (output + validation + summary).

        Returns the attempt directory path.
        """
        attempt_dir = self._root / "agents" / agent.value / f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        _dump_yaml(output, attempt_dir / "output.yaml")
        _dump_yaml(validation, attempt_dir / "validation.yaml")

        summary = self._build_summary(agent, attempt, validation)
        _write_text(summary, attempt_dir / "summary.md")

        logger.debug("[session] Saved %s attempt %d", agent.value, attempt)
        return attempt_dir

    # ─── Top-level session files ─────────────────────────────────────

    def save_manifest(self, manifest: SceneManifest) -> None:
        _dump_yaml(manifest, self._root / "manifest.yaml")

    def save_result(self, result: dict[str, Any]) -> None:
        _dump_yaml(result, self._root / "result.yaml")

    def save_final_review(self, review: dict[str, Any]) -> None:
        _dump_yaml(review, self._root / "final_review.yaml")

    # ─── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _build_summary(
        agent: AgentRole,
        attempt: int,
        validation: StepValidationResult,
    ) -> str:
        status = "PASSED" if validation.passed else "FAILED"
        lines = [
            f"# {agent.value} — Attempt {attempt}",
            "",
            f"**Status:** {status}",
            "",
        ]
        if validation.error_summary:
            lines.extend(["## Errors", "", validation.error_summary, ""])
        if validation.checks:
            lines.append("## Checks")
            lines.append("")
            for c in validation.checks:
                mark = "✓" if c.get("passed") else "✗"
                lines.append(f"- {mark} **{c.get('name', '?')}** — {c.get('message', '')}")
            lines.append("")
        return "\n".join(lines)
