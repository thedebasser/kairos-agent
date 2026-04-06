"""Creative pipeline orchestrator — cascade re-run logic.

Runs: Set Designer → Path Setter → Connector Agent sequentially,
validating after each step.  On failure the cascade re-runs from
the *responsible* agent, not from the start:

    Set Designer fail  → re-run [1, 2, 3]
    Path Setter fail   → re-run [2, 3]
    Connector fail     → re-run [3] only

Iteration limits:
    per_agent_max:      10
    pipeline_total_max: 30
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from kairos.pipelines.domino.creative.models import (
    AgentRole,
    AttemptRecord,
    ConnectorOutput,
    IterationHistory,
    PathOutput,
    SceneManifest,
    StepValidationResult,
)
from kairos.pipelines.domino.creative.connector_agent import ConnectorAgent
from kairos.pipelines.domino.creative.path_setter import PathSetterAgent
from kairos.pipelines.domino.creative.set_designer import SetDesignerAgent
from kairos.pipelines.domino.creative.session import SessionStorage
from kairos.pipelines.domino.creative.validator import (
    validate_connectors,
    validate_path,
    validate_scene,
)
from kairos.skills.catalogue import AssetEntry

logger = logging.getLogger(__name__)

PER_AGENT_MAX = 10
PIPELINE_TOTAL_MAX = 30


# ─── Cascade mapping ────────────────────────────────────────────────

# When an agent fails, which agents must re-run (in order)?
_CASCADE: dict[AgentRole, list[AgentRole]] = {
    AgentRole.SET_DESIGNER: [AgentRole.SET_DESIGNER, AgentRole.PATH_SETTER, AgentRole.CONNECTOR],
    AgentRole.PATH_SETTER: [AgentRole.PATH_SETTER, AgentRole.CONNECTOR],
    AgentRole.CONNECTOR: [AgentRole.CONNECTOR],
}


class PipelineExhausted(Exception):
    """Raised when the pipeline exhausts all allowed attempts."""


class CreativePipelineResult:
    """Immutable result container for a creative pipeline run."""

    __slots__ = ("manifest", "path", "connectors", "history", "success")

    def __init__(
        self,
        *,
        manifest: SceneManifest | None,
        path: PathOutput | None,
        connectors: ConnectorOutput | None,
        history: IterationHistory,
        success: bool,
    ) -> None:
        self.manifest = manifest
        self.path = path
        self.connectors = connectors
        self.history = history
        self.success = success


class CreativePipeline:
    """Orchestrates the three creative agents with cascade re-run logic.

    Usage::

        pipeline = CreativePipeline(assets=catalogue, output_dir=Path("output/run1"))
        result = await pipeline.run(domino_count=300)
    """

    def __init__(
        self,
        *,
        assets: list[AssetEntry],
        output_dir: Path,
        session_id: str = "default",
        force_theme: str | None = None,
    ) -> None:
        self._set_designer = SetDesignerAgent(assets, force_theme=force_theme)
        self._path_setter = PathSetterAgent()
        self._connector = ConnectorAgent()
        self._storage = SessionStorage(output_dir, session_id)
        self._history = IterationHistory()
        self._agent_attempts: dict[AgentRole, int] = {
            AgentRole.SET_DESIGNER: 0,
            AgentRole.PATH_SETTER: 0,
            AgentRole.CONNECTOR: 0,
        }

    async def run(self, *, domino_count: int = 300) -> CreativePipelineResult:
        """Execute the creative pipeline with cascade re-run on failure.

        Returns a CreativePipelineResult (success or exhausted).
        """
        manifest: SceneManifest | None = None
        path_output: PathOutput | None = None
        connector_output: ConnectorOutput | None = None

        # Start with the full cascade (all three agents)
        agents_to_run = list(_CASCADE[AgentRole.SET_DESIGNER])

        while self._history.total_pipeline_attempts < PIPELINE_TOTAL_MAX:
            for agent_role in agents_to_run:
                if self._agent_attempts[agent_role] >= PER_AGENT_MAX:
                    logger.error(
                        "[pipeline] %s exhausted (%d attempts)",
                        agent_role.value,
                        PER_AGENT_MAX,
                    )
                    return self._finish(manifest, path_output, connector_output, success=False)

                self._agent_attempts[agent_role] += 1
                attempt_num = self._agent_attempts[agent_role]

                if agent_role == AgentRole.SET_DESIGNER:
                    manifest = await self._set_designer.design_scene(
                        domino_count=domino_count,
                        history=self._history,
                    )
                    validation = validate_scene(manifest)
                    self._record(agent_role, attempt_num, validation)
                    self._storage.save_attempt(agent_role, attempt_num, manifest, validation)
                    if manifest and attempt_num == 1:
                        self._storage.save_manifest(manifest)
                    if not validation.passed:
                        agents_to_run = list(_CASCADE[AgentRole.SET_DESIGNER])
                        break  # restart from set_designer

                elif agent_role == AgentRole.PATH_SETTER:
                    assert manifest is not None
                    path_output = await self._path_setter.plan_path(
                        manifest,
                        history=self._history,
                    )
                    validation = validate_path(path_output, manifest)
                    self._record(agent_role, attempt_num, validation)
                    self._storage.save_attempt(agent_role, attempt_num, path_output, validation)
                    if not validation.passed:
                        agents_to_run = list(_CASCADE[AgentRole.PATH_SETTER])
                        break  # restart from path_setter

                elif agent_role == AgentRole.CONNECTOR:
                    assert path_output is not None
                    connector_output = await self._connector.resolve_connectors(
                        path_output,
                        history=self._history,
                    )
                    validation = validate_connectors(connector_output, path_output)
                    self._record(agent_role, attempt_num, validation)
                    self._storage.save_attempt(agent_role, attempt_num, connector_output, validation)
                    if not validation.passed:
                        agents_to_run = list(_CASCADE[AgentRole.CONNECTOR])
                        break  # restart connector only
            else:
                # All agents in agents_to_run passed — pipeline success
                logger.info(
                    "[pipeline] Creative pipeline succeeded after %d total attempts",
                    self._history.total_pipeline_attempts,
                )
                return self._finish(manifest, path_output, connector_output, success=True)

        # Total pipeline attempts exhausted
        logger.error(
            "[pipeline] Pipeline exhausted (%d total attempts)",
            self._history.total_pipeline_attempts,
        )
        return self._finish(manifest, path_output, connector_output, success=False)

    # ─── Internal helpers ────────────────────────────────────────────

    def _record(
        self,
        agent: AgentRole,
        attempt: int,
        validation: StepValidationResult,
    ) -> None:
        self._history.add(
            AttemptRecord(
                agent=agent,
                attempt=attempt,
                passed=validation.passed,
                error_summary=validation.error_summary,
            )
        )

    def _finish(
        self,
        manifest: SceneManifest | None,
        path: PathOutput | None,
        connectors: ConnectorOutput | None,
        *,
        success: bool,
    ) -> CreativePipelineResult:
        result_data = {
            "success": success,
            "total_attempts": self._history.total_pipeline_attempts,
            "agent_attempts": {k.value: v for k, v in self._agent_attempts.items()},
        }
        self._storage.save_result(result_data)

        if not success:
            review = self._build_final_review()
            self._storage.save_final_review(review)

        return CreativePipelineResult(
            manifest=manifest,
            path=path,
            connectors=connectors,
            history=self._history,
            success=success,
        )

    def _build_final_review(self) -> dict[str, Any]:
        """Build a final review summarising failures and recommendations."""
        issues: list[dict[str, str]] = []
        for record in self._history.attempts:
            if not record.passed:
                issues.append({
                    "agent": record.agent.value,
                    "attempt": str(record.attempt),
                    "error": record.error_summary,
                })
        return {
            "status": "EXHAUSTED",
            "total_attempts": self._history.total_pipeline_attempts,
            "issues": issues,
            "recommendation": (
                "Review the latest validation errors and consider "
                "adjusting asset catalogue, theme constraints, or "
                "scene layout parameters."
            ),
        }
