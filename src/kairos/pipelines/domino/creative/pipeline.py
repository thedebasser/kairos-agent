"""Creative pipeline orchestrator — cascade re-run logic.

Runs: Set Designer → Path Setter → Connector Agent → Camera Router
sequentially, validating after each step.  A Final Reviewer then
assesses the complete output.  On failure the cascade re-runs from
the *responsible* agent, not from the start:

    Set Designer fail  → re-run [1, 2, 3, 4]
    Path Setter fail   → re-run [2, 3, 4]
    Connector fail     → re-run [3, 4]
    Camera Router fail → re-run [4] only

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
    CameraOutput,
    ConnectorOutput,
    FinalReviewResult,
    IterationHistory,
    PathOutput,
    SceneManifest,
    StepValidationResult,
)
from kairos.pipelines.domino.creative.camera_router import CameraRouter
from kairos.pipelines.domino.creative.camera_validator import validate_camera
from kairos.pipelines.domino.creative.connector_agent import ConnectorAgent
from kairos.pipelines.domino.creative.final_reviewer import FinalReviewer
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
    AgentRole.SET_DESIGNER: [
        AgentRole.SET_DESIGNER, AgentRole.PATH_SETTER,
        AgentRole.CONNECTOR, AgentRole.CAMERA_ROUTER,
    ],
    AgentRole.PATH_SETTER: [
        AgentRole.PATH_SETTER, AgentRole.CONNECTOR, AgentRole.CAMERA_ROUTER,
    ],
    AgentRole.CONNECTOR: [AgentRole.CONNECTOR, AgentRole.CAMERA_ROUTER],
    AgentRole.CAMERA_ROUTER: [AgentRole.CAMERA_ROUTER],
}


class PipelineExhausted(Exception):
    """Raised when the pipeline exhausts all allowed attempts."""


class CreativePipelineResult:
    """Immutable result container for a creative pipeline run."""

    __slots__ = (
        "manifest", "path", "connectors", "camera",
        "final_review", "history", "success",
    )

    def __init__(
        self,
        *,
        manifest: SceneManifest | None,
        path: PathOutput | None,
        connectors: ConnectorOutput | None,
        camera: CameraOutput | None = None,
        final_review: FinalReviewResult | None = None,
        history: IterationHistory,
        success: bool,
    ) -> None:
        self.manifest = manifest
        self.path = path
        self.connectors = connectors
        self.camera = camera
        self.final_review = final_review
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
        self._camera_router = CameraRouter()
        self._final_reviewer = FinalReviewer()
        self._storage = SessionStorage(output_dir, session_id)
        self._history = IterationHistory()
        self._agent_attempts: dict[AgentRole, int] = {
            AgentRole.SET_DESIGNER: 0,
            AgentRole.PATH_SETTER: 0,
            AgentRole.CONNECTOR: 0,
            AgentRole.CAMERA_ROUTER: 0,
        }

    async def run(self, *, domino_count: int = 300) -> CreativePipelineResult:
        """Execute the creative pipeline with cascade re-run on failure.

        Returns a CreativePipelineResult (success or exhausted).
        """
        manifest: SceneManifest | None = None
        path_output: PathOutput | None = None
        connector_output: ConnectorOutput | None = None
        camera_output: CameraOutput | None = None

        # Start with the full cascade (all four agents)
        agents_to_run = list(_CASCADE[AgentRole.SET_DESIGNER])

        while self._history.total_pipeline_attempts < PIPELINE_TOTAL_MAX:
            for agent_role in agents_to_run:
                if self._agent_attempts[agent_role] >= PER_AGENT_MAX:
                    logger.error(
                        "[pipeline] %s exhausted (%d attempts)",
                        agent_role.value,
                        PER_AGENT_MAX,
                    )
                    return self._finish(
                        manifest, path_output, connector_output,
                        camera_output, success=False,
                    )

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
                        break

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
                        break

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
                        break

                elif agent_role == AgentRole.CAMERA_ROUTER:
                    assert connector_output is not None
                    assert manifest is not None
                    camera_output = self._camera_router.compute_trajectory(
                        connector_output, manifest,
                    )
                    validation = validate_camera(
                        camera_output, connector_output, manifest,
                    )
                    self._record(agent_role, attempt_num, validation)
                    self._storage.save_attempt(
                        agent_role, attempt_num, camera_output, validation,
                    )
                    if not validation.passed:
                        agents_to_run = list(_CASCADE[AgentRole.CAMERA_ROUTER])
                        break
            else:
                # All agents passed — run final review
                assert manifest is not None
                assert path_output is not None
                assert connector_output is not None
                final_review = await self._final_reviewer.review(
                    manifest, path_output, connector_output, camera_output,
                    history=self._history,
                )

                if final_review.passed:
                    logger.info(
                        "[pipeline] Creative pipeline succeeded after %d total attempts",
                        self._history.total_pipeline_attempts,
                    )
                    return self._finish(
                        manifest, path_output, connector_output,
                        camera_output, success=True, final_review=final_review,
                    )

                # Final reviewer failed — cascade from attributed agent
                cascade_from = final_review.cascade_from or AgentRole.SET_DESIGNER
                self._record(
                    AgentRole.FINAL_REVIEWER,
                    self._history.total_pipeline_attempts,
                    StepValidationResult(
                        agent=AgentRole.FINAL_REVIEWER,
                        passed=False,
                        error_summary=final_review.summary,
                    ),
                )
                agents_to_run = list(_CASCADE.get(
                    cascade_from, _CASCADE[AgentRole.SET_DESIGNER],
                ))
                logger.warning(
                    "[pipeline] Final review failed, cascading from %s",
                    cascade_from.value,
                )
                continue  # restart the while loop with new agents_to_run

        # Total pipeline attempts exhausted
        logger.error(
            "[pipeline] Pipeline exhausted (%d total attempts)",
            self._history.total_pipeline_attempts,
        )
        return self._finish(
            manifest, path_output, connector_output,
            camera_output, success=False,
        )

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
        camera: CameraOutput | None = None,
        *,
        success: bool,
        final_review: FinalReviewResult | None = None,
    ) -> CreativePipelineResult:
        result_data = {
            "success": success,
            "total_attempts": self._history.total_pipeline_attempts,
            "agent_attempts": {k.value: v for k, v in self._agent_attempts.items()},
        }
        self._storage.save_result(result_data)

        if not success:
            review_data = self._build_final_review()
            self._storage.save_final_review(review_data)

        return CreativePipelineResult(
            manifest=manifest,
            path=path,
            connectors=connectors,
            camera=camera,
            final_review=final_review,
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
