"""Final Reviewer Agent — holistic end-to-end review of a complete pipeline run.

Watches the full pipeline outputs (scene manifest, path, connectors, camera)
and checks for emergent issues that per-step validation cannot catch:
  - Chain completeness (needs physics bake data)
  - Camera + scene coherence
  - Transition smoothness
  - Overall pacing

Uses VLM (via ``call_llm``) when frame data is available.  Falls back to
structural checks when no rendered frames are provided.

Produces a FinalReviewResult with failure attribution and cascade routing.
"""

from __future__ import annotations

import logging
from typing import Any

from kairos.ai.llm.config import get_step_config
from kairos.ai.llm.routing import call_llm
from kairos.pipelines.domino.creative.models import (
    AgentRole,
    CameraOutput,
    ConnectorOutput,
    FinalReviewResult,
    IterationHistory,
    PathOutput,
    ReviewIssue,
    SceneManifest,
)

logger = logging.getLogger(__name__)


# ─── Structural checks (no VLM needed) ──────────────────────────────


def _check_chain_coverage(
    path_output: PathOutput,
    connector_output: ConnectorOutput,
) -> list[ReviewIssue]:
    """Check that the connected path covers all path segments."""
    issues: list[ReviewIssue] = []

    transition_ids = {s.id for s in path_output.segments if s.needs_connector}
    resolved_ids = {c.for_segment for c in connector_output.connectors}
    missing = transition_ids - resolved_ids
    if missing:
        issues.append(ReviewIssue(
            description=f"Unresolved transitions remain: {missing}",
            attributed_to=AgentRole.CONNECTOR,
            reason="Connector agent did not resolve all transition segments",
            suggested_fix="Re-run connector agent to fill missing transitions",
            severity="blocking",
        ))

    if len(connector_output.complete_path_waypoints) < 4:
        issues.append(ReviewIssue(
            description="Complete path has too few waypoints for a viable domino chain",
            attributed_to=AgentRole.PATH_SETTER,
            reason="Path is too short or too few segments produced",
            suggested_fix="Re-plan path with more segments covering more objects",
            severity="blocking",
        ))

    return issues


def _check_camera_coverage(
    camera_output: CameraOutput | None,
) -> list[ReviewIssue]:
    """Check camera trajectory quality from structural data."""
    issues: list[ReviewIssue] = []
    if camera_output is None:
        issues.append(ReviewIssue(
            description="No camera trajectory produced",
            attributed_to=AgentRole.CAMERA_ROUTER,
            reason="Camera router was not run or failed",
            suggested_fix="Re-run camera router",
            severity="blocking",
        ))
        return issues

    if len(camera_output.keyframes) < 2:
        issues.append(ReviewIssue(
            description="Camera trajectory has insufficient keyframes",
            attributed_to=AgentRole.CAMERA_ROUTER,
            reason="Too few keyframes for smooth tracking",
            suggested_fix="Recompute trajectory with smaller key_interval",
            severity="blocking",
        ))

    # Too many occlusions
    total = max(1, camera_output.total_frames)
    occluded_frames = sum(
        occ.frame_end - occ.frame_start for occ in camera_output.occlusion_events
    )
    if occluded_frames / total > 0.10:
        issues.append(ReviewIssue(
            description=f"Excessive occlusion: {occluded_frames}/{total} frames ({occluded_frames / total:.0%})",
            attributed_to=AgentRole.CAMERA_ROUTER,
            reason="Occlusion avoidance did not resolve enough occlusions",
            suggested_fix="Increase follow distance or reposition more aggressively",
            severity="blocking",
        ))

    return issues


def _check_scene_coherence(
    manifest: SceneManifest,
    path_output: PathOutput,
) -> list[ReviewIssue]:
    """Check that path actually visits the scene's functional objects."""
    issues: list[ReviewIssue] = []

    functional_ids = {
        o.asset_id for o in manifest.objects
        if o.role.value == "functional"
    }
    referenced_surfaces = set()
    for seg in path_output.segments:
        if seg.surface_ref:
            # surface_ref format: "asset_id.surface_name"
            asset_id = seg.surface_ref.split(".")[0]
            referenced_surfaces.add(asset_id)

    unvisited = functional_ids - referenced_surfaces
    if unvisited and len(functional_ids) > 0:
        # This is a warning, not blocking — some objects may be decorative context
        issues.append(ReviewIssue(
            description=f"Functional objects not visited by path: {unvisited}",
            attributed_to=AgentRole.PATH_SETTER,
            reason="Path does not route through all functional surfaces",
            suggested_fix="Re-plan path to include more functional objects",
            severity="warning",
        ))

    return issues


# ─── Cascade determination ───────────────────────────────────────────


def _earliest_responsible_agent(issues: list[ReviewIssue]) -> AgentRole | None:
    """Return the earliest agent in the pipeline that has a blocking issue."""
    # Priority order: earliest in pipeline gets the cascade
    priority = [
        AgentRole.SET_DESIGNER,
        AgentRole.PATH_SETTER,
        AgentRole.CONNECTOR,
        AgentRole.CAMERA_ROUTER,
    ]
    blocking_agents = {
        issue.attributed_to
        for issue in issues
        if issue.severity == "blocking"
    }
    for agent in priority:
        if agent in blocking_agents:
            return agent
    return None


# ─── Final Reviewer ─────────────────────────────────────────────────


class FinalReviewer:
    """Reviews the complete creative pipeline output for emergent issues.

    Runs structural checks and optionally VLM analysis on rendered frames.
    Produces a FinalReviewResult with failure attribution.
    """

    async def review(
        self,
        manifest: SceneManifest,
        path_output: PathOutput,
        connector_output: ConnectorOutput,
        camera_output: CameraOutput | None = None,
        *,
        rendered_frames: list[str] | None = None,
        history: IterationHistory | None = None,
    ) -> FinalReviewResult:
        """Run the final review.

        Args:
            manifest: Scene layout.
            path_output: Routed path.
            connector_output: Connected path.
            camera_output: Camera trajectory (if computed).
            rendered_frames: Paths to rendered frame images (for VLM).
            history: Pipeline iteration history.

        Returns:
            FinalReviewResult with pass/fail and attribution.
        """
        all_issues: list[ReviewIssue] = []

        # Structural checks
        all_issues.extend(_check_chain_coverage(path_output, connector_output))
        all_issues.extend(_check_camera_coverage(camera_output))
        all_issues.extend(_check_scene_coherence(manifest, path_output))

        # VLM review on rendered frames (if available)
        if rendered_frames:
            vlm_issues = await self._vlm_review(rendered_frames, manifest)
            all_issues.extend(vlm_issues)

        blocking = [i for i in all_issues if i.severity == "blocking"]
        passed = len(blocking) == 0
        cascade_from = _earliest_responsible_agent(all_issues) if not passed else None

        summary_parts = []
        if passed:
            summary_parts.append("All checks passed.")
        else:
            for issue in blocking:
                summary_parts.append(
                    f"[{issue.attributed_to.value}] {issue.description}"
                )

        result = FinalReviewResult(
            passed=passed,
            issues=all_issues,
            cascade_from=cascade_from,
            summary="; ".join(summary_parts),
        )

        logger.info(
            "[final_reviewer] %s — %d issues (%d blocking), cascade=%s",
            "PASS" if passed else "FAIL",
            len(all_issues),
            len(blocking),
            cascade_from.value if cascade_from else "none",
        )
        return result

    async def _vlm_review(
        self,
        frame_paths: list[str],
        manifest: SceneManifest,
    ) -> list[ReviewIssue]:
        """Use a VLM to analyse rendered frames for visual issues.

        This is optional — the pipeline works with structural checks alone.
        VLM adds detection of physics anomalies and visual coherence issues
        that can't be caught from data alone.
        """
        issues: list[ReviewIssue] = []

        try:
            step_cfg = _get_step_config_safe("video_review")
            model = _resolve_model(step_cfg)

            prompt = (
                "You are reviewing frames from a domino run video.\n"
                f"Scene theme: {manifest.theme}\n"
                f"Narrative: {manifest.narrative}\n\n"
                "Check for:\n"
                "1. Chain breaks (dominos not toppling in sequence)\n"
                "2. Camera occlusion (dominos not visible)\n"
                "3. Physics anomalies (clipping, flying dominos, explosions)\n"
                "4. Visual coherence issues\n\n"
                "For each issue found, respond with JSON array of objects:\n"
                '  {"description": "...", "attributed_to": "set_designer|path_setter|connector|camera_router", '
                '"severity": "blocking|warning"}\n'
                "If no issues, respond with an empty array: []"
            )

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Review these {len(frame_paths)} frames:"},
                    *[
                        {"type": "image_url", "image_url": {"url": f"file://{p}"}}
                        for p in frame_paths[:16]  # cap at 16 frames
                    ],
                ]},
            ]

            response = await call_llm(
                model=model,
                messages=messages,
                cache_step="final_reviewer_vlm",
            )

            # Parse VLM response — expect JSON array
            import json
            text = response if isinstance(response, str) else str(response)
            # Extract JSON from response
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                for item in parsed:
                    if isinstance(item, dict) and "description" in item:
                        agent_str = item.get("attributed_to", "camera_router")
                        try:
                            agent = AgentRole(agent_str)
                        except ValueError:
                            agent = AgentRole.CAMERA_ROUTER
                        issues.append(ReviewIssue(
                            description=item["description"],
                            attributed_to=agent,
                            severity=item.get("severity", "warning"),
                        ))

        except Exception:
            logger.debug("[final_reviewer] VLM review failed, using structural checks only", exc_info=True)

        return issues


# ─── Helpers ─────────────────────────────────────────────────────────


def _get_step_config_safe(step_name: str):
    try:
        return get_step_config(step_name)
    except Exception:
        return None


def _resolve_model(step_cfg) -> str:
    if step_cfg:
        try:
            return step_cfg.resolve_model()
        except Exception:
            pass
    return "video-review"
