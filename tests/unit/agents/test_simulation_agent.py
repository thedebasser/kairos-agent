"""Unit tests for PhysicsSimulationAgent.

Tests simulation code generation, sandbox execution, validation,
stats extraction, and the full run_loop.
All LLM and sandbox calls are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kairos.exceptions import (
    SimulationExecutionError,
    SimulationOOMError,
    SimulationTimeoutError,
)
from kairos.schemas.contracts import (
    ConceptBrief,
    PipelineState,
    PipelineStatus,
    SimulationLoopResult,
    SimulationResult,
    SimulationStats,
    ValidationCheck,
    ValidationResult,
)
from kairos.schemas.simulation import AdjustedSimulationConfig, SimulationCode, SimulationConfigOutput
from kairos.pipelines.physics.simulation_agent import PhysicsSimulationAgent

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent() -> PhysicsSimulationAgent:
    """Create a PhysicsSimulationAgent with test settings."""
    return PhysicsSimulationAgent()


@pytest.fixture
def passing_sim_result() -> SimulationResult:
    """A successful simulation result with video output."""
    return SimulationResult(
        returncode=0,
        stdout=(
            "Rendering frames...\n"
            "PAYOFF_TIMESTAMP=42.5\n"
            "PEAK_BODY_COUNT=247\n"
            "Done. 1950 frames written."
        ),
        stderr="",
        output_files=["/workspace/output/simulation.mp4"],
        execution_time_sec=90.0,
    )


@pytest.fixture
def failing_sim_result() -> SimulationResult:
    """A simulation result with no output file."""
    return SimulationResult(
        returncode=1,
        stdout="",
        stderr="ImportError: No module named 'nonexistent'",
        output_files=[],
        execution_time_sec=2.0,
    )


@pytest.fixture
def passing_validation() -> ValidationResult:
    """All checks pass."""
    return ValidationResult(
        passed=True,
        tier1_passed=True,
        checks=[
            ValidationCheck(name="valid_mp4", passed=True, message="OK"),
            ValidationCheck(name="resolution", passed=True, message="1080x1920"),
            ValidationCheck(name="fps", passed=True, message="30"),
            ValidationCheck(name="duration", passed=True, message="65.0s"),
        ],
    )


@pytest.fixture
def failing_validation() -> ValidationResult:
    """Resolution check fails."""
    return ValidationResult(
        passed=False,
        tier1_passed=False,
        checks=[
            ValidationCheck(name="valid_mp4", passed=True, message="OK"),
            ValidationCheck(
                name="resolution",
                passed=False,
                message="1920x1080 (expected 1080x1920)",
                value="1920x1080",
                threshold="1080x1920",
            ),
            ValidationCheck(name="duration", passed=True, message="65.0s"),
        ],
    )


# ---------------------------------------------------------------------------
# generate_simulation
# ---------------------------------------------------------------------------


class TestGenerateSimulation:
    """Tests for generate_simulation."""

    @patch("kairos.pipelines.physics.simulation_agent.llm_routing")
    async def test_generates_code_from_concept(
        self,
        mock_routing: MagicMock,
        agent: PhysicsSimulationAgent,
        sample_concept: ConceptBrief,
    ) -> None:
        """Should call the LLM with a category-specific prompt and return code."""
        mock_routing.call_llm = AsyncMock(
            return_value=SimulationConfigOutput(
                config={"layout": "grid", "ball_count": 50},
                reasoning="Simple test",
            )
        )

        code = await agent.generate_simulation(sample_concept)

        assert isinstance(code, str)
        mock_routing.call_llm.assert_awaited_once()

    @patch("kairos.pipelines.physics.simulation_agent.llm_routing")
    async def test_prompt_contains_concept_details(
        self,
        mock_routing: MagicMock,
        agent: PhysicsSimulationAgent,
        sample_concept: ConceptBrief,
    ) -> None:
        """The generated config call should use concept details."""
        mock_routing.call_llm = AsyncMock(
            return_value=SimulationConfigOutput(
                config={"layout": "grid", "ball_count": 50},
                reasoning="Simple test",
            )
        )

        code = await agent.generate_simulation(sample_concept)

        # Verify the LLM was called with messages containing concept info
        call_kwargs = mock_routing.call_llm.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages", [])
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        assert sample_concept.title in user_msg

    @patch("kairos.pipelines.physics.simulation_agent.llm_routing")
    async def test_returns_json_config(
        self,
        mock_routing: MagicMock,
        agent: PhysicsSimulationAgent,
        sample_concept: ConceptBrief,
    ) -> None:
        """Should return a JSON string from the LLM config output."""
        mock_routing.call_llm = AsyncMock(
            return_value=SimulationConfigOutput(
                config={"layout": "grid"},
                reasoning="test",
            )
        )

        code = await agent.generate_simulation(sample_concept)
        parsed = json.loads(code)
        assert parsed["layout"] == "grid"


# ---------------------------------------------------------------------------
# execute_simulation
# ---------------------------------------------------------------------------


class TestExecuteSimulation:
    """Tests for execute_simulation."""

    @patch("kairos.pipelines.physics.simulation_agent.run_blender_script")
    async def test_delegates_to_sandbox(
        self,
        mock_blender: MagicMock,
        agent: PhysicsSimulationAgent,
        passing_sim_result: SimulationResult,
    ) -> None:
        """Should delegate to run_blender_script (async)."""
        mock_blender.return_value = passing_sim_result

        result = await agent.execute_simulation('{"scene": "test"}')

        assert result.returncode == 0
        assert result.output_files == ["/workspace/output/simulation.mp4"]
        mock_blender.assert_called_once()

    @patch("kairos.pipelines.physics.simulation_agent.run_blender_script")
    async def test_propagates_timeout_error(
        self,
        mock_blender: MagicMock,
        agent: PhysicsSimulationAgent,
    ) -> None:
        """Should propagate SimulationTimeoutError from executor."""
        mock_blender.side_effect = SimulationTimeoutError("Exceeded 300s timeout")

        with pytest.raises(SimulationTimeoutError):
            await agent.execute_simulation("import time; time.sleep(999)")

    @patch("kairos.pipelines.physics.simulation_agent.run_blender_script")
    async def test_propagates_oom_error(
        self,
        mock_blender: MagicMock,
        agent: PhysicsSimulationAgent,
    ) -> None:
        """Should propagate SimulationOOMError from executor."""
        mock_blender.side_effect = SimulationOOMError("OOM kill")

        with pytest.raises(SimulationOOMError):
            await agent.execute_simulation("x = [0] * 10**10")

        with pytest.raises(SimulationOOMError):
            await agent.execute_simulation("x = [0] * 10**10")


# ---------------------------------------------------------------------------
# validate_output
# ---------------------------------------------------------------------------


class TestValidateOutput:
    """Tests for validate_output."""

    @patch("kairos.pipelines.physics.simulation_agent.validation")
    async def test_delegates_to_validation_service(
        self,
        mock_validation: MagicMock,
        agent: PhysicsSimulationAgent,
        passing_validation: ValidationResult,
    ) -> None:
        """Should call validation.validate_simulation and return the result."""
        mock_validation.validate_simulation = AsyncMock(return_value=passing_validation)

        result = await agent.validate_output("/path/to/video.mp4")

        assert result.passed is True
        mock_validation.validate_simulation.assert_called_once_with(
            "/path/to/video.mp4", run_tier2=False, skip_checks={"audio_present"}
        )


# ---------------------------------------------------------------------------
# get_simulation_stats
# ---------------------------------------------------------------------------


class TestGetSimulationStats:
    """Tests for get_simulation_stats."""

    @patch.object(PhysicsSimulationAgent, "_ffprobe")
    async def test_extracts_stats_from_ffprobe(
        self,
        mock_ffprobe: MagicMock,
        agent: PhysicsSimulationAgent,
    ) -> None:
        """Should extract duration, fps, frame count, file size from probe data."""
        mock_ffprobe.return_value = {
            "format": {"duration": "65.0", "size": "45000000"},
            "streams": [
                {
                    "codec_type": "video",
                    "r_frame_rate": "30/1",
                    "nb_frames": "1950",
                }
            ],
        }

        stats = await agent.get_simulation_stats("/path/to/video.mp4")

        assert stats.duration_sec == 65.0
        assert stats.avg_fps == 30.0
        assert stats.total_frames == 1950
        assert stats.file_size_bytes == 45000000

    @patch.object(PhysicsSimulationAgent, "_ffprobe")
    async def test_handles_missing_ffprobe_data(
        self,
        mock_ffprobe: MagicMock,
        agent: PhysicsSimulationAgent,
    ) -> None:
        """Should return zeros for missing probe data."""
        mock_ffprobe.return_value = {}

        stats = await agent.get_simulation_stats("/path/to/video.mp4")

        assert stats.duration_sec == 0.0
        assert stats.total_frames == 0


class TestEnrichStatsFromStdout:
    """Tests for _enrich_stats_from_stdout."""

    def test_parses_payoff_and_peak(self, agent: PhysicsSimulationAgent) -> None:
        """Should extract PAYOFF_TIMESTAMP and PEAK_BODY_COUNT from stdout."""
        base = SimulationStats(
            duration_sec=65.0,
            peak_body_count=0,
            avg_fps=30.0,
            min_fps=30.0,
            payoff_timestamp_sec=0.0,
            total_frames=1950,
            file_size_bytes=45000000,
        )
        stdout = "PAYOFF_TIMESTAMP=42.5\nPEAK_BODY_COUNT=247\n"

        enriched = agent._enrich_stats_from_stdout(base, stdout)  # noqa: SLF001

        assert enriched.payoff_timestamp_sec == 42.5
        assert enriched.peak_body_count == 247

    def test_handles_no_markers(self, agent: PhysicsSimulationAgent) -> None:
        """Should return original stats if no markers in stdout."""
        base = SimulationStats(
            duration_sec=65.0,
            peak_body_count=0,
            avg_fps=30.0,
            min_fps=30.0,
            payoff_timestamp_sec=0.0,
            total_frames=1950,
            file_size_bytes=45000000,
        )

        result = agent._enrich_stats_from_stdout(base, "no markers here")  # noqa: SLF001

        assert result is base  # Same object — no enrichment needed


# ---------------------------------------------------------------------------
# run_loop
# ---------------------------------------------------------------------------


class TestRunLoop:
    """Tests for the full simulation iteration loop."""

    async def test_passes_on_first_iteration(
        self,
        agent: PhysicsSimulationAgent,
        sample_concept: ConceptBrief,
        passing_sim_result: SimulationResult,
        passing_validation: ValidationResult,
    ) -> None:
        """Happy path: generate → execute → validate passes on first try."""
        with (
            patch.object(agent, "generate_simulation", new_callable=AsyncMock, return_value='{"scene": "domino_course"}'),
            patch.object(agent, "execute_simulation", new_callable=AsyncMock, return_value=passing_sim_result),
            patch.object(agent, "validate_output", new_callable=AsyncMock, return_value=passing_validation),
            patch.object(agent, "_ffprobe", return_value={
                "format": {"duration": "65.0", "size": "45000000"},
                "streams": [{"codec_type": "video", "r_frame_rate": "30/1", "nb_frames": "1950"}],
            }),
        ):
            result = await agent.run_loop(sample_concept)

        assert result.simulation_iteration == 1
        assert result.simulation_code == '{"scene": "domino_course"}'
        assert result.raw_video_path == "/workspace/output/simulation.mp4"
        assert result.validation_result is not None
        assert result.validation_result.passed is True
        assert result.simulation_stats is not None

    async def test_adjusts_and_retries_on_failure(
        self,
        agent: PhysicsSimulationAgent,
        sample_concept: ConceptBrief,
        passing_sim_result: SimulationResult,
        passing_validation: ValidationResult,
        failing_validation: ValidationResult,
    ) -> None:
        """Should retry execution when validation fails."""
        with (
            patch.object(agent, "generate_simulation", new_callable=AsyncMock, return_value="original code"),
            patch.object(agent, "execute_simulation", new_callable=AsyncMock, return_value=passing_sim_result),
            patch.object(agent, "validate_output", new_callable=AsyncMock, side_effect=[failing_validation, passing_validation]),
            patch.object(agent, "_ffprobe", return_value={
                "format": {"duration": "65.0", "size": "45000000"},
                "streams": [{"codec_type": "video", "r_frame_rate": "30/1", "nb_frames": "1950"}],
            }),
            patch.object(agent, "_check_completion_from_stdout", return_value=(True, 1.0, "OK")),
        ):
            result = await agent.run_loop(sample_concept)

        # The agent generates once then retries execution with the same config
        assert result.simulation_iteration == 2
        assert result.validation_result is not None
        assert result.validation_result.passed is True

    async def test_handles_execution_failure(
        self,
        agent: PhysicsSimulationAgent,
        sample_concept: ConceptBrief,
        passing_sim_result: SimulationResult,
        passing_validation: ValidationResult,
    ) -> None:
        """Should catch execution errors and retry."""
        with (
            patch.object(agent, "generate_simulation", new_callable=AsyncMock, return_value="code"),
            patch.object(agent, "execute_simulation", new_callable=AsyncMock,
                side_effect=[SimulationExecutionError("Docker failed"), passing_sim_result]),
            patch.object(agent, "validate_output", new_callable=AsyncMock, return_value=passing_validation),
            patch.object(agent, "_ffprobe", return_value={
                "format": {"duration": "65.0", "size": "45000000"},
                "streams": [{"codec_type": "video", "r_frame_rate": "30/1", "nb_frames": "1950"}],
            }),
        ):
            result = await agent.run_loop(sample_concept)

        assert len(result.errors) >= 1
        assert result.simulation_iteration == 2
        assert result.validation_result is not None
        assert result.validation_result.passed is True

    async def test_max_iterations_reached(
        self,
        agent: PhysicsSimulationAgent,
        sample_concept: ConceptBrief,
        passing_sim_result: SimulationResult,
        failing_validation: ValidationResult,
    ) -> None:
        """Should stop after max iterations and record error."""
        # Override max iterations to 2 for faster test
        agent._settings.max_simulation_iterations = 2  # noqa: SLF001

        with (
            patch.object(agent, "generate_simulation", new_callable=AsyncMock, return_value="code"),
            patch.object(agent, "execute_simulation", new_callable=AsyncMock, return_value=passing_sim_result),
            patch.object(agent, "validate_output", new_callable=AsyncMock, return_value=failing_validation),
            patch.object(agent, "_ffprobe", return_value={
                "format": {"duration": "65.0", "size": "45000000"},
                "streams": [{"codec_type": "video", "r_frame_rate": "30/1", "nb_frames": "1950"}],
            }),
        ):
            result = await agent.run_loop(sample_concept)

        assert result.simulation_iteration == 2
        assert result.validation_result is not None
        assert result.validation_result.passed is False
        assert any("Max iterations" in e for e in result.errors)

    async def test_no_video_output_triggers_retry(
        self,
        agent: PhysicsSimulationAgent,
        sample_concept: ConceptBrief,
    ) -> None:
        """Should retry when sandbox produces no MP4 file."""
        agent._settings.max_simulation_iterations = 2  # noqa: SLF001

        # Both executions succeed but produce no video
        no_video_result = SimulationResult(
            returncode=0,
            stdout="done",
            stderr="",
            output_files=["/workspace/output/debug.log"],
        )

        with (
            patch.object(agent, "generate_simulation", new_callable=AsyncMock, return_value="code"),
            patch.object(agent, "execute_simulation", new_callable=AsyncMock, return_value=no_video_result),
        ):
            result = await agent.run_loop(sample_concept)

        assert result.simulation_iteration == 2
        assert any("No MP4" in e for e in result.errors)


# ---------------------------------------------------------------------------
# _find_video
# ---------------------------------------------------------------------------


class TestFindVideo:
    """Tests for the _find_video helper."""

    def test_finds_mp4(self) -> None:
        result = SimulationResult(
            returncode=0,
            output_files=["/workspace/output/debug.log", "/workspace/output/simulation.mp4"],
        )
        assert PhysicsSimulationAgent._find_video(result) == "/workspace/output/simulation.mp4"

    def test_finds_uppercase_mp4(self) -> None:
        result = SimulationResult(
            returncode=0,
            output_files=["/workspace/output/VIDEO.MP4"],
        )
        assert PhysicsSimulationAgent._find_video(result) == "/workspace/output/VIDEO.MP4"

    def test_returns_none_when_no_mp4(self) -> None:
        result = SimulationResult(
            returncode=0,
            output_files=["/workspace/output/debug.log"],
        )
        assert PhysicsSimulationAgent._find_video(result) is None

    def test_returns_none_when_empty(self) -> None:
        result = SimulationResult(returncode=0, output_files=[])
        assert PhysicsSimulationAgent._find_video(result) is None


# ---------------------------------------------------------------------------
# Adapter integration
# ---------------------------------------------------------------------------


class TestAdapterIntegration:
    """Test that the adapter returns the simulation agent correctly."""

    def test_adapter_returns_simulation_agent(self) -> None:
        """PhysicsPipelineAdapter.get_simulation_agent should return PhysicsSimulationAgent."""
        from kairos.pipelines.adapters.physics_adapter import PhysicsPipelineAdapter

        adapter = PhysicsPipelineAdapter()
        agent = adapter.get_simulation_agent()
        assert isinstance(agent, PhysicsSimulationAgent)
