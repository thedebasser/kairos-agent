"""Unit tests for PhysicsSimulationAgent.

Tests simulation code generation, sandbox execution, validation,
parameter adjustment, stats extraction, and the full run_loop.
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
from kairos.models.contracts import (
    ConceptBrief,
    PipelineState,
    PipelineStatus,
    SimulationResult,
    SimulationStats,
    ValidationCheck,
    ValidationResult,
)
from kairos.models.simulation import AdjustedSimulationCode, SimulationCode
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
        pipeline_state: PipelineState,
    ) -> None:
        """Should call the LLM with a category-specific prompt and return code."""
        mock_routing.call_llm = AsyncMock(
            return_value=SimulationCode(
                code="import pygame\nprint('hello')",
                reasoning="Simple test",
            )
        )

        code = await agent.generate_simulation(sample_concept, pipeline_state)

        assert "import pygame" in code
        mock_routing.call_llm.assert_awaited_once()
        call_kwargs = mock_routing.call_llm.call_args
        assert call_kwargs.kwargs.get("model") or call_kwargs.args[0] == "simulation-first-pass"

    @patch("kairos.pipelines.physics.simulation_agent.llm_routing")
    async def test_prompt_contains_concept_details(
        self,
        mock_routing: MagicMock,
        agent: PhysicsSimulationAgent,
        sample_concept: ConceptBrief,
        pipeline_state: PipelineState,
    ) -> None:
        """The prompt should include concept title, body counts, colours, etc."""
        mock_routing.call_llm = AsyncMock(
            return_value=SimulationCode(code="code", reasoning="test")
        )

        # Verify prompt building works
        prompt = agent._build_generation_prompt(sample_concept)  # noqa: SLF001
        assert sample_concept.title in prompt
        assert str(sample_concept.simulation_requirements.body_count_initial) in prompt
        assert str(sample_concept.simulation_requirements.body_count_max) in prompt
        assert sample_concept.simulation_requirements.background_colour in prompt

    def test_prompt_template_missing_raises(
        self,
        agent: PhysicsSimulationAgent,
        sample_concept: ConceptBrief,
    ) -> None:
        """Should raise FileNotFoundError if the template doesn't exist."""
        agent._prompts_dir = Path("/nonexistent/prompts")  # noqa: SLF001
        with pytest.raises(FileNotFoundError):
            agent._build_generation_prompt(sample_concept)  # noqa: SLF001


# ---------------------------------------------------------------------------
# execute_simulation
# ---------------------------------------------------------------------------


class TestExecuteSimulation:
    """Tests for execute_simulation."""

    @patch("kairos.pipelines.physics.simulation_agent.sandbox")
    async def test_delegates_to_sandbox(
        self,
        mock_sandbox: MagicMock,
        agent: PhysicsSimulationAgent,
        passing_sim_result: SimulationResult,
    ) -> None:
        """Should delegate to sandbox.execute_simulation in an executor."""
        mock_sandbox.execute_simulation = MagicMock(return_value=passing_sim_result)

        result = await agent.execute_simulation("import pygame")

        assert result.returncode == 0
        assert result.output_files == ["/workspace/output/simulation.mp4"]
        mock_sandbox.execute_simulation.assert_called_once()

    @patch("kairos.pipelines.physics.simulation_agent.sandbox")
    async def test_propagates_timeout_error(
        self,
        mock_sandbox: MagicMock,
        agent: PhysicsSimulationAgent,
    ) -> None:
        """Should propagate SimulationTimeoutError from sandbox."""
        mock_sandbox.execute_simulation = MagicMock(
            side_effect=SimulationTimeoutError("Exceeded 300s timeout")
        )

        with pytest.raises(SimulationTimeoutError):
            await agent.execute_simulation("import time; time.sleep(999)")

    @patch("kairos.pipelines.physics.simulation_agent.sandbox")
    async def test_propagates_oom_error(
        self,
        mock_sandbox: MagicMock,
        agent: PhysicsSimulationAgent,
    ) -> None:
        """Should propagate SimulationOOMError from sandbox."""
        mock_sandbox.execute_simulation = MagicMock(
            side_effect=SimulationOOMError("OOM kill")
        )

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
        mock_validation.validate_simulation = MagicMock(return_value=passing_validation)

        result = await agent.validate_output("/path/to/video.mp4")

        assert result.passed is True
        mock_validation.validate_simulation.assert_called_once_with(
            "/path/to/video.mp4", run_tier2=False
        )


# ---------------------------------------------------------------------------
# adjust_parameters
# ---------------------------------------------------------------------------


class TestAdjustParameters:
    """Tests for adjust_parameters."""

    @patch("kairos.pipelines.physics.simulation_agent.llm_routing")
    async def test_sends_failed_checks_to_llm(
        self,
        mock_routing: MagicMock,
        agent: PhysicsSimulationAgent,
        failing_validation: ValidationResult,
    ) -> None:
        """Should include failed check details in the adjustment prompt."""
        mock_routing.call_with_quality_fallback = AsyncMock(
            return_value=AdjustedSimulationCode(
                code="import pygame\n# fixed",
                changes_made=["Changed resolution to 1080x1920"],
                reasoning="Swapped width and height",
            )
        )

        code = await agent.adjust_parameters(
            "import pygame\n# original", failing_validation, 2
        )

        assert "fixed" in code
        call_args = mock_routing.call_with_quality_fallback.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[2]
        user_msg = messages[-1]["content"]
        assert "resolution" in user_msg.lower()
        assert "1920x1080" in user_msg

    @patch("kairos.pipelines.physics.simulation_agent.llm_routing")
    async def test_uses_quality_fallback(
        self,
        mock_routing: MagicMock,
        agent: PhysicsSimulationAgent,
        failing_validation: ValidationResult,
    ) -> None:
        """Should use sim-param-adjust as primary with simulation-debugger fallback."""
        mock_routing.call_with_quality_fallback = AsyncMock(
            return_value=AdjustedSimulationCode(code="fixed", changes_made=[])
        )

        await agent.adjust_parameters("code", failing_validation, 1)

        call_args = mock_routing.call_with_quality_fallback.call_args
        primary = call_args.kwargs.get("primary_model") or call_args.args[0]
        fallback = call_args.kwargs.get("fallback_model") or call_args.args[1]
        assert primary == "sim-param-adjust"
        assert fallback == "simulation-debugger"


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

    @patch("kairos.pipelines.physics.simulation_agent.validation")
    @patch("kairos.pipelines.physics.simulation_agent.sandbox")
    @patch("kairos.pipelines.physics.simulation_agent.llm_routing")
    async def test_passes_on_first_iteration(
        self,
        mock_routing: MagicMock,
        mock_sandbox: MagicMock,
        mock_validation: MagicMock,
        agent: PhysicsSimulationAgent,
        sample_concept: ConceptBrief,
        pipeline_state: PipelineState,
        passing_sim_result: SimulationResult,
        passing_validation: ValidationResult,
    ) -> None:
        """Happy path: generate → execute → validate passes on first try."""
        mock_routing.call_llm = AsyncMock(
            return_value=SimulationCode(code="import pygame\n# simulation", reasoning="ok")
        )
        mock_sandbox.execute_simulation = MagicMock(return_value=passing_sim_result)
        mock_validation.validate_simulation = MagicMock(return_value=passing_validation)

        # Mock ffprobe via the agent method
        with patch.object(agent, "_ffprobe", return_value={
            "format": {"duration": "65.0", "size": "45000000"},
            "streams": [{"codec_type": "video", "r_frame_rate": "30/1", "nb_frames": "1950"}],
        }):
            state = await agent.run_loop(sample_concept, pipeline_state)

        assert state.status == PipelineStatus.SIMULATION_PHASE
        assert state.simulation_iteration == 1
        assert state.simulation_code == "import pygame\n# simulation"
        assert state.raw_video_path == "/workspace/output/simulation.mp4"
        assert state.validation_result is not None
        assert state.validation_result.passed is True
        assert state.simulation_stats is not None

    @patch("kairos.pipelines.physics.simulation_agent.validation")
    @patch("kairos.pipelines.physics.simulation_agent.sandbox")
    @patch("kairos.pipelines.physics.simulation_agent.llm_routing")
    async def test_adjusts_and_retries_on_failure(
        self,
        mock_routing: MagicMock,
        mock_sandbox: MagicMock,
        mock_validation: MagicMock,
        agent: PhysicsSimulationAgent,
        sample_concept: ConceptBrief,
        pipeline_state: PipelineState,
        passing_sim_result: SimulationResult,
        passing_validation: ValidationResult,
        failing_validation: ValidationResult,
    ) -> None:
        """Should adjust code and retry when validation fails."""
        mock_routing.call_llm = AsyncMock(
            return_value=SimulationCode(code="original code", reasoning="ok")
        )
        mock_routing.call_with_quality_fallback = AsyncMock(
            return_value=AdjustedSimulationCode(
                code="fixed code", changes_made=["Fix resolution"]
            )
        )
        mock_sandbox.execute_simulation = MagicMock(return_value=passing_sim_result)
        # First validation fails, second passes
        mock_validation.validate_simulation = MagicMock(
            side_effect=[failing_validation, passing_validation]
        )

        with patch.object(agent, "_ffprobe", return_value={
            "format": {"duration": "65.0", "size": "45000000"},
            "streams": [{"codec_type": "video", "r_frame_rate": "30/1", "nb_frames": "1950"}],
        }):
            state = await agent.run_loop(sample_concept, pipeline_state)

        assert state.simulation_iteration == 2
        assert state.simulation_code == "fixed code"
        assert state.validation_result is not None
        assert state.validation_result.passed is True
        assert mock_routing.call_with_quality_fallback.await_count == 1

    @patch("kairos.pipelines.physics.simulation_agent.validation")
    @patch("kairos.pipelines.physics.simulation_agent.sandbox")
    @patch("kairos.pipelines.physics.simulation_agent.llm_routing")
    async def test_handles_execution_failure(
        self,
        mock_routing: MagicMock,
        mock_sandbox: MagicMock,
        mock_validation: MagicMock,
        agent: PhysicsSimulationAgent,
        sample_concept: ConceptBrief,
        pipeline_state: PipelineState,
        passing_sim_result: SimulationResult,
        passing_validation: ValidationResult,
    ) -> None:
        """Should catch execution errors, adjust, and retry."""
        mock_routing.call_llm = AsyncMock(
            return_value=SimulationCode(code="broken code", reasoning="ok")
        )
        mock_routing.call_with_quality_fallback = AsyncMock(
            return_value=AdjustedSimulationCode(
                code="fixed code", changes_made=["Fix import"]
            )
        )
        # First execution fails, second succeeds
        mock_sandbox.execute_simulation = MagicMock(
            side_effect=[
                SimulationExecutionError("Docker failed"),
                passing_sim_result,
            ]
        )
        mock_validation.validate_simulation = MagicMock(return_value=passing_validation)

        with patch.object(agent, "_ffprobe", return_value={
            "format": {"duration": "65.0", "size": "45000000"},
            "streams": [{"codec_type": "video", "r_frame_rate": "30/1", "nb_frames": "1950"}],
        }):
            state = await agent.run_loop(sample_concept, pipeline_state)

        assert len(state.errors) >= 1
        assert state.simulation_iteration == 2
        assert state.validation_result is not None
        assert state.validation_result.passed is True

    @patch("kairos.pipelines.physics.simulation_agent.validation")
    @patch("kairos.pipelines.physics.simulation_agent.sandbox")
    @patch("kairos.pipelines.physics.simulation_agent.llm_routing")
    async def test_max_iterations_reached(
        self,
        mock_routing: MagicMock,
        mock_sandbox: MagicMock,
        mock_validation: MagicMock,
        agent: PhysicsSimulationAgent,
        sample_concept: ConceptBrief,
        pipeline_state: PipelineState,
        passing_sim_result: SimulationResult,
        failing_validation: ValidationResult,
    ) -> None:
        """Should stop after max iterations and record error."""
        # Override max iterations to 2 for faster test
        agent._settings.max_simulation_iterations = 2  # noqa: SLF001

        mock_routing.call_llm = AsyncMock(
            return_value=SimulationCode(code="code", reasoning="ok")
        )
        mock_routing.call_with_quality_fallback = AsyncMock(
            return_value=AdjustedSimulationCode(code="code v2", changes_made=["fix"])
        )
        mock_sandbox.execute_simulation = MagicMock(return_value=passing_sim_result)
        mock_validation.validate_simulation = MagicMock(return_value=failing_validation)

        with patch.object(agent, "_ffprobe", return_value={
            "format": {"duration": "65.0", "size": "45000000"},
            "streams": [{"codec_type": "video", "r_frame_rate": "30/1", "nb_frames": "1950"}],
        }):
            state = await agent.run_loop(sample_concept, pipeline_state)

        assert state.simulation_iteration == 2
        assert state.validation_result is not None
        assert state.validation_result.passed is False
        assert any("Max iterations" in e for e in state.errors)

    @patch("kairos.pipelines.physics.simulation_agent.sandbox")
    @patch("kairos.pipelines.physics.simulation_agent.llm_routing")
    async def test_no_video_output_triggers_retry(
        self,
        mock_routing: MagicMock,
        mock_sandbox: MagicMock,
        agent: PhysicsSimulationAgent,
        sample_concept: ConceptBrief,
        pipeline_state: PipelineState,
    ) -> None:
        """Should retry when sandbox produces no MP4 file."""
        agent._settings.max_simulation_iterations = 2  # noqa: SLF001

        mock_routing.call_llm = AsyncMock(
            return_value=SimulationCode(code="code", reasoning="ok")
        )
        mock_routing.call_with_quality_fallback = AsyncMock(
            return_value=AdjustedSimulationCode(code="code v2", changes_made=["fix"])
        )
        # Both executions succeed but produce no video
        no_video_result = SimulationResult(
            returncode=0,
            stdout="done",
            stderr="",
            output_files=["/workspace/output/debug.log"],
        )
        mock_sandbox.execute_simulation = MagicMock(return_value=no_video_result)

        state = await agent.run_loop(sample_concept, pipeline_state)

        assert state.simulation_iteration == 2
        assert any("No MP4" in e for e in state.errors)


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
        from kairos.pipelines.physics.adapter import PhysicsPipelineAdapter

        adapter = PhysicsPipelineAdapter()
        agent = adapter.get_simulation_agent()
        assert isinstance(agent, PhysicsSimulationAgent)
