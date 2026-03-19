"""Unit tests for the Physics Video Editor Agent.

Tests all four VideoEditorAgent methods:
- select_music() — programmatic music selection
- generate_captions() — LLM-powered hook caption generation
- generate_title() — LLM-powered title generation
- compose_video() — FFmpeg video assembly

Also tests:
- Response models (HookCaptionResponse, VideoTitleResponse)
- FFmpeg compositor service
- Adapter integration
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kairos.exceptions import VideoAssemblyError
from kairos.schemas.contracts import (
    AudioBrief,
    Caption,
    CaptionSet,
    CaptionType,
    ConceptBrief,
    EnergyLevel,
    MusicTrackMetadata,
    ScenarioCategory,
    SimulationRequirements,
    SimulationStats,
    VideoOutput,
)
from kairos.schemas.video_editor import HookCaptionResponse, VideoTitleResponse
from kairos.pipelines.physics.video_editor_agent import (
    _caption_writer_model,
    _title_writer_model,
    PhysicsVideoEditorAgent,
)
from kairos.services.caption import (
    CAPTION_FADE_IN_SEC,
    CAPTION_FADE_OUT_SEC,
    CAPTION_FONT,
    CAPTION_FONT_SIZE,
    CAPTION_MAX_WORDS,
    CAPTION_POSITION_Y_PCT,
    CAPTION_STROKE_WIDTH,
    build_caption_set,
    build_ffmpeg_caption_filter,
    ffmpeg_escape_path,
    validate_caption_text,
)
from kairos.services.ffmpeg_compositor import (
    MUSIC_FADE_OUT_SEC,
    MUSIC_VOLUME_DB,
    OUTPUT_AUDIO_BITRATE,
    OUTPUT_AUDIO_CODEC,
    OUTPUT_CODEC,
    OUTPUT_CRF,
    OUTPUT_FPS,
    OUTPUT_HEIGHT,
    OUTPUT_PIXEL_FORMAT,
    OUTPUT_WIDTH,
    build_audio_filter,
    build_caption_filters,
    build_ffmpeg_command,
    build_watermark_filter,
    check_ffmpeg_available,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def agent(tmp_path: Path) -> PhysicsVideoEditorAgent:
    """Create a test video editor agent with temp directories."""
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return PhysicsVideoEditorAgent(
        music_dir=music_dir,
        output_dir=output_dir,
        pipeline_run_id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
    )


@pytest.fixture
def sample_stats() -> SimulationStats:
    """Valid simulation stats for testing."""
    return SimulationStats(
        duration_sec=65.0,
        peak_body_count=247,
        avg_fps=32.0,
        min_fps=28.0,
        payoff_timestamp_sec=48.0,
        total_frames=1950,
        file_size_bytes=45_000_000,
    )


@pytest.fixture
def music_library() -> list[MusicTrackMetadata]:
    """A test music library with multiple tracks."""
    return [
        MusicTrackMetadata(
            track_id="upbeat_120bpm_01",
            filename="upbeat_120bpm_01.mp3",
            source="pixabay",
            duration_sec=120.0,
            bpm=120,
            mood=["upbeat", "energetic"],
            energy_curve="building",
            contentid_status="cleared",
            use_count=0,
        ),
        MusicTrackMetadata(
            track_id="chill_80bpm_01",
            filename="chill_80bpm_01.mp3",
            source="pixabay",
            duration_sec=90.0,
            bpm=80,
            mood=["calm", "relaxing"],
            energy_curve="low",
            contentid_status="cleared",
            use_count=2,
        ),
        MusicTrackMetadata(
            track_id="uncleared_track",
            filename="uncleared.mp3",
            source="pixabay",
            duration_sec=100.0,
            bpm=110,
            mood=["upbeat", "energetic"],
            energy_curve="building",
            contentid_status="unknown",
            use_count=0,
        ),
    ]


# =============================================================================
# Response Model Tests
# =============================================================================


class TestHookCaptionResponse:
    """Tests for HookCaptionResponse Pydantic model."""

    def test_valid_hook(self):
        response = HookCaptionResponse(
            hook_text="Watch them all collide",
            reasoning="Creates curiosity about the collision outcome",
        )
        assert response.hook_text == "Watch them all collide"
        assert len(response.hook_text.split()) <= 6

    def test_hook_strips_whitespace(self):
        response = HookCaptionResponse(
            hook_text="  Watch them collide  ",
            reasoning="test",
        )
        assert response.hook_text == "Watch them collide"

    def test_hook_rejects_more_than_six_words(self):
        with pytest.raises(ValueError, match="<=6 words"):
            HookCaptionResponse(
                hook_text="This hook text has way too many words",
                reasoning="test",
            )

    def test_hook_exactly_six_words(self):
        response = HookCaptionResponse(
            hook_text="One two three four five six",
            reasoning="test",
        )
        assert len(response.hook_text.split()) == 6

    def test_hook_single_word(self):
        response = HookCaptionResponse(
            hook_text="Satisfying",
            reasoning="test",
        )
        assert response.hook_text == "Satisfying"


class TestVideoTitleResponse:
    """Tests for VideoTitleResponse Pydantic model."""

    def test_valid_title(self):
        response = VideoTitleResponse(
            title="When 500 Balls Meet a Marble Funnel",
            description="The most satisfying physics simulation you'll see today.",
        )
        assert response.title == "When 500 Balls Meet a Marble Funnel"

    def test_title_strips_whitespace(self):
        response = VideoTitleResponse(
            title="  Great Title  ",
            description="",
        )
        assert response.title == "Great Title"

    def test_title_rejects_empty(self):
        with pytest.raises(ValueError, match="must not be empty"):
            VideoTitleResponse(title="  ", description="")

    def test_title_default_description(self):
        response = VideoTitleResponse(title="Good Title")
        assert response.description == ""


# =============================================================================
# Music Selection Tests
# =============================================================================


class TestSelectMusic:
    """Tests for PhysicsVideoEditorAgent.select_music()."""

    async def test_selects_matching_track(
        self,
        agent: PhysicsVideoEditorAgent,
        sample_concept: ConceptBrief,
        sample_stats: SimulationStats,
        music_library: list[MusicTrackMetadata],
    ):
        """Agent selects best-matching track from library."""
        with patch(
            "kairos.pipelines.physics.video_editor_agent.load_music_library",
            return_value=music_library,
        ), patch(
            "kairos.pipelines.physics.video_editor_agent.select_music",
            return_value=music_library[0],
        ):
            result = await agent.select_music(sample_concept, sample_stats)
            assert result.track_id == "upbeat_120bpm_01"
            assert result.contentid_status == "cleared"

    async def test_raises_on_empty_library(
        self,
        agent: PhysicsVideoEditorAgent,
        sample_concept: ConceptBrief,
        sample_stats: SimulationStats,
    ):
        """Agent raises VideoAssemblyError when no tracks in library."""
        with patch(
            "kairos.pipelines.physics.video_editor_agent.load_music_library",
            return_value=[],
        ):
            with pytest.raises(VideoAssemblyError, match="No music tracks found"):
                await agent.select_music(sample_concept, sample_stats)

    async def test_raises_when_no_match(
        self,
        agent: PhysicsVideoEditorAgent,
        sample_concept: ConceptBrief,
        sample_stats: SimulationStats,
        music_library: list[MusicTrackMetadata],
    ):
        """Agent raises when selector returns None (no suitable track)."""
        with patch(
            "kairos.pipelines.physics.video_editor_agent.load_music_library",
            return_value=music_library,
        ), patch(
            "kairos.pipelines.physics.video_editor_agent.select_music",
            return_value=None,
        ):
            with pytest.raises(VideoAssemblyError, match="No suitable music track"):
                await agent.select_music(sample_concept, sample_stats)


# =============================================================================
# Caption Generation Tests
# =============================================================================


class TestGenerateCaptions:
    """Tests for PhysicsVideoEditorAgent.generate_captions()."""

    async def test_generates_hook_from_llm(
        self,
        agent: PhysicsVideoEditorAgent,
        sample_concept: ConceptBrief,
    ):
        """Agent calls caption-writer LLM and builds caption set."""
        mock_response = HookCaptionResponse(
            hook_text="Watch every ball collide",
            reasoning="Creates visual curiosity",
        )
        with patch(
            "kairos.pipelines.physics.video_editor_agent.call_llm",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_call:
            result = await agent.generate_captions(sample_concept)

            # Verify LLM was called with correct model
            mock_call.assert_called_once()
            assert mock_call.call_args.kwargs["model"] == _caption_writer_model()
            assert mock_call.call_args.kwargs["response_model"] is HookCaptionResponse

            # Verify caption set
            assert len(result.captions) == 1
            hook = result.hook
            assert hook is not None
            assert hook.caption_type == CaptionType.HOOK
            assert hook.text == "Watch every ball collide"
            assert hook.start_sec == 0.0
            assert hook.end_sec == 2.5

    async def test_falls_back_to_concept_hook_on_llm_error(
        self,
        agent: PhysicsVideoEditorAgent,
        sample_concept: ConceptBrief,
    ):
        """Agent uses concept hook_text when LLM fails."""
        with patch(
            "kairos.pipelines.physics.video_editor_agent.call_llm",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM unavailable"),
        ):
            result = await agent.generate_captions(sample_concept)

            hook = result.hook
            assert hook is not None
            assert hook.text == sample_concept.hook_text

    async def test_falls_back_when_llm_returns_invalid_hook(
        self,
        agent: PhysicsVideoEditorAgent,
        sample_concept: ConceptBrief,
    ):
        """Agent uses concept hook_text when LLM returns invalid text."""
        mock_response = HookCaptionResponse(
            hook_text="Valid",
            reasoning="test",
        )
        with patch(
            "kairos.pipelines.physics.video_editor_agent.call_llm",
            new_callable=AsyncMock,
            return_value=mock_response,
        ), patch(
            "kairos.pipelines.physics.video_editor_agent.validate_caption_text",
            return_value=False,
        ):
            result = await agent.generate_captions(sample_concept)

            hook = result.hook
            assert hook is not None
            assert hook.text == sample_concept.hook_text

    async def test_caption_prompt_includes_concept_details(
        self,
        agent: PhysicsVideoEditorAgent,
        sample_concept: ConceptBrief,
    ):
        """Caption writer prompt contains category, title, and visual brief."""
        mock_response = HookCaptionResponse(
            hook_text="Watch them collide",
            reasoning="test",
        )
        with patch(
            "kairos.pipelines.physics.video_editor_agent.call_llm",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_call:
            await agent.generate_captions(sample_concept)

            messages = mock_call.call_args.kwargs["messages"]
            user_msg = messages[1]["content"]
            assert sample_concept.category.value in user_msg
            assert sample_concept.title in user_msg
            assert sample_concept.visual_brief in user_msg


# =============================================================================
# Title Generation Tests
# =============================================================================


class TestGenerateTitle:
    """Tests for PhysicsVideoEditorAgent.generate_title()."""

    async def test_generates_title_from_llm(
        self,
        agent: PhysicsVideoEditorAgent,
        sample_concept: ConceptBrief,
    ):
        """Agent calls title-writer LLM and returns title string."""
        mock_response = VideoTitleResponse(
            title="When 500 Balls Meet a Marble Funnel",
            description="Satisfying physics simulation",
        )
        with patch(
            "kairos.pipelines.physics.video_editor_agent.call_llm",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_call:
            result = await agent.generate_title(sample_concept)

            assert result == "When 500 Balls Meet a Marble Funnel"
            mock_call.assert_called_once()
            assert mock_call.call_args.kwargs["model"] == _title_writer_model()
            assert mock_call.call_args.kwargs["response_model"] is VideoTitleResponse

    async def test_falls_back_to_concept_title_on_error(
        self,
        agent: PhysicsVideoEditorAgent,
        sample_concept: ConceptBrief,
    ):
        """Agent returns concept.title when LLM fails."""
        with patch(
            "kairos.pipelines.physics.video_editor_agent.call_llm",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM timeout"),
        ):
            result = await agent.generate_title(sample_concept)
            assert result == sample_concept.title

    async def test_title_prompt_includes_concept_details(
        self,
        agent: PhysicsVideoEditorAgent,
        sample_concept: ConceptBrief,
    ):
        """Title writer prompt contains category, title, visual brief, and hook."""
        mock_response = VideoTitleResponse(
            title="Test Title",
            description="test",
        )
        with patch(
            "kairos.pipelines.physics.video_editor_agent.call_llm",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_call:
            await agent.generate_title(sample_concept)

            messages = mock_call.call_args.kwargs["messages"]
            user_msg = messages[1]["content"]
            assert sample_concept.category.value in user_msg
            assert sample_concept.title in user_msg
            assert sample_concept.hook_text in user_msg


# =============================================================================
# Video Composition Tests
# =============================================================================


class TestComposeVideo:
    """Tests for PhysicsVideoEditorAgent.compose_video()."""

    async def test_compose_builds_and_runs_ffmpeg(
        self,
        agent: PhysicsVideoEditorAgent,
        sample_concept: ConceptBrief,
        sample_caption_set: CaptionSet,
        sample_music_track: MusicTrackMetadata,
    ):
        """compose_video builds correct FFmpeg command and returns VideoOutput."""
        with patch(
            "kairos.pipelines.physics.video_editor_agent.check_ffmpeg_available",
            return_value=True,
        ), patch(
            "kairos.pipelines.physics.video_editor_agent.run_ffmpeg",
            new_callable=AsyncMock,
            return_value=(0, "", ""),
        ) as mock_ffmpeg, patch(
            "kairos.pipelines.physics.video_editor_agent.call_llm",
            new_callable=AsyncMock,
            return_value=VideoTitleResponse(title="Great Video", description="test"),
        ):
            result = await agent.compose_video(
                raw_video_path="/tmp/raw.mp4",
                music=sample_music_track,
                captions=sample_caption_set,
                concept=sample_concept,
            )

            assert isinstance(result, VideoOutput)
            assert result.title == "Great Video"
            assert result.captions == sample_caption_set
            assert result.music_track == sample_music_track
            assert result.final_video_path.endswith(".mp4")

            # FFmpeg was called
            mock_ffmpeg.assert_called_once()

    async def test_compose_raises_when_ffmpeg_missing(
        self,
        agent: PhysicsVideoEditorAgent,
        sample_concept: ConceptBrief,
        sample_caption_set: CaptionSet,
        sample_music_track: MusicTrackMetadata,
    ):
        """Raises VideoAssemblyError when FFmpeg not found."""
        with patch(
            "kairos.pipelines.physics.video_editor_agent.check_ffmpeg_available",
            return_value=False,
        ):
            with pytest.raises(VideoAssemblyError, match="FFmpeg not found"):
                await agent.compose_video(
                    "/tmp/raw.mp4",
                    sample_music_track,
                    sample_caption_set,
                    sample_concept,
                )

    async def test_compose_raises_on_ffmpeg_failure(
        self,
        agent: PhysicsVideoEditorAgent,
        sample_concept: ConceptBrief,
        sample_caption_set: CaptionSet,
        sample_music_track: MusicTrackMetadata,
    ):
        """Raises VideoAssemblyError when FFmpeg returns non-zero."""
        with patch(
            "kairos.pipelines.physics.video_editor_agent.check_ffmpeg_available",
            return_value=True,
        ), patch(
            "kairos.pipelines.physics.video_editor_agent.run_ffmpeg",
            new_callable=AsyncMock,
            return_value=(1, "", "Error: invalid codec"),
        ):
            with pytest.raises(VideoAssemblyError, match="FFmpeg composition failed"):
                await agent.compose_video(
                    "/tmp/raw.mp4",
                    sample_music_track,
                    sample_caption_set,
                    sample_concept,
                )


# =============================================================================
# FFmpeg Compositor Service Tests
# =============================================================================


class TestBuildAudioFilter:
    """Tests for FFmpeg audio filter generation."""

    def test_audio_filter_volume_and_fade(self):
        result = build_audio_filter(65.0)
        assert f"volume={MUSIC_VOLUME_DB}dB" in result
        assert "afade=t=out" in result
        # Fade starts at duration - fade_out_sec
        assert f"st={65.0 - MUSIC_FADE_OUT_SEC:.1f}" in result
        assert f"d={MUSIC_FADE_OUT_SEC:.1f}" in result

    def test_audio_filter_custom_volume(self):
        result = build_audio_filter(60.0, volume_db=-12)
        assert "volume=-12dB" in result

    def test_audio_filter_custom_fade(self):
        result = build_audio_filter(60.0, fade_out_sec=5.0)
        assert "st=55.0" in result
        assert "d=5.0" in result

    def test_audio_filter_short_duration_no_negative_fade_start(self):
        """Fade start should not go negative for short durations."""
        result = build_audio_filter(2.0, fade_out_sec=5.0)
        assert "st=0" in result


class TestBuildCaptionFilters:
    """Tests for caption filter generation."""

    def test_builds_filters_for_all_captions(self, sample_caption_set: CaptionSet):
        filters = build_caption_filters(sample_caption_set)
        assert len(filters) == len(sample_caption_set.captions)

    def test_filter_contains_drawtext(self, sample_caption_set: CaptionSet):
        filters = build_caption_filters(sample_caption_set)
        for f in filters:
            assert "drawtext=" in f

    def test_empty_caption_set(self):
        empty = CaptionSet(captions=[])
        filters = build_caption_filters(empty)
        assert filters == []


class TestBuildWatermarkFilter:
    """Tests for watermark filter generation."""

    def test_default_bottom_right(self):
        result = build_watermark_filter("logo.png")
        assert "logo.png" in result
        assert "W-w-" in result
        assert "H-h-" in result

    def test_top_left_position(self):
        result = build_watermark_filter("logo.png", position="top_left")
        assert "x=20" in result
        assert "y=20" in result

    def test_custom_opacity(self):
        result = build_watermark_filter("logo.png", opacity=0.3)
        assert "aa=0.3" in result

    def test_custom_size(self):
        result = build_watermark_filter("logo.png", size=80)
        assert "scale=-1:80" in result


class TestBuildFfmpegCommand:
    """Tests for complete FFmpeg command generation."""

    @patch("kairos.services.ffmpeg_compositor._probe_has_audio", return_value=False)
    @patch("kairos.services.ffmpeg_compositor._get_ffmpeg_path", return_value="ffmpeg")
    def test_command_structure(self, _mock_ffmpeg, _mock_probe, sample_caption_set: CaptionSet):
        cmd = build_ffmpeg_command(
            raw_video_path="/tmp/input.mp4",
            music_path="/tmp/music.mp3",
            output_path="/tmp/output.mp4",
            captions=sample_caption_set,
            duration_sec=65.0,
        )
        assert cmd[0] == "ffmpeg"
        assert "-y" in cmd
        assert "/tmp/input.mp4" in cmd
        assert "/tmp/output.mp4" in cmd

    def test_codec_settings(self, sample_caption_set: CaptionSet):
        cmd = build_ffmpeg_command(
            raw_video_path="input.mp4",
            music_path="music.mp3",
            output_path="output.mp4",
            captions=sample_caption_set,
            duration_sec=65.0,
        )
        assert "-c:v" in cmd
        idx = cmd.index("-c:v")
        assert cmd[idx + 1] == OUTPUT_CODEC

        assert "-c:a" in cmd
        idx = cmd.index("-c:a")
        assert cmd[idx + 1] == OUTPUT_AUDIO_CODEC

    def test_pixel_format(self, sample_caption_set: CaptionSet):
        cmd = build_ffmpeg_command(
            raw_video_path="input.mp4",
            music_path="music.mp3",
            output_path="output.mp4",
            captions=sample_caption_set,
            duration_sec=65.0,
        )
        assert "-pix_fmt" in cmd
        idx = cmd.index("-pix_fmt")
        assert cmd[idx + 1] == OUTPUT_PIXEL_FORMAT

    def test_duration_limit(self, sample_caption_set: CaptionSet):
        cmd = build_ffmpeg_command(
            raw_video_path="input.mp4",
            music_path="music.mp3",
            output_path="output.mp4",
            captions=sample_caption_set,
            duration_sec=65.0,
        )
        assert "-t" in cmd
        idx = cmd.index("-t")
        assert cmd[idx + 1] == "65.0"

    def test_fps_setting(self, sample_caption_set: CaptionSet):
        cmd = build_ffmpeg_command(
            raw_video_path="input.mp4",
            music_path="music.mp3",
            output_path="output.mp4",
            captions=sample_caption_set,
            duration_sec=65.0,
        )
        assert "-r" in cmd
        idx = cmd.index("-r")
        assert cmd[idx + 1] == str(OUTPUT_FPS)

    def test_filter_complex_present(self, sample_caption_set: CaptionSet):
        cmd = build_ffmpeg_command(
            raw_video_path="input.mp4",
            music_path="music.mp3",
            output_path="output.mp4",
            captions=sample_caption_set,
            duration_sec=65.0,
        )
        assert "-filter_complex" in cmd

    def test_resolution_scaling(self, sample_caption_set: CaptionSet):
        cmd = build_ffmpeg_command(
            raw_video_path="input.mp4",
            music_path="music.mp3",
            output_path="output.mp4",
            captions=sample_caption_set,
            duration_sec=65.0,
        )
        idx = cmd.index("-filter_complex")
        filter_str = cmd[idx + 1]
        assert f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}" in filter_str

    @patch("kairos.services.ffmpeg_compositor._probe_has_audio", return_value=False)
    @patch("kairos.services.ffmpeg_compositor._get_ffmpeg_path", return_value="ffmpeg")
    def test_audio_filter_in_command(self, _mock_ffmpeg, _mock_probe, sample_caption_set: CaptionSet):
        cmd = build_ffmpeg_command(
            raw_video_path="input.mp4",
            music_path="music.mp3",
            output_path="output.mp4",
            captions=sample_caption_set,
            duration_sec=65.0,
        )
        idx = cmd.index("-filter_complex")
        filter_str = cmd[idx + 1]
        # Music removed — silence branch uses anullsrc
        assert "anullsrc" in filter_str
        assert "atrim=duration=" in filter_str


class TestCaptionFilterIntegration:
    """Tests for FFmpeg caption filter details (using caption.py)."""

    def test_drawtext_filter_contains_required_elements(self):
        caption = Caption(
            caption_type=CaptionType.HOOK,
            text="Watch them collide",
            start_sec=0.0,
            end_sec=2.5,
        )
        result = build_ffmpeg_caption_filter(caption)

        assert "drawtext=" in result
        assert ffmpeg_escape_path(CAPTION_FONT) in result
        assert f"borderw={CAPTION_STROKE_WIDTH}" in result
        assert "enable='between" in result
        assert "alpha=" in result

    def test_caption_position_lower_third(self):
        caption = Caption(
            caption_type=CaptionType.HOOK,
            text="Test",
            start_sec=0.0,
            end_sec=2.0,
        )
        result = build_ffmpeg_caption_filter(caption, video_height=1920)
        y_pos = int(1920 * CAPTION_POSITION_Y_PCT)
        assert f"y={y_pos}" in result

    def test_caption_centered_horizontally(self):
        caption = Caption(
            caption_type=CaptionType.HOOK,
            text="Test",
            start_sec=0.0,
            end_sec=2.0,
        )
        result = build_ffmpeg_caption_filter(caption)
        assert "x=max(40\\,(w-text_w)/2)" in result

    def test_caption_timing_enable(self):
        caption = Caption(
            caption_type=CaptionType.HOOK,
            text="Hook",
            start_sec=0.5,
            end_sec=3.0,
        )
        result = build_ffmpeg_caption_filter(caption)
        assert "between(t,0.5,3.0)" in result

    def test_caption_fade_in_out(self):
        caption = Caption(
            caption_type=CaptionType.HOOK,
            text="Hook",
            start_sec=0.0,
            end_sec=2.5,
        )
        result = build_ffmpeg_caption_filter(caption)
        assert str(CAPTION_FADE_IN_SEC) in result
        assert str(CAPTION_FADE_OUT_SEC) in result


# =============================================================================
# Adapter Integration Tests
# =============================================================================


class TestAdapterIntegration:
    """Tests that the adapter returns PhysicsVideoEditorAgent."""

    def test_adapter_returns_video_editor_agent(self):
        from kairos.pipelines.adapters.physics_adapter import PhysicsPipelineAdapter

        adapter = PhysicsPipelineAdapter()
        agent = adapter.get_video_editor_agent()
        assert isinstance(agent, PhysicsVideoEditorAgent)

    def test_agent_is_base_video_editor_agent(self):
        from kairos.pipelines.contracts import VideoEditorAgent
        from kairos.pipelines.adapters.physics_adapter import PhysicsPipelineAdapter

        adapter = PhysicsPipelineAdapter()
        agent = adapter.get_video_editor_agent()
        assert isinstance(agent, VideoEditorAgent)
