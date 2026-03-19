"""Physics Video Editor Agent.

Implements VideoEditorAgent for the physics simulation pipeline.
Assembles final platform-ready video from raw simulation + metadata.

Subagents:
- Music Selector (programmatic — tag/mood-based filtering)
- Caption Writer (Claude Sonnet via call_llm — hook caption only for POC)
- Title Generator (Llama 3.1 8B local via call_llm)
- Final Compositor (FFmpeg assembly)
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID, uuid4

from kairos.pipelines.contracts import VideoEditorAgent
from kairos.config import get_settings
from kairos.exceptions import VideoAssemblyError
from kairos.schemas.contracts import (
    CaptionSet,
    ConceptBrief,
    MusicTrackMetadata,
    SimulationStats,
    VideoOutput,
)
from kairos.schemas.video_editor import HookCaptionResponse, VideoTitleResponse
from kairos.ai.prompts.physics.builder import (
    build_user_prompt,
    load_system_prompt,
)
from kairos.services.caption import build_caption_set, validate_caption_text
from kairos.services.ffmpeg_compositor import (
    build_ffmpeg_command,
    check_ffmpeg_available,
    run_ffmpeg,
)
from kairos.ai.llm.config import get_step_config
from kairos.ai.llm.routing import call_llm
from kairos.services.music_selector import load_music_library, select_music

logger = logging.getLogger(__name__)


def _caption_writer_model() -> str:
    return get_step_config("caption_writer").resolve_model()


def _title_writer_model() -> str:
    return get_step_config("title_writer").resolve_model()


class PhysicsVideoEditorAgent(VideoEditorAgent):
    """Video Editor Agent for physics simulation pipeline.

    Takes a raw simulation video and produces a platform-ready video
    with music, captions, title, and channel watermark.
    """

    def __init__(
        self,
        *,
        music_dir: Path | None = None,
        output_dir: Path | None = None,
        watermark_path: str | None = None,
        pipeline_run_id: UUID | None = None,
    ) -> None:
        """Initialise the video editor agent.

        Args:
            music_dir: Path to music library directory (default from settings).
            output_dir: Path to output directory (default from settings).
            watermark_path: Optional path to channel watermark PNG.
            pipeline_run_id: Pipeline run ID for tracing.
        """
        settings = get_settings()
        self._music_dir = music_dir or settings.music_dir
        self._output_dir = output_dir or settings.output_dir
        self._watermark_path = watermark_path
        self._pipeline_run_id = pipeline_run_id

    async def select_music(
        self,
        concept: ConceptBrief,
        stats: SimulationStats,
    ) -> MusicTrackMetadata:
        """Select a music track from the curated library.

        Programmatic selection — no LLM. Uses tag/mood-based filtering
        with ContentID safety, duration matching, and recency penalty.

        Args:
            concept: The concept brief with audio requirements.
            stats: Simulation stats (duration, payoff timestamp, etc.).

        Returns:
            Selected MusicTrackMetadata.

        Raises:
            VideoAssemblyError: If no suitable track is found.
        """
        logger.info("Selecting music for concept: %s", concept.title)

        library = load_music_library(self._music_dir)

        if not library:
            raise VideoAssemblyError(
                "No music tracks found in library",
                pipeline_run_id=self._pipeline_run_id,
            )

        selected = select_music(concept, stats, library=library)

        if selected is None:
            raise VideoAssemblyError(
                "No suitable music track found matching concept requirements",
                pipeline_run_id=self._pipeline_run_id,
            )

        logger.info(
            "Selected music: %s (bpm=%d, mood=%s)",
            selected.track_id,
            selected.bpm,
            selected.mood,
        )
        return selected

    async def generate_captions(
        self,
        concept: ConceptBrief,
        *,
        theme_name: str = "",
    ) -> CaptionSet:
        """Generate captions for the video.

        POC: Hook caption only (0-2s, <=6 words).
        Uses Claude Sonnet for hook quality — highest-leverage text
        element for viewer retention.

        Falls back to the concept's hook_text if LLM call fails.

        Args:
            concept: The concept brief.

        Returns:
            CaptionSet with hook caption.
        """
        logger.info("Generating captions for concept: %s", concept.title)

        try:
            response = await call_llm(
                model=_caption_writer_model(),
                messages=[
                    {"role": "system", "content": load_system_prompt("caption_writer").text},
                    {
                        "role": "user",
                        "content": build_user_prompt("caption_writer", {
                            "category": concept.category.value,
                            "title": concept.title,
                            "visual_brief": concept.visual_brief,
                            "hook_text": concept.hook_text,
                            "theme_name": theme_name or "default",
                        }).text,
                    },
                ],
                response_model=HookCaptionResponse,
                cache_step="caption_writer",
            )

            hook_text = response.hook_text
            logger.info("LLM generated hook: '%s' (reason: %s)", hook_text, response.reasoning)

        except Exception:
            # Fall back to concept's hook text
            logger.warning(
                "Caption writer LLM failed, using concept hook_text fallback",
                exc_info=True,
            )
            hook_text = concept.hook_text

        # Validate and build caption set
        if not validate_caption_text(hook_text):
            logger.warning(
                "Generated hook '%s' failed validation, using concept fallback",
                hook_text,
            )
            hook_text = concept.hook_text

        caption_set = build_caption_set(hook_text)
        logger.info("Caption set built: %d caption(s)", len(caption_set.captions))
        return caption_set

    async def generate_title(
        self,
        concept: ConceptBrief,
    ) -> str:
        """Generate video title for platform upload.

        Uses local LLM (Llama 3.1 8B) — one line, well within
        local model capability.

        Falls back to concept title if LLM call fails.

        Args:
            concept: The concept brief.

        Returns:
            Video title string.
        """
        logger.info("Generating title for concept: %s", concept.title)

        try:
            response = await call_llm(
                model=_title_writer_model(),
                messages=[
                    {"role": "system", "content": load_system_prompt("title_writer").text},
                    {
                        "role": "user",
                        "content": build_user_prompt("title_writer", {
                            "category": concept.category.value,
                            "title": concept.title,
                            "visual_brief": concept.visual_brief,
                            "hook_text": concept.hook_text,
                        }).text,
                    },
                ],
                response_model=VideoTitleResponse,
                cache_step="title_writer",
            )

            title = response.title
            logger.info("LLM generated title: '%s'", title)
            return title

        except Exception:
            logger.warning(
                "Title writer LLM failed, using concept title fallback",
                exc_info=True,
            )
            return concept.title

    async def compose_video(
        self,
        raw_video_path: str,
        music: MusicTrackMetadata,
        captions: CaptionSet,
        concept: ConceptBrief,
    ) -> VideoOutput:
        """Assemble final video using FFmpeg.

        Assembly: raw video + music (-18dB, fade out last 3s) + captions
        (Inter Bold, white with black stroke, lower third) + watermark.

        Output: 9:16, H.264, 62-68s, AAC audio.

        Args:
            raw_video_path: Path to raw simulation video.
            music: Selected music track metadata.
            captions: CaptionSet to overlay.
            concept: The concept brief.

        Returns:
            VideoOutput with final video details.

        Raises:
            VideoAssemblyError: If FFmpeg is not available or composition fails.
        """
        logger.info("Composing final video for concept: %s", concept.title)

        if not check_ffmpeg_available():
            raise VideoAssemblyError(
                "FFmpeg not found on system PATH",
                pipeline_run_id=self._pipeline_run_id,
            )

        # Resolve paths
        music_path = str(self._music_dir / "tracks" / music.filename)
        if not Path(music_path).exists():
            # Try contentid_cleared directory
            music_path = str(self._music_dir / "contentid_cleared" / music.filename)

        output_dir = Path(self._output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_id = uuid4()
        output_filename = f"{output_id}.mp4"
        output_path = str(output_dir / output_filename)

        # Determine duration
        duration_sec = float(concept.target_duration_sec)

        # Build FFmpeg command
        cmd = build_ffmpeg_command(
            raw_video_path=raw_video_path,
            music_path=music_path,
            output_path=output_path,
            captions=captions,
            duration_sec=duration_sec,
            watermark_path=self._watermark_path,
        )

        # Execute FFmpeg
        returncode, stdout, stderr = await run_ffmpeg(cmd)

        if returncode != 0:
            raise VideoAssemblyError(
                f"FFmpeg composition failed (rc={returncode}): {stderr[-500:]}",
                pipeline_run_id=self._pipeline_run_id,
            )

        # Generate title (may use LLM)
        title = await self.generate_title(concept)

        # Build output record
        output = VideoOutput(
            output_id=output_id,
            pipeline_run_id=self._pipeline_run_id or uuid4(),
            simulation_id=concept.concept_id,
            final_video_path=output_path,
            captions=captions,
            music_track=music,
            title=title,
            description=f"Oddly satisfying {concept.category.value} physics simulation",
        )

        logger.info(
            "Video composed: %s (title='%s')",
            output_path,
            title,
        )
        return output
