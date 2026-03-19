"""Marble Video Editor Agent.

Implements VideoEditorAgent for the Blender marble pipeline.
Reuses the same music selection, caption generation, and FFmpeg
composition logic as the physics pipeline.
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
    try:
        return get_step_config("caption_writer").resolve_model()
    except Exception:
        return "anthropic/claude-sonnet-4-20250514"


def _title_writer_model() -> str:
    try:
        return get_step_config("title_writer").resolve_model()
    except Exception:
        return "anthropic/claude-sonnet-4-20250514"


class MarbleVideoEditorAgent(VideoEditorAgent):
    """Video Editor Agent for marble course pipeline.

    Same workflow as the physics editor:
    1. Select music (programmatic tag matching)
    2. Generate hook caption (LLM)
    3. Compose with FFmpeg (music + captions)
    """

    def __init__(
        self,
        *,
        music_dir: Path | None = None,
        output_dir: Path | None = None,
        pipeline_run_id: UUID | None = None,
    ) -> None:
        settings = get_settings()
        self._music_dir = music_dir or settings.music_dir
        self._output_dir = output_dir or settings.output_dir
        self._pipeline_run_id = pipeline_run_id

    async def select_music(
        self,
        concept: ConceptBrief,
        stats: SimulationStats,
    ) -> MusicTrackMetadata:
        """Select music from the curated library."""
        logger.info("Selecting music for marble concept: %s", concept.title)

        library = load_music_library(self._music_dir)
        if not library:
            raise VideoAssemblyError(
                "No music tracks found in library",
                pipeline_run_id=self._pipeline_run_id,
            )

        selected = select_music(concept, stats, library=library)
        if selected is None:
            raise VideoAssemblyError(
                "No suitable music track found",
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
        """Generate hook caption via LLM, fallback to concept hook_text."""
        logger.info("Generating captions for marble: %s", concept.title)

        try:
            response = await call_llm(
                model=_caption_writer_model(),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You write short, punchy hook captions for marble "
                            "course videos. The hook appears in the first 2 seconds. "
                            "Max 6 words. Make viewers stop scrolling."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Title: {concept.title}\n"
                            f"Visual: {concept.visual_brief}\n"
                            f"Existing hook suggestion: {concept.hook_text}\n\n"
                            "Write the perfect hook caption (≤6 words)."
                        ),
                    },
                ],
                response_model=HookCaptionResponse,
                cache_step="marble_caption_writer",
            )
            hook_text = response.hook_text
            logger.info("LLM generated hook: '%s'", hook_text)
        except Exception:
            logger.warning(
                "Caption writer LLM failed, using fallback",
                exc_info=True,
            )
            hook_text = concept.hook_text

        if not validate_caption_text(hook_text):
            logger.warning("Hook '%s' failed validation, using fallback", hook_text)
            hook_text = concept.hook_text

        return build_caption_set(hook_text)

    async def generate_title(
        self,
        concept: ConceptBrief,
    ) -> str:
        """Generate video title via LLM, fallback to concept title."""
        logger.info("Generating title for marble: %s", concept.title)

        try:
            response = await call_llm(
                model=_title_writer_model(),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You write short YouTube Shorts titles for marble "
                            "course videos. Catchy, under 60 chars, no emoji spam."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Concept: {concept.title}\n"
                            f"Visual: {concept.visual_brief}\n"
                            "Write a title."
                        ),
                    },
                ],
                response_model=VideoTitleResponse,
                cache_step="marble_title_writer",
            )
            logger.info("LLM generated title: '%s'", response.title)
            return response.title
        except Exception:
            logger.warning("Title writer failed, using concept title", exc_info=True)
            return concept.title

    async def compose_video(
        self,
        raw_video_path: str,
        music: MusicTrackMetadata,
        captions: CaptionSet,
        concept: ConceptBrief,
    ) -> VideoOutput:
        """Assemble final video: rendered marble video + music + captions."""
        logger.info("Composing marble video for: %s", concept.title)

        if not check_ffmpeg_available():
            raise VideoAssemblyError(
                "FFmpeg not found on system PATH",
                pipeline_run_id=self._pipeline_run_id,
            )

        # Resolve music path
        music_path = str(self._music_dir / "tracks" / music.filename)
        if not Path(music_path).exists():
            music_path = str(self._music_dir / "contentid_cleared" / music.filename)

        # Output dir is set to the versioned folder inside runs/
        # (e.g. runs/{run_id}/output/v1/) by the graph orchestrator.
        output_dir = Path(self._output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = str(output_dir / "final.mp4")

        duration_sec = float(concept.target_duration_sec)

        cmd = build_ffmpeg_command(
            raw_video_path=raw_video_path,
            music_path=music_path,
            output_path=output_path,
            captions=captions,
            duration_sec=duration_sec,
        )

        returncode, stdout, stderr = await run_ffmpeg(cmd)

        if returncode != 0:
            raise VideoAssemblyError(
                f"FFmpeg composition failed (rc={returncode}): {stderr[-500:]}",
                pipeline_run_id=self._pipeline_run_id,
            )

        title = await self.generate_title(concept)

        output = VideoOutput(
            pipeline_run_id=self._pipeline_run_id or uuid4(),
            simulation_id=concept.concept_id,
            final_video_path=output_path,
            captions=captions,
            music_track=music,
            title=title,
            description=f"Marble course: {concept.title}",
        )

        logger.info("Marble video composed: %s", output_path)
        return output
