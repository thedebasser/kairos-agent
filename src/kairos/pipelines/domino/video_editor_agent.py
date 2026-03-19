"""Domino Video Editor Agent.

Implements BaseVideoEditorAgent for the Blender domino pipeline.
Reuses the same music selection, caption generation, and FFmpeg
composition logic as the marble pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID, uuid4

from kairos.agents.base import BaseVideoEditorAgent
from kairos.config import get_settings
from kairos.exceptions import VideoAssemblyError
from kairos.models.contracts import (
    CaptionSet,
    ConceptBrief,
    MusicTrackMetadata,
    SimulationStats,
    VideoOutput,
)
from kairos.models.video_editor import HookCaptionResponse, VideoTitleResponse
from kairos.services.caption import build_caption_set, validate_caption_text
from kairos.services.ffmpeg_compositor import (
    build_ffmpeg_command,
    check_ffmpeg_available,
    run_ffmpeg,
)
from kairos.services.llm_config import get_step_config
from kairos.services.llm_routing import call_llm
from kairos.services.music_selector import load_music_library, select_music
from kairos.services.response_cache import get_cache

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


class DominoVideoEditorAgent(BaseVideoEditorAgent):
    """Video Editor Agent for domino run pipeline.

    Same workflow as the marble editor:
    1. Select music (programmatic tag matching)
    2. Generate hook caption (LLM, cached)
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
        logger.info("Selecting music for domino concept: %s", concept.title)

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
            selected.track_id, selected.bpm, selected.mood,
        )
        return selected

    async def generate_captions(
        self,
        concept: ConceptBrief,
        *,
        theme_name: str = "",
    ) -> CaptionSet:
        """Generate hook caption via LLM, with caching. Fallback to concept hook_text."""
        logger.info("Generating captions for domino: %s", concept.title)

        # ── Cache check ──────────────────────────────────────────────
        cache = get_cache()
        if cache:
            cached = cache.get_step("domino_captions")
            if cached:
                logger.info("[domino_editor] Cache HIT for captions — skipping LLM")
                return CaptionSet.model_validate(cached)

        try:
            theme_line = f"\nEnvironment Theme: {theme_name}" if theme_name else ""
            response = await call_llm(
                model=_caption_writer_model(),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You write short, punchy hook captions for domino "
                            "run videos. The hook appears in the first 2 seconds. "
                            "Max 6 words. Make viewers stop scrolling. "
                            "The caption should match the overall visual mood "
                            "and theme of the video."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Title: {concept.title}\n"
                            f"Visual: {concept.visual_brief}"
                            f"{theme_line}\n"
                            f"Existing hook suggestion: {concept.hook_text}\n\n"
                            "Write the perfect hook caption (<=6 words) that "
                            "matches the visual mood."
                        ),
                    },
                ],
                response_model=HookCaptionResponse,
                cache_step="domino_caption_writer",
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

        caption_set = build_caption_set(hook_text)

        # ── Cache store ──────────────────────────────────────────────
        if cache:
            cache.put_step("domino_captions", caption_set.model_dump(mode="json"))

        return caption_set

    async def generate_title(
        self,
        concept: ConceptBrief,
    ) -> str:
        """Generate video title via LLM, with caching. Fallback to concept title."""
        logger.info("Generating title for domino: %s", concept.title)

        # ── Cache check ──────────────────────────────────────────────
        cache = get_cache()
        if cache:
            cached = cache.get_step("domino_title")
            if cached:
                logger.info("[domino_editor] Cache HIT for title — skipping LLM")
                return cached.get("title", concept.title)

        try:
            response = await call_llm(
                model=_title_writer_model(),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You write short YouTube Shorts titles for domino "
                            "run videos. Catchy, under 60 chars, no emoji spam."
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
                cache_step="domino_title_writer",
            )
            title = response.title
            logger.info("LLM generated title: '%s'", title)
        except Exception:
            logger.warning("Title writer failed, using concept title", exc_info=True)
            title = concept.title

        # ── Cache store ──────────────────────────────────────────────
        if cache:
            cache.put_step("domino_title", {"title": title})

        return title

    async def compose_video(
        self,
        raw_video_path: str,
        music: MusicTrackMetadata,
        captions: CaptionSet,
        concept: ConceptBrief,
    ) -> VideoOutput:
        """Assemble final video: rendered domino video + music + captions + TTS."""
        logger.info("Composing domino video for: %s", concept.title)

        if not check_ffmpeg_available():
            raise VideoAssemblyError(
                "FFmpeg not found on system PATH",
                pipeline_run_id=self._pipeline_run_id,
            )

        run_id = self._pipeline_run_id or uuid4()
        # Output dir is set to the versioned folder inside runs/
        # (e.g. runs/{run_id}/output/v1/) by the graph orchestrator.
        output_dir = Path(self._output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / "final.mp4"

        music_path = self._music_dir / "tracks" / music.filename
        if not music_path.exists():
            raise VideoAssemblyError(
                f"Music file not found: {music_path}",
                pipeline_run_id=self._pipeline_run_id,
            )

        # Generate TTS voice-over from hook caption
        tts_path: str | None = None
        caption_colour = ""
        caption_stroke_colour = ""
        if captions.captions:
            hook_text = captions.captions[0].text
            try:
                from kairos.services.tts import generate_tts

                # Read theme config for TTS voice and caption colours
                theme_name = ""
                theme_cfg = Path(raw_video_path).parent / "theme_config.json"
                if theme_cfg.exists():
                    import json
                    theme_data = json.loads(
                        theme_cfg.read_text(encoding="utf-8")
                    )
                    theme_name = theme_data.get("theme_name", "")
                    caption_colour = theme_data.get("caption_colour", "")
                    caption_stroke_colour = theme_data.get("caption_stroke_colour", "")

                tts_file = await generate_tts(
                    hook_text,
                    output_dir,
                    theme_name=theme_name,
                )
                if tts_file:
                    tts_path = str(tts_file)
                    logger.info("[domino_editor] TTS generated: %s", tts_path)
            except Exception as exc:
                logger.warning("[domino_editor] TTS failed (non-fatal): %s", exc)

        cmd = build_ffmpeg_command(
            raw_video_path=raw_video_path,
            music_path=str(music_path),
            output_path=str(final_path),
            captions=captions,
            duration_sec=concept.target_duration_sec,
            tts_path=tts_path,
            caption_colour=caption_colour,
            caption_stroke_colour=caption_stroke_colour,
        )

        await run_ffmpeg(cmd)

        return VideoOutput(
            pipeline_run_id=run_id,
            simulation_id=uuid4(),
            final_video_path=str(final_path),
            captions=captions,
            music_track=music,
            title=concept.title,
        )
