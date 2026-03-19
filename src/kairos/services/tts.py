"""Text-to-Speech service using edge-tts (Microsoft TTS).

Generates voice-over audio for hook captions. The voice is selected
to match the theme/mood for cohesive video personality.

Usage:
    path = await generate_tts("Wait for the last one", theme_name="deep_space")
    # -> Path to .mp3 file
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Theme → voice mapping (Microsoft TTS voices)
# See: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support
_THEME_VOICES: dict[str, str] = {
    "deep_space": "en-US-GuyNeural",        # Deep male — cinematic
    "enchanted_forest": "en-GB-SoniaNeural", # Soft British female — whimsical
    "golden_hour": "en-US-JennyNeural",      # Warm female — friendly
    "arctic_lab": "en-US-EricNeural",        # Clear male — technical
    "neon_city": "en-US-AriaNeural",         # Energetic female — urban
    "candy_land": "en-US-JennyNeural",       # Warm female — playful
    "lava_world": "en-US-GuyNeural",         # Deep male — intense
}
_DEFAULT_VOICE = "en-US-GuyNeural"

# Speaking rate/pitch adjustments per theme
_THEME_PROSODY: dict[str, dict[str, str]] = {
    "deep_space": {"rate": "-10%", "pitch": "-5Hz"},
    "enchanted_forest": {"rate": "-5%", "pitch": "+0Hz"},
    "golden_hour": {"rate": "-5%", "pitch": "+0Hz"},
    "arctic_lab": {"rate": "+0%", "pitch": "+0Hz"},
    "neon_city": {"rate": "+10%", "pitch": "+2Hz"},
    "candy_land": {"rate": "+0%", "pitch": "+0Hz"},
    "lava_world": {"rate": "-15%", "pitch": "-5Hz"},
}


async def generate_tts(
    text: str,
    output_dir: Path,
    *,
    theme_name: str = "",
    filename: str = "tts_hook.mp3",
) -> Path | None:
    """Generate TTS audio for the given text.

    Args:
        text: Text to speak (typically the hook caption).
        output_dir: Directory to write the audio file.
        theme_name: Theme name for voice selection.
        filename: Output filename.

    Returns:
        Path to the generated .mp3 file, or None if TTS fails.
    """
    try:
        import edge_tts
    except ImportError:
        logger.warning("[tts] edge-tts not installed — skipping TTS")
        return None

    if not text or not text.strip():
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    voice = _THEME_VOICES.get(theme_name, _DEFAULT_VOICE)
    prosody = _THEME_PROSODY.get(theme_name, {"rate": "+0%", "pitch": "+0Hz"})

    logger.info(
        "[tts] Generating TTS: voice=%s, text='%s', theme=%s",
        voice, text, theme_name,
    )

    try:
        communicate = edge_tts.Communicate(
            text=text.strip(),
            voice=voice,
            rate=prosody.get("rate", "+0%"),
            pitch=prosody.get("pitch", "+0Hz"),
        )
        await communicate.save(str(output_path))

        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info("[tts] TTS saved: %s (%d bytes)", output_path, output_path.stat().st_size)
            return output_path

        logger.warning("[tts] TTS file empty or missing")
        return None

    except Exception as exc:
        logger.warning("[tts] TTS generation failed: %s", exc)
        return None
