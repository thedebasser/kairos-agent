"""SFX variation pool — avoids repeat fatigue via gain randomisation.

Applies simple gain variation only (no FFmpeg pitch shifting) to avoid
resampling artifacts on very short synthetic collision sounds.
The source WAV files already contain frequency/timbre jitter from
synthetic_sfx.py, so pitch shifting is unnecessary.
"""

from __future__ import annotations

import logging
import random
import shutil
import struct
import wave
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
SFX_WORK_DIR = _PROJECT_ROOT / "assets" / "sfx" / "varied"


def _apply_gain_wav(src: Path, dst: Path, gain_db: float) -> None:
    """Apply gain to a WAV file using pure Python — no FFmpeg, no resampling.

    Reads 16-bit PCM, scales samples, writes back.  Zero artifacts.
    """
    linear = 10 ** (gain_db / 20.0)
    with wave.open(str(src), "rb") as r:
        params = r.getparams()
        raw = r.readframes(params.nframes)

    # 16-bit signed samples
    fmt = f"<{params.nframes * params.nchannels}h"
    samples = list(struct.unpack(fmt, raw))
    samples = [max(-32768, min(32767, int(s * linear))) for s in samples]
    raw_out = struct.pack(fmt, *samples)

    with wave.open(str(dst), "wb") as w:
        w.setparams(params)
        w.writeframes(raw_out)


class SFXPool:
    """Pool of collision sounds with per-use gain variation.

    Maintains a ring buffer of recent sounds to avoid exact repeats.
    No FFmpeg processing — gain is applied directly on PCM samples
    to eliminate resampling noise on short synthetic sounds.
    """

    def __init__(
        self,
        paths: list[Path],
        pitch_range: tuple[float, float] = (-2.0, 2.0),
        gain_range_db: tuple[float, float] = (-3.0, 2.0),
    ) -> None:
        self.paths = list(paths)
        # pitch_range kept in signature for API compat but unused
        self.gain_range_db = gain_range_db
        self._recent: list[Path] = []
        self._counter = 0

        SFX_WORK_DIR.mkdir(parents=True, exist_ok=True)

    def next(self) -> Path | None:
        """Return path to a fresh gain-varied WAV collision sound.

        Returns None if the pool is empty or processing fails.
        """
        if not self.paths:
            return None

        # Avoid last 3 repeats
        avoid = set(self._recent[-3:]) if len(self._recent) >= 3 else set()
        available = [p for p in self.paths if p not in avoid]
        src = random.choice(available or self.paths)
        self._recent.append(src)

        self._counter += 1
        out_path = SFX_WORK_DIR / f"sfx_{self._counter:05d}.wav"

        gain_db = random.uniform(*self.gain_range_db)

        try:
            if abs(gain_db) > 0.1:
                _apply_gain_wav(src, out_path, gain_db)
            else:
                shutil.copy2(src, out_path)
        except Exception as exc:
            logger.debug("[sfx_pool] Gain variation failed: %s — copying raw", exc)
            try:
                shutil.copy2(src, out_path)
            except Exception:
                return None

        return out_path

    def cleanup(self) -> None:
        """Remove temporary varied SFX files."""
        if SFX_WORK_DIR.exists():
            for f in SFX_WORK_DIR.glob("sfx_*.wav"):
                f.unlink(missing_ok=True)
