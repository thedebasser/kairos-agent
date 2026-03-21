# ADR-007: Two-Tier Validation Strategy

**Status:** Accepted
**Date:** 2026-02-20

## Context

Simulation output must be validated before the pipeline proceeds to video editing. Validation needs to catch:
- Completely broken outputs (wrong format, empty file, corrupt video)
- Spec violations (wrong resolution, duration, FPS)
- Subtle quality issues (blank frames, frozen animation, poor visual quality)

Running an AI vision model on every simulation output is expensive (~$0.01/call) and slow (~5s). But programmatic checks alone miss visual quality problems that a human would immediately notice.

## Decision

Implement a two-tier validation strategy:

**Tier 1 — Programmatic (FFprobe, ~100ms):**
1. `check_valid_mp4()` — FFprobe can parse the file
2. `check_file_size()` — File is >10KB (not empty/corrupt)
3. `check_duration()` — Within 62-68s target range
4. `check_resolution()` — 1080×1920 (9:16 portrait)
5. `check_fps()` — Consistent 30fps
6. `check_audio_present()` — Audio stream exists (if expected)

**Tier 2 — AI (vision model, ~5s):**
- Extract frames at key timestamps (0s, 15s, 30s, 45s, 60s)
- Vision model evaluates visual quality, motion, and colour
- Blank frame detection via pixel variance analysis

**Gate rule:** All Tier 1 checks must pass before Tier 2 runs. This saves ~5s and one LLM call on the ~60% of simulations that fail on basic spec violations.

## Consequences

**Positive:**
- Fast feedback loop — obvious failures caught in <1s with zero LLM cost.
- Tier 2 only runs on plausible videos — no wasting AI on clearly broken output.
- Per-check granularity — the Simulation Agent sees exactly which check failed and can adjust accordingly.
- `ValidationResult` carries both `tier1_passed` and `tier2_passed` flags, enabling callers to make nuanced routing decisions.

**Negative:**
- Two codepaths to maintain (programmatic + AI).
- Tier 2 depends on vision model availability (moondream2 was removed from Ollama — currently using Qwen3-VL via video review agent instead).
- Some quality issues fall between tiers — too subtle for FFprobe, too obvious for AI (e.g., audio-video sync drift).

**Validation metrics:**
- Tier 1 catches ~60% of bad simulations (wrong resolution is the most common failure).
- Tier 2 catches an additional ~15% (mostly blank/frozen frame issues).
- ~25% of failures are only caught by human review.
