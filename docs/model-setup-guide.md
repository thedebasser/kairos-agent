# Model Setup & Review Agents Guide

## Overview

The Kairos pipeline now includes two automated review agents that run on every video before human review:

```
Idea Agent → Simulation Agent → Video Editor Agent
  → Video Review → Audio Review → Human Review → Publish
```

Both review agents inspect `final.mp4` and produce structured pass/fail results. On failure, the pipeline automatically retries the upstream agent (video failure → re-simulate, audio failure → re-edit).

---

## Quick Start

### 1. Pull Required Models

**Windows (PowerShell):**
```powershell
.\scripts\setup_ollama_models.ps1 -Group All
```

**Linux/macOS (Bash):**
```bash
chmod +x scripts/setup_ollama_models.sh
./scripts/setup_ollama_models.sh --all
```

Pull subsets:
```powershell
.\scripts\setup_ollama_models.ps1 -Group Core     # existing pipeline models only
.\scripts\setup_ollama_models.ps1 -Group Review    # video + audio review models only
.\scripts\setup_ollama_models.ps1 -Group Blender   # Blender coding models only
```

### 2. Verify Models
```bash
ollama list
```

---

## New Models

### Video Review Agent

| Role | Model | VRAM | LiteLLM Alias | Config Key |
|------|-------|------|---------------|------------|
| **Default** | Qwen3-VL-8B-Instruct | ~16 GB FP16 | `video-reviewer-default` | `video_reviewer` |
| **Escalation** | Qwen3-VL-30B-A3B-Instruct | ~17-20 GB Q4_K_M | `video-reviewer-escalation` | `video_reviewer_escalation` |

**How it works:**
1. Extracts frames from `final.mp4` at 1 FPS (8B) or 0.5 FPS (30B)
2. Sends frames + context prompt to the VLM
3. VLM returns structured JSON: pass/fail + issue list + confidence
4. If confidence < 0.7 → escalates to 30B model for re-review
5. Optional pre-checks (DOVER, aesthetic scoring) can auto-reject before VLM

**Checks performed:**
- Broken physics (clipping, unrealistic trajectories)
- Incomplete simulation (dominos stopping, marble not reaching end)
- Objects flying off screen or freezing
- Bad framing / camera not following action
- Caption/text overlay placement issues
- Overall visual polish

### Audio Review Agent

| Role | Model | VRAM | LiteLLM Alias | Config Key |
|------|-------|------|---------------|------------|
| **Default** | Qwen2.5-Omni-7B + FFmpeg | ~17 GB FP16 | `audio-reviewer-default` | `audio_reviewer` |
| **Specialist** | Whisper + BEATs + FFmpeg | ~8-12 GB | — | `audio_reviewer_specialist` |
| **Escalation** | Qwen3-Omni-30B-A3B | ~20 GB Q4_K_M | `audio-reviewer-escalation` | `audio_reviewer_escalation` |

**How it works:**
1. **FFmpeg ebur128 loudness analysis** runs on every video (always, regardless of model)
2. Hard fail if integrated loudness outside -14 to -16 LUFS or true peak > -1 dBTP
3. Primary reviewer (omni-modal LLM) listens to full 65s mixed audio
4. Returns structured JSON: pass/fail + issue list + loudness metrics
5. Optional DNSMOS P.835 speech quality check on TTS segments

**Checks performed:**
- Background static / noise artifacts
- Unexpected sounds that don't belong
- TTS accuracy (voiceover correctness)
- Theme/vibe match (satisfying, cozy, relaxing)
- Volume level consistency
- LUFS loudness compliance

### Blender Coding Models

| Role | Model | VRAM | LiteLLM Alias | Config Key |
|------|-------|------|---------------|------------|
| **Default** | Qwen3.5-27B | ~18.5 GB Q5_K_M | `blender-coder-default` | `blender_code_generation` |
| **Option 2** | Devstral Small 2 24B | ~19 GB Q6_K | `blender-coder-option2` | — |
| **Option 3** | Qwen3-Coder-30B-A3B | MoE | `blender-coder-option3` | — |
| **Fallback** | Qwen2.5-Coder-32B | ~18.5 GB Q4_K_M | `blender-coder-fallback` | — |

The `blender_code_generation` step in `llm_config.yaml` has an `enabled: true/false` toggle. Set `enabled: false` for code-only mode (no LLM, manual scripts only).

---

## Configuration

### Switching Models

All model routing is configured in `llm_config.yaml`. To switch the video reviewer to the escalation model as default:

```yaml
steps:
  video_reviewer:
    litellm_alias_local: video-reviewer-escalation   # was: video-reviewer-default
```

To switch the Blender coder to Devstral:
```yaml
steps:
  blender_code_generation:
    litellm_alias_local: blender-coder-option2       # was: blender-coder-default
```

### Disabling Review Agents

The review agents are unconditional — they run on every video. To skip them temporarily, you can set the model alias to null in `llm_config.yaml`. The pipeline will proceed with a warning if the reviewer is unavailable.

### Adjusting Thresholds

**Video review escalation threshold** (default: 0.7):
```yaml
steps:
  video_reviewer_escalation:
    escalation:
      confidence_threshold: 0.7  # lower = fewer escalations
```

**Audio loudness thresholds**:
```yaml
steps:
  audio_reviewer:
    loudness:
      target_lufs_min: -16.0
      target_lufs_max: -14.0
      true_peak_max_dbtp: -1.0
```

**Optional pre-checks** (disabled by default):
```yaml
steps:
  video_reviewer:
    pre_checks:
      dover:
        enabled: true
        threshold_technical: 0.4
      aesthetic_predictor:
        enabled: true
        threshold: 3.0
```

### Enabling DNSMOS Speech Quality Check
```yaml
steps:
  audio_reviewer:
    dnsmos:
      enabled: true
      threshold_ovrl: 3.0   # MOS 1-5 scale
```

---

## VRAM Budget (RTX 3090 — 24 GB)

Only **one large model** loads at a time. Ollama swaps models automatically.

| Model | VRAM | Notes |
|-------|------|-------|
| Core models (Mistral 7B, Llama 3.1 8B) | ~4-8 GB | Can coexist |
| Qwen3-VL-8B | ~16 GB FP16 | Default video reviewer |
| Qwen3-VL-30B-A3B | ~17-20 GB | Escalation only |
| Qwen2.5-Omni-7B | ~17 GB FP16 | Default audio reviewer |
| Qwen3-Omni-30B-A3B | ~20 GB Q4_K_M | Tight fit, thinker-only mode |
| Qwen3.5-27B | ~18.5 GB Q5_K_M | Default Blender coder |
| Devstral Small 24B | ~19 GB Q6_K | Higher quant than default |
| Qwen2.5-Coder-32B | ~18.5 GB Q4_K_M | Limited context headroom |

The pipeline runs models sequentially, never concurrently, so a single 24 GB GPU handles all steps.

---

## Pipeline Retry Logic

| Review Agent | On Failure | Max Retries |
|-------------|------------|-------------|
| Video Review | → Re-run Simulation Agent | 2 |
| Audio Review | → Re-run Video Editor Agent | 2 |

After max retries, the pipeline proceeds to Human Review with the accumulated issues logged for manual decision.

---

## Architecture

```
src/kairos/
├── agents/
│   └── base.py                    # ABCs: BaseVideoReviewAgent, BaseAudioReviewAgent
├── services/
│   ├── video_review.py            # VideoReviewAgent (VLM frame analysis)
│   ├── audio_review.py            # AudioReviewAgent (omni-modal + FFmpeg)
│   ├── llm_config.py              # StepConfig resolution
│   └── llm_routing.py             # LLM call infrastructure
├── models/
│   └── contracts.py               # VideoReviewResult, AudioReviewResult, etc.
├── pipeline/
│   └── graph.py                   # LangGraph nodes: video_review_node, audio_review_node
└── pipelines/
    ├── physics/adapter.py         # get_video_review_agent(), get_audio_review_agent()
    ├── marble/adapter.py          # (same)
    └── domino/adapter.py          # (same)
```

Config files:
- `llm_config.yaml` — step configs, model routing, thresholds
- `litellm_config.yaml` — model alias → real provider/model mapping
