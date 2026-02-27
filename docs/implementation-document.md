# Simulation Content Pipeline — Implementation Document

**Status:** Ready for Development
**Repo:** New (standalone)
**Last updated:** February 2026
**POC Scope:** Pipeline 1 — Oddly Satisfying Physics

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Core Design Principles](#2-core-design-principles)
3. [Architecture Overview](#3-architecture-overview)
4. [Agent Architecture](#4-agent-architecture)
5. [LLM Routing & Model Strategy](#5-llm-routing--model-strategy)
6. [Simulation Sandbox](#6-simulation-sandbox)
7. [Music & Audio Strategy](#7-music--audio-strategy)
8. [Human Review Gate](#8-human-review-gate)
9. [Upload Scheduling & Distribution](#9-upload-scheduling--distribution)
10. [Database Schema](#10-database-schema)
11. [Tech Stack](#11-tech-stack)
12. [Testing Strategy](#12-testing-strategy)
13. [Fine-Tuning Strategy](#13-fine-tuning-strategy)
14. [Pipeline Backlog](#14-pipeline-backlog)
15. [Cost Model](#15-cost-model)
16. [Monetisation & Distribution](#16-monetisation--distribution)
17. [POC Success Criteria](#17-poc-success-criteria)
18. [Implementation Plan](#18-implementation-plan)

---

## 1. Project Overview

A fully automated content production system that generates short-form videos from programmatic simulations. No filming, no manual editing, no human on camera. The system produces 2–3 unique videos per day across multiple channels and platforms.

Each video is generated end-to-end by an agent pipeline:

1. An **Idea Agent** selects a scenario and generates a concept
2. A **Simulation Agent** writes and validates the code that produces the visual
3. A **Video Editor Agent** assembles the final video with music and captions
4. A **Human Review Gate** approves before publish (lightweight — designed for a time-constrained PM)
5. A **Distribution Service** publishes to platforms on a schedule

The same three-agent architecture runs across every pipeline. What changes between pipelines is only the simulation engine — the orchestration, idea generation, video assembly, and distribution logic are shared infrastructure.

### POC Scope

The POC targets **Pipeline 1: Oddly Satisfying Physics** using Pygame + Pymunk. It is limited to four scenario categories: ball pit / collision cascade, marble funnel / sorting, domino chains, and destruction / stacking. The goal is a working end-to-end pipeline that produces validated, reviewable, publishable 9:16 short-form videos.

---

## 2. Core Design Principles

**Deterministic output** — Simulations produce consistent, verifiable results. The agent loop can validate output programmatically without human judgment on every run.

**Infinite variety from finite rules** — Small parameter changes (mass, velocity, body count, scenario seed) produce visually distinct videos from the same simulation template. Each pipeline has effectively unlimited scenario capacity before repetition risk.

**Prediction-based content** — The highest-performing formats give viewers a stake before the video ends. "Which car makes the jump?" or "Which marble wins?" drives comments, replays, and shares. Prioritise formats with a winner/loser or outcome the viewer can predict.

**No human bottleneck** — The system should be capable of producing and publishing daily without human intervention beyond periodic quality review. The human review gate exists for quality, not production capacity. The reviewer is a project manager with limited time — the review interface must be fast and frictionless.

**Shared infrastructure, swappable engine** — Agent logic, database schemas, orchestration, and distribution are shared. Adding a new pipeline = writing a new simulation adapter, not a new system.

**Test-driven confidence** — Every component is tested programmatically. The testing framework is built before the pipeline so that quality is enforced from day one and new pipelines can integrate tests easily.

**Fail-fast error handling** — Errors must never be swallowed or silently ignored. The system uses specific, typed exceptions so that reading the exception type alone tells you what went wrong (inspired by C#-style exception hierarchies). If something unexpected happens, the pipeline crashes immediately, logs the full context, and sends a Slack alert. For expected failure modes (LLM returns invalid output, sandbox times out), the system retries or escalates per the defined strategy. For unexpected failures (database down, Docker daemon crash, disk full), the pipeline raises, crashes, and waits for manual investigation.

Exception hierarchy:

```python
class PipelineError(Exception):
    """Base for all pipeline errors."""

class ConceptGenerationError(PipelineError):
    """Idea Agent failed to produce a valid concept."""

class SimulationExecutionError(PipelineError):
    """Simulation code failed to execute in sandbox."""

class SimulationTimeoutError(SimulationExecutionError):
    """Simulation exceeded maximum execution time."""

class SimulationOOMError(SimulationExecutionError):
    """Simulation exceeded memory limit."""

class ValidationError(PipelineError):
    """Produced output failed validation checks."""

class VideoAssemblyError(PipelineError):
    """FFmpeg composition failed."""

class LLMRoutingError(PipelineError):
    """Both local and cloud LLM calls failed."""

class PublishError(PipelineError):
    """Upload to platform failed after all retries."""

class InfrastructureError(PipelineError):
    """Database, Redis, Docker, or other infrastructure failure. Always fatal."""
```

`PipelineError` subclasses are caught and handled (retry, escalate, re-route). `InfrastructureError` is never caught — it crashes the process. All exceptions include the `pipeline_run_id` for tracing.

**Recovery options (exposed via CLI and review dashboard):**

When a pipeline run fails, the operator has two options:

1. **Restart from failed step** — Uses LangGraph's PostgreSQL checkpoint to resume from the last successful node. Preferred when the failure was transient (e.g., Docker daemon hiccup, temporary disk full). Command: `pipeline resume <pipeline_run_id>`
2. **Restart entire pipeline** — Discards all state and re-runs from scratch with a new `pipeline_run_id`. Preferred when the concept itself is the problem or state is corrupted. Command: `pipeline restart <pipeline_run_id>`

Both options send a Slack notification on failure and on recovery attempt. The review dashboard shows failed runs with one-click buttons for both options.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LangGraph Orchestrator                       │
│                  (PostgreSQL-checkpointed state machine)             │
├────────────┬──────────────────┬───────────────────┬─────────────────┤
│ Idea Agent │ Simulation Agent │ Video Editor Agent│ Review + Publish│
│            │                  │                   │                 │
│ Claude     │ Claude (1st pass)│ Claude (captions) │ Review Dashboard│
│ Sonnet     │ Mistral (tweaks) │ Mistral (titles)  │ Publish Queue   │
│ Mistral    │ Moondream (vis.) │ FFmpeg            │ Upload-Post API │
│ (rotation) │ Docker sandbox   │ Pixabay (music)   │                 │
└────────────┴──────────────────┴───────────────────┴─────────────────┘
        │              │                │                    │
        └──────────────┴────────────────┴────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │  Shared Services     │
                    │  - LiteLLM proxy     │
                    │  - PostgreSQL        │
                    │  - ChromaDB (RAG)    │
                    │  - Ollama            │
                    │  - Redis (queue)     │
                    │  - Langfuse (obs.)   │
                    └─────────────────────┘
```

### Pipeline State (LangGraph)

```
idea_agent
    │
    ├─(success)──→ simulation_agent
    │                   │
    │                   ├─(success)──→ video_editor_agent
    │                   │                   │
    │                   │                   └─(complete)──→ human_review
    │                   │                                       │
    │                   │                                 ├─(approved)──→ publish_queue
    │                   │                                 ├─(needs_edit)──→ video_editor_agent
    │                   │                                 └─(bad_concept)──→ idea_agent
    │                   │
    │                   ├─(retry, ≤5)──→ simulation_agent
    │                   └─(too_complex)──→ idea_agent
    │
    └─(no_concept)──→ idea_agent (regenerate)
```

State is checkpointed to PostgreSQL at each node. A failure at any stage does not restart from the beginning.

### Key Architecture Decisions

- **LangGraph pinned to a specific version** for stability. Agent logic is written as plain Python functions that LangGraph orchestrates — no deep coupling to LangChain abstractions. This keeps the option open to swap orchestrators later without rewriting agents.
- **Instructor library** used for all LLM calls that require structured output. Provides Pydantic-validated JSON responses with automatic retries on malformed output. Works with both Anthropic and Ollama models.
- **LiteLLM** as unified LLM proxy with built-in fallback routing (local → cloud escalation).
- **All validation is programmatic first, LLM-assisted second.** Duration, FPS, frame count, aspect ratio, audio levels — these are arithmetic checks in Python, not LLM calls.

---

## 4. Agent Architecture

All pipelines share the same three-agent structure, orchestrated via LangGraph.

### Agent 1: Idea Agent

Responsible for selecting a scenario category, generating ranked concepts, and returning one for production.

**Subagents:**

- **Inventory Analyst** — reads database, produces category saturation and performance breakdown
- **Category Selector** — applies rotation rules (no repeat of last category, streak limits, boost underused categories)
- **Concept Developer** — generates 3 ranked concepts for the selected category, returns the top one
- **Trend Scout** *(periodic, optional)* — web search for trending content signals in the niche

**Model routing:**

| Subagent | Model | Rationale |
|---|---|---|
| Inventory Analyst | Programmatic (SQL) | No LLM needed — pure database queries |
| Category Selector | Mistral 7B 4-bit (local) | Structured, rule-based decision |
| Concept Developer | Claude Sonnet (cloud) | Most important creative decision in pipeline |
| Trend Scout | Claude Sonnet + web search (cloud) | Requires real-time retrieval |

**Output:** Concept JSON (Pydantic-validated via Instructor) including visual brief, simulation requirements, audio brief (mood, tempo, energy curve), novelty score, feasibility score, target duration (65s default), and scenario category.

**Category rotation rules:**

- Hard block: cannot repeat same category as previous video
- Soft block: deprioritise categories >30% of last 30 days output
- Boost: categories with <5 total videos
- Streak break: force switch after 3 consecutive videos in same category

**POC scenario categories (Oddly Satisfying Physics):**

1. Ball pit / collision cascade
2. Marble funnel / sorting
3. Domino chains
4. Destruction / stacking

---

### Agent 2: Simulation Agent

Responsible for writing, iterating, validating, and rendering the simulation that produces the raw video.

**Agent loop:**

```
Write code → Execute in sandbox → Programmatic validation → Inspect frames (if needed) → Adjust → Re-run → Render final
```

Loops until all validation checks pass or max iterations (5) reached. On max iterations exceeded, escalates to Idea Agent for concept replacement.

**Model routing:**

| Task | Model | Rationale |
|---|---|---|
| Simulation first pass (code gen) | Claude Sonnet (cloud) | Strong code reasoning, complex spatial logic |
| Parameter adjustments | Mistral 7B 4-bit (local) | Mechanical edits, well-defined scope |
| Complex debugging | Claude Sonnet (cloud) | Subtle bug reasoning |
| Frame/visual inspection | Moondream2 1.8B (local) | Basic visual content checks only |

**MCP Tools:**

| Tool | Description |
|---|---|
| `query_knowledge` | RAG search over simulation patterns, known bugs, visual design principles |
| `write_simulation_code` | Write or overwrite the simulation file |
| `execute_simulation` | Run in Docker sandbox, return stdout/stderr and basic stats |
| `extract_frame_sequence` | Generate contact sheet of frames for visual inspection |
| `get_simulation_stats` | Duration, peak body count, avg FPS, payoff timestamp |
| `adjust_parameters` | Targeted parameter edits without full rewrite |
| `validate_output` | Run all programmatic validation checks |
| `render_final` | Full-quality render to MP4 |

**Validation — two tiers:**

Tier 1 — Programmatic (mandatory, no LLM):

- Duration within target range (62–68s, target 65s)
- FPS stable throughout (≥30)
- Frame count matches expected at render FPS
- Output file is valid MP4, correct resolution (1080×1920)
- File size within reasonable bounds (not 0, not >500MB)
- No duplicate/frozen frames (frame hash comparison)
- Motion detected (pixel variance between frames)
- Colour space valid (not all black, all white, or single-colour)

Tier 2 — AI-assisted (optional, supplements Tier 1):

- Moondream2 frame inspection: visual content present and progressing across sampled frames
- Payoff detection: activity/climax visible in final 20% of frames
- No obvious visual glitches (objects stuck at borders, clipping through each other)

**Important:** Moondream2 is used only for basic content verification ("are there objects on screen?", "is the frame blank?"), not for subjective aesthetic judgments. It cannot reliably answer "does this look satisfying?" — that is deferred to human review.

**RAG knowledge base structure:**

**Ingestion tool: LlamaIndex.** Rather than building a custom document ingestion pipeline, use LlamaIndex's `IngestionPipeline` for all RAG data loading. LlamaIndex handles document loading (web pages, markdown, code files), code-aware chunking, embedding generation, deduplication, and incremental updates. It feeds directly into ChromaDB (our vector store). This means:

- Pymunk/Pygame API docs are ingested via `WebBaseLoader` or `SimpleDirectoryReader`
- Code patterns and scenario templates are ingested with language-aware splitting (preserves function boundaries)
- Approved simulations are auto-ingested after human review — the pipeline grows its own knowledge base over time
- LlamaIndex's deduplication ensures re-ingesting updated docs only processes changes

At runtime, the Simulation Agent queries ChromaDB directly (no LlamaIndex dependency in the hot path — LlamaIndex is ingestion-only). This keeps the runtime simple and fast.

```
/knowledge/
├── {engine}_patterns/     # Engine-specific code patterns and gotchas
├── visual_design/         # Colour palettes, pacing, body sizing
├── scenario_templates/    # Per-category implementation guides
├── common_bugs/           # Known issues and fixes per engine
└── cloud_learnings/       # Auto-populated from cloud fallback successes
```

**RAG Learning Loop:** When a local LLM fails a task and the cloud model succeeds (via quality-based fallback), the cloud model's successful output is automatically recorded as a new RAG entry under `cloud_learnings/`. This creates a feedback loop: the local model's knowledge base grows over time with real examples of what works, reducing future cloud escalations. Combined with the fine-tuning pipeline, this means the local models continuously improve from the cloud model's corrections. The `agent_runs` table tracks which calls were escalated, providing the data source for this loop.

---

### Agent 3: Video Editor Agent

Responsible for assembling the final video from raw simulation output. No voiceover. Music and minimal captions only.

**Subagents (parallel where possible):**

- **Music Selector** — selects from pre-downloaded Pixabay library, matches mood/tempo from simulation metadata, aligns musical build to payoff timestamp
- **Caption Writer** — writes 3–4 captions timed to key moments
- **Final Compositor** — FFmpeg assembly: raw video + music + captions, reframe to 9:16, encode to platform specs

**Model routing:**

| Task | Model | Rationale |
|---|---|---|
| Caption writing | Claude Sonnet (cloud) | Viewer-facing, high-leverage for retention. Quality matters disproportionately. |
| Title generation | Llama 3.1 8B (local) | One line, well within local capability |
| Music selection | Programmatic (metadata matching) | No LLM needed — tag/mood-based filtering |
| FFmpeg composition | Programmatic | No LLM needed |

**Note on caption routing:** The hook caption is routed to Claude Sonnet (from the original plan's Mistral 7B) because it is the single highest-leverage element for viewer retention. A weak hook at 0–2s means people scroll past. For ~$0.01/video additional cost, the quality improvement is significant. Captions move local only after sufficient training examples are collected.

**POC scope:** Hook caption only. The content (oddly satisfying physics simulations) speaks for itself after the initial hook. Additional caption types (rule, tension, payoff) are designed into the framework but deferred until retention data indicates they would help.

**Caption framework (POC — hook only):**

For the POC, captions are limited to a single hook at the start of the video. The content speaks for itself — no mid-video or payoff captions. This keeps the Video Editor Agent simple and avoids over-captioning, which can hurt retention for "oddly satisfying" content where viewers want uninterrupted visuals.

| Type | Timing | Purpose | Example |
|---|---|---|---|
| Hook | 0–2s | Question or intrigue | "What happens when every collision spawns a new ball?" |

The hook is the single highest-leverage text element for retention. A weak hook at 0–2s means viewers scroll past. Investing in Claude Sonnet for this one caption is worth the ~$0.01/video cost.

**Future expansion:** The framework below is designed to support additional caption types (rule, tension, payoff) when data shows they improve retention. The Pydantic model and FFmpeg compositor support multi-caption rendering, but POC uses hook only.

| Type | Timing | Purpose | Status |
|---|---|---|---|
| Rule | 3–6s | Simple mechanic explanation | Deferred — add if retention data supports it |
| Tension | Mid | Build anticipation | Deferred |
| Payoff | Last 10s | Minimal or silent | Deferred |

**Audio rules:**

- Music only. No voiceover. No AI narration.
- Music volume: -18dB under ambient mix
- Fade out: last 3 seconds
- No copyrighted music. Pixabay-licensed assets only.

**Caption rendering rules:**

> ⚠️ **CONFIRM BEFORE IMPLEMENTATION** — The following caption style is a sensible default based on high-performing short-form content in the "oddly satisfying" niche. Review against actual test renders before committing to the FFmpeg compositor.

- **Font:** Inter Bold (or Montserrat Bold as fallback) — clean sans-serif, widely used in short-form content
- **Size:** 72px minimum, scaled to ~5% of frame height
- **Colour:** White (#FFFFFF) with black stroke outline (3px) for readability on any background
- **Shadow:** Subtle drop shadow (2px offset, 50% opacity black) — adds depth without clutter
- **Position:** Lower third of frame (y: ~75%), horizontally centred, never covering the main action area
- **Animation:** None for POC — static text with fade-in over 0.3s and fade-out over 0.3s. Motion graphics (pop-in, bounce, typewriter) deferred until retention data justifies the complexity.
- **Max words:** 6 per caption line
- **Duration on screen:** 2–3 seconds for hook caption
- **Render clean** — no platform watermarks baked in

---

## 5. LLM Routing & Model Strategy

### Hardware

RTX 3090 (24GB VRAM), dedicated local machine (developer's personal workstation — no cloud hosting costs). Ollama for local model management and serving. **Deployment: localhost only.** All services (review dashboard, API, Ollama, PostgreSQL, Redis) run on this machine and are not exposed to the internet. No auth system needed for POC — the only user is the developer/PM sitting at the machine or on the local network.

> **Channel identity:** Channel names, branding, and social media accounts are TBD — to be created separately before Step 12 (Upload & Publishing). This does not block development of the pipeline itself.

### Routing Table (Updated)

| Task | Model | Where | Rationale |
|---|---|---|---|
| Concept generation | Claude Sonnet | Cloud | Most important creative decision in pipeline |
| Trend Scout | Claude Sonnet + web search | Cloud | Requires real-time retrieval |
| Simulation first pass | Claude Sonnet | Cloud | Strong code reasoning, complex spatial logic |
| Simulation parameter adjustments | Mistral 7B 4-bit | Local | Mechanical edits, well-defined scope |
| Simulation debugging (complex) | Claude Sonnet | Cloud | Subtle bug reasoning |
| Frame/visual inspection | Moondream2 1.8B | Local | Basic content verification only |
| Category selection + rotation logic | Mistral 7B 4-bit | Local | Structured, rule-based |
| Caption writing | Claude Sonnet | Cloud | Viewer-facing, high-leverage (moved from local) |
| Title generation | Llama 3.1 8B | Local | One line, well within local capability |
| Validation / QA | Programmatic (Python) | Local | Arithmetic checks — no LLM needed |

### VRAM Allocation

Phases are sequential — models loaded/unloaded between phases via Ollama.

| Phase | Models Active | VRAM Used |
|---|---|---|
| Idea generation | Mistral 7B 4-bit | ~4GB |
| Simulation loop | Moondream2 (~2GB) + Mistral 7B (~4GB) | ~6GB |
| Video editing | Llama 3.1 8B | ~5GB |
| Cloud calls | — | 0GB local |

Peak local VRAM: ~6GB. Well within 24GB headroom. Remaining VRAM available for simulation rendering.

### LiteLLM Configuration

```yaml
model_list:
  # Cloud models
  - model_name: concept-developer
    litellm_params:
      model: claude-sonnet-4-6
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: simulation-first-pass
    litellm_params:
      model: claude-sonnet-4-6
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: caption-writer
    litellm_params:
      model: claude-sonnet-4-6
      api_key: os.environ/ANTHROPIC_API_KEY

  # Local models
  - model_name: idea-agent-local
    litellm_params:
      model: ollama/mistral:7b-instruct-q4_0
      api_base: http://localhost:11434

  - model_name: sim-param-adjust
    litellm_params:
      model: ollama/mistral:7b-instruct-q4_0
      api_base: http://localhost:11434

  - model_name: title-writer
    litellm_params:
      model: ollama/llama3.1:8b
      api_base: http://localhost:11434

  - model_name: frame-inspector
    litellm_params:
      model: ollama/moondream2
      api_base: http://localhost:11434

router_settings:
  # Fallback: if local caption-writer existed, fall back to cloud
  fallbacks:
    - sim-param-adjust: ["simulation-first-pass"]
  num_retries: 2
  timeout: 120
  enable_pre_call_checks: true
```

### Fallback Strategy

LiteLLM handles error-based fallback natively (local model timeout/crash → automatically retry on cloud model). For **quality-based fallback** (local model output fails validation → retry on cloud), a thin wrapper is needed:

```python
async def call_with_quality_fallback(
    primary_model: str,
    fallback_model: str,
    messages: list,
    validator: Callable,
    response_model: BaseModel,  # Instructor Pydantic model
    pipeline_run_id: UUID | None = None,
) -> BaseModel:
    """Try primary (local), validate output, fall back to cloud if invalid.
    Records successful cloud fallbacks as RAG learnings for local model improvement."""
    try:
        result = await client.chat.completions.create(
            model=primary_model,
            messages=messages,
            response_model=response_model,
        )
        if validator(result):
            return result
        logger.warning(f"Quality check failed on {primary_model}, falling back")
    except Exception as e:
        logger.warning(f"{primary_model} failed: {e}, falling back")

    # Cloud fallback
    cloud_result = await client.chat.completions.create(
        model=fallback_model,
        messages=messages,
        response_model=response_model,
    )

    # Learning loop: record successful cloud output for future local model improvement
    if validator(cloud_result):
        await record_cloud_learning(
            pipeline_run_id=pipeline_run_id,
            primary_model=primary_model,
            fallback_model=fallback_model,
            messages=messages,
            successful_output=cloud_result,
        )

    return cloud_result
```

**Learning loop:** Every time the cloud model succeeds where the local model failed, the input/output pair is stored in two places: (1) the `agent_runs` table with `status='escalated'` for tracking, and (2) the ChromaDB RAG knowledge base under `cloud_learnings/` so the local model can retrieve it in future similar tasks. This creates a continuous improvement cycle — the local model gets better over time as its knowledge base grows with real examples.

### Version Pinning

All model versions must be pinned explicitly. Ollama model versions shift on `pull` — pin versions in the Dockerfile and only update deliberately:

```bash
ollama pull mistral:7b-instruct-q4_0
ollama pull llama3.1:8b
ollama pull moondream:latest  # Pin to specific revision in code
```

---

## 6. Simulation Sandbox

The Simulation Agent generates and executes arbitrary Python code. This is a security concern. All simulation code runs inside a sandboxed Docker container with strict resource limits.

### Sandbox Design

```
Host machine
└── Docker: simulation-sandbox
    ├── Base image: python:3.12-slim + pygame + pymunk
    ├── Mounted volume: /workspace (read-write, simulation code + output)
    ├── No network access (--network=none)
    ├── Resource limits:
    │   ├── Memory: 4GB (--memory=4g)
    │   ├── CPU: 2 cores (--cpus=2)
    │   └── Timeout: 300s (5 min max execution)
    ├── Read-only filesystem except /workspace
    └── No GPU passthrough needed (Pygame renders on CPU)
```

### Sandbox Dockerfile

```dockerfile
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    pygame==2.6.1 \
    pymunk==6.8.1 \
    numpy==1.26.4 \
    Pillow==10.4.0

RUN useradd -m -s /bin/bash sandbox
USER sandbox
WORKDIR /workspace

# No ENTRYPOINT — command provided at runtime
```

### Execution Flow

```python
def execute_simulation(code: str, timeout: int = 300) -> SimulationResult:
    """Execute simulation code in Docker sandbox."""
    # Write code to temp directory
    workspace = Path(tempfile.mkdtemp())
    (workspace / "simulation.py").write_text(code)

    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "--network=none",
            "--memory=4g",
            "--cpus=2",
            f"--stop-timeout={timeout}",
            "-v", f"{workspace}:/workspace",
            "simulation-sandbox:latest",
            "python", "/workspace/simulation.py"
        ],
        capture_output=True,
        text=True,
        timeout=timeout + 30,  # Grace period beyond Docker timeout
    )

    return SimulationResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        output_files=list(workspace.glob("output/*")),
    )
```

### Why 4GB Memory Limit

- Pymunk simulations with 500+ rigid bodies typically use 200–800MB
- Pygame surface rendering for 1080×1920 at 60fps uses ~100MB
- MP4 encoding buffer adds ~500MB
- 4GB provides comfortable headroom without risking the host machine (which needs VRAM and system RAM for Ollama and other services)
- If a simulation leaks memory or enters an infinite spawn loop, the OOM killer terminates the container cleanly

---

## 7. Music & Audio Strategy

### Recommendation: Pre-Downloaded Pixabay Library

**Primary source: Pixabay Music** — royalty-free, commercially licensed, no attribution required. Explicitly permits use in monetised videos across YouTube, TikTok, and all target platforms.

**Approach: pre-download a curated library rather than API calls at runtime.**

Rationale:

1. **Pixabay's API does not expose music search/download endpoints** — it covers images and videos only. Music must be downloaded from the website.
2. **ContentID risk mitigation** — some Pixabay contributors register their tracks with YouTube's Content ID system. While you have the right to use the music (and can dispute claims), automated claims are friction. By curating the library upfront, you can test tracks against ContentID before they enter rotation.
3. **Consistency** — a curated set of 50–100 tracks tagged by mood, tempo, and energy curve gives you deterministic matching without runtime API dependencies.
4. **Offline operation** — no dependency on external service availability during video production.

### Music Library Structure

```
/music/
├── metadata.json          # Track index with tags
├── tracks/
│   ├── upbeat_120bpm_01.mp3
│   ├── tense_90bpm_01.mp3
│   ├── satisfying_ambient_01.mp3
│   └── ...
└── contentid_cleared/     # Tracks verified to not trigger ContentID
```

### Metadata Schema

```json
{
  "track_id": "upbeat_120bpm_01",
  "filename": "tracks/upbeat_120bpm_01.mp3",
  "source": "pixabay",
  "pixabay_id": 12345,
  "artist": "contributor_name",
  "license": "pixabay_content_license",
  "duration_sec": 120,
  "bpm": 120,
  "mood": ["upbeat", "energetic"],
  "energy_curve": "building",
  "genre": "electronic",
  "contentid_status": "cleared",
  "last_used_at": null,
  "use_count": 0
}
```

### Music Selection Logic

The Music Selector subagent is programmatic (no LLM needed):

1. Read simulation metadata: mood, energy curve, payoff timestamp, duration
2. Filter tracks by mood match and minimum duration (≥ video duration)
3. Prefer tracks with `energy_curve` matching the simulation (building → climax aligns with payoff)
4. Deprioritise recently-used tracks (avoid repetition)
5. Select top match, trim/fade to video duration using FFmpeg

### Initial Curation

For the POC, manually curate 50–80 tracks from Pixabay across these categories:

| Mood | Count | Use Case |
|---|---|---|
| Upbeat / energetic | 15 | Ball pits, high-action collisions |
| Tense / building | 15 | Domino chains, stacking before collapse |
| Chill / ambient | 10 | Slow marble runs, sorting |
| Dramatic / epic | 10 | Destruction, large-scale events |

**Avoid tracks with the ContentID shield icon on Pixabay.** For any tracks that do trigger ContentID claims during testing, remove them from the library and document the track ID to avoid re-adding.

> **Critical — YouTube Content ID verification:** A track being "royalty-free" under Pixabay's license does not guarantee it won't trigger YouTube's Content ID system. Some contributors register tracks with Content ID independently. Since our videos are >60 seconds, **any Content ID claim will cause YouTube to block the video globally** (per YouTube's 3-minute Shorts policy). Every track must be tested against Content ID before entering the `contentid_cleared/` directory. Test method: upload a private unlisted YouTube video using the track and monitor for claims over 48 hours. Only tracks that receive zero claims enter rotation.

---

## 8. Human Review Gate

### Design Principles

The reviewer is a project manager with limited time. The review interface must be:

- **Fast** — one-click approve/reject, no navigation required
- **Self-contained** — all context visible on one screen (no downloading files, no opening terminals)
- **Lightweight** — no complex dashboard, no auth system for POC (single-user)
- **Actionable** — rejection captures a reason code that feeds back into the system

### Review Dashboard

A minimal FastAPI + Jinja2 web application. Single page, no auth for POC (restrict to localhost or local network).

**Review page shows:**

```
┌─────────────────────────────────────────────────────┐
│  REVIEW QUEUE (3 pending)                           │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ▶ [Video Player - inline, auto-plays]              │
│                                                     │
│  ── Quick Summary ──────────────────────────────    │
│  Category:    Ball Pit Collision                    │
│  Concept:     "Every collision spawns a new ball.   │
│               200 balls fill the screen by payoff"  │
│  Duration:    65s                                   │
│  Iterations:  3 (of 5 max)                          │
│  Cost:        $0.16                                 │
│  Captions:    "What happens when..." (0s)           │
│                                                     │
│  ── Stats ──────────────────────────────────────    │
│  FPS: 32 avg | Bodies: 247 peak | Payoff: 48s      │
│  Validation: ✅ All checks passed                   │
│                                                     │
│  ── Actions ────────────────────────────────────    │
│  [✅ Approve]  [❌ Bad Concept]  [❌ Bad Sim]       │
│  [❌ Bad Edit]  [🔄 Request Re-edit]                │
│                                                     │
│  Optional feedback: [________________________]      │
│                                                     │
│  ← Previous    Next →    (2 more in queue)          │
└─────────────────────────────────────────────────────┘
```

### Rejection Reason Codes

| Code | Meaning | Pipeline Action |
|---|---|---|
| `bad_concept` | Idea is boring, repetitive, or doesn't work | Route back to Idea Agent for replacement |
| `bad_simulation` | Simulation has visual issues, glitches, or is unsatisfying | Route back to Simulation Agent (or Idea Agent if fundamentally broken) |
| `bad_edit` | Video assembly issue (music mismatch, bad captions, timing) | Route back to Video Editor Agent |
| `request_reedit` | Minor tweaks needed (caption wording, music swap) | Route back to Video Editor with feedback |

### Notification

When a video enters the review queue, send a notification to the PM. For POC, a simple **Slack webhook** or **email** with a link to the review page. No need for a complex notification system.

### Queue Management

Videos queue up for review. The PM can batch-review at a convenient time (e.g., review 3 videos in 5 minutes once per day). Approved videos automatically enter the publish queue — no further action needed.

**Failed pipeline runs** also appear in the dashboard with two recovery options: "Resume from failed step" (uses LangGraph checkpoint) or "Restart entire pipeline" (new run from scratch). See Section 2, Recovery Options.

---

## 9. Upload Scheduling & Distribution

### Upload Service: Upload-Post API

For the POC, use **Upload-Post** (upload-post.com) — a managed API service that handles multi-platform publishing from a single endpoint.

**Why Upload-Post over custom integrations:**

- Single API for YouTube Shorts, TikTok, Instagram Reels, Facebook Reels, Snapchat Spotlight
- Built-in scheduling (exact time or queue-based)
- Handles platform-specific encoding requirements
- Manages OAuth/auth tokens
- Free tier (10 uploads/month) for initial testing, paid plans scale cheaply
- No need to maintain 5 separate platform integrations

**If Upload-Post is unsuitable** (pricing, reliability, feature gaps), fallback options:

- YouTube: official Data API v3 (well-documented, reliable)
- TikTok: `tiktok-uploader` Python package (session-cookie based — fragile, requires periodic cookie refresh)
- Instagram: Graph API (business accounts only)
- Build custom for YouTube first, add other platforms incrementally

### Publishing Schedule

```python
# Platform-specific optimal posting times (UTC, adjust for audience timezone)
PUBLISH_SCHEDULE = {
    "youtube_shorts": {
        "channel_1": {"times": ["09:00", "17:00"], "max_daily": 2},
    },
    "tiktok": {
        "account_1": {"times": ["08:00", "12:00", "18:00"], "max_daily": 3},
    },
    "instagram_reels": {
        "account_1": {"times": ["11:00"], "max_daily": 1},
    },
    "facebook_reels": {
        "account_1": {"times": ["10:00"], "max_daily": 1},
    },
    "snapchat_spotlight": {
        "account_1": {"times": ["09:00", "14:00"], "max_daily": 2},
    },
}
```

### Publishing Flow

```
Approved video → publish_queue table →
  Scheduler (Celery beat / cron) checks queue every 15 min →
    For each platform+account at scheduled time:
      Pop next queued video →
      Upload via Upload-Post API (or platform API) →
      Record in publish_log →
      Update status
```

### Platform Duration Requirements

**YouTube Shorts:** Maximum 3 minutes (180 seconds) as of October 15, 2024. No minimum duration for monetisation. Shorts over 1 minute with any Content ID claim are blocked globally — royalty-free music is mandatory for videos >60s. Revenue share: 45% to creator from Shorts Feed ad pool.

**TikTok Creator Rewards:** Minimum 60 seconds for monetisation eligibility. No maximum (TikTok supports up to 60 minutes). Views must be ≥5 seconds and from unique accounts to count as "qualified." Revenue is RPM-based, not a fixed pool.

**Target duration: 65 seconds.** This comfortably exceeds TikTok's 60-second minimum while staying short enough for high completion rates. Acceptable range: 62–68 seconds. The FFmpeg compositor should add minimal padding if the raw simulation falls short of 62 seconds.

> **Note:** TikTok's Creator Rewards Program requires content to be "original, high-quality content that is filmed, designed, and produced entirely by yourself." Programmatic simulation content sits in a gray area — existing channels in this niche appear to be monetised, but this should be monitored. **Mitigation: add subtle branding/watermark** to all videos. This establishes channel identity and strengthens the "produced by you" argument. The watermark should be minimal (small logo in corner, consistent across all videos) and baked into the FFmpeg composition step. Track whether content is flagged as non-original during production burn-in — if flagged, escalate to more aggressive visual style variation or intro cards.

### Platform-Specific Metadata

Each platform gets tailored metadata. Titles and descriptions are generated once, then adapted:

```python
# Example: platform-specific title adaptation
base_title = "What happens when every collision spawns a new ball?"
platform_titles = {
    "youtube": base_title,  # YouTube favours searchable, keyword-rich
    "tiktok": base_title + " #oddlysatisfying #physics #simulation",
    "instagram": base_title,  # Cleaner, fewer hashtags
}
```

---

## 10. Database Schema

```sql
-- ============================================================
-- Core tables shared across all pipelines
-- ============================================================

-- Top-level pipeline execution record
CREATE TABLE pipeline_runs (
  pipeline_run_id UUID PRIMARY KEY,
  pipeline VARCHAR(50) NOT NULL,
  idea_id UUID,
  simulation_id UUID,
  output_id UUID,
  status VARCHAR(50) DEFAULT 'running',
  -- running, idea_phase, simulation_phase, editing_phase,
  -- pending_review, approved, published, failed, cancelled
  started_at TIMESTAMP DEFAULT NOW(),
  completed_at TIMESTAMP,
  total_cost_usd DECIMAL(8,4),
  total_duration_sec INTEGER
);

-- Ideas / concepts
CREATE TABLE video_ideas (
  idea_id UUID PRIMARY KEY,
  pipeline_run_id UUID REFERENCES pipeline_runs(pipeline_run_id),
  pipeline VARCHAR(50) NOT NULL,
  concept JSONB NOT NULL,
  category VARCHAR(100),
  novelty_score DECIMAL(3,1),
  feasibility_score DECIMAL(3,1),
  status VARCHAR(50) DEFAULT 'pending',
  -- pending, in_production, approved, rejected, published, cancelled
  performance_data JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Category rotation cache (refreshed periodically, not source of truth)
-- Source of truth: derived from video_ideas + publish_log
CREATE TABLE category_stats (
  pipeline VARCHAR(50) NOT NULL,
  category VARCHAR(100) NOT NULL,
  total_count INTEGER DEFAULT 0,
  last_used_at TIMESTAMP,
  avg_performance DECIMAL(5,2),
  streak_count INTEGER DEFAULT 0,
  videos_last_30_days INTEGER DEFAULT 0,
  PRIMARY KEY (pipeline, category)
);

-- Simulations
CREATE TABLE simulations (
  simulation_id UUID PRIMARY KEY,
  idea_id UUID REFERENCES video_ideas(idea_id),
  pipeline_run_id UUID REFERENCES pipeline_runs(pipeline_run_id),
  pipeline VARCHAR(50) NOT NULL,
  code_path TEXT,
  raw_video_path TEXT,
  frame_contact_sheet TEXT,
  stats JSONB,
  -- {duration_sec, peak_body_count, avg_fps, payoff_timestamp_sec}
  scene_metadata JSONB,
  -- {key_moments, mood, palette, energy_curve, etc.}
  validation_passed BOOLEAN DEFAULT false,
  iteration_count INTEGER DEFAULT 0,
  iteration_history JSONB DEFAULT '[]',
  -- [{iteration: 1, code_hash: "...", error: null, validation: {...}}, ...]
  created_at TIMESTAMP DEFAULT NOW()
);

-- Final assembled videos
CREATE TABLE outputs (
  output_id UUID PRIMARY KEY,
  simulation_id UUID REFERENCES simulations(simulation_id),
  pipeline_run_id UUID REFERENCES pipeline_runs(pipeline_run_id),
  final_video_path TEXT,
  captions JSONB,
  music_track TEXT,
  music_metadata JSONB,
  -- {track_id, source, bpm, mood, artist}
  title TEXT,
  description TEXT,
  status VARCHAR(50) DEFAULT 'pending_review',
  -- pending_review, approved, rejected, published
  review_action VARCHAR(50),
  -- null, approved, bad_concept, bad_simulation, bad_edit, request_reedit
  review_feedback TEXT,
  reviewed_at TIMESTAMP,
  cost_usd DECIMAL(6,4),
  created_at TIMESTAMP DEFAULT NOW()
);

-- Publishing queue (decouples production from distribution)
CREATE TABLE publish_queue (
  queue_id UUID PRIMARY KEY,
  output_id UUID REFERENCES outputs(output_id),
  platform VARCHAR(50) NOT NULL,
  account VARCHAR(100),
  scheduled_for TIMESTAMP,
  platform_title TEXT,
  platform_description TEXT,
  platform_tags JSONB,
  status VARCHAR(50) DEFAULT 'queued',
  -- queued, publishing, published, failed, cancelled
  attempts INTEGER DEFAULT 0,
  last_error TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Publish log (after successful upload)
CREATE TABLE publish_log (
  publish_id UUID PRIMARY KEY,
  output_id UUID REFERENCES outputs(output_id),
  queue_id UUID REFERENCES publish_queue(queue_id),
  platform VARCHAR(50) NOT NULL,
  account VARCHAR(100),
  platform_video_id TEXT,
  published_at TIMESTAMP,
  -- Engagement metrics (populated by analytics sync job)
  views_7d INTEGER,
  views_30d INTEGER,
  likes INTEGER,
  comments INTEGER,
  shares INTEGER,
  avg_view_duration_sec DECIMAL(6,2),
  retention_rate DECIMAL(5,4),
  revenue_usd DECIMAL(8,4)
);

-- Agent execution log (every LLM call and significant operation)
CREATE TABLE agent_runs (
  run_id UUID PRIMARY KEY,
  pipeline_run_id UUID REFERENCES pipeline_runs(pipeline_run_id),
  idea_id UUID,
  agent_name VARCHAR(100) NOT NULL,
  -- 'idea_agent', 'simulation_agent', 'video_editor_agent'
  step_name VARCHAR(100),
  -- 'concept_developer', 'frame_inspector', 'caption_writer', etc.
  model_used VARCHAR(100),
  -- 'claude-sonnet-4-6', 'ollama/mistral:7b', 'programmatic', etc.
  input_summary JSONB,
  output_summary JSONB,
  tokens_in INTEGER,
  tokens_out INTEGER,
  cost_usd DECIMAL(8,6),
  latency_ms INTEGER,
  status VARCHAR(50),
  -- 'success', 'failed', 'retried', 'escalated'
  error_message TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Training examples for fine-tuning
CREATE TABLE training_examples (
  example_id UUID PRIMARY KEY,
  simulation_id UUID REFERENCES simulations(simulation_id),
  pipeline VARCHAR(50) NOT NULL,
  concept_brief JSONB NOT NULL,
  simulation_code TEXT NOT NULL,
  validation_passed BOOLEAN NOT NULL,
  human_approved BOOLEAN NOT NULL,
  rejection_reason VARCHAR(50),
  created_at TIMESTAMP DEFAULT NOW()
);

-- Pipeline configuration (per-pipeline defaults)
CREATE TABLE pipeline_config (
  pipeline VARCHAR(50) PRIMARY KEY,
  engine VARCHAR(100) NOT NULL,
  target_duration_min INTEGER DEFAULT 62,
  target_duration_max INTEGER DEFAULT 68,
  target_fps INTEGER DEFAULT 30,
  target_resolution VARCHAR(20) DEFAULT '1080x1920',
  max_iterations INTEGER DEFAULT 5,
  categories JSONB NOT NULL,
  -- ["ball_pit", "marble_funnel", "domino_chain", "destruction"]
  active BOOLEAN DEFAULT true,
  config JSONB DEFAULT '{}',
  -- Engine-specific configuration
  created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- Indexes
-- ============================================================

CREATE INDEX idx_pipeline_runs_status ON pipeline_runs(pipeline, status);
CREATE INDEX idx_pipeline_runs_started ON pipeline_runs(started_at DESC);
CREATE INDEX idx_ideas_pipeline_status ON video_ideas(pipeline, status);
CREATE INDEX idx_ideas_category ON video_ideas(pipeline, category, created_at DESC);
CREATE INDEX idx_category_stats_pipeline ON category_stats(pipeline);
CREATE INDEX idx_simulations_pipeline ON simulations(pipeline, created_at DESC);
CREATE INDEX idx_outputs_status ON outputs(status);
CREATE INDEX idx_publish_queue_status ON publish_queue(status, scheduled_for);
CREATE INDEX idx_publish_log_platform ON publish_log(platform, published_at DESC);
CREATE INDEX idx_agent_runs_pipeline ON agent_runs(pipeline_run_id);
CREATE INDEX idx_agent_runs_model ON agent_runs(model_used, created_at DESC);
CREATE INDEX idx_training_approved ON training_examples(pipeline, human_approved, validation_passed);
```

### Schema Notes

- **`category_stats`** is a materialised cache, not the source of truth. It is refreshed periodically via cron or trigger. Source of truth for category performance is derived from `video_ideas` + `publish_log`.
- **`pipeline_config`** stores per-pipeline defaults so that adding a new pipeline is a database row, not a code change. Engine-specific config goes in the JSONB `config` field.
- **`simulations.iteration_history`** stores a summary of each iteration attempt (code hash, errors, validation results) for debugging and fine-tuning data collection.
- **`outputs.review_action` and `review_feedback`** capture structured rejection reasons that feed back into the system and can be used for training data curation.
- All tables with `pipeline` columns support future multi-pipeline operation without schema changes.

---

## 11. Tech Stack

| Component | Technology | Notes |
|---|---|---|
| Orchestration | LangGraph (pinned version) | Agent logic as plain Python functions, loosely coupled |
| Structured LLM output | Instructor | Pydantic-validated JSON from all LLM calls |
| Physics (2D) | Pygame 2.6 + Pymunk 6.8 | POC engine |
| Physics (3D, planned) | Blender Python API | Future pipeline |
| Vehicle sim (planned) | BeamNG.tech + BeamNGpy | Future pipeline, requires license |
| Orbital sim (planned) | Custom n-body (numpy) | Future pipeline |
| Video assembly | FFmpeg | Industry standard |
| Local LLM serving | Ollama (pinned model versions) | Model management and serving |
| LLM routing | LiteLLM proxy | Unified API, fallback routing, cost tracking |
| Knowledge base / RAG | ChromaDB + LlamaIndex (ingestion) | LlamaIndex for doc loading/chunking/embedding, ChromaDB for storage. Migrate to pgvector later. |
| Database | PostgreSQL (JSONB) | State, content, execution logs, publishing |
| Task queue | Redis + Celery (or arq) | Background pipeline runs, publish scheduler |
| Service layer | Python FastAPI | API + review dashboard |
| Review dashboard | FastAPI + Jinja2 | Minimal, single-page, inline video preview |
| Monitoring / Observability | Langfuse | LLM tracing, cost tracking, quality scoring |
| Simulation sandbox | Docker (--network=none, 4GB RAM limit) | Isolated code execution |
| Upload / Distribution | Upload-Post API | Multi-platform publishing |
| Containerisation | Docker Compose | All services on dedicated local machine |
| Testing | pytest + markers | Unit / integration / quality / LLM tiers |

### What Is NOT in the Stack (and why)

| Excluded | Reason |
|---|---|
| Kubernetes | Premature — Docker Compose on dedicated machine is correct for single-GPU, 2–3 videos/day |
| Kafka / RabbitMQ | Redis is sufficient for the task queue at this scale |
| MLflow / W&B | Not needed until fine-tuning phase — use `training_examples` table for now |
| LangChain agent abstractions | Intentionally excluded — LangGraph is used for orchestration only, agent logic is plain Python. LlamaIndex is used for RAG ingestion only (not for agent/chain abstractions). |

---

## 12. Testing Strategy & Code Quality Enforcement

> **Full specification:** See the companion document **Code Quality & Testing Enforcement Addendum** for complete configuration files, interface definitions, CI pipeline YAML, and Pydantic model contracts.

### Principles

- Test framework and quality gates are built **before** the pipeline
- Every component is testable in isolation
- AI-assisted tests supplement (never replace) programmatic checks
- Test structure mirrors pipeline structure — adding a new pipeline = adding a new test directory
- Tests are tiered by speed, cost, and dependency
- **No code enters `main` unless it passes every quality gate** — whether written by a human or an AI agent
- Interface enforcement via ABCs + mypy strict mode makes it structurally impossible to add a pipeline without implementing required methods

### Test Directory Structure

```
tests/
├── unit/
│   ├── agents/
│   │   ├── test_idea_agent.py
│   │   ├── test_simulation_agent.py
│   │   └── test_video_editor_agent.py
│   ├── services/
│   │   ├── test_category_rotation.py
│   │   ├── test_validation_engine.py
│   │   ├── test_music_selector.py
│   │   ├── test_caption_framework.py
│   │   ├── test_publish_queue.py
│   │   └── test_sandbox_executor.py
│   ├── models/
│   │   └── test_pydantic_models.py       # All data contracts validate/reject correctly
│   └── db/
│       └── test_db_operations.py
│
├── integration/
│   ├── test_pipeline_e2e.py              # Full pipeline with mocked LLMs
│   ├── test_simulation_sandbox.py        # Docker sandbox isolation & limits
│   ├── test_checkpoint_recovery.py       # Kill & resume from LangGraph checkpoint
│   ├── test_review_flow.py              # Approve/reject → correct routing
│   ├── test_fallback_routing.py         # Local fail → cloud escalation
│   └── test_llm_integration.py          # Real LLM calls (marked slow)
│
├── quality/
│   ├── test_video_output.py              # Programmatic quality checks on produced videos
│   ├── test_ai_quality.py                # AI-assisted checks (Moondream2)
│   └── golden_set/
│       ├── concepts/                     # Known-good concept JSONs
│       └── simulations/                  # Known-good simulation scripts + outputs
│
├── pipelines/
│   ├── test_pipeline_interface.py        # ⭐ Shared interface tests (auto-run per pipeline)
│   ├── physics/
│   │   ├── fixtures/
│   │   │   ├── valid_output.mp4
│   │   │   └── invalid/
│   │   │       ├── blank_video.mp4
│   │   │       ├── wrong_resolution.mp4
│   │   │       └── frozen_frames.mp4
│   │   ├── test_physics_pipeline.py      # Physics-specific tests
│   │   ├── test_pymunk_patterns.py       # Pymunk code generation patterns
│   │   └── test_scenario_categories.py   # Each category produces runnable code
│   └── (future pipelines add directories here)
│
├── conftest.py                           # Shared fixtures, DB setup, mock LLM
└── pytest.ini                            # Markers and configuration
```

When you add Pipeline 2 (BeamNG), you add `tests/pipelines/beamng/` with engine-specific tests and fixtures. The shared `test_pipeline_interface.py` **automatically discovers and tests the new pipeline** via the pipeline registry — no manual wiring needed.

### pytest Markers

```ini
# pytest.ini
[pytest]
markers =
    unit: Fast, no external dependencies
    integration: Needs DB, may need Docker
    llm: Calls real LLM (slow, costs money) — skip in CI
    quality: Runs on produced video files
    pipeline: Pipeline-specific tests (use with --pipeline=physics)
    slow: Tests taking >30s
```

### 12.1 Unit Tests

**Idea Agent:**

| Test | Assertion |
|---|---|
| Category rotation — hard block | Same category as previous video is never selected |
| Category rotation — soft block | Category >30% of last 30 days is deprioritised |
| Category rotation — boost | Category with <5 total videos is boosted |
| Category rotation — streak break | Category switches after 3 consecutive |
| Category rotation — empty database | First video ever selects any category |
| Category rotation — single category | Handles edge case of only one category |
| Concept JSON schema | Output validates against Pydantic model |
| Novelty score | Given existing concepts, score computed correctly |

**Simulation Agent:**

| Test | Assertion |
|---|---|
| Validation — duration in range | 62–68s passes, outside fails |
| Validation — FPS check | ≥30 passes, below fails |
| Validation — frame count | Matches expected (duration × FPS) within 1% |
| Validation — file validity | Valid MP4 passes, corrupt file fails |
| Validation — resolution check | 1080×1920 passes, wrong resolution fails |
| Validation — motion detection | Static image fails, animated content passes |
| Validation — blank detection | All-black or all-white frames fail |
| Iteration limit | Max iterations (5) respected, escalation triggered |
| Code template | Generated skeleton is syntactically valid Python |
| Parameter adjustment | Given adjustment instruction, correct code modification |

**Video Editor Agent:**

| Test | Assertion |
|---|---|
| FFmpeg command generation | Correct command string for given inputs |
| Caption timing | No overlaps, within duration, framework respected |
| Caption length | All captions ≤6 words |
| Audio levels | -18dB verified in output |
| Output specs | 9:16 ratio, correct codec, correct duration |
| Music selection | Correct mood match, no repeat within N videos |

**Shared Infrastructure:**

| Test | Assertion |
|---|---|
| LiteLLM routing | Mock responses → correct model selection per task |
| Database CRUD | All tables: create, read, update, delete |
| State transitions | All valid/invalid state transitions in pipeline graph |
| Publish queue | Correct scheduling, status updates, retry logic |
| Instructor models | All Pydantic models validate/reject correctly |

### 12.2 Integration Tests

| Test | What It Covers |
|---|---|
| End-to-end with mocked LLMs | Full pipeline: idea → sim → video → review (mock all LLM calls) |
| Checkpoint recovery | Kill pipeline mid-run, restart, verify resumption |
| Simulation sandbox | Known-good script runs correctly, known-bad script errors cleanly |
| Sandbox isolation | Container cannot access network, write outside /workspace |
| Sandbox resource limits | OOM behaviour at 4GB limit, timeout at 300s |
| Review flow | Submit video → approve → enters publish queue → correct status |
| Review rejection | Reject with each reason code → correct pipeline re-routing |
| Fallback routing | Local model failure → automatic cloud escalation via LiteLLM |

### 12.3 AI-Assisted Quality Tests

Run automatically on every produced video before it enters the review queue.

**Programmatic (no LLM, fast):**

| Check | Method |
|---|---|
| Valid MP4 | FFprobe returns valid metadata |
| Correct resolution | FFprobe: 1080×1920 |
| Correct FPS | FFprobe: ≥30fps |
| Duration in range | FFprobe: 62–68s (target 65s) |
| Audio present | FFprobe: audio stream exists |
| Audio levels | FFmpeg loudnorm filter: -18dB ±3dB |
| File size reasonable | 1MB < size < 500MB |
| No frozen frames | Frame hash comparison: <5 consecutive identical frames |
| Motion present | Pixel variance between sampled frames > threshold |
| Colour valid | Mean pixel value not at extremes (0 or 255) |
| Captions parseable | JSON validates, timestamps within video duration |

**AI-assisted (Moondream2, slower):**

| Check | Method |
|---|---|
| Content present | Sample 5 frames evenly → "Is there visual content on screen?" |
| Progression | First frame vs last frame differ meaningfully |
| No border clipping | "Are objects stuck at the edges of the frame?" |
| Payoff visible | Sample 3 frames from final 20% → "Is there activity in the scene?" |

### 12.4 Regression Tests

- Maintain a **golden set** of 10–15 concept briefs with known-good simulation outputs
- Re-run weekly against the current pipeline
- Track over time: success rate, iteration count, cost, validation pass rate
- Alert if success rate drops below 80% or average iteration count increases by >1

### 12.5 Running Tests

```bash
# Fast unit tests (every commit)
pytest -m unit

# Integration tests (every PR)
pytest -m integration

# Full suite excluding real LLM calls
pytest -m "not llm"

# Quality checks on a produced video
pytest -m quality --video-path=/path/to/video.mp4

# Pipeline-specific tests
pytest tests/pipelines/physics/

# Everything (including real LLM calls — expensive)
pytest --run-llm
```

---

## 13. Fine-Tuning Strategy

### Goal

Fine-tune a local model on simulation code generation so the first-pass moves off Claude Sonnet. Estimated savings: ~$0.08/video.

### Base Model

Code Llama 7B or Llama 3.1 8B via LoRA fine-tuning on the 3090. LoRA adapters: 50–200MB. Fits comfortably in 4-bit on 24GB VRAM.

### Training Data Collection

Every simulation that passes full validation AND human review is a candidate training example. Quality gates — only add if ALL pass:

- Simulation ran without errors
- Duration within target range
- FPS stable throughout
- All programmatic validation checks passed
- Human review approved (not just "not rejected")

**Training pair format:**

```
INPUT:  concept brief JSON
OUTPUT: working simulation file (complete Python source)
```

Rejected simulations are also stored (with `human_approved = false`) — negative examples are valuable for understanding failure modes.

### Viability Timeline

| Examples Collected | Expected Capability |
|---|---|
| 20–30 | Understands basic engine patterns |
| 50 | Handles most common scenarios reliably |
| 100 | Covers edge cases and complex mechanics |
| 200+ | Comparable to frontier model for this narrow task |

### Why It Works

The domain is extremely narrow (Pymunk uses a small, consistent API subset), output is objectively verifiable (runs or doesn't, passes validation or doesn't), base models are already strong at Python, and the frontier model becomes teacher/validator rather than workhorse. Each pipeline's fine-tuned model covers only its own engine — small, specialised, fast.

---

## 14. Pipeline Backlog

All future pipelines use the same three-agent architecture with a new simulation adapter. Adding a pipeline means: writing a new sandbox Dockerfile (with the engine's dependencies), adding scenario categories to `pipeline_config`, adding engine-specific patterns to the RAG knowledge base, and adding pipeline-specific tests to `tests/pipelines/`.

### Pipeline 1: Oddly Satisfying Physics (POC — Active)

**Engine:** Pygame + Pymunk (2D rigid body physics)
**Categories:** Ball pit, marble funnel, domino chains, destruction/stacking
**Content type:** Ball physics, collisions, sorting, destruction

### Pipeline 2: BeamNG Vehicle Simulations (Planned)

**Engine:** BeamNG.tech + BeamNGpy
**Content type:** "Which car makes this jump?", drag races, braking tests
**Note:** Requires BeamNG.tech license — contact BeamNG before building.

### Pipeline 3: Marble Races (Planned)

**Engine:** Pygame + Pymunk (shares engine with Pipeline 1)
**Content type:** Single races, multi-round brackets, surface variants
**Note:** Likely a scenario category extension of Pipeline 1 rather than a full separate pipeline.

### Pipeline 4: Space & Orbital Simulations (Planned)

**Engine:** Custom n-body (numpy + matplotlib or Pygame renderer)
**Content type:** Planet collisions, rogue bodies, orbital decay, galaxy formation

### Pipeline 5: 3D Physics & Destruction (Planned)

**Engine:** Blender Python API (Bullet physics, GPU rendering)
**Content type:** Trebuchets, structural collapse, 3D marble runs
**Note:** Significant visual quality upgrade. Blender headless rendering on RTX 3090.

### Future Ideas (Validated for Automation Potential)

| Idea | Engine | Format |
|---|---|---|
| Powder/sand material sims | Custom renderer | Material vs material |
| Evolution simulations | Custom (neural net + physics) | Timelapse + highlights |
| Racing bar charts | Matplotlib animated | Historical comparisons |
| Animal size comparisons | Matplotlib / custom | Scale animations |
| Sports career stats races | Matplotlib animated | Player vs player |

---

## 15. Cost Model

### Per-Video Cost (POC — Updated Routing)

| Task | Model | Est. Cost |
|---|---|---|
| Concept development | Claude Sonnet | $0.05 |
| Simulation first pass | Claude Sonnet | $0.08 |
| Simulation iterations (avg 2) | Mistral 7B local | $0.00 |
| Frame analysis | Moondream2 local | $0.00 |
| Caption writing | Claude Sonnet (moved from local) | $0.01 |
| Title generation | Llama 3.1 8B local | $0.00 |
| Music selector | Programmatic | $0.00 |
| FFmpeg render | — | $0.00 |
| Upload-Post API | — | ~$0.02 |
| **Total** | | **~$0.16–$0.22** |

### Cost Risk

If the Simulation Agent frequently escalates to Claude for debugging (not just Mistral for parameter tweaks), average cost could reach $0.30–$0.40/video. The `agent_runs` table tracks actual cost per video from day one. Set an alert if 7-day rolling average exceeds $0.30.

### Post Fine-Tuning Target

~$0.06/video (simulation first-pass moves local).

### Monthly Infrastructure Cost

| Item | Cost |
|---|---|
| Electricity (dedicated machine, ~200W avg) | ~$30 |
| Upload-Post API (100 uploads/month) | ~$10–$20 |
| Anthropic API (~60 videos × $0.15) | ~$9 |
| Domain / hosting (review dashboard) | $0 (localhost on local machine) |
| **Total monthly** | **~$50–$60** |

---

## 16. Monetisation & Distribution

### Platform Requirements

**YouTube (YouTube Partner Program):**

| Tier | Requirements | Unlocks |
|---|---|---|
| Tier 1 | 500 subs + 3M Shorts views/90 days | Memberships, Super Thanks |
| Tier 2 | 1,000 subs + 10M Shorts views/90 days | Ad revenue (45% to creator) |

**TikTok (Creator Rewards Program):**

- 10,000 followers + 100,000 views in last 30 days
- Videos must be ≥60 seconds (1 minute) for monetisation eligibility
- Personal account only (Business Accounts not eligible)
- Content must be original — programmatic content is a gray area, monitor during burn-in
- RPM-based: typically $0.40–$1.00/1K qualified views, up to $6/1K for high-retention niche content

**Other platforms:** Instagram Reels (invite-only monetisation), Facebook Reels, Snapchat Spotlight (25 posts/month minimum for monetisation).

### Revenue Projections

**Conservative** (50k avg views/video, 30 videos/month):

| Platform | Monthly Views | Est. Revenue |
|---|---|---|
| YouTube Shorts (2 channels) | 3M | ~$270 |
| TikTok | 1.5M | ~$900 |
| Facebook Reels | 900k | ~$80 |
| Instagram Reels | 900k | ~$30 |
| Snapchat | 750k | ~$500 |
| **Total** | **~7M** | **~$1,780/mo** |

### Realistic Timeline

| Period | Expectation |
|---|---|
| Months 1–2 | Zero revenue. Building to thresholds. |
| Months 3–4 | Hit YouTube Tier 1. First TikTok monetisation. |
| Months 4–6 | First real ad revenue. $200–$600/month combined. |
| Month 6+ | Compounds. One viral video can pull forward by 2–3 months. |

---

## 17. POC Success Criteria

- [ ] Idea Agent generates non-repetitive concepts respecting all category rotation rules
- [ ] Simulation Agent produces validated output in ≤5 iterations for 80%+ of concepts
- [ ] Final video correctly formatted (9:16, 65s target / 62–68s acceptable, clean audio, hook caption)
- [ ] Full pipeline runs end-to-end in <15 minutes per video
- [ ] Cost per video stays under $0.25 (updated to reflect caption routing change)
- [ ] Human review gate works: PM can approve/reject with one click in <2 minutes per video
- [ ] Approved videos automatically enter publish queue
- [ ] Publishing to at least YouTube Shorts + TikTok works end-to-end
- [ ] Training examples accumulate correctly with all quality gates enforced
- [ ] 2 videos/day sustained output achievable without intervention
- [ ] At least 20 training examples collected in first 2 weeks of operation
- [ ] Cost tracking matches projections (within 2x of target)
- [ ] Pipeline recovers from any single-agent failure without manual intervention
- [ ] All unit and integration tests pass in CI
- [ ] Rejected videos include structured reason codes that feed back into system

---

## 18. Implementation Plan

Steps are ordered by dependency. Each step produces a testable deliverable. No step requires more than one to be "in progress" at a time, though independent steps can be parallelised by multiple developers.

### Phase 1: Foundation

**Step 1 — Repository, Environment & Quality Gates Setup**

- Initialise repository with project structure, `.gitignore`, `pyproject.toml`
- Set up Docker Compose with services: PostgreSQL, Redis, Ollama
- Configure Python environment with dependency pinning
- **Configure Ruff** with full rule set (see Addendum Section 2)
- **Configure mypy strict mode** with Pydantic plugin (see Addendum Section 3)
- **Set up pre-commit hooks**: Ruff, mypy, Gitleaks, debug-statements, large file check, no-commit-to-branch (see Addendum Section 1)
- **Set up CI pipeline** (GitHub Actions): lint → unit tests → integration tests → pipeline interface tests. Configure branch protection on `main` to block merges on failure (see Addendum Section 7)
- **Define abstract base classes** for `PipelineAdapter`, `IdeaAgent`, `SimulationAgent`, `VideoEditorAgent` (see Addendum Section 4)
- **Define Pydantic models** for all data contracts: `ConceptBrief`, `SimulationResult`, `VideoOutput`, `ValidationResult`, `CaptionSet` (see Addendum Section 5)
- Create `tests/` directory structure with `conftest.py`, `pytest.ini`, markers, and shared interface test suite
- Create pipeline registry (`pipeline/registry.py`)
- Pull and pin Ollama model versions (Mistral 7B, Llama 3.1 8B, Moondream2)
- **Deliverable:** `docker-compose up` starts all infrastructure services. `pytest` runs (with no tests yet). Pre-commit hooks block bad code. CI pipeline runs on PRs. ABCs and Pydantic models define the contracts all agents must implement.

**Step 2 — Database Schema & Migrations**

- Apply full database schema from Section 10
- Seed `pipeline_config` table with Physics pipeline configuration
- Seed `category_stats` with initial categories (ball_pit, marble_funnel, domino_chain, destruction)
- Write database access layer (async SQLAlchemy or similar)
- Write unit tests for all CRUD operations
- **Deliverable:** All database tables exist, CRUD tests pass.

**Step 3 — LiteLLM Proxy & Instructor Setup**

- Configure LiteLLM proxy with all model routes (cloud + local)
- Configure fallback routing
- Set up Instructor with Pydantic models for all structured outputs (ConceptBrief, SimulationStats, CaptionSet, ValidationResult, etc.)
- Write unit tests: mock LLM responses, assert correct routing and Pydantic validation
- **Deliverable:** LiteLLM proxy serves requests. Instructor parses structured output from both Anthropic and Ollama models.

**Step 4 — Simulation Sandbox**

- Build sandbox Docker image with Pygame + Pymunk
- Implement `execute_simulation()` function with resource limits
- Test: known-good script executes and returns output
- Test: known-bad script (infinite loop, OOM) terminates cleanly
- Test: sandbox cannot access network, cannot write outside /workspace
- **Deliverable:** Simulation code executes in isolated Docker container with enforced resource limits.

### Phase 2: Agents (Individual)

**Step 5 — Validation Engine**

- Implement all Tier 1 programmatic validation checks (duration, FPS, frame count, resolution, motion, colour, frozen frames)
- Implement Tier 2 AI-assisted checks (Moondream2 frame inspection)
- Write unit tests for every validation check with known-good and known-bad inputs
- **Deliverable:** `validate_simulation(video_path) → ValidationResult` works for all checks.

**Step 6 — Simulation Agent**

**6a — Prompt Development Harness (prerequisite)**

Before wiring the Simulation Agent into LangGraph, build a standalone harness for iterating on prompts manually. This is a critical de-risking step — the entire pipeline depends on LLMs generating working Pymunk code, and that assumption must be validated before building automation around it.

The harness is a CLI tool that:

1. Generates a concept brief (either from the Idea Agent or a hand-crafted fixture)
2. Displays the concept and the prompt that would be sent to the LLM
3. Allows the developer to copy the prompt into any LLM chat interface (Claude, ChatGPT, local model)
4. Accepts the generated simulation code back as input
5. Executes it in the sandbox and runs all validation checks
6. Records the result: concept → prompt → code → validation outcome → manual notes
7. Stores successful runs as golden set fixtures and RAG knowledge entries

The harness produces three outputs:
- **Refined prompt templates** that work reliably for each scenario category
- **Initial golden set** for regression testing (10–15 working concept→simulation pairs)
- **First RAG entries** (working patterns the Simulation Agent can retrieve)

> **Note:** The base prompts below are intentionally minimal starting points. They MUST be expanded and refined through hands-on testing with the harness before the Simulation Agent is considered functional. Prompt quality is the single highest-leverage factor in pipeline reliability.

**6b — Initial Base Prompt (simulation first pass)**

```
You are a physics simulation developer. You write Python code using Pygame and Pymunk to create visually satisfying 2D physics simulations that render to MP4 video.

## Requirements
- Use Pygame 2.6 + Pymunk 6.8
- Render at 1080x1920 (9:16 portrait) at 60 FPS
- Target duration: 65 seconds
- Run headless (no SDL display) — render frames to a surface, save as MP4
- Output file: /workspace/output/simulation.mp4
- Use pygame.image.save() for frames, then FFmpeg to encode MP4
- All physics bodies must be visible and use distinct colours

## Concept
{concept_brief_json}

## Simulation Guidelines
- Start with a clear setup phase (2-5 seconds)
- Build tension through the middle section
- Include a satisfying payoff/climax in the final 20% of the video
- Use a dark background (#1a1a2e or similar) for visual contrast
- Bodies should be colourful and easy to distinguish
- Ensure bodies don't clip through boundaries
- Add slight camera shake or zoom during high-energy moments if appropriate

## Output Format
Return ONLY the complete Python file. No explanations, no markdown. The code must be directly executable.
```

**6c — Simulation Agent implementation**

- Implement Simulation Agent as a plain Python class with methods for: code generation (Claude), parameter adjustment (Mistral), frame extraction, validation orchestration
- Implement iteration loop: generate → execute in sandbox → validate → adjust → repeat
- Implement escalation logic (max iterations → back to Idea Agent)
- Build RAG knowledge base using **LlamaIndex IngestionPipeline**: auto-ingest Pymunk/Pygame docs via web loader, seed initial code patterns and scenario templates, configure auto-ingestion of approved simulations after human review
- Wire up ChromaDB for `query_knowledge` tool (LlamaIndex feeds ChromaDB at ingestion time; runtime queries go direct to ChromaDB)
- **Implement cloud fallback learning loop:** when local LLM fails and cloud succeeds, automatically store the successful output in RAG under `cloud_learnings/` (see Section 4, RAG Learning Loop)
- Write unit tests for iteration logic, escalation, parameter adjustment parsing
- Write integration test: given a mock concept, agent produces a valid simulation (with mocked LLM)
- **Deliverable:** Simulation Agent can receive a concept brief and produce a validated simulation video. Prompt templates have been validated through the harness with at least 10 successful runs across all 4 scenario categories.

**Step 7 — Idea Agent**

- Implement category rotation logic as pure functions (testable without LLM)
- Implement Inventory Analyst (SQL queries against database)
- Implement Category Selector (Mistral 7B, structured output via Instructor)
- Implement Concept Developer (Claude Sonnet, structured output via Instructor)
- Write comprehensive unit tests for rotation rules (all edge cases from Section 12.1)
- Write integration test: given database state, agent produces a valid concept
- **Deliverable:** Idea Agent generates non-repetitive concepts respecting rotation rules.

**Step 8 — Video Editor Agent**

- Implement Music Selector (programmatic, no LLM — tag-based filtering from local library)
- Curate initial Pixabay music library (50–80 tracks, tagged, **verified against YouTube Content ID** — upload private test videos and monitor for claims over 48 hours before clearing tracks for rotation)
- Implement Caption Writer — POC uses hook-only (Claude Sonnet, single caption at 0–2s, ≤6 words)
- Implement Final Compositor (FFmpeg assembly: video + music + hook caption + **channel watermark**, 9:16 reframe, target 65s)
- **Implement channel branding overlay** — subtle logo/watermark in corner, baked into FFmpeg composition. Required for TikTok originality compliance (see Section 9).
- Write unit tests: FFmpeg command generation, caption timing, audio levels, output specs
- Write integration test: given a raw simulation video, agent produces final assembled video
- **Deliverable:** Video Editor Agent takes raw simulation + metadata and produces a platform-ready video.

### Phase 3: Orchestration & Review

**Step 9 — LangGraph Pipeline**

- Define LangGraph state machine matching the pipeline diagram (Section 3)
- Wire all three agents into the graph as nodes
- Configure PostgreSQL checkpointing
- Implement all edge conditions (success, retry, escalate, too_complex, bad_concept)
- Implement `agent_runs` logging at every node (model used, tokens, cost, latency, status)
- Write integration test: full end-to-end pipeline with mocked LLMs
- Write integration test: kill and resume from checkpoint
- **Deliverable:** Pipeline runs end-to-end from concept to final video, with state checkpointed at every step.

**Step 10 — Human Review Dashboard**

- Build FastAPI + Jinja2 review page (single page as spec'd in Section 8)
- Inline video player, concept summary, stats, captions, action buttons
- Implement approve/reject actions with reason codes
- Implement feedback text capture
- Wire approved videos into `publish_queue`
- Wire rejections into correct pipeline re-routing (bad_concept → Idea Agent, etc.)
- Send Slack webhook notification when video enters review queue
- **Deliverable:** PM can open a URL, watch a video, approve/reject in one click.

**Step 11 — Monitoring & Observability**

- Set up Langfuse for LLM call tracing (integrates with LiteLLM)
- Create a simple dashboard or alerting for: videos produced per day, success rate, cost per video, queue depth, model latency
- Set alert: 7-day rolling cost average > $0.30/video
- Set alert: success rate drops below 80%
- **Deliverable:** All LLM calls are traced. Cost and success rate are visible.

### Phase 4: Distribution

**Step 12 — Upload & Publishing Service**

- Integrate Upload-Post API (or YouTube Data API v3 as minimum)
- Implement `publish_queue` consumer: Celery beat task checks queue on schedule, publishes next video
- Implement platform-specific metadata generation (titles, descriptions, tags per platform)
- Implement duration padding (ensure ≥62s for TikTok monetisation eligibility)
- Handle upload failures: retry with backoff, alert on repeated failure
- Write integration test: approved video → queued → published (mock API)
- **Deliverable:** Approved videos publish automatically to at least YouTube Shorts and TikTok on schedule.

**Step 13 — Analytics Sync**

- Build a periodic job (daily or weekly) that pulls engagement metrics from platform APIs (YouTube Analytics, TikTok Analytics)
- Write metrics to `publish_log` (views, likes, comments, retention, revenue)
- Update `category_stats` based on real performance data
- **Deliverable:** Engagement data flows back into the database, closing the feedback loop to the Idea Agent.

### Phase 5: Hardening

**Step 14 — Regression Test Suite**

- Create golden set of 10–15 concept briefs with known-good outputs
- Automate weekly regression runs
- Track success rate, iteration count, cost trends over time
- **Deliverable:** Automated regression catches quality degradation.

**Step 15 — Production Burn-In**

- Run the full pipeline for 2 weeks at 2 videos/day
- Monitor all metrics: cost, success rate, review approval rate, queue depth
- Fix issues as they surface
- Collect first 20+ training examples
- Tune prompts based on observed failure modes
- **Deliverable:** Pipeline runs reliably at target throughput. PM is comfortable with review cadence.

**Step 16 — Documentation & Handoff for Pipeline Expansion**

- Document: how to add a new pipeline (sandbox Dockerfile, pipeline_config row, RAG patterns, test directory)
- Document: how to add a new scenario category to an existing pipeline
- Document: prompt templates and what makes them work
- Document: common failure modes and how the system recovers
- **Deliverable:** A new developer can add Pipeline 2 without needing to understand the entire codebase.
