# Kairos Pipeline: Logging, Observability & Continuous Improvement Spec

> Version: 1.0
> Date: March 13, 2026
> Status: Draft
> Scope: Extend existing pipeline logging to enable run-level debugging, LLM call tracing, and continuous improvement via structured outcome data.

---

## Problem Statement

The current pipeline produces output files per run (`run_summary.json`, per-step JSON files, Blender artifacts) but critical data is missing or hollow:

- `llm_calls` arrays are **empty in every step file** despite LLM calls clearly executing (thinking is captured in metadata but call details are not).
- There is **no token/cost tracking** — `total_cost_usd` is always 0.0 even when cloud models are used.
- There is **no record of the raw prompt or response** for any LLM call — making it impossible to debug why a model produced a bad output.
- There is **no diff between what the LLM returned and what the pipeline actually used** after overrides (e.g., locked domino physics params).
- The `output/` and `runs/` folders are split — final videos live in a separate output directory, disconnected from the run data that produced them.
- There is **no version tracking** for re-renders after review feedback.
- There is **no structured human feedback** mechanism — runs end at `pending_review` with no way to record what the human decided or why.
- There is **no cross-run comparison** — each run is an island with no aggregated view of pipeline performance over time.

The result: you cannot currently trace a bad video back to the LLM call that caused it, cannot measure cost per video, cannot identify which agents or models are underperforming, and cannot build training data for the learning loop.

---

## Goals

1. **Run-level debugging:** Open a single run folder and understand exactly what happened, what each agent produced, what it was thinking, and where things went wrong.
2. **LLM call tracing:** Full provenance from prompt → response → parsed output → applied output for every LLM call.
3. **Cost and token tracking:** Know exactly what each run costs, broken down by agent and model.
4. **Continuous improvement data:** Structured signals (quality gate results, review findings, human feedback) that accumulate into a dataset for identifying weaknesses and improving prompts/models.
5. **Version tracking:** Handle iterative re-renders cleanly when the new video/audio review agents request changes.

---

## Non-Goals

- Production database schema or API design — this is local filesystem POC.
- Real-time dashboards or monitoring.
- Automated retraining pipelines — this spec produces the *data* that would feed those, not the pipelines themselves.

---

## 1. Folder Structure

Consolidate everything for a run into a single directory. Move final outputs into the run folder. Add version tracking for review iterations.

### Current structure (fragmented)
```
runs/{run_id}/
  blender/
    domino_course.blend
    render.mp4
  01_idea_agent.json
  02_simulation_agent.json
  03_video_editor.json
  ...

output/{some_other_uuid}/        ← disconnected from run
  final.mp4
```

### Proposed structure
```
runs/{run_id}/
  run_summary.json               # enriched (see §2)
  run_narrative.md               # human-readable story (see §6)
  human_review.json              # filled post-review (see §8)

  steps/
    01_idea_agent.json
    02_simulation_agent.json
    03_video_editor.json
    04_video_review.json         # NEW — video review agent findings
    05_audio_review.json         # NEW — audio review agent findings

  llm_calls/                     # NEW — raw prompt/response per call
    call_001_concept-developer_prompt.txt
    call_001_concept-developer_response.txt
    call_002_caption-writer_prompt.txt
    call_002_caption-writer_response.txt
    call_003_video-reviewer-default_prompt.txt
    call_003_video-reviewer-default_response.txt

  blender/
    domino_course.blend
    generation_result.json
    validation.json
    render.mp4

  output/                        # final composed videos, versioned
    v1/
      final.mp4
      composition_metadata.json  # FFmpeg command, music track, caption overlay details
    v2/                          # created if re-rendered after review
      final.mp4
      composition_metadata.json
      changes_from_v1.json       # what changed and why
    v3/
      ...
```

### Key changes
- Final video moves inside `runs/{run_id}/output/v{n}/` — no more separate output directory.
- Step files move into `steps/` subdirectory for cleanliness.
- `llm_calls/` subdirectory stores raw prompts and responses (kept separate from step JSON to avoid bloating structured data).
- Versioned `output/v{n}/` directories handle review-triggered re-renders.
- `human_review.json` provides structured feedback that closes the learning loop.

---

## 2. Enriched Run Summary

The current `run_summary.json` has step-level status and duration. Add aggregated metrics that let you scan a run and immediately understand what happened.

### Fields to add

```jsonc
{
  // ... existing fields (pipeline_run_id, pipeline, started_at, etc.) ...

  "totals": {
    "llm_calls_count": 3,
    "total_prompt_tokens": 8420,
    "total_completion_tokens": 1847,
    "total_thinking_tokens": 4200,
    "total_tokens": 14467,
    "total_cost_usd": 0.0089,
    "models_used": ["claude-sonnet-4-6", "ollama/qwen3.5-9b"],
    "local_calls": 1,
    "cloud_calls": 2,
    "escalations": 0,
    "quality_gate_failures": 0
  },

  "output_version": 1,          // which version is the current final
  "video_iterations": 1,        // how many render/review cycles

  "review_verdicts": {           // null if reviews haven't run yet
    "video_review": "pass",
    "audio_review": "pass"
  },

  "music_match": {               // compare requested vs selected mood
    "requested_mood": ["calm", "flowing", "uplifting", "satisfying"],
    "selected_mood": ["upbeat", "energetic", "happy"],
    "overlap_tags": [],
    "match_quality": "poor"      // none/poor/partial/good/exact
  },

  "human_review": null           // populated when human reviews (see §8)
}
```

### Music match rationale
The current run data shows the concept requested `["calm", "flowing", "uplifting", "satisfying"]` but the music selector picked `["upbeat", "energetic", "happy"]` — zero overlap. Logging this structurally means you can track music selection quality before the audio review agent even exists.

---

## 3. LLM Call Logging (Highest-Value Fix)

Every LLM call — including failed attempts and retries — must be logged in the step file's `llm_calls` array. This is the single most impactful change in this spec.

### Schema per call

```jsonc
{
  "call_id": "uuid",
  "call_sequence": 1,                    // 1st, 2nd, 3rd call in this step
  "model_alias": "concept-developer",    // config alias from llm_config.yaml
  "model_resolved": "claude-sonnet-4-6", // actual model string that ran
  "provider": "anthropic",               // anthropic, ollama, litellm
  "call_pattern": "direct",              // direct, quality_fallback
  "routing_outcome": "cloud",            // local, cloud, local_then_cloud

  "tokens": {
    "prompt": 2847,
    "completion": 412,
    "thinking": 1893,                    // null if not applicable
    "cache_read": 0,                     // prompt cache hits
    "cache_write": 2847,                 // prompt cache writes
    "total": 5152
  },

  "latency_ms": 3420,
  "cost_usd": 0.0034,

  "quality_gate": {                      // null if no gate for this call
    "applied": true,
    "passed": true,
    "checks_run": ["contains_pygame", "contains_pymunk", "min_length_500"],
    "failure_reason": null
  },

  "escalation": {                        // null if no escalation occurred
    "escalated_from_model": null,
    "escalated_from_alias": null,
    "escalation_reason": null
  },

  "prompt_file": "llm_calls/call_001_concept-developer_prompt.txt",
  "response_file": "llm_calls/call_001_concept-developer_response.txt",

  "parsed_successfully": true,           // did JSON/structured parse succeed
  "parse_error": null,                   // error message if parse failed

  "thinking_summary": "Generate a domino run concept for cascade archetype with ocean palette."
  // Brief summary of the model's thinking — not the full chain, just enough to scan
}
```

### Raw prompt/response files

Store full prompts and responses as separate text files in `llm_calls/`. The step JSON references them by relative path. This keeps the structured JSON scannable while preserving full provenance.

File naming: `call_{NNN}_{model_alias}_{prompt|response}.txt`

For thinking-enabled models, also store: `call_{NNN}_{model_alias}_thinking.txt`

### What this enables
- Trace any bad output back to the exact prompt that caused it.
- Compare token usage across models to validate cost assumptions.
- Identify which calls escalate most often (learning loop priority targets).
- Build training examples from successful calls (the existing `training_examples` table, but now with full context).

---

## 4. LLM Output Delta Tracking

The domino/marble pipelines strip locked physics parameters from LLM output and overwrite them in code. The physics pipeline's config generation also applies validation rules. Currently there's no record of what the LLM originally returned vs what got used.

### Add to each LLM call that produces structured output

```jsonc
{
  "output_delta": {
    "fields_overridden": ["domino_mass", "spacing_ratio", "domino_friction"],
    "fields_added": [],             // pipeline added fields the LLM didn't produce
    "fields_removed": ["some_hallucinated_field"],
    "override_details": [
      {
        "field": "domino_mass",
        "llm_value": 0.5,
        "applied_value": 0.3,
        "reason": "locked_parameter"
      }
    ]
  }
}
```

### What this enables
- Identify prompt problems: if the LLM keeps trying to change locked params, the prompt needs to be clearer.
- Spot creative drift: track how often the LLM produces values that get overridden, and whether overrides correlate with lower-quality outputs.

---

## 5. Review Agent Findings (New Agents)

The new video and audio review agents need structured output that goes beyond pass/fail.

### Video review findings schema

```jsonc
{
  "step_name": "video_review",
  "reviewer_model": "video-reviewer-default",
  "model_resolved": "ollama/qwen3-vl-8b-instruct",
  "video_reviewed": "output/v1/final.mp4",
  "review_duration_ms": 95000,

  "verdict": "pass",                    // pass, fail, borderline
  "confidence": 0.87,
  "escalated": false,                   // was 30B escalation model used?

  "findings": [
    {
      "check": "chain_completion",
      "result": "pass",
      "confidence": 0.95,
      "detail": "All dominoes appear to fall. Chain propagation looks complete.",
      "timestamp_sec": null
    },
    {
      "check": "camera_framing",
      "result": "minor_issue",
      "confidence": 0.62,
      "detail": "Camera slightly tight on right edge during cascade turn.",
      "timestamp_sec": 32.0
    },
    {
      "check": "caption_visibility",
      "result": "pass",
      "confidence": 0.91,
      "detail": "Hook caption 'One push paints the ocean' visible and on-screen at 0-2.5s.",
      "timestamp_sec": 0.0
    },
    {
      "check": "physics_stability",
      "result": "pass",
      "confidence": 0.93,
      "detail": "No objects flying off screen or clipping through surfaces.",
      "timestamp_sec": null
    },
    {
      "check": "overall_polish",
      "result": "pass",
      "confidence": 0.85,
      "detail": "Video looks polished and professional. Smooth cascade, good colour cycling.",
      "timestamp_sec": null
    }
  ],

  "objective_prescores": {              // null if objective pre-checks disabled
    "dover_technical": null,
    "dover_aesthetic": null,
    "laion_aesthetic_keyframe": null
  }
}
```

### Audio review findings schema

```jsonc
{
  "step_name": "audio_review",
  "reviewer_model": "audio-reviewer-default",
  "model_resolved": "ollama/qwen2.5-omni-7b",
  "audio_reviewed": "output/v1/final.mp4",
  "review_duration_ms": 45000,

  "verdict": "pass",
  "confidence": 0.78,

  "findings": [
    {
      "check": "background_artifacts",
      "result": "pass",
      "confidence": 0.82,
      "detail": "No static, hiss, or background noise detected."
    },
    {
      "check": "unexpected_sounds",
      "result": "pass",
      "confidence": 0.75,
      "detail": "No unexpected sounds. Collision SFX sounds natural."
    },
    {
      "check": "tts_accuracy",
      "result": "pass",
      "confidence": 0.90,
      "detail": "TTS narration matches expected text.",
      "expected_text": "One push paints the ocean",
      "transcribed_text": "One push paints the ocean",
      "wer": 0.0
    },
    {
      "check": "theme_match",
      "result": "minor_issue",
      "confidence": 0.55,
      "detail": "Music mood (upbeat, energetic) doesn't closely match concept brief (calm, flowing). Acceptable but not ideal."
    },
    {
      "check": "loudness",
      "result": "pass",
      "confidence": 1.0,
      "detail": "Integrated loudness within target range.",
      "lufs": -14.8,
      "true_peak_dbtp": -1.2,
      "loudness_range_lu": 6.3
    }
  ],

  "objective_metrics": {
    "ffmpeg_loudness": {
      "integrated_lufs": -14.8,
      "true_peak_dbtp": -1.2,
      "loudness_range_lu": 6.3,
      "target_lufs_min": -16.0,
      "target_lufs_max": -14.0,
      "within_target": true
    },
    "dnsmos": null                      // null if DNSMOS not enabled
  }
}
```

---

## 6. Run Narrative (Human-Readable Summary)

Auto-generate a `run_narrative.md` after each run completes. This is the file you open first when reviewing a run. It stitches together the step data, LLM thinking, and review findings into a readable story.

### Example output

```markdown
# Run 9c2e9623 — 300 Ocean Dominoes Cascade

**Pipeline:** domino | **Duration:** 12m 30s | **Cost:** $0.0089 | **Status:** pending_review

## Step 1: Idea Agent (11.5s)
- **Archetype:** cascade | **Palette:** ocean | **Finale:** tower
- **Concept:** 300 dominoes in ocean colours sweeping in wide rows with U-turns
- **Hook:** "Ocean waves in dominoes!"
- **LLM calls:** 1 (concept-developer → claude-sonnet-4-6, 412 tokens, $0.003)

### Concept Developer Thinking
> Generate a domino run concept for cascade archetype with ocean palette.

## Step 2: Simulation Agent (10m 12s)
- **Blender generate:** domino_course.blend — 300 dominos placed
- **Validation:** 10/10 checks passed
  - Chain propagation: 100%
  - Physics stability: no flyaways
  - All rigid bodies configured
- **Render:** 1950 frames baked → render.mp4 (16MB)
- **LLM calls:** 0 (Blender subprocess only)

## Step 3: Video Editor (2m 5s)
- **Music:** upbeat_electronic_01 (120bpm, electronic)
  - ⚠️ **Mood mismatch:** requested [calm, flowing] → selected [upbeat, energetic] (0 overlap)
- **Hook caption:** "One push paints the ocean" (0-2.5s)
- **LLM calls:** 1 (caption-writer → claude-sonnet-4-6, 189 tokens, $0.001)

### Caption Writer Thinking
> The video features ocean-themed dominoes... I think "One push paints the ocean"
> is strong — it's evocative and mysterious.

## Step 4: Video Review
*Not yet implemented*

## Step 5: Audio Review
*Not yet implemented*

---

## Summary
| Metric | Value |
|--------|-------|
| Total LLM calls | 2 |
| Total tokens | 5,152 |
| Total cost | $0.0089 |
| Models used | claude-sonnet-4-6 |
| Escalations | 0 |
| Music mood match | ⚠️ poor |
| Output version | v1 |

## Human Review
*Pending*
```

### Implementation
Generate this by iterating through step files and `llm_calls/` at the end of a pipeline run. Template-driven — no LLM needed for this.

---

## 7. Version Tracking for Review Iterations

When the video or audio review agent flags an issue that triggers a re-render or re-composition, the pipeline creates a new version.

### Version lifecycle

```
v1: initial render + composition
  → video review: pass
  → audio review: fail (music mood mismatch)
  → action: re-compose with different music selection

v2: re-composed with new music
  → video review: skip (video unchanged)
  → audio review: pass
  → status: pending_human_review
```

### changes_from_v{n}.json schema

```jsonc
{
  "previous_version": "v1",
  "current_version": "v2",
  "trigger": "audio_review_fail",
  "trigger_detail": "Music mood mismatch — requested calm/flowing, got upbeat/energetic",
  "changes": [
    {
      "component": "music",
      "field": "track_id",
      "old_value": "upbeat_electronic_01",
      "new_value": "ambient_ocean_03",
      "reason": "Better mood alignment with concept brief"
    }
  ],
  "unchanged": ["raw_video", "captions", "tts"]
}
```

### Version cap
Set a configurable maximum version count (default: 3). If the pipeline reaches v3 and still fails review, escalate to `requires_manual_intervention` status rather than looping indefinitely.

---

## 8. Human Review Feedback

When you manually review a run, record structured feedback that becomes training data.

### human_review.json schema

```jsonc
{
  "reviewed_at": "2026-03-08T14:30:00Z",
  "reviewed_version": "v1",
  "verdict": "approved_with_notes",     // approved, approved_with_notes, rejected, rejected_with_notes
  "would_publish": true,
  "rating": 7,                          // 1-10 overall quality

  "category_ratings": {                 // optional granular ratings
    "visual_quality": 8,
    "physics_accuracy": 9,
    "camera_work": 7,
    "audio_quality": 5,
    "caption_quality": 8,
    "music_fit": 3,
    "overall_satisfaction": 7
  },

  "issues_spotted": [
    {
      "category": "music_mood_mismatch",
      "severity": "moderate",
      "detail": "Music is upbeat electronic, concept is calm ocean — doesn't fit",
      "agent_caught": false             // did the pipeline's own review catch this?
    }
  ],

  "notes": "Good cascade visually. Music is completely wrong for the ocean mood. Hook caption is strong. Would publish with different music.",

  "comparison_to_automated_review": {
    "video_review_agreed": true,        // did you agree with video review verdict?
    "audio_review_agreed": false,       // did you agree with audio review verdict?
    "false_negatives": ["music_mood_mismatch"],  // issues you caught that review missed
    "false_positives": []                         // issues review flagged that you disagree with
  }
}
```

### What this enables
- Every `agent_caught: false` entry is a labelled training example for improving that review agent's prompts.
- Every `false_positive` tells you the reviewer is being too strict.
- Over time, `comparison_to_automated_review` builds a precision/recall dataset for the review agents.
- The `category_ratings` give you a fine-grained quality signal that can be correlated with pipeline parameters (archetype, palette, model used, etc.).

---

## 9. Cross-Run Index

Maintain a single `runs/index.jsonl` (JSON Lines — one line per run, append-only) that enables cross-run analysis.

### Per-run entry

```jsonc
{
  "run_id": "9c2e9623",
  "pipeline": "domino",
  "archetype": "cascade",
  "palette": "ocean",
  "started_at": "2026-03-08T13:06:26Z",
  "duration_ms": 750557,
  "status": "pending_review",
  "output_version": 1,
  "llm_calls_count": 2,
  "total_tokens": 5152,
  "total_cost_usd": 0.0089,
  "models_used": ["claude-sonnet-4-6"],
  "escalations": 0,
  "video_review_verdict": null,
  "audio_review_verdict": null,
  "music_match_quality": "poor",
  "human_verdict": null,
  "human_rating": null,
  "human_issues": []
}
```

### What this enables
- "What percentage of cascade runs pass first time?"
- "Which palette has the highest human approval rate?"
- "What's my average cost per approved video?"
- "How often does the music selector mismatch mood?"
- "Which model produces the most escalations?"

Use JSONL format (not a JSON array) so entries can be appended without reading/rewriting the whole file.

---

## 10. Implementation Priority

Ordered by value delivered per effort:

| Priority | Item | Effort | Value | Rationale |
|----------|------|--------|-------|-----------|
| **P0** | Populate `llm_calls` array (§3) | Medium | Critical | Currently flying blind on what LLMs actually do |
| **P0** | Consolidate output into runs folder (§1) | Low | High | Eliminates the fragmented output/runs split |
| **P1** | Raw prompt/response files (§3) | Low | High | Enables tracing bad outputs to bad prompts |
| **P1** | Enriched run_summary (§2) | Low | High | Aggregated metrics for quick scanning |
| **P1** | Version tracking (§7) | Medium | High | Required for review agent iteration loop |
| **P2** | Review agent findings schema (§5) | Medium | High | Structured output for new review agents |
| **P2** | Run narrative (§6) | Low | Medium | QOL — the file you actually open first |
| **P2** | Music match detection (§2) | Low | Medium | Catches an obvious current issue |
| **P2** | Cross-run index (§9) | Low | Medium | Enables trend analysis across runs |
| **P3** | LLM output delta tracking (§4) | Medium | Medium | Identifies prompt problems over time |
| **P3** | Human review feedback (§8) | Low | High (long-term) | Training data for learning loop |

P0 items should ship with the review agents. P1 items should follow immediately. P2/P3 can be added incrementally.

---

## Open Questions

1. **Token counting for Ollama:** Does the current Ollama setup return token counts in the response? If not, the `tokens` field in `llm_calls` will need to be estimated or left null for local models until a counting mechanism is added.

2. ~~**Cost calculation for local models:** Should local model calls report $0.00 cost (since they're free) or should we estimate an equivalent cost based on electricity/time for comparison purposes?~~ **DECIDED:** All local models report $0.00. Cloud models use `litellm.completion_cost()` with per-model fallback pricing from capabilities layer.

3. **Review agent iteration cap:** The spec proposes a max of 3 versions before escalating to manual intervention. Is this the right cap, or should it be configurable per pipeline?

4. **Narrative generation timing:** Should `run_narrative.md` be generated at the end of every run automatically, or only on demand? Recommendation: auto-generate — it's cheap (string formatting only, no LLM) and always useful.

5. **Index file management:** JSONL files grow indefinitely. Should we rotate (e.g., monthly files) or keep a single file? For POC with <1000 runs, a single file is fine.

---

## Implementation Decisions Log

Decisions made during implementation, recorded for traceability.

### D1: Thinking Extraction Strategy
**Decision:** Manual extraction with strategy pattern — one `ModelCapabilities` ABC that branches per model family. No scattered if-statements.
**Rationale:** User requested "one interface that branches down the correct path depending on your LLM model." Strategy pattern via `model_capabilities.py` with regex-based registry.
**File:** `src/kairos/services/model_capabilities.py`

### D2: Thinking Enabled for All Capable Models
**Decision:** Thinking is enabled for all models that support it (Anthropic Claude, Qwen3).
**Rationale:** User preference — "quality is more important than time."
**Impact:** Anthropic uses `extended_thinking` param; Qwen3 thinks by default (parse `<think>` tags from response).

### D3: LLM Call Buffer Pattern
**Decision:** Receipt box pattern — `_record_llm_call()` accumulates call records into a module-level buffer. `collect_llm_calls()` drains the buffer at step boundaries (called by graph.py nodes before `save_step()`).
**Rationale:** Mirrors the existing `collect_thinking()` pattern. Avoids threading call data through return values.
**File:** `src/kairos/services/llm_routing.py`

### D4: Model Pricing by Model Type
**Decision:** `ModelType.LOCAL` → $0.00 always. `ModelType.CLOUD` → `litellm.completion_cost()` with per-family fallback from `get_pricing()`.
**Rationale:** User confirmed "All locals will be $0.00" and "include the cost for cloud models."
**File:** `src/kairos/services/model_capabilities.py` (pricing), `llm_routing.py` (`_extract_usage`)

### D5: Folder Structure Migration
**Decision:** Apply new structure (`steps/`, `llm_calls/`, `output/v{n}/`) going forward. Old flat structure remains until user confirms migration.
**Rationale:** User preferred incremental migration — "I will remove the old structure once I've confirmed the new structure is working."
**File:** `src/kairos/services/step_artifacts.py`

### D6: Version Tracking
**Decision:** `output_version` integer field in `PipelineGraphState` and `PipelineState`. Incremented each time `video_editor_node` runs. Versioned output dirs created via `RunArtifacts.get_output_version_dir(version)`.
**Rationale:** Required for review agent iteration loop (§7). Version bumps on re-render allow before/after comparison.
**Files:** `src/kairos/pipeline/graph.py`, `src/kairos/models/contracts.py`, `src/kairos/services/step_artifacts.py`
