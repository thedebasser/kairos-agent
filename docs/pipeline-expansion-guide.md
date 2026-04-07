# Kairos Agent — Pipeline Expansion Guide

This document covers how to extend the Kairos Agent platform with new pipelines, categories, and customizations.

## Table of Contents

1. [Adding a New Pipeline](#adding-a-new-pipeline)
2. [Adding a New Scenario Category](#adding-a-new-scenario-category)
3. [Prompt Templates](#prompt-templates)
4. [Common Failure Modes & Recovery](#common-failure-modes--recovery)
5. [Architecture Overview](#architecture-overview)

---

## Adding a New Pipeline

A "pipeline" in Kairos is a complete content production workflow — from idea generation through video editing. The default pipeline is `physics` (Blender simulations). To add a new pipeline (e.g., `3d_renders`, `data_viz`):

### 1. Create the Sandbox Dockerfile

```
sandbox/<pipeline_name>/Dockerfile
```

The sandbox is where simulation code runs in isolation. Base it on the existing physics sandbox:

```dockerfile
FROM python:3.13-slim
RUN pip install --no-cache-dir <your-engine-packages>
COPY entrypoint.sh /entrypoint.sh
WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
```

### 2. Create Agent Implementations

Create a new directory under `src/kairos/pipelines/<pipeline_name>/` with:

```
src/kairos/pipelines/<pipeline_name>/
├── __init__.py
├── adapter.py         # Factory functions returning your agents
├── idea_agent.py      # Implements BaseIdeaAgent
├── simulation_agent.py # Implements BaseSimulationAgent
└── video_editor_agent.py  # Implements BaseVideoEditorAgent
```

Each agent must implement the abstract base classes in `src/kairos/agents/base.py`:

```python
from kairos.agents.base import BaseIdeaAgent, BaseSimulationAgent, BaseVideoEditorAgent

class MyIdeaAgent(BaseIdeaAgent):
    async def generate_concept(self, ...) -> ConceptBrief:
        ...

class MySimulationAgent(BaseSimulationAgent):
    async def generate_simulation(self, concept: ConceptBrief) -> SimulationResult:
        ...

class MyVideoEditorAgent(BaseVideoEditorAgent):
    async def compose_video(self, ...) -> VideoOutput:
        ...
```

### 3. Create the Adapter

The adapter in `adapter.py` provides factory functions:

```python
def get_idea_agent() -> MyIdeaAgent:
    return MyIdeaAgent()

def get_simulation_agent() -> MySimulationAgent:
    return MySimulationAgent()

def get_video_editor_agent() -> MyVideoEditorAgent:
    return MyVideoEditorAgent()
```

### 4. Register the Pipeline

Use the `@register_pipeline` decorator from `src/kairos/pipeline/registry.py`:

```python
from kairos.pipeline.registry import register_pipeline

@register_pipeline("my_pipeline")
def create_my_pipeline():
    return {
        "idea_agent": get_idea_agent(),
        "simulation_agent": get_simulation_agent(),
        "video_editor_agent": get_video_editor_agent(),
    }
```

### 5. Add Database Configuration

Insert a row in `pipeline_config`:

```sql
INSERT INTO pipeline_config (pipeline, engine, categories)
VALUES ('my_pipeline', 'my_engine', '["cat_1", "cat_2", "cat_3"]');
```

### 6. Create Tests

```
tests/unit/pipelines/<pipeline_name>/
├── __init__.py
├── test_idea_agent.py
├── test_simulation_agent.py
└── test_video_editor_agent.py
```

### 7. Add Golden Set Briefs

Create concept briefs for regression testing in `tests/golden_set/`:

```json
[
  {
    "concept_id": "...",
    "pipeline": "my_pipeline",
    "category": "cat_1",
    "title": "...",
    ...
  }
]
```

---

## Adding a New Scenario Category

Categories are defined in the `ScenarioCategory` enum in `src/kairos/models/contracts.py`:

### 1. Add the Enum Value

```python
class ScenarioCategory(str, Enum):
    GRAVITY_CHAOS = "gravity_chaos"
    DOMINO_CASCADE = "domino_cascade"
    # Add your new category:
    MY_NEW_CATEGORY = "my_new_category"
```

### 2. Update Category Rotation Config

The category rotation system in `src/kairos/services/category_rotation.py` uses these categories. Add any category-specific weights or constraints if needed.

### 3. Add Prompt Knowledge

If your category has specific simulation patterns, add them to the RAG knowledge base under `rag/patterns/<category>/`. The Idea Agent will use this context when generating concepts.

### 4. Update the Database

Insert a `category_stats` row:

```sql
INSERT INTO category_stats (pipeline, category, total_count, streak_count)
VALUES ('physics', 'my_new_category', 0, 0);
```

### 5. Add Golden Set Briefs

Add 1-2 concept briefs for the new category to `tests/golden_set/concept_briefs.json`.

---

## Prompt Templates

### What Makes Them Work

The prompt templates in Kairos follow these principles:

1. **Structured Output**: Every prompt targets a Pydantic model via Instructor. The LLM must return valid JSON matching the schema. This eliminates parsing failures.

2. **Context Injection**: Prompts include:
   - Recent concepts (for novelty checking)
   - Category rotation state (what was recently used)
   - Previous simulation failures (for debug iterations)
   - Validation results (specific failure reasons)

3. **Few-Shot Examples**: Where possible, include 1-2 examples of good output in the system prompt.

4. **Constraint Specification**: Explicitly state constraints:
   - Maximum word counts
   - Required fields
   - Value ranges
   - Forbidden patterns

### Key Prompt Locations

| Agent | Prompt Location | Model |
|-------|----------------|-------|
| Idea Agent (concept) | `PhysicsIdeaAgent.generate_concept()` | `concept-developer` (Claude Sonnet) |
| Idea Agent (local) | `PhysicsIdeaAgent.generate_concept()` | `idea-agent-local` (Mistral 7B) |
| Simulation (first pass) | `PhysicsSimulationAgent.generate_code()` | `simulation-first-pass` (Claude Sonnet) |
| Simulation (debug) | `PhysicsSimulationAgent.debug_code()` | `simulation-debugger` (Claude Sonnet) |
| Video Editor (captions) | `PhysicsVideoEditorAgent.generate_captions()` | `caption-writer` (Claude Sonnet) |
| Video Editor (title) | `PhysicsVideoEditorAgent.generate_title()` | `title-writer` (Llama 3.1 8B) |

### Tuning Tips

- **If concepts are too similar**: Increase the novelty context window (pass more recent concepts)
- **If simulation code fails**: Add more constraints to the system prompt (banned functions, required imports)
- **If captions are too long**: Reduce max_length in the Pydantic model and add explicit word count to the prompt
- **If titles are generic**: Add category-specific title patterns to the prompt

---

## Common Failure Modes & Recovery

### 1. Simulation Code Doesn't Compile

**Cause**: LLM generates invalid Python (missing imports, syntax errors).

**Recovery**: The Simulation Agent automatically retries with the `simulation-debugger` model, passing the error traceback as context. Up to `MAX_SIMULATION_ITERATIONS` (5) retries.

**Prevention**: The system prompt includes a list of allowed imports and a code template.

### 2. Simulation Runs But Produces Bad Output

**Cause**: Animation is too short, static, or visually broken.

**Recovery**: The Validation Engine checks:
- Duration (must be ≥ `target_duration_sec`)
- Frame rate (must be ≥ 24fps)
- Motion detection (frames must change)
- Object count (minimum objects must appear)

Failed validations trigger re-generation with the specific failure reason.

### 3. Video Duration Too Short for TikTok

**Cause**: Simulation produces < 62 seconds of video.

**Recovery**: The `needs_duration_padding()` function in `src/kairos/services/uploader.py` detects this. The Video Editor Agent can loop the end of the simulation or add an outro to meet the 62s minimum.

### 4. LLM Quality Fallback

**Cause**: Local model (Mistral/Llama) produces low-quality output.

**Recovery**: The `call_with_quality_fallback()` function in `src/kairos/services/llm_routing.py` automatically falls back to the cloud model (Claude Sonnet) and records the successful output for future fine-tuning.

### 5. Publish Upload Failures

**Cause**: Platform API rate limits, authentication expiry, network issues.

**Recovery**: Exponential backoff retry (2s, 4s, 8s). After `MAX_RETRY_ATTEMPTS` (3), a Slack alert is sent. The queue item remains in `failed` status for manual retry.

### 6. Cost Threshold Exceeded

**Cause**: Too many cloud model fallbacks or excessive retries.

**Recovery**: The `AlertManager` in `src/kairos/services/monitoring.py` checks the 7-day rolling cost average. When it exceeds `$0.30/video`, a warning alert fires. Investigate the cloud fallback ratio and tune local model prompts.

### 7. Pipeline Checkpoint Failure

**Cause**: Process crash during pipeline execution.

**Recovery**: LangGraph checkpointing (PostgreSQL-backed) saves state after each node. Use `resume_pipeline()` to restart from the last checkpoint. The `review_action` state persists across restarts.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    CLI / Schedule                     │
│                  (src/kairos/cli.py)                  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              LangGraph Pipeline                      │
│           (src/kairos/pipeline/graph.py)             │
│                                                      │
│  idea_node → simulation_node → video_editor_node    │
│       ↓            ↓               ↓                │
│  human_review_node → publish_node                   │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼──────┐ ┌─────▼──────┐ ┌────▼────────┐
│  Idea Agent  │ │ Simulation │ │ Video Editor│
│              │ │   Agent    │ │    Agent    │
└───────┬──────┘ └─────┬──────┘ └────┬────────┘
        │              │              │
┌───────▼──────────────▼──────────────▼────────┐
│                Services Layer                 │
│  • LLM Routing (LiteLLM + Instructor)        │
│  • Validation Engine                          │
│  • Category Rotation                          │
│  • Sandbox Execution (Docker)                 │
│  • FFmpeg Compositor                          │
│  • Monitoring (Langfuse)                      │
│  • Publishing (YouTube, TikTok)               │
│  • Analytics Sync                             │
└───────┬──────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────┐
│              Database Layer                    │
│  • PostgreSQL (pipeline_runs, outputs, etc.)  │
│  • Redis (caching, queue)                     │
│  • ChromaDB (RAG knowledge base)              │
└──────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Pipeline Registry Pattern**: Pipelines register via decorator, enabling runtime selection and future multi-pipeline support.

2. **TypedDict State Graph**: LangGraph uses `PipelineGraphState(TypedDict)` instead of bare `dict` — critical for proper state merging across nodes.

3. **Agent ABCs**: All agents implement abstract base classes, making it straightforward to swap implementations for different content types.

4. **LLM Routing with Fallback**: Local models (Ollama) are tried first, with automatic cloud fallback. Successful cloud outputs are logged for fine-tuning.

5. **Human-in-the-Loop**: LangGraph's interrupt mechanism pauses the pipeline at `human_review_node`. The FastAPI dashboard provides the review UI.

6. **Checkpoint Recovery**: PostgreSQL-backed checkpointing survives process crashes. `resume_pipeline()` continues from the last completed node.
