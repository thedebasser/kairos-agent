# ADR-008: Pipeline Adapter Pattern

**Status:** Accepted
**Date:** 2026-02-25

## Context

The system needs to support multiple simulation pipelines (physics, domino, marble, potentially more in the future). All pipelines use Blender as the simulation engine but have different:
- Agent implementations (different prompts, different config schemas)
- Execution environments (Docker sandbox vs Blender subprocess)
- Output characteristics (render settings, camera angles, post-processing)
- Category support (ball_pit/destruction vs domino_chain vs marble_funnel)

We needed a way to add new pipelines without modifying the orchestrator or shared services.

## Decision

Implement a **pipeline adapter pattern** with:

1. **ABCs** (`contracts.py`) — Define the interface every pipeline must implement: `PipelineAdapter` (factory) + 5 agent ABCs.
2. **Registry** (`registry.py`) — `@register_pipeline("physics")` decorator + auto-discovery via `pkgutil.walk_packages`.
3. **Per-pipeline directories** — Each pipeline has its own agents, models, templates, and config schemas.

```python
@register_pipeline("physics")
class PhysicsPipelineAdapter(PipelineAdapter):
    pipeline_name = "physics"
    engine_name = "blender"
    categories = [ScenarioCategory.BALL_PIT, ScenarioCategory.DESTRUCTION]

    def get_idea_agent(self) -> IdeaAgent:
        return PhysicsIdeaAgent()

    def get_simulation_agent(self) -> SimulationAgent:
        return PhysicsSimulationAgent()
    # ...
```

The orchestrator resolves the adapter at runtime:
```python
adapter = get_pipeline(state["pipeline"])  # "physics" → PhysicsPipelineAdapter
agent = adapter.get_idea_agent()           # → PhysicsIdeaAgent
```

## Consequences

**Positive:**
- **Zero orchestrator changes** to add a new pipeline. Create adapter → implement agents → register.
- **Auto-discovery test** (`test_pipeline_interface.py`) verifies all registered pipelines implement every required method.
- **Runtime flexibility** — `pipeline run --pipeline domino` works with no code changes.
- **Shared services stay shared** — validation, FFmpeg, music selection work for all pipelines.

**Negative:**
- Each pipeline is a significant amount of code (3-5 agent implementations + models + templates).
- Some agents share logic across pipelines (video editor, review) but the adapter pattern doesn't naturally support sharing. Solved by shared base classes in `ai/review/`.
- The registry is global mutable state. Mitigated by `clear_registry()` for testing.

**Current pipelines:**
| Pipeline | Engine | Status |
|----------|--------|--------|
| physics | Blender | Active, primary |
| domino | Blender | Active |
| marble | Blender | Under redesign |
