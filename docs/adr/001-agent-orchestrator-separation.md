# ADR-001: Agent–Orchestrator Separation

**Status:** Accepted
**Date:** 2026-02-15

## Context

The pipeline requires multiple AI agents (Idea, Simulation, Video Editor, Review) to collaborate in a complex workflow with retries, branching, and human intervention. LangGraph is the natural choice for orchestrating this workflow, but we needed to decide how tightly agents should couple to the framework.

Two options:
1. **Agents as LangGraph nodes** — agents inherit from LangGraph abstractions, access state directly, and are defined inline in the graph.
2. **Agents as plain Python** — agents are regular classes behind abstract interfaces. The graph calls them but they have no knowledge of LangGraph.

## Decision

Agents are plain Python classes implementing ABCs defined in `pipelines/contracts.py`. They accept narrow DTOs and return Pydantic models. The LangGraph graph nodes are thin wrappers that extract inputs from state, call the agent, and write results back to state.

```python
# Agent (no LangGraph imports)
class PhysicsIdeaAgent(IdeaAgent):
    async def generate_concept(self, input: IdeaAgentInput) -> ConceptBrief: ...

# Graph node (thin wrapper)
async def idea_node(state: PipelineGraphState) -> dict:
    agent = get_pipeline(state["pipeline"]).get_idea_agent()
    concept = await agent.generate_concept(IdeaAgentInput(pipeline=state["pipeline"]))
    return {"concept": concept.model_dump(), ...}
```

## Consequences

**Positive:**
- Agents are fully testable in isolation — no graph, no state machine, just call the method.
- Agents are framework-portable. If LangGraph is superseded, only the orchestration layer changes.
- Type safety is enforced by ABCs + mypy. Can't forget to implement a method.
- Agent code is readable by anyone — no framework magic to learn first.

**Negative:**
- Thin wrapper nodes are boilerplate (7 nodes × ~30 lines each).
- State serialization/deserialization happens manually in each node.
- Agents can't access other agents' state (by design, but occasionally inconvenient).

**Risk mitigated:** LangGraph is a fast-moving project (v0.3 at time of writing). Our agents will survive API changes because they don't import from LangGraph.
