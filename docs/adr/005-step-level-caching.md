# ADR-005: Step-Level Input-Hash Caching

**Status:** Accepted
**Date:** 2026-03-01

## Context

Pipeline runs frequently fail partway through — a simulation might succeed but the video editor fails, or human review rejects and the pipeline restarts from an earlier step. Without caching, every restart re-executes all steps from the beginning, wasting LLM tokens and sandbox compute.

LangGraph's built-in checkpointing saves graph state at each node, enabling resume after crash. But it doesn't help with intentional re-runs (new pipeline run with same concept) or partial retries (video review rejection → re-simulate but keep the same concept).

## Decision

Each graph node computes an `_state_input_hash` from its relevant input fields before execution. The hash is checked against a key-value cache (`ai/llm/cache.py`). If a match exists, the cached result is returned without executing the agent.

```python
async def simulation_node(state: PipelineGraphState) -> dict:
    input_hash = _state_input_hash(state, ["concept", "simulation_iteration", "pipeline"])
    cached = get_cache().get(input_hash)
    if cached:
        logger.info("Cache hit for simulation (hash=%s)", input_hash[:8])
        return cached
    # ... execute agent ...
    get_cache().set(input_hash, result)
    return result
```

Cache keys include:
- **Idea node:** `pipeline` name
- **Simulation node:** `concept` + `simulation_iteration` + `pipeline`
- **Video editor node:** `raw_video_path` + `concept` + `pipeline`
- **Review nodes:** `final_video_path` + relevant review inputs

## Consequences

**Positive:**
- Reruns with identical inputs are free (zero LLM cost, zero compute).
- Failed runs can be resumed cheaply — only the failed step and onwards re-execute.
- Content-addressed keys mean cache hits are correct by construction (same input → same output).
- Cache is transparent to agents — they don't know about it.

**Negative:**
- Stale cache risk — if a template or prompt changes, old cached results may be invalid. Mitigated by including relevant config in the hash.
- Memory usage — cache stores full step results. Bounded by `max_cache_size_mb` (default 2GB).
- Cache invalidation is manual — no automatic TTL. Cleared on code deployments.

**Observation:** In a typical development session with 5-10 pipeline runs, cache hit rate is ~60%, saving approximately $0.30-0.50 in LLM costs.
