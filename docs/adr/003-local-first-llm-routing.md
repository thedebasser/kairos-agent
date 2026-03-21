# ADR-003: Local-First LLM Routing

**Status:** Accepted
**Date:** 2026-02-22

## Context

The pipeline makes 5-15 LLM calls per run across concept generation, simulation config, caption writing, and quality review. At ~$0.01-0.03 per Claude Sonnet call, a fully cloud-based run costs ~$0.15. At 10 runs/day, that's $45/month — significant for a side project.

Local models (Mistral 7B, Llama 3.1 8B via Ollama) run free on the RTX 3090 but produce lower quality output, especially for creative generation tasks.

## Decision

Implement a **local-first with cloud fallback** routing strategy:

1. Each pipeline step declares a `litellm_alias_local` and `litellm_alias_cloud` in `llm_config.yaml`.
2. `call_with_quality_fallback()` tries the local model first with a caller-supplied validator function.
3. If the local output fails validation, the cloud model handles the request.
4. Every successful cloud response is stored as training data for future local model fine-tuning.
5. A global `use_local_llms: false` toggle skips local attempts entirely (for testing, CI, or when local GPU is unavailable).

## Consequences

**Positive:**
- ~85% cost reduction when local models handle routine calls (config adjustment, caption formatting).
- Cloud responses accumulate as training data — the system gets cheaper over time.
- Graceful degradation — if Ollama is down, cloud handles everything transparently.
- The `use_local_llms` toggle makes cloud-only mode trivial for CI/testing.

**Negative:**
- Local-then-cloud adds latency on fallback (~10s local attempt + 5s cloud = 15s vs 5s cloud-only).
- Validator functions must be carefully designed — too strict means always falling back to cloud.
- Two model ecosystems to maintain (Ollama pull + LiteLLM proxy config).

**Metrics observed:**
- Concept generation: ~30% local success rate (creative task, usually falls back to Claude)
- Config adjustment: ~80% local success rate (constrained output, local handles well)
- Caption generation: ~70% local success rate
- Average cost: ~$0.02/run (down from ~$0.15/run cloud-only)
