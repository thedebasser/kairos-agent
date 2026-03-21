# ADR-002: Config-Based Simulation (Not Raw Code Generation)

**Status:** Accepted
**Date:** 2026-02-20

## Context

The Simulation Agent needs to produce runnable Pygame+Pymunk code from concept briefs. The naive approach — having the LLM generate complete Python source code — was the original implementation. It suffered from:

- **High hallucination rate.** LLMs confuse Pymunk APIs, generate invalid physics parameters, import non-existent modules.
- **Unbounded output space.** Any Python code is valid output, making validation nearly impossible before execution.
- **Expensive iteration.** When a simulation fails, the LLM must regenerate the entire program, not just fix one parameter.
- **Non-deterministic.** Same concept → wildly different code structures → inconsistent results.

## Decision

The LLM generates a **JSON config** that matches a per-category Pydantic schema (e.g., `BallPitConfig`, `DestructionConfig`). A **fixed Python template** per category renders the config into runnable code. Instructor enforces the schema via structured output.

```
LLM → JSON Config → Template → Runnable Code
         ↑ (on failure)
    adjust_config()
```

On validation failure, the LLM adjusts the **config** (e.g., change gravity, ball count, colours) — not the code. The search space for fixes is dramatically smaller.

## Consequences

**Positive:**
- Hallucination reduction — the LLM can't call non-existent APIs because it never writes API calls.
- Config adjustment is cheap (~500 tokens vs ~3000 tokens for full code regeneration).
- Deterministic code structure — same template always produces the same code shape.
- Configs are human-readable and diffable. Easy to understand what changed between iterations.
- Schema validation catches invalid configs before execution (no sandbox cost).

**Negative:**
- Expressiveness ceiling — complex simulations may need template changes, not just config changes.
- Template maintenance burden — each new category needs a new template + config schema.
- Config schemas must be carefully designed to balance flexibility with constraint.

**Validation:** After switching from raw code to config-based generation, simulation success rate improved from ~40% to ~75% on first attempt across 50 test runs.
