# Lessons Learned

Honest post-mortem on building Kairos Agent. What worked, what didn't, and what I'd change if starting over.

---

## What Worked Well

### 1. Config-Based Simulation Was the Single Best Decision

Early versions had the LLM generate complete Pygame+Pymunk code. The success rate was ~40% on first attempt. Switching to JSON configs rendered by fixed templates brought it to ~75%.

The key insight: **constrain the LLM's output space.** A JSON config with 15 numeric fields is a tiny target compared to 200 lines of Python. The LLM is still doing useful creative work (choosing colours, physics parameters, object counts), but it can't hallucinate API calls or import non-existent modules.

An unexpected benefit: config adjustment on failure is ~6× cheaper than full code regeneration (500 tokens vs 3000 tokens), because you're asking "change gravity from 900 to 600" instead of "rewrite the whole program."

### 2. Agent–Orchestrator Separation Paid Dividends Immediately

Making agents plain Python classes behind ABCs seemed like over-engineering at first. It proved its value in three ways:
- **Testing speed.** Agent unit tests run in <1s each because there's no graph setup, no state machine, no checkpointer.
- **Debugging clarity.** When a simulation fails, you can call `agent.generate_simulation(concept)` in a REPL without any orchestration context.
- **Framework insulation.** LangGraph went through breaking API changes between v0.2 and v0.3. Agent code was untouched — only the graph wrappers needed updating.

### 3. Structured Output via Instructor Eliminated Parsing Bugs

Using Instructor with Pydantic response models means the LLM's output is type-checked before our code ever sees it. Zero JSON parsing bugs in production. Zero "the LLM returned a string instead of a dict" errors.

The one caveat: Instructor's retry mechanism (`max_retries`) is aggressive — it silently re-prompts if validation fails. We set `max_retries=1` everywhere to fail fast and let our own retry logic handle it at the graph level.

### 4. The Tracing System Made Debugging Possible

Before the refactor, debugging a failed run meant `grep`-ing through disconnected log files. After implementing the unified tracing system (Phase 2), every run produces a self-contained `runs/<id>/` directory with:
- A machine-readable event timeline
- A human-readable console feed
- Every prompt sent and response received
- Every decision the agent made

This turned 30-minute debugging sessions into 2-minute "read the timeline" sessions.

---

## What Didn't Work

### 1. Raw Code Generation (Abandoned)

The original approach of having the LLM write complete simulation programs was a dead end for production use. Specific failure modes:
- **API confusion.** The LLM mixed up Pymunk and Pygame APIs, created non-existent methods, used deprecated parameters.
- **Resource leaks.** Generated code often forgot to quit Pygame or close file handles, causing Docker sandbox to hang.
- **Resolution bugs.** Despite explicit prompting, the LLM frequently generated landscape (1920×1080) instead of portrait (1080×1920) simulations.
- **Inconsistent structure.** Every generated program was structured differently, making validation heuristics brittle.

Config-based generation solved all of these. See [ADR-002](adr/002-config-based-simulation.md).

### 2. Vision Model Availability (moondream2)

We designed Tier 2 validation around the moondream2 vision model for frame inspection. It was removed from Ollama's model registry in February 2026 with no warning. The feature is implemented and tested, but can't run in production.

**Lesson:** Don't build critical features on a single model without a fallback. We now use Qwen3-VL for video review (a related but different feature), but frame-level inspection quality isn't the same.

### 3. Three Logging Systems Before the Refactor

The pre-refactor codebase had logging going to `logs/` text files, `runs/steps/` JSON files, and the database — with no cross-referencing. This seems obviously wrong in hindsight, but it happened gradually:
- Text logs came first (Python's logging module, easy default)
- Step JSON files were added for run replay
- Database logging was added for the dashboard

Each was individually reasonable. The problem was never unifying them. The Phase 2 tracing refactor (pluggable sinks) was the fix.

### 4. Premature Dependencies

The original `pyproject.toml` included Redis, Celery, ChromaDB, and llama-index — all intended for features that were never built (task queue, vector store, RAG). They added ~500MB to the install, caused C++ build tool requirements, and created confusing health check failures.

Phase 4 removed all of them. **Lesson:** Don't add dependencies for planned features. Add them when the feature is implemented.

---

## What I'd Change

### 1. Start with Config-Based Simulation from Day One

Spent ~3 weeks on raw code generation before discovering the config-based approach. If I'd recognized earlier that "constrain the output space" is the meta-strategy for AI reliability, I'd have saved significant time and frustration.

### 2. Use Alembic from the Start (Not Raw SQL Migrations)

The first 5 migrations were raw SQL files. They work, but they don't support downgrade, don't track which migrations have been applied, and can't be generated automatically from model changes. Alembic was added in Phase 4. Should have been day one.

### 3. Design for Multiple Pipelines Earlier

The physics pipeline was built monolithically. When domino and marble pipelines were added, significant refactoring was needed to extract common logic. The adapter pattern (Phase 1) should have been the starting architecture. The effort to retrofit was considerable.

### 4. Build the Eval Harness Before the Agents

The eval harness was built after agents were already working. This meant weeks of "run it and see" debugging. A proper eval suite with curated test cases would have provided faster feedback during development.

This is a common omission in AI projects — everyone builds the feature first and the eval second. Building eval first forces you to define "what does success look like" before writing code.

---

## Cost Observations

After ~200 pipeline runs:

| Metric | Value |
|--------|-------|
| Average cost per successful run (local-first) | $0.02 |
| Average cost per successful run (cloud-only) | $0.15 |
| Most expensive single run | $0.41 (5 simulation retries + cloud fallback on all) |
| Cheapest successful run | $0.003 (full cache hit except concept generation) |
| Local model success rate (concept) | ~30% |
| Local model success rate (config) | ~80% |
| Monthly spend at 10 runs/day | ~$6 (local-first) |

The learning loop is starting to show effects — local model success rates have improved ~10% over the last month as more training examples accumulate.

---

## Technical Debt Remaining

1. **Marble pipeline** needs ramp geometry redesign and Blender camera placement fixes.
2. **Publishing adapters** (TikTok, YouTube Shorts, Instagram Reels) are data-modelled but not implemented.
3. **The 2 flaky test_graph.py tests** depend on mock state ordering. Known, low priority, not fixing until the next graph refactor.
4. **No CI/CD pipeline.** Tests run locally. GitHub Actions or equivalent would provide confidence on PR merges.
5. **Windows-only testing.** Docker sandbox works via WSL2, but macOS/Linux paths aren't exercised.
