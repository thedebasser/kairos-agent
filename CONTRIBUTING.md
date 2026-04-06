# Contributing to Kairos Agent

Guidelines for developing on and contributing to Kairos Agent.

---

## Development Setup

See the [README quick start](README.md#quick-start) or the detailed [setup guide](docs/setup-guide.md) for full environment setup.

**TL;DR:**

```bash
git clone https://github.com/thedebasser/kairos-agent.git
cd kairos-agent
cp .env.example .env
docker compose up -d postgres
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable. All tests pass. Merge via PR. |
| `refactor/phase-*` | Feature branches for refactor phases. Branch from previous phase. |
| `feature/*` | New features. Branch from `main`. |
| `fix/*` | Bug fixes. Branch from `main`. |

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): short description

- Bullet point details
- Another detail

600 passed (test summary)
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`.

---

## Code Conventions

### Python Style

- **Python 3.12+** — use modern syntax (`type` unions, `match`, f-strings).
- **Type hints everywhere** — all function signatures and return types annotated.
- **Pydantic for data** — all DTOs, configs, and API models are Pydantic `BaseModel`.
- **Async by default** — all I/O-bound functions are `async def`. Use `async_subprocess.py` instead of `subprocess.run`.
- **No framework imports in agents** — agents are plain Python behind ABCs. No LangGraph, FastAPI, or similar.

### Naming

| Entity | Convention | Example |
|--------|-----------|---------|
| Files | `snake_case.py` | `idea_agent.py`, `mix_audio.py` |
| Classes | `PascalCase` | `PhysicsIdeaAgent`, `ValidationResult` |
| Functions | `snake_case` | `generate_concept()`, `run_async()` |
| Constants | `UPPER_SNAKE` | `MAX_SIMULATION_ITERATIONS` |
| Pydantic fields | `snake_case` | `pipeline_run_id`, `total_cost_usd` |

### Imports

Standard library → third-party → local, separated by blank lines:

```python
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from pydantic import BaseModel

from kairos.schemas.contracts import ConceptBrief
from kairos.services.validation import validate_simulation
```

---

## Testing

### Run Tests

```bash
pytest tests/ -q --timeout=60        # Full suite (~600 tests, ~50s)
pytest tests/unit/ -m unit           # Unit tests only
pytest tests/integration/            # Requires Docker
pytest tests/pipelines/              # Pipeline interface tests
pytest tests/quality/                # Video quality checks
pytest tests/golden_set/             # Regression tests
```

### Writing Tests

- **Async tests:** Use `async def` — pytest-asyncio with `mode=auto` handles the rest.
- **Mocking:** Use `unittest.mock.AsyncMock` for async functions, `MagicMock` for sync.
- **Markers:** Apply `@pytest.mark.unit`, `@pytest.mark.integration`, etc. as appropriate.
- **Test location mirrors source:** `src/kairos/services/validation.py` → `tests/unit/services/test_validation_engine.py`.

### Test Structure

```python
class TestValidateSimulation:
    """Tests for the full validation pipeline."""

    async def test_nonexistent_video(self, tmp_path):
        result = await validate_simulation(str(tmp_path / "missing.mp4"))
        assert result.passed is False
        assert result.tier1_passed is False
```

---

## Adding a New Pipeline

1. Create a directory under `src/kairos/pipelines/<name>/` with:
   - `idea_agent.py` — implements `IdeaAgent` ABC
   - `simulation_agent.py` — implements `SimulationAgent` ABC
   - `video_editor_agent.py` — implements `VideoEditorAgent` ABC
   - `models.py` — pipeline-specific Pydantic models

2. Create an adapter in `src/kairos/pipelines/adapters/<name>_adapter.py`:
   ```python
   @register_pipeline("myengine")
   class MyPipelineAdapter(PipelineAdapter):
       pipeline_name = "myengine"
       engine_name = "blender"  # or "pymunk", etc.
       ...
   ```

3. Add prompt templates in `src/kairos/ai/prompts/<name>/`.

4. Add a `ScenarioCategory` enum value in `src/kairos/schemas/contracts.py`.

5. Add a sample concept in `src/kairos/tools/prompt_harness.py`.

6. `pytest tests/pipelines/test_pipeline_interface.py` will auto-discover and verify the new pipeline.

---

## Adding a New Tracing Sink

1. Create a class implementing `TracingSink` in `src/kairos/ai/tracing/sinks/`:
   ```python
   class MySink(TracingSink):
       async def on_event(self, event: TraceEvent) -> None: ...
       async def flush(self) -> None: ...
       async def close(self) -> None: ...
   ```

2. Register it in the tracer initialization (see `RunTracer.init_run()`).

---

## Project Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| Architecture | [docs/architecture.md](docs/architecture.md) | Technical walkthrough |
| ADRs | [docs/adr/](docs/adr/) | Decision records |
| Lessons learned | [docs/lessons-learned.md](docs/lessons-learned.md) | Post-mortem |
| Agent reference | [docs/agent-reference.md](docs/agent-reference.md) | Per-agent details |
| Design doc | [docs/refactor-design-doc.md](docs/refactor-design-doc.md) | Refactor plan |
