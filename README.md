# Kairos Agent

Automated simulation content pipeline for generating short-form videos from programmatic physics simulations.

## Overview

Kairos Agent uses LLM-powered agents orchestrated by LangGraph to:
1. **Generate concepts** — Idea Agent selects categories (with rotation rules) and creates visual briefs
2. **Build simulations** — Simulation Agent generates Pygame+Pymunk code, executes in Docker sandbox, validates output
3. **Assemble videos** — Video Editor Agent adds captions, music, and produces final 9:16 portrait videos
4. **Human review** — FastAPI dashboard for approve/reject workflow
5. **Publish** — Queue-based publishing to multiple platforms

## Quick Start

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- NVIDIA GPU with CUDA (for Ollama local models)
- FFmpeg

### Setup

```bash
# Clone and enter
cd kairos-agent

# Copy environment file
cp .env.example .env
# Edit .env with your API keys

# Start infrastructure
docker compose up -d

# Create Python environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -e ".[dev]"

# Run database migration
psql -h localhost -U kairos -d kairos -f migrations/001_initial_schema.sql

# Pull Ollama models
ollama pull mistral:7b-instruct-q4_0
ollama pull llama3.1:8b
ollama pull moondream2

# Build sandbox image
docker build -t kairos-sandbox sandbox/

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/unit/ -m unit
```

### Run Pipeline

```bash
pipeline run --pipeline physics
pipeline resume <run-id>
pipeline status
```

## Architecture

```
LangGraph State Machine
├── Idea Agent (concept generation + category rotation)
├── Simulation Agent (code gen → sandbox exec → validate → iterate)
├── Video Editor Agent (captions + music + FFmpeg assembly)
├── Human Review (FastAPI dashboard, LangGraph interrupt)
└── Publish Queue (Celery + platform adapters)
```

## Project Structure

```
src/kairos/
├── agents/          # LLM agent implementations
├── pipeline/        # Pipeline registry & orchestration
├── pipelines/       # Pipeline adapters (physics, future: chemistry, etc.)
├── services/        # Business logic (validation, sandbox, captions, etc.)
├── db/              # Database models & operations
├── models/          # Pydantic data contracts
├── config.py        # Settings (from .env)
├── exceptions.py    # Exception hierarchy
└── cli.py           # CLI entry point
```

## Testing

```bash
pytest tests/unit/ -m unit              # Fast unit tests
pytest tests/integration/ -m integration # Requires Docker services
pytest tests/pipelines/ -m pipeline      # Pipeline interface tests
pytest tests/quality/ -m quality         # Video quality checks
```

## Implementation Plan

This project follows a 16-step implementation plan across 5 phases. See [docs/implementation-document.md](docs/implementation-document.md) for full spec.

### Phase 1: Foundation
| Step | Description | Status |
|------|-------------|--------|
| 1 | Repository, Environment & Quality Gates | ✅ Complete |
| 2 | Database Schema & Migrations | ✅ Complete |
| 3 | LiteLLM Proxy & Instructor Setup | ✅ Complete |
| 4 | Simulation Sandbox | ✅ Complete |

### Phase 2: Agents (Individual)
| Step | Description | Status |
|------|-------------|--------|
| 5 | Validation Engine | ✅ Complete |
| 6 | Simulation Agent | ✅ Complete |
| 7 | Idea Agent | ✅ Complete |
| 8 | Video Editor Agent | ✅ Complete |

### Phase 3: Orchestration & Review
| Step | Description | Status |
|------|-------------|--------|
| 9 | LangGraph Pipeline | ⬜ Not started |
| 10 | Human Review Dashboard | ⬜ Not started |
| 11 | Monitoring & Observability | ⬜ Not started |

### Phase 4: Distribution
| Step | Description | Status |
|------|-------------|--------|
| 12 | Upload & Publishing Service | ⬜ Not started |
| 13 | Analytics Sync | ⬜ Not started |

### Phase 5: Hardening
| Step | Description | Status |
|------|-------------|--------|
| 14 | Regression Test Suite | ⬜ Not started |
| 15 | Production Burn-In | ⬜ Not started |
| 16 | Documentation & Handoff | ⬜ Not started |

## License

Private — All rights reserved.
