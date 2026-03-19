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

---

## New Machine Setup Guide (Windows)

Step-by-step instructions for getting Kairos Agent running from scratch on a fresh Windows machine. Last verified: 2026-02-28.

### Prerequisites

Install these before anything else:

| Tool | Min Version | Install |
|------|-------------|---------|
| **Python** | 3.12+ | https://www.python.org/downloads/ |
| **Docker Desktop** | 4.x+ | https://www.docker.com/products/docker-desktop/ |
| **Git** | 2.x+ | https://git-scm.com/downloads |
| **FFmpeg** | 6.x+ | `winget install Gyan.FFmpeg` (restart terminal after) |
| **NVIDIA GPU + CUDA** | — | Required for local LLMs via Ollama |
| **Ollama** | latest | https://ollama.com/download or run via Docker |

**Microsoft C++ Build Tools** are also required for `chromadb` (the `chroma-hnswlib` package compiles from source on Python 3.13+):

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" --accept-package-agreements --accept-source-agreements
```

> **Note:** If you already have Visual Studio 2022 Community/Pro, ensure the "Desktop development with C++" workload is installed via the Visual Studio Installer.

Verify prerequisites:

```powershell
python --version      # 3.12+
docker --version      # 20+
git --version         # 2+
ffmpeg -version       # 6+
nvidia-smi            # Should show your GPU
```

### Step 1: Clone & Enter

```powershell
git clone <your-repo-url> kairos-agent
cd kairos-agent
```

### Step 2: Configure Environment

Copy or create `.env` from `.env.example`. Key values:

```env
ANTHROPIC_API_KEY="sk-ant-..."
DATABASE_URL="postgresql+asyncpg://kairos:changeme@localhost:5434/kairos"
DATABASE_URL_SYNC="postgresql://kairos:changeme@localhost:5434/kairos"
POSTGRES_PORT="5434"
OLLAMA_BASE_URL="http://localhost:11434"
REDIS_URL="redis://localhost:6379/0"
```

> **Port conflict note:** The Docker Postgres runs on port **5434** to avoid conflicts with any locally installed PostgreSQL (which commonly uses 5432 or 5433). If you have no local Postgres, you can change this back to 5433 in both `docker-compose.yml` and `.env`.

### Step 3: Start Docker Services

```powershell
# Start Postgres, Redis, and ChromaDB (not Ollama — see Step 5)
docker compose up -d postgres redis chromadb
```

If you see container name conflicts from a previous setup:
```powershell
docker rm -f kairos-postgres kairos-redis kairos-chromadb
docker compose up -d postgres redis chromadb
```

Verify they're running:
```powershell
docker ps --filter "name=kairos"
# All 3 should show "healthy" after ~30 seconds
```

The database migration runs automatically — the SQL file is mounted to `docker-entrypoint-initdb.d`. Verify:
```powershell
docker exec kairos-postgres psql -U kairos -d kairos -c "\dt"
# Should show 10 tables
```

### Step 4: Create Virtual Environment & Install Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

This installs all 200+ dependencies including LangGraph, LiteLLM, Instructor, ChromaDB, FastAPI, etc. The `chroma-hnswlib` package compiles from C++ source — this is why the Build Tools are required.

Verify:
```powershell
python -c "import kairos; print('OK')"
pipeline --help
```

### Step 5: Set Up Ollama (Local LLMs)

For best GPU performance on RTX 3090, run Ollama **natively** on the host (not in Docker). If you already have Ollama running (e.g., via Open WebUI), skip the install.

If Ollama is not installed:
```powershell
winget install Ollama.Ollama
# Restart terminal, then:
ollama serve  # Start the server (or it may auto-start as a service)
```

Pull the required models:
```powershell
# If Ollama CLI is on PATH:
ollama pull mistral:7b-instruct-q4_0
ollama pull llama3.1:8b

# If running Ollama in Docker:
docker exec <ollama-container-name> ollama pull mistral:7b-instruct-q4_0
docker exec <ollama-container-name> ollama pull llama3.1:8b
```

> **Note:** `moondream2` (vision model for Tier 2 frame inspection) is no longer available in the Ollama registry as of Feb 2026. This feature is not yet implemented and is safely skipped.

Verify Ollama has the models:
```powershell
# Via API:
Invoke-RestMethod http://localhost:11434/api/tags | Select-Object -ExpandProperty models | Select-Object name
# Should show: mistral:7b-instruct-q4_0, llama3.1:8b
```

### Step 6: Enable Local LLMs

Edit `llm_config.yaml` and set:
```yaml
use_local_llms: true
```

This tells the pipeline to try local Ollama models first before falling back to Claude (cloud). See the config file for which steps use local vs cloud models.

### Step 7: Build Sandbox Image

The sandbox is a Docker image used to safely execute agent-generated simulation code:

```powershell
docker build -t kairos-sandbox sandbox/
```

### Step 8: Verify Everything

Run the full verification:

```powershell
python -c "
import redis, httpx, psycopg2
# Postgres
conn = psycopg2.connect(host='127.0.0.1', port=5434, dbname='kairos', user='kairos', password='changeme')
cur = conn.cursor(); cur.execute('SELECT count(*) FROM information_schema.tables WHERE table_schema=''public''')
print(f'Postgres: {cur.fetchone()[0]} tables'); cur.close(); conn.close()
# Redis
r = redis.Redis.from_url('redis://localhost:6379/0'); r.ping(); print('Redis: OK')
# Ollama
resp = httpx.get('http://localhost:11434/api/tags')
print(f'Ollama models: {[m[\"name\"] for m in resp.json()[\"models\"]]}')
# ChromaDB
resp = httpx.get('http://localhost:8000/api/v2/heartbeat')
print(f'ChromaDB: OK')
"
```

Run unit tests:
```powershell
pytest tests/unit/ -q --timeout=30
# Expect ~280+ passed
```

### Run the Pipeline

```powershell
pipeline run --pipeline physics
pipeline status
pipeline resume <run-id>
```

### Troubleshooting

| Issue | Fix |
|-------|-----|
| `chroma-hnswlib` build fails | Install C++ Build Tools (see Prerequisites) |
| Postgres password auth fails | Port conflict with local Postgres — check `netstat -ano \| Select-String ":5434"` |
| FFmpeg not found after install | Refresh PATH: `$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")` |
| Doppler env vars cause `extra_forbidden` error | Already fixed — `config.py` uses `extra="ignore"` |
| ChromaDB healthcheck shows "unhealthy" | The v1 API is deprecated in recent ChromaDB images; the service still works fine |
