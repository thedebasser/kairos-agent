# Kairos Agent — Setup Guide

> **Last updated:** 2026-02-25
> **Target:** Cloud-only mode (no local GPU / Ollama required)

This guide walks you through every dependency, key, and configuration value needed to run Kairos Agent end-to-end on a machine without a local GPU.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone & Install](#2-clone--install)
3. [Docker Services (Postgres + Redis)](#3-docker-services-postgres--redis)
4. [Run Database Migrations](#4-run-database-migrations)
5. [Anthropic API Key (Claude)](#5-anthropic-api-key-claude)
6. [Discord Webhook (Notifications)](#6-discord-webhook-notifications)
7. [Langfuse (Observability — Optional)](#7-langfuse-observability--optional)
8. [Build the Simulation Sandbox Image](#8-build-the-simulation-sandbox-image)
9. [LLM Config (Local/Cloud Toggle)](#9-llm-config-localcloud-toggle)
10. [Final `.env` File](#10-final-env-file)
11. [Verify Everything Works](#11-verify-everything-works)
12. [Optional: YouTube Upload API Key](#12-optional-youtube-upload-api-key)

---

## 1. Prerequisites

Install these before anything else:

| Tool | Minimum Version | Download |
|---|---|---|
| **Python** | 3.12+ | https://www.python.org/downloads/ |
| **Docker Desktop** | 4.x | https://www.docker.com/products/docker-desktop/ |
| **Git** | 2.x | https://git-scm.com/downloads |
| **FFmpeg** | 6.x+ | https://www.gyan.dev/ffmpeg/builds/ (Windows: download "release full", add `bin/` to PATH) |

### Verify prerequisites

```powershell
python --version      # 3.12+
docker --version      # 20+
git --version         # 2+
ffmpeg -version       # 6+
```

---

## 2. Clone & Install

```powershell
# Clone the repo
git clone <your-repo-url> kairos-agent
cd kairos-agent

# Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install the project in editable mode with dev dependencies
pip install -e ".[dev]"
```

> If `pip install` fails on `psycopg2-binary`, you may need the [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

---

## 3. Docker Services (Postgres + Redis)

Kairos uses PostgreSQL for all persistent data (pipeline runs, training examples, agent logs) and Redis as the Celery task broker.

### Step 1 — Start Docker Desktop
Open Docker Desktop and make sure the engine is running (green icon in system tray).

### Step 2 — Start all services
```powershell
docker-compose up -d postgres redis
```

### Step 3 — Verify they're running
```powershell
docker ps
# You should see kairos-postgres and kairos-redis with status "Up"
```

### Step 4 — Verify Postgres is accepting connections
```powershell
docker exec kairos-postgres pg_isready -U kairos
# Expected: "localhost:5432 - accepting connections"
```

> **Defaults:** The database is created with user `kairos`, password `changeme`, database name `kairos` on port `5433`. You can change these in `.env` — just keep `DATABASE_URL` and `DATABASE_URL_SYNC` in sync.

---

## 4. Run Database Migrations

The initial schema creates all tables (pipeline_runs, video_ideas, simulations, agent_runs, training_examples, etc.).

```powershell
# Apply the initial migration
docker exec -i kairos-postgres psql -U kairos -d kairos < migrations/001_initial_schema.sql
```

Verify:
```powershell
docker exec kairos-postgres psql -U kairos -d kairos -c "\dt"
# Should list: pipeline_runs, video_ideas, category_stats, simulations,
#              outputs, publish_queue, publish_log, agent_runs,
#              training_examples, pipeline_config
```

---

## 5. Anthropic API Key (Claude)

This is the **only required API key**. Every LLM step uses Claude Sonnet when running in cloud-only mode.

### Step 1 — Create an Anthropic account
Go to https://console.anthropic.com/ and sign up (or log in).

### Step 2 — Add billing
Go to **Settings → Billing** (https://console.anthropic.com/settings/billing) and add a payment method. Claude API usage is pay-per-token.

### Step 3 — Create an API key
Go to **Settings → API Keys** (https://console.anthropic.com/settings/keys).
Click **"Create Key"**, give it a name like `kairos-agent`, and copy the key.

### Step 4 — Paste into your `.env`
```dotenv
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> **Cost estimate:** Each full pipeline run (concept → simulation → video) costs roughly $0.10–$0.30 in Claude API tokens. The `cost_alert_threshold_usd` in `.env` will warn if the 7-day rolling average exceeds $0.30/run.

---

## 6. Discord Webhook (Notifications)

Discord webhooks notify you when a video enters the review queue or when publishing fails.

### Step 1 — Open your Discord server
Open Discord (app or browser) and navigate to the server where you want notifications.

### Step 2 — Create a channel (or pick an existing one)
Right-click your server name → **Create Channel** → name it something like `#kairos-notifications` → select **Text Channel** → click **Create Channel**.

### Step 3 — Open channel settings
Click the **gear icon** (⚙️) next to the channel name to open Channel Settings.

### Step 4 — Go to Integrations
In the left sidebar, click **Integrations**.

### Step 5 — Create a webhook
Click **Webhooks** → **New Webhook**.

### Step 6 — Configure the webhook
- Give it a name like `Kairos Agent`
- Optionally upload an avatar
- Make sure the correct channel is selected in the dropdown

### Step 7 — Copy the webhook URL
Click **Copy Webhook URL**. It will look like:
```
https://discord.com/api/webhooks/1234567890/abcDEF...
```

### Step 8 — Paste into your `.env`
```dotenv
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/1234567890/abcDEF...
```

> **What you'll receive:** Embed messages when a new video enters review (title, category, output ID) and alerts when publishing fails (platform, error, retry count).

---

## 7. Langfuse (Observability — Optional)

Langfuse gives you a dashboard for every LLM call: latency, tokens, cost, success/failure. Highly recommended but not required to run.

### Option A — Langfuse Cloud (easiest)

#### Step 1 — Sign up
Go to https://cloud.langfuse.com and create a free account.

#### Step 2 — Create a project
Click **New Project**, name it `kairos-agent`.

#### Step 3 — Get your keys
Go to **Settings → API Keys** inside the project. Copy the **Public Key** and **Secret Key**.

#### Step 4 — Paste into your `.env`
```dotenv
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Option B — Self-hosted (Docker)

```powershell
# Add to docker-compose.yml or run separately
docker run -d --name langfuse \
  -p 3000:3000 \
  -e DATABASE_URL=postgresql://kairos:changeme@host.docker.internal:5432/langfuse \
  langfuse/langfuse:latest
```

Then use:
```dotenv
LANGFUSE_PUBLIC_KEY=<from localhost:3000 UI>
LANGFUSE_SECRET_KEY=<from localhost:3000 UI>
LANGFUSE_HOST=http://localhost:3000
```

> **If skipping Langfuse:** Leave the keys blank. The app will run fine — you just won't have LLM call tracing.

---

## 8. Build the Simulation Sandbox Image

The simulation agent runs generated Pygame+Pymunk code inside a Docker container for isolation.

```powershell
docker build -t simulation-sandbox:latest ./sandbox
```

Verify:
```powershell
docker images simulation-sandbox
# Should show "simulation-sandbox   latest   <hash>   <size>"
```

---

## 9. LLM Config (Local/Cloud Toggle)

The file `llm_config.yaml` in the repo root controls which models each pipeline step uses.

**For your current setup (no GPU), it's already configured correctly:**

```yaml
use_local_llms: false          # Skip Ollama, go straight to cloud
always_store_training_data: true  # Save cloud responses for future fine-tuning
```

**You don't need to change anything.** When you later have a machine with a GPU:
1. Install [Ollama](https://ollama.ai/download) natively
2. Pull the required models:
   ```powershell
   ollama pull mistral:7b-instruct-q4_0
   ollama pull llama3.1:8b
   ollama pull moondream2
   ```
3. Flip the toggle in `llm_config.yaml`:
   ```yaml
   use_local_llms: true
   ```

> **Training data** is always stored regardless of this toggle — in the `agent_runs` and `training_examples` Postgres tables, and as JSON files in `knowledge/cloud_learnings/`.

---

## 10. Final `.env` File

Copy the example and fill in your values:

```powershell
copy .env.example .env
```

Here's what your completed `.env` should look like:

```dotenv
# =============================================================================
# Kairos Agent — Environment Configuration
# =============================================================================

# --- Anthropic (Claude) --- [REQUIRED]
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here

# --- PostgreSQL --- [REQUIRED — defaults work with docker-compose]
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_DB=kairos
POSTGRES_USER=kairos
POSTGRES_PASSWORD=changeme
DATABASE_URL=postgresql+asyncpg://kairos:changeme@localhost:5433/kairos
DATABASE_URL_SYNC=postgresql://kairos:changeme@localhost:5433/kairos

# --- Redis --- [REQUIRED — defaults work with docker-compose]
REDIS_URL=redis://localhost:6379/0

# --- Ollama --- [NOT NEEDED when use_local_llms: false]
OLLAMA_BASE_URL=http://localhost:11434

# --- LiteLLM ---
LITELLM_CONFIG_PATH=litellm_config.yaml

# --- Langfuse (Observability) --- [OPTIONAL]
LANGFUSE_PUBLIC_KEY=pk-lf-your-key-here
LANGFUSE_SECRET_KEY=sk-lf-your-key-here
LANGFUSE_HOST=https://cloud.langfuse.com

# --- Discord Notifications --- [OPTIONAL but recommended]
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your-webhook-url-here

# --- Upload-Post API --- [OPTIONAL — for YouTube publishing]
UPLOAD_POST_API_KEY=

# --- Pipeline Defaults ---
DEFAULT_PIPELINE=physics
TARGET_DURATION_SEC=65
MAX_SIMULATION_ITERATIONS=5
SANDBOX_TIMEOUT_SEC=300
SANDBOX_MEMORY_LIMIT=4g
SANDBOX_CPU_LIMIT=2
```

### Quick reference — what each key is for

| Key | Required? | What uses it | Where to get it |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | **Yes** | All LLM steps (concept, simulation, captions, titles, debugging) | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) |
| `POSTGRES_*` / `DATABASE_URL` | **Yes** | Pipeline state, training data, agent logs, publish queue | Auto-created by `docker-compose up -d postgres` |
| `REDIS_URL` | **Yes** | Celery task queue | Auto-created by `docker-compose up -d redis` |
| `DISCORD_WEBHOOK_URL` | Optional | Review notifications, publish failure alerts | Discord channel → Integrations → Webhooks (see Section 6) |
| `LANGFUSE_*` | Optional | LLM call tracing dashboard (latency, cost, tokens) | [cloud.langfuse.com](https://cloud.langfuse.com) or self-hosted |
| `OLLAMA_BASE_URL` | No (cloud mode) | Local LLM serving — unused when `use_local_llms: false` | [ollama.ai](https://ollama.ai) |
| `UPLOAD_POST_API_KEY` | No (manual) | YouTube / TikTok upload automation | Platform-specific OAuth (not needed for POC) |
| `LITELLM_CONFIG_PATH` | Auto | Points to `litellm_config.yaml` for model routing | Already set — don't change |

---

## 11. Verify Everything Works

### Check 1 — Python imports
```powershell
python -c "from kairos.config import get_settings; s = get_settings(); print('DB:', s.database_url[:30] + '...'); print('Anthropic key set:', bool(s.anthropic_api_key))"
```

### Check 2 — Database connection
```powershell
python -c "
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from kairos.config import get_settings
async def check():
    engine = create_async_engine(get_settings().database_url)
    async with engine.connect() as conn:
        result = await conn.execute(__import__('sqlalchemy').text('SELECT 1'))
        print('Postgres OK:', result.scalar())
    await engine.dispose()
asyncio.run(check())
"
```

### Check 3 — LLM config loads
```powershell
python -c "
from kairos.services.llm_config import use_local_llms, get_step_config
print('use_local_llms:', use_local_llms())
step = get_step_config('concept_developer')
print('concept_developer model:', step.resolve_model())
"
# Expected:
#   use_local_llms: False
#   concept_developer model: concept-developer
```

### Check 4 — Sandbox image exists
```powershell
docker images simulation-sandbox --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
```

---

## 12. Optional: YouTube Upload API Key

YouTube publishing uses the YouTube Data API v3 via OAuth2. This is only needed when you're ready to auto-publish.

### Step 1 — Go to Google Cloud Console
Navigate to https://console.cloud.google.com/

### Step 2 — Create a project
Click the project dropdown at the top → **New Project** → name it `kairos-agent` → **Create**.

### Step 3 — Enable the YouTube Data API v3
Go to **APIs & Services → Library** → search for "YouTube Data API v3" → click **Enable**.

### Step 4 — Create OAuth credentials
Go to **APIs & Services → Credentials** → **Create Credentials → OAuth 2.0 Client ID**.
- Application type: **Desktop app**
- Name: `kairos-agent`
- Click **Create** and download the JSON file.

### Step 5 — Configure consent screen
Go to **APIs & Services → OAuth consent screen** → fill in the required fields → add the `youtube.upload` scope.

### Step 6 — Place credentials
Save the downloaded JSON as `credentials.json` in the project root (it's already in `.gitignore`).

> **Note:** The upload service is pluggable — the `UPLOAD_POST_API_KEY` in `.env` is a placeholder for future platform-specific integrations. For now, videos are produced and reviewed locally.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `psycopg2` install fails | Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) or use `psycopg2-binary` (already in deps) |
| Docker says "port 5432 already in use" | The Docker Postgres is mapped to port 5433 to avoid conflicts with native Postgres installs. If you still have issues, check `netstat -aon \| Select-String ":5433"` |
| `ModuleNotFoundError: No module named 'kairos'` | Make sure you ran `pip install -e ".[dev]"` with the venv activated |
| LLM calls fail with 401 | Double-check `ANTHROPIC_API_KEY` in `.env` — it must start with `sk-ant-` |
| Discord notifications not arriving | Verify the webhook URL in `.env`, test it with: `curl -X POST -H "Content-Type: application/json" -d "{\"content\":\"test\"}" YOUR_WEBHOOK_URL` |
| Sandbox container OOM-killed | Increase `SANDBOX_MEMORY_LIMIT` in `.env` (e.g., `6g`) |
