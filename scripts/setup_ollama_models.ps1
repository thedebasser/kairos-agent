# =============================================================================
# Kairos Agent — Ollama Model Setup Script (Windows PowerShell)
# =============================================================================
#
# Pulls all required Ollama models for the Kairos pipeline.
# Run this once on a fresh machine, or when new models are added.
#
# Usage:
#   .\scripts\setup_ollama_models.ps1 [-Group All|Core|Review|Blender]
#
# Requirements:
#   - Ollama installed and running (http://localhost:11434)
#   - RTX 3090 (24GB VRAM) recommended
#   - ~80GB disk space for all models
#
# =============================================================================

param(
    [ValidateSet("All", "Core", "Review", "Blender")]
    [string]$Group = "All"
)

$OllamaBaseUrl = if ($env:OLLAMA_BASE_URL) { $env:OLLAMA_BASE_URL } else { "http://localhost:11434" }

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

$CoreModels = @(
    "mistral:7b-instruct-q4_0"     # Category selector, param adjustment
    "llama3.1:8b"                   # Title writer
    "moondream:latest"              # Frame inspector (Tier 2, planned)
)

$VideoReviewModels = @(
    "qwen3-vl:8b"                  # Default video reviewer (~16 GB FP16)
    "qwen3-vl:30b-a3b"             # Escalation video reviewer (~17-20 GB Q4_K_M)
)

$AudioReviewModels = @(
    "qwen2.5-omni:7b"              # Default audio reviewer (~17 GB FP16)
    "qwen3-omni:30b-a3b"           # Escalation audio reviewer (~20 GB Q4_K_M)
)

$BlenderModels = @(
    "qwen3.5:27b"                  # Default Blender coder (~18.5 GB Q5_K_M)
    "devstral-small:24b"           # Option 2: agentic coding (~19 GB Q6_K)
    "qwen3-coder:30b-a3b"          # Option 3: MoE code specialist
    "qwen2.5-coder:32b"            # Fallback: strong function-level coding
)

# ---------------------------------------------------------------------------

function Test-OllamaReachable {
    try {
        $null = Invoke-RestMethod -Uri "$OllamaBaseUrl/api/tags" -TimeoutSec 5
        Write-Host "[OK]    Ollama is reachable at $OllamaBaseUrl" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[ERROR] Ollama is not reachable at $OllamaBaseUrl" -ForegroundColor Red
        Write-Host "        Make sure Ollama is installed and running." -ForegroundColor Red
        Write-Host "        Install: https://ollama.ai/download" -ForegroundColor Red
        Write-Host "        Start:   ollama serve" -ForegroundColor Red
        return $false
    }
}

function Pull-Model {
    param([string]$Model)
    Write-Host "[INFO]  Pulling $Model..." -ForegroundColor Cyan
    try {
        & ollama pull $Model 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK]    Successfully pulled $Model" -ForegroundColor Green
        } else {
            Write-Host "[WARN]  Failed to pull $Model — it may not be available yet" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "[WARN]  Failed to pull $Model — $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

function Pull-ModelGroup {
    param(
        [string]$GroupName,
        [string[]]$Models
    )
    Write-Host ""
    Write-Host "[INFO]  === $GroupName ($($Models.Length) models) ===" -ForegroundColor Cyan
    foreach ($model in $Models) {
        Pull-Model -Model $model
    }
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

Write-Host "============================================================"
Write-Host "  Kairos Agent — Ollama Model Setup"
Write-Host "============================================================"
Write-Host ""

if (-not (Test-OllamaReachable)) {
    exit 1
}

switch ($Group) {
    "Core" {
        Pull-ModelGroup -GroupName "Core Pipeline Models" -Models $CoreModels
    }
    "Review" {
        Pull-ModelGroup -GroupName "Video Review Models" -Models $VideoReviewModels
        Pull-ModelGroup -GroupName "Audio Review Models" -Models $AudioReviewModels
    }
    "Blender" {
        Pull-ModelGroup -GroupName "Blender Coding Models" -Models $BlenderModels
    }
    default {
        Pull-ModelGroup -GroupName "Core Pipeline Models" -Models $CoreModels
        Pull-ModelGroup -GroupName "Video Review Models" -Models $VideoReviewModels
        Pull-ModelGroup -GroupName "Audio Review Models" -Models $AudioReviewModels
        Pull-ModelGroup -GroupName "Blender Coding Models" -Models $BlenderModels
    }
}

Write-Host ""
Write-Host "[WARN]  === VRAM Budget Notes (RTX 3090 — 24GB) ===" -ForegroundColor Yellow
Write-Host "  Only ONE large model can be loaded at a time."
Write-Host "  Ollama handles loading/unloading automatically."
Write-Host ""
Write-Host "  Approximate VRAM usage:"
Write-Host "    Core models:          ~4-8 GB"
Write-Host "    Qwen3-VL-8B:          ~16 GB FP16"
Write-Host "    Qwen3-VL-30B-A3B:     ~17-20 GB Q4_K_M"
Write-Host "    Qwen2.5-Omni-7B:      ~17 GB FP16"
Write-Host "    Qwen3-Omni-30B-A3B:   ~20 GB Q4_K_M"
Write-Host "    Qwen3.5-27B:          ~18.5 GB Q5_K_M"
Write-Host "    Devstral Small 24B:   ~19 GB Q6_K"
Write-Host "    Qwen2.5-Coder-32B:    ~18.5 GB Q4_K_M"
Write-Host ""
Write-Host "[OK]    Model setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  To verify: ollama list"
Write-Host "  To test:   ollama run qwen3-vl:8b 'Hello, world'"
Write-Host ""
