#!/usr/bin/env bash
# =============================================================================
# Kairos Agent — Ollama Model Setup Script
# =============================================================================
#
# Pulls all required Ollama models for the Kairos pipeline.
# Run this once on a fresh machine, or when new models are added.
#
# Usage:
#   chmod +x scripts/setup_ollama_models.sh
#   ./scripts/setup_ollama_models.sh [--all | --core | --review | --blender]
#
# Options:
#   --all      Pull all models (default)
#   --core     Pull only core pipeline models (existing)
#   --review   Pull only review agent models (video + audio)
#   --blender  Pull only Blender coding models
#
# Requirements:
#   - Ollama installed and running (http://localhost:11434)
#   - RTX 3090 (24GB VRAM) recommended
#   - ~80GB disk space for all models
#
# =============================================================================

set -euo pipefail

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

# ANSI colours
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

# Core pipeline models (existing)
CORE_MODELS=(
    "mistral:7b-instruct-q4_0"     # Category selector, param adjustment
    "llama3.1:8b"                   # Title writer
    "moondream:latest"              # Frame inspector (Tier 2 validation, planned)
)

# Video review models
VIDEO_REVIEW_MODELS=(
    "qwen3-vl:8b"                  # Default video reviewer (~16 GB FP16)
    "qwen3-vl:30b-a3b"             # Escalation video reviewer (~17-20 GB Q4_K_M)
)

# Audio review models
AUDIO_REVIEW_MODELS=(
    "qwen2.5-omni:7b"              # Default audio reviewer (~17 GB FP16)
    "qwen3-omni:30b-a3b"           # Escalation audio reviewer (~20 GB Q4_K_M)
)

# Blender coding models
BLENDER_MODELS=(
    "qwen3.5:27b"                  # Default Blender coder (~18.5 GB Q5_K_M)
    "devstral-small:24b"           # Option 2: agentic coding (~19 GB Q6_K)
    "qwen3-coder:30b-a3b"          # Option 3: MoE code specialist
    "qwen2.5-coder:32b"            # Fallback: strong function-level coding
)

# ---------------------------------------------------------------------------

check_ollama() {
    if ! curl -sf "${OLLAMA_BASE_URL}/api/tags" > /dev/null 2>&1; then
        log_error "Ollama is not reachable at ${OLLAMA_BASE_URL}"
        log_error "Make sure Ollama is installed and running."
        log_error "  Install: https://ollama.ai/download"
        log_error "  Start:   ollama serve"
        exit 1
    fi
    log_ok "Ollama is reachable at ${OLLAMA_BASE_URL}"
}

pull_model() {
    local model="$1"
    log_info "Pulling ${model}..."
    if ollama pull "${model}" 2>&1; then
        log_ok "Successfully pulled ${model}"
    else
        log_warn "Failed to pull ${model} — it may not be available yet"
    fi
}

pull_group() {
    local group_name="$1"
    shift
    local models=("$@")

    echo ""
    log_info "=== ${group_name} (${#models[@]} models) ==="
    for model in "${models[@]}"; do
        pull_model "${model}"
    done
}

show_vram_warning() {
    echo ""
    log_warn "=== VRAM Budget Notes (RTX 3090 — 24GB) ==="
    echo "  Only ONE large model can be loaded at a time."
    echo "  Ollama handles loading/unloading automatically."
    echo ""
    echo "  Approximate VRAM usage:"
    echo "    Core models:          ~4-8 GB"
    echo "    Qwen3-VL-8B:          ~16 GB FP16"
    echo "    Qwen3-VL-30B-A3B:     ~17-20 GB Q4_K_M"
    echo "    Qwen2.5-Omni-7B:      ~17 GB FP16"
    echo "    Qwen3-Omni-30B-A3B:   ~20 GB Q4_K_M"
    echo "    Qwen3.5-27B:          ~18.5 GB Q5_K_M"
    echo "    Devstral Small 24B:   ~19 GB Q6_K"
    echo "    Qwen2.5-Coder-32B:    ~18.5 GB Q4_K_M"
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODE="${1:---all}"

echo "============================================================"
echo "  Kairos Agent — Ollama Model Setup"
echo "============================================================"
echo ""

check_ollama

case "${MODE}" in
    --core)
        pull_group "Core Pipeline Models" "${CORE_MODELS[@]}"
        ;;
    --review)
        pull_group "Video Review Models" "${VIDEO_REVIEW_MODELS[@]}"
        pull_group "Audio Review Models" "${AUDIO_REVIEW_MODELS[@]}"
        ;;
    --blender)
        pull_group "Blender Coding Models" "${BLENDER_MODELS[@]}"
        ;;
    --all|*)
        pull_group "Core Pipeline Models" "${CORE_MODELS[@]}"
        pull_group "Video Review Models" "${VIDEO_REVIEW_MODELS[@]}"
        pull_group "Audio Review Models" "${AUDIO_REVIEW_MODELS[@]}"
        pull_group "Blender Coding Models" "${BLENDER_MODELS[@]}"
        ;;
esac

show_vram_warning

echo ""
log_ok "Model setup complete!"
echo ""
echo "  To verify: ollama list"
echo "  To test:   ollama run qwen3-vl:8b 'Hello, world'"
echo ""
