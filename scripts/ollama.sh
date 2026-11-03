#!/usr/bin/env bash
# Start a local Ollama server tuned for PatchPilot (32k context fits our largest prompt, ~17k tokens).
# Usage: scripts/ollama.sh            (foreground)   |   scripts/ollama.sh pull   (pull the model)
set -euo pipefail
OLLAMA_BIN="${OLLAMA_BIN:-$(command -v ollama || echo "$HOME/.local/ollama/bin/ollama")}"
MODEL="${OLLAMA_MODEL:-qwen2.5-coder:7b}"

export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-32768}"
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"   # halves KV memory → 32k ctx fits in 8 GB VRAM
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"

if [[ "${1:-}" == "pull" ]]; then
  exec "$OLLAMA_BIN" pull "$MODEL"
fi
exec "$OLLAMA_BIN" serve
