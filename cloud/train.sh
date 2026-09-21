#!/usr/bin/env bash
# The only sanctioned way to launch GPU training on Vast.ai.
# Runs the repository preflight before handing the job to the direct Vast API
# controller. The controller downloads and verifies artifacts before teardown.
#
# Usage:
#   cloud/train.sh python lib/train.py --epochs 10
#   cloud/train.sh bash cloud/main2025_v2_horizon_run.sh 1
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ $# -eq 0 ]; then
  echo "usage: cloud/train.sh <training command...>" >&2
  exit 1
fi

# Do deterministic checks before contacting Vast.ai or allocating a worker.
# The resulting source digest is passed to the worker because .vastignore
# intentionally omits .git.
TRAIN_CMD="$*"
PREFLIGHT_PY="${PREFLIGHT_PY:-python3}"
if ! command -v "$PREFLIGHT_PY" >/dev/null 2>&1; then
  echo "ERROR: Python is required for the zero-compute cloud preflight." >&2
  exit 1
fi
SOURCE_SHA="$($PREFLIGHT_PY cloud/preflight.py --print-source-digest)" || {
  echo "ERROR: unable to compute the source preflight digest; refusing to launch." >&2
  exit 1
}
if ! PREFLIGHT_SOURCE_SHA256="$SOURCE_SHA" "$PREFLIGHT_PY" cloud/preflight.py \
    --phase local --train-cmd "$TRAIN_CMD"; then
  echo "ERROR: local cloud preflight blocked this launch." >&2
  exit 1
fi

# Prefer the repository's ignored .env, then support the standard Hugging Face
# token file. Only the explicit HF_TOKEN and optional OPENROUTER_API_KEY entries
# are parsed; unrelated .env secrets are never sent to the instance.
HF_TOKEN_VALUE="${HF_TOKEN:-}"
if [ -z "$HF_TOKEN_VALUE" ] && [ -f "$ROOT_DIR/.env" ]; then
  HF_TOKEN_VALUE="$(awk -F= '$1 == "HF_TOKEN" {sub(/^[^=]*=/, ""); print; exit}' "$ROOT_DIR/.env")"
fi
HF_TOKEN_VALUE="${HF_TOKEN_VALUE#\"}"
HF_TOKEN_VALUE="${HF_TOKEN_VALUE%\"}"
HF_TOKEN_VALUE="${HF_TOKEN_VALUE#\'}"
HF_TOKEN_VALUE="${HF_TOKEN_VALUE%\'}"
HF_TOKEN_FILE="$HOME/.cache/huggingface/token"
if [ -z "$HF_TOKEN_VALUE" ] && [ -f "$HF_TOKEN_FILE" ]; then
  HF_TOKEN_VALUE="$(<"$HF_TOKEN_FILE")"
fi
if [ -n "$HF_TOKEN_VALUE" ]; then
  export HF_TOKEN="$HF_TOKEN_VALUE"
else
  echo "ERROR: no HF_TOKEN found in $ROOT_DIR/.env, the HF_TOKEN environment variable, or $HF_TOKEN_FILE; refusing to launch because the instance disk is ephemeral." >&2
  exit 1
fi
unset HF_TOKEN_VALUE

# Optional LLM credential for the explicitly bound LLM pilot only. It is
# parsed independently so unrelated .env secrets never enter the cloud job.
OPENROUTER_KEY_VALUE="${OPENROUTER_API_KEY:-}"
if [ "$TRAIN_CMD" = "bash cloud/llm_pilot_run.sh" ] && [ -z "$OPENROUTER_KEY_VALUE" ] && [ -f "$ROOT_DIR/.env" ]; then
  OPENROUTER_KEY_VALUE="$(awk -F= '$1 == "OPENROUTER_API_KEY" {sub(/^[^=]*=/, ""); print; exit}' "$ROOT_DIR/.env")"
fi
OPENROUTER_KEY_VALUE="${OPENROUTER_KEY_VALUE#\"}"
OPENROUTER_KEY_VALUE="${OPENROUTER_KEY_VALUE%\"}"
OPENROUTER_KEY_VALUE="${OPENROUTER_KEY_VALUE#\'}"
OPENROUTER_KEY_VALUE="${OPENROUTER_KEY_VALUE%\'}"
if [ "$TRAIN_CMD" = "bash cloud/llm_pilot_run.sh" ] && [ -n "$OPENROUTER_KEY_VALUE" ]; then
  export OPENROUTER_API_KEY="$OPENROUTER_KEY_VALUE"
fi
unset OPENROUTER_KEY_VALUE

exec bash cloud/vast_api_train.sh "$TRAIN_CMD" "$SOURCE_SHA"
