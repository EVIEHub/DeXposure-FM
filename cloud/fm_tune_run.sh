#!/usr/bin/env bash
# One worker: verify recovered checkpoints, validation-only search, final evaluation.
set -euo pipefail
ROOT="$PWD"
export DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$ROOT:$ROOT/paper:$ROOT/cloud/train_src"
export LD_LIBRARY_PATH="$(echo "$ROOT"/.venv/lib/python3.12/site-packages/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH:-}"
exec "$ROOT/.venv/bin/python" cloud/train_src/tune_fm.py --run
