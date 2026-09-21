#!/usr/bin/env bash
# One worker: evaluate pinned EVIEHub DeXposure-FM v1.0 weights on the 2025 holdout.
set -euo pipefail
ROOT="$PWD"
export DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$ROOT:$ROOT/paper:$ROOT/cloud/train_src"
export LD_LIBRARY_PATH="$(echo "$ROOT"/.venv/lib/python3.12/site-packages/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH:-}"
exec "$ROOT/.venv/bin/python" cloud/train_src/eval_paper.py --run
