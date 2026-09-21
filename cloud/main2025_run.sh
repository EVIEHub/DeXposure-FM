#!/usr/bin/env bash
# Main causal-harness FM run.  This is the primary cloud binding.
#
# The primary configuration is deliberately h=4 only, holdout-start
# 2025-01-01, validation window 24 weeks, seed 42, and 20 epochs.  Broader
# horizon sweeps belong to the separately named RQ1 binding.
set -euo pipefail

ROOT="$PWD"
export PATH="$HOME/.local/bin:$PATH"
PY="$ROOT/.venv/bin/python"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
if [ ! -x "$PY" ]; then
  (cd "$ROOT" && uv sync --frozen)
fi
test -x "$PY" || { echo "FATAL: venv bootstrap failed" >&2; exit 1; }

"$PY" cloud/preflight.py --phase host \
  --train-cmd "bash cloud/main2025_run.sh"

export DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1
export PYTHONPATH="$ROOT:$ROOT/paper:$ROOT/cloud/train_src"
export LD_LIBRARY_PATH="$(echo "$ROOT"/.venv/lib/python3.12/site-packages/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH:-}"

TRAIN_HORIZONS=4
echo "=== Main 2025 FM run (h=4, holdout=2025-01-01, val=24, epochs=20, seed=42) ==="
"$PY" cloud/train_src/run_full_experiment.py \
  --mode dexposure-fm --holdout-start 2025-01-01 \
  --val-weeks 24 --epochs 20 --seed 42 --horizons "$TRAIN_HORIZONS" \
  --output-dir checkpoints/main2025_train

echo "=== Stage primary h=4 checkpoint ==="
mkdir -p checkpoints/main2025
src="checkpoints/main2025_train/finetuned/best_model_h4.pt"
test -s "$src" || { echo "FATAL: $src not found or empty" >&2; exit 1; }
cp "$src" checkpoints/main2025/dexposure-fm-h4.pt
schema_src="checkpoints/main2025_train/finetuned/feature_schema.json"
test -s "$schema_src" || { echo "FATAL: $schema_src not found or empty" >&2; exit 1; }
cp "$schema_src" checkpoints/main2025/feature_schema.json
ls -lh checkpoints/main2025/

echo "MAIN 2025 H4 RUN DONE"
