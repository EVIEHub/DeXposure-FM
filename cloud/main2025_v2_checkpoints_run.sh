#!/usr/bin/env bash
# Reconstruct all four DeXposure-FM v2 forecasting checkpoints.
set -euo pipefail

ROOT="$PWD"
export PATH="$HOME/.local/bin:$PATH"
PY="$ROOT/.venv/bin/python"
test -x "$PY" || { echo "FATAL: Python environment is unavailable" >&2; exit 1; }

PREFLIGHT_CMD="bash cloud/main2025_v2_checkpoints_run.sh"
"$PY" cloud/preflight.py --phase host --train-cmd "$PREFLIGHT_CMD"

GPU_SELECTOR="${CUDA_VISIBLE_DEVICES:-0}"
GPU_SELECTOR="${GPU_SELECTOR%%,*}"
GPU_NAME="$(nvidia-smi -i "$GPU_SELECTOR" --query-gpu=name --format=csv,noheader | head -n1)"
GPU_MEMORY_MIB="$(nvidia-smi -i "$GPU_SELECTOR" --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
case "$GPU_NAME" in
  *Blackwell*)
    echo "FATAL: allocated GPU is incompatible with locked PyTorch 2.2.1: $GPU_NAME" >&2
    exit 1
    ;;
esac
if [ "$GPU_MEMORY_MIB" -gt 49152 ]; then
  echo "FATAL: allocated GPU exceeds the 48 GiB ceiling: $GPU_NAME ($GPU_MEMORY_MIB MiB)" >&2
  exit 1
fi
echo "=== Allocated GPU: $GPU_NAME ($GPU_MEMORY_MIB MiB) ==="

export DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$ROOT:$ROOT/paper:$ROOT/cloud/train_src"
export LD_LIBRARY_PATH="$(echo "$ROOT"/.venv/lib/python3.12/site-packages/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH:-}"

TRAIN_HORIZONS=1,4,8,12
TRAIN_ROOT="checkpoints/main2025_v2_checkpoints_train"
TRAIN_OUTPUT="$TRAIN_ROOT/finetuned"
RELEASE_ROOT="checkpoints/main2025_v2_rerun"

echo "=== DeXposure-FM v2 checkpoint reconstruction (h=1,4,8,12; holdout=2025-01-01; val=24; epochs=20; seed=42) ==="
"$PY" cloud/train_src/run_full_experiment.py \
  --mode dexposure-fm --holdout-start 2025-01-01 \
  --val-weeks 24 --epochs 20 --seed 42 --horizons "$TRAIN_HORIZONS" \
  --output-dir "$TRAIN_ROOT"

mkdir -p "$RELEASE_ROOT"
for h in 1 4 8 12; do
  src="$TRAIN_OUTPUT/best_model_h${h}.pt"
  test -s "$src" || { echo "FATAL: $src not found or empty" >&2; exit 1; }
  cp "$src" "$RELEASE_ROOT/dexposure-fm-h${h}.pt"
done
schema_src="$TRAIN_OUTPUT/feature_schema.json"
test -s "$schema_src" || { echo "FATAL: $schema_src not found or empty" >&2; exit 1; }
cp "$schema_src" "$RELEASE_ROOT/feature_schema.json"
shasum -a 256 "$RELEASE_ROOT"/*.pt | tee "$RELEASE_ROOT/SHA256SUMS"

echo "MAIN 2025 V2 FOUR-CHECKPOINT RUN DONE"
