#!/usr/bin/env bash
# Reconstruct one DeXposure-FM v2 checkpoint on the fixed 2025 holdout.
set -euo pipefail

HORIZON="${1:?usage: cloud/main2025_v2_horizon_run.sh <1|4|8|12>}"
case "$HORIZON" in
  1|4|8|12) ;;
  *) echo "FATAL: horizon must be one of 1, 4, 8, or 12" >&2; exit 1 ;;
esac
EPOCHS=20
PATIENCE=3
PREFLIGHT_CMD="bash cloud/main2025_v2_horizon_run.sh $HORIZON"
if [ "${2:-}" = --extended ] && [ "$HORIZON" = 12 ] && [ "$#" = 2 ]; then
  EPOCHS=40
  PATIENCE=5
  PREFLIGHT_CMD="$PREFLIGHT_CMD --extended"
elif [ "$#" != 1 ]; then
  echo "FATAL: only h=12 accepts --extended" >&2
  exit 1
fi

ROOT="$PWD"
export PATH="$HOME/.local/bin:$PATH"
PY="$ROOT/.venv/bin/python"
test -x "$PY" || { echo "FATAL: Python environment is unavailable" >&2; exit 1; }

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

TRAIN_ROOT="checkpoints/main2025_v2_h${HORIZON}_train"
TRAIN_OUTPUT="$TRAIN_ROOT/finetuned"
RELEASE_ROOT="checkpoints/main2025_v2_h${HORIZON}"

echo "=== DeXposure-FM h=$HORIZON (283 snapshots; holdout=2025-01-01; val=24; epochs=$EPOCHS; patience=$PATIENCE; seed=42) ==="
"$PY" cloud/train_src/run_full_experiment.py \
  --mode dexposure-fm --holdout-start 2025-01-01 \
  --val-weeks 24 --epochs "$EPOCHS" --patience "$PATIENCE" --seed 42 --horizons "$HORIZON" \
  --output-dir "$TRAIN_ROOT"

MODEL_SOURCE="$TRAIN_OUTPUT/best_model_h${HORIZON}.pt"
SCHEMA_SOURCE="$TRAIN_OUTPUT/feature_schema.json"
METRICS_SOURCE="$TRAIN_OUTPUT/metrics.json"
test -s "$MODEL_SOURCE" || { echo "FATAL: $MODEL_SOURCE not found or empty" >&2; exit 1; }
test -s "$SCHEMA_SOURCE" || { echo "FATAL: $SCHEMA_SOURCE not found or empty" >&2; exit 1; }
test -s "$METRICS_SOURCE" || { echo "FATAL: $METRICS_SOURCE not found or empty" >&2; exit 1; }

mkdir -p "$RELEASE_ROOT"
cp "$MODEL_SOURCE" "$RELEASE_ROOT/dexposure-fm-h${HORIZON}.pt"
cp "$SCHEMA_SOURCE" "$RELEASE_ROOT/feature_schema.json"
cp "$METRICS_SOURCE" "$RELEASE_ROOT/task1_metrics.json"
HORIZON="$HORIZON" EPOCHS="$EPOCHS" PATIENCE="$PATIENCE" GPU_NAME="$GPU_NAME" GPU_MEMORY_MIB="$GPU_MEMORY_MIB" \
  "$PY" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

horizon = int(os.environ["HORIZON"])
payload = {
    "artifact_status": "complete",
    "forecast_horizon_weeks": horizon,
    "holdout_start": "2025-01-01",
    "validation_weeks": 24,
    "epochs": int(os.environ["EPOCHS"]),
    "early_stop_patience": int(os.environ["PATIENCE"]),
    "experiment_kind": "h12_extended_training" if int(os.environ["EPOCHS"]) == 40 else "v2_reconstruction",
    "seed": 42,
    "network_snapshots": 283,
    "source_sha256": os.environ["PREFLIGHT_SOURCE_SHA256"],
    "gpu_name": os.environ["GPU_NAME"],
    "gpu_memory_mib": int(os.environ["GPU_MEMORY_MIB"]),
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
}
path = Path(f"checkpoints/main2025_v2_h{horizon}/run_config.json")
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
(
  cd "$RELEASE_ROOT"
  shasum -a 256 \
    "dexposure-fm-h${HORIZON}.pt" feature_schema.json task1_metrics.json run_config.json \
    > SHA256SUMS
  shasum -a 256 -c SHA256SUMS
)

echo "MAIN 2025 V2 H${HORIZON} CHECKPOINT RUN DONE"
