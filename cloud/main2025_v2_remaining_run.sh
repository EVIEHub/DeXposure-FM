#!/usr/bin/env bash
# Reconstruct h=4,8,12, or all four horizons with --all, on one worker.
set -euo pipefail

ROOT="$PWD"
export PATH="$HOME/.local/bin:$PATH"
PY="$ROOT/.venv/bin/python"
test -x "$PY" || { echo "FATAL: Python environment is unavailable" >&2; exit 1; }

PREFLIGHT_CMD="bash cloud/main2025_v2_remaining_run.sh"
TRAIN_HORIZONS=4,8,12
case "${*:-}" in
  "") ;;
  --all) TRAIN_HORIZONS=1,4,8,12; PREFLIGHT_CMD="$PREFLIGHT_CMD --all" ;;
  *) echo "FATAL: only --all is supported" >&2; exit 2 ;;
esac
HORIZON_LIST="${TRAIN_HORIZONS//,/ }"
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

TRAIN_ROOT="checkpoints/main2025_v2_hremaining_train"
TRAIN_OUTPUT="$TRAIN_ROOT/finetuned"
RELEASE_ROOT="checkpoints/main2025_v2_hremaining"

echo "=== DeXposure-FM v2 remaining checkpoints (h=$TRAIN_HORIZONS; 283 snapshots; holdout=2025-01-01; val=24; epochs=20; seed=42) ==="
"$PY" cloud/train_src/run_full_experiment.py \
  --mode dexposure-fm --holdout-start 2025-01-01 \
  --val-weeks 24 --epochs 20 --seed 42 --horizons "$TRAIN_HORIZONS" \
  --output-dir "$TRAIN_ROOT"

SCHEMA_SOURCE="$TRAIN_OUTPUT/feature_schema.json"
METRICS_SOURCE="$TRAIN_OUTPUT/metrics.json"
test -s "$SCHEMA_SOURCE" || { echo "FATAL: $SCHEMA_SOURCE not found or empty" >&2; exit 1; }
test -s "$METRICS_SOURCE" || { echo "FATAL: $METRICS_SOURCE not found or empty" >&2; exit 1; }

mkdir -p "$RELEASE_ROOT"
for h in $HORIZON_LIST; do
  model_source="$TRAIN_OUTPUT/best_model_h${h}.pt"
  test -s "$model_source" || { echo "FATAL: $model_source not found or empty" >&2; exit 1; }
  cp "$model_source" "$RELEASE_ROOT/dexposure-fm-h${h}.pt"
done
cp "$SCHEMA_SOURCE" "$RELEASE_ROOT/feature_schema.json"
cp "$METRICS_SOURCE" "$RELEASE_ROOT/task1_metrics.json"

GPU_NAME="$GPU_NAME" GPU_MEMORY_MIB="$GPU_MEMORY_MIB" \
  INPUT_REVISION="${HF_V2_INPUT_REVISION:-cd8fcf4264054b58109e8a50c35cd69f6b2ff72a}" \
  TRAIN_HORIZONS="$TRAIN_HORIZONS" RELEASE_ROOT="$RELEASE_ROOT" METRICS_SOURCE="$METRICS_SOURCE" "$PY" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

metrics_path = Path(os.environ["METRICS_SOURCE"])
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
horizons = tuple(map(int, os.environ["TRAIN_HORIZONS"].split(",")))
missing_metrics = {f"h{h}" for h in horizons} - set(metrics)
if missing_metrics:
    raise SystemExit(f"FATAL: Task I metrics missing horizons: {sorted(missing_metrics)}")

release_root = Path(os.environ["RELEASE_ROOT"])
common = {
    "artifact_status": "complete",
    "holdout_start": "2025-01-01",
    "validation_weeks": 24,
    "epochs": 20,
    "seed": 42,
    "network_snapshots": 283,
    "source_sha256": os.environ["PREFLIGHT_SOURCE_SHA256"],
    "input_revision": os.environ["INPUT_REVISION"],
    "gpu_name": os.environ["GPU_NAME"],
    "gpu_memory_mib": int(os.environ["GPU_MEMORY_MIB"]),
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
}
for horizon in horizons:
    payload = {**common, "forecast_horizon_weeks": horizon}
    path = release_root / f"run_config_h{horizon}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

(
  cd "$RELEASE_ROOT"
  files=()
  for h in $HORIZON_LIST; do files+=("dexposure-fm-h${h}.pt" "run_config_h${h}.json"); done
  shasum -a 256 "${files[@]}" feature_schema.json task1_metrics.json > SHA256SUMS
  shasum -a 256 -c SHA256SUMS
  expected_files=$'SHA256SUMS\ndexposure-fm-h12.pt\ndexposure-fm-h4.pt\ndexposure-fm-h8.pt\nfeature_schema.json\nrun_config_h12.json\nrun_config_h4.json\nrun_config_h8.json\ntask1_metrics.json'
  if [ "$TRAIN_HORIZONS" = 1,4,8,12 ]; then
    expected_files="$(printf '%s\n' "$expected_files" dexposure-fm-h1.pt run_config_h1.json | LC_ALL=C sort)"
  fi
  actual_files="$(find . -mindepth 1 -maxdepth 1 -type f -print | sed 's#^./##' | LC_ALL=C sort)"
  [ "$actual_files" = "$expected_files" ] || {
    echo "FATAL: release directory contains missing or unexpected files" >&2
    printf 'Expected:\n%s\nActual:\n%s\n' "$expected_files" "$actual_files" >&2
    exit 1
  }
)

echo "MAIN 2025 V2 CHECKPOINT RUN DONE: h=$TRAIN_HORIZONS"
