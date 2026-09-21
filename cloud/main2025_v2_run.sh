#!/usr/bin/env bash
# Reconstruct the DeXposure-FM v2 forecasting model family on the 2025 holdout.
set -euo pipefail

ROOT="$PWD"
export PATH="$HOME/.local/bin:$PATH"
PY="$ROOT/.venv/bin/python"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
if [ ! -x "$PY" ]; then
  (cd "$ROOT" && uv sync --frozen)
fi
test -x "$PY" || { echo "FATAL: venv bootstrap failed" >&2; exit 1; }

PREFLIGHT_CMD="bash cloud/main2025_v2_run.sh"
"$PY" cloud/preflight.py --phase host --train-cmd "$PREFLIGHT_CMD"

export DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1
export PYTHONPATH="$ROOT:$ROOT/paper:$ROOT/cloud/train_src"
export LD_LIBRARY_PATH="$(echo "$ROOT"/.venv/lib/python3.12/site-packages/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH:-}"

TRAIN_HORIZONS=1,4,8,12
TRAIN_ROOT="checkpoints/main2025_v2_train"
RELEASE_ROOT="checkpoints/main2025_v2"
echo "=== DeXposure-FM v2 reconstruction (h=1,4,8,12, holdout=2025-01-01, val=24, epochs=20, seed=42) ==="
"$PY" cloud/train_src/run_full_experiment.py \
  --mode all --holdout-start 2025-01-01 \
  --val-weeks 24 --epochs 20 --seed 42 --horizons "$TRAIN_HORIZONS" \
  --output-dir "$TRAIN_ROOT"

echo "=== Stage four horizon-specific checkpoints ==="
mkdir -p "$RELEASE_ROOT"
for h in 1 4 8 12; do
  src="$TRAIN_ROOT/finetuned/best_model_h${h}.pt"
  test -s "$src" || { echo "FATAL: $src not found or empty" >&2; exit 1; }
  cp "$src" "$RELEASE_ROOT/dexposure-fm-h${h}.pt"
done
schema_src="$TRAIN_ROOT/feature_schema.json"
test -s "$schema_src" || { echo "FATAL: $schema_src not found or empty" >&2; exit 1; }
cp "$schema_src" "$RELEASE_ROOT/feature_schema.json"
shasum -a 256 "$RELEASE_ROOT"/*.pt | tee "$RELEASE_ROOT/SHA256SUMS"
ls -lh "$RELEASE_ROOT"/

echo "=== Protect Task I checkpoints before downstream evaluation ==="
if [ -n "${HF_TOKEN:-}" ]; then
  EARLY_TAG="main2025_v2_$(date +%Y%m%d_%H%M%S)_task1"
  uvx --from 'huggingface_hub[cli]' hf upload losdwind/graph-dexposure-ckpt \
    "$RELEASE_ROOT" "runs/$EARLY_TAG/checkpoints_main2025_v2" --repo-type model \
    --commit-message "early v2 Task I checkpoint push: $EARLY_TAG"
  uvx --from 'huggingface_hub[cli]' hf upload losdwind/graph-dexposure-ckpt \
    "$TRAIN_ROOT" "runs/$EARLY_TAG/checkpoints_main2025_v2_train" --repo-type model \
    --commit-message "early v2 Task I result push: $EARLY_TAG"
fi

echo "=== Reuse Task I checkpoints for Task II forecast-then-measure evaluation ==="
mkdir -p output/model_cache
for h in 1 4 8 12; do
  cp "$TRAIN_ROOT/finetuned/best_model_h${h}.pt" "output/model_cache/dexposure_fm__h${h}.pt"
done
"$PY" archive/code/run_task2_model_based.py \
  --experiment all --epochs 20 --seed 42 \
  --forward-horizons "$TRAIN_HORIZONS" \
  --contagion-horizons "$TRAIN_HORIZONS" \
  --output-dir output/task2_model_based_v2

echo "=== Rebuild the v2 sector spillover figure from its dated snapshot ==="
"$PY" DeXposure_FM_V2/scripts/make_spillover_matrix_figure.py \
  --date 2025-06-30 \
  --data-path data/historical-network_week_2025-07-01.json \
  --meta-path data/meta_df.csv \
  --out output/task2_model_based_v2/figures/fig_spillover_matrix_example.pdf

find output/task2_model_based_v2 -type f -print0 | sort -z | xargs -0 shasum -a 256 \
  > output/task2_model_based_v2/SHA256SUMS

echo "MAIN 2025 V2 FULL REPRODUCTION RUN DONE"
