#!/usr/bin/env bash
# Reconstruct the arXiv v2 Task I and Task II experiments.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PY=(uv run python)
HORIZONS=1,4,8,12
TASK1_DIR=output/v2_task1
TASK2_DIR=output/v2_task2

test -s data/historical-network_week_2020-03-30.json || {
  echo "ERROR: missing main weekly-network dataset" >&2
  exit 1
}
test -s checkpoints/graphpfn-v1.ckpt || {
  echo "ERROR: missing GraphPFN base checkpoint" >&2
  exit 1
}

"${PY[@]}" run_full_experiment.py \
  --mode all \
  --holdout-start 2025-01-01 \
  --val-weeks 24 \
  --epochs 20 \
  --seed 42 \
  --horizons "$HORIZONS" \
  --output-dir "$TASK1_DIR"

mkdir -p output/model_cache
for h in 1 4 8 12; do
  checkpoint="$TASK1_DIR/finetuned/best_model_h${h}.pt"
  test -s "$checkpoint" || {
    echo "ERROR: missing reconstructed h=${h} checkpoint" >&2
    exit 1
  }
  cp "$checkpoint" "output/model_cache/dexposure_fm__h${h}.pt"
done

"${PY[@]}" run_task2_model_based.py \
  --experiment all \
  --epochs 20 \
  --seed 42 \
  --forward-horizons "$HORIZONS" \
  --contagion-horizons "$HORIZONS" \
  --output-dir "$TASK2_DIR"

"${PY[@]}" scripts/make_spillover_matrix_figure.py \
  --date 2025-06-30 \
  --data-path data/historical-network_week_2025-07-01.json \
  --meta-path data/meta_df.csv \
  --out "$TASK2_DIR/figures/fig_spillover_matrix_example.pdf"

find "$TASK1_DIR" "$TASK2_DIR" -type f -print0 \
  | sort -z \
  | xargs -0 shasum -a 256 > output/v2-SHA256SUMS

echo "Experiment sequence finished; compare outputs with the paper before claiming matching results."
echo "Task I: $TASK1_DIR"
echo "Task II: $TASK2_DIR"
echo "Hashes: output/v2-SHA256SUMS"
