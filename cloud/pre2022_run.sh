#!/usr/bin/env bash
# Exploratory pre-Terra FM retrain, crisis-window backtest, and prompt export.
# Runs on Vast through cloud/train.sh. It is separate from the main 2025 h=4
# causal-harness run.
set -euo pipefail

ROOT="$PWD"
PY="$ROOT/.venv/bin/python"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
if [ ! -x "$PY" ]; then
  (cd "$ROOT" && uv sync --frozen)
fi
test -x "$PY" || { echo "FATAL: venv bootstrap failed" >&2; exit 1; }

MODE="crisis_h4"
case "${1:-}" in
  "") ;;
  --rq1-all-horizons)
    MODE="rq1"
    [ "$#" -eq 1 ] || { echo "FATAL: RQ1 binding accepts no extra arguments" >&2; exit 2; }
    ;;
  *)
    echo "usage: cloud/pre2022_run.sh [--rq1-all-horizons]" >&2
    exit 2
    ;;
esac

case "$MODE" in
  crisis_h4) PREFLIGHT_CMD="bash cloud/pre2022_run.sh" ;;
  rq1) PREFLIGHT_CMD="bash cloud/pre2022_run.sh --rq1-all-horizons" ;;
esac
"$PY" cloud/preflight.py --phase host --train-cmd "$PREFLIGHT_CMD"

# The crisis evaluator and FM API must use the same authoritative source,
# metadata, and edge threshold recorded by the cloud preflight.
export DEXPOSURE_FM_PI_MIN="$PREFLIGHT_PI_MIN"
export DEXPOSURE_AUTHORITATIVE=1
export DEXPOSURE_DATA_PATH="$ROOT/data/historical-network_week_2020-03-30.json"
export DEXPOSURE_META_PATH="$ROOT/data/meta_df.csv"

export DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1
export PYTHONPATH="$ROOT:$ROOT/paper:$ROOT/cloud/train_src"
export LD_LIBRARY_PATH="$(echo "$ROOT"/.venv/lib/python3.12/site-packages/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH:-}"

if [ "$MODE" = "rq1" ]; then
  TRAIN_HORIZONS="1,4,8,12"
  STAGE_HORIZONS="1 4 8 12"
else
  TRAIN_HORIZONS="4"
  STAGE_HORIZONS="4"
fi

echo "=== [1/4] Train pre-Terra FM (holdout >= 2022-04-01, horizons=$TRAIN_HORIZONS) ==="
"$PY" cloud/train_src/run_full_experiment.py \
  --mode dexposure-fm --holdout-start 2022-04-01 \
  --val-weeks 24 --epochs 20 --seed 42 --horizons "$TRAIN_HORIZONS" \
  --output-dir checkpoints/pre2022_train

echo "=== [2/4] Stage checkpoints in fm_predictor layout ==="
mkdir -p checkpoints/pre2022
for h in $STAGE_HORIZONS; do
  src="checkpoints/pre2022_train/finetuned/best_model_h${h}.pt"
  test -s "$src" || { echo "FATAL: $src not found or empty" >&2; exit 1; }
  case "$h" in
    8) cp "$src" checkpoints/pre2022/dexposure-fm-h8-h12.pt ;;
    12) cp "$src" checkpoints/pre2022/dexposure-fm-h12.pt ;;
    *) cp "$src" "checkpoints/pre2022/dexposure-fm-h${h}.pt" ;;
  esac
done
schema_src="checkpoints/pre2022_train/finetuned/feature_schema.json"
test -s "$schema_src" || { echo "FATAL: $schema_src not found or empty" >&2; exit 1; }
cp "$schema_src" checkpoints/pre2022/feature_schema.json
ls -lh checkpoints/pre2022/

# Push weights before the backtest/export stages when a token is available.
# The Vast.ai remote runner still performs the required final upload. The local
# controller downloads and verifies the artifacts before it destroys the instance.
if [ -n "${HF_TOKEN:-}" ]; then
  EARLY_TAG="pre2022_$(date +%Y%m%d_%H%M%S)_weights"
  uvx --from 'huggingface_hub[cli]' hf upload losdwind/graph-dexposure-ckpt \
    checkpoints/pre2022 "runs/$EARLY_TAG/checkpoints_pre2022" --repo-type model \
    --commit-message "early weight push: $EARLY_TAG" \
    || echo "WARN: early HF push failed -- continuing" >&2
fi

echo "=== [3/4] Crisis backtest: m5_fm_rules vs m1_persistence_rules ==="
export DEXPOSURE_FM_CKPT_DIR="$ROOT/checkpoints/pre2022"
export DEXPOSURE_DATA_DIR="$ROOT/data"
cd "$ROOT/paper"
DEXPOSURE_HORIZONS="$TRAIN_HORIZONS" CRISIS_SPLIT_KEYS=terra_luna,ftx,svb \
  "$PY" scripts/run_fm_vs_persistence_crisis.py
mkdir -p "$ROOT/logs/crisis_backtest"
cp results/run_fm_vs_persistence_crisis/b1_crisis.json \
  "$ROOT/logs/crisis_backtest/b1_crisis.json"
cp results/run_fm_vs_persistence_crisis/b5_crisis.json \
  "$ROOT/logs/crisis_backtest/b5_crisis.json"

echo "=== [4/4] Export M6 evidence prompts for offline M7 replay ==="
"$PY" -m uvicorn dexposure_agent.serve:app --host 127.0.0.1 --port 8000 \
  > "$ROOT/logs/serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT
ok=0
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then ok=1; break; fi
  sleep 5
done
test "$ok" = 1 || { echo "FATAL: FM API failed to start"; tail -50 "$ROOT/logs/serve.log"; exit 1; }
curl -fsS http://127.0.0.1:8000/health; echo

for WIN in "2022-04~2022-07" "2022-10~2023-01" "2023-01~2023-05"; do
  "$PY" experiments/llm_eval_b5.py --export-prompts-only \
    --method m6_fm_llm --test-split "$WIN" --horizon 4
  RUN_DIR=$(find results -maxdepth 1 -type d -name 'llm_eval_*' -print | sort | tail -1)
  test -n "$RUN_DIR" && test -f "$RUN_DIR/prompts_m6_fm_llm.json" || {
    echo "FATAL: M6 prompt artifact not found for $WIN" >&2
    exit 1
  }
  PROMPT_SUFFIX="${WIN//\~/__}"
  cp "$RUN_DIR/prompts_m6_fm_llm.json" \
    "$ROOT/logs/crisis_backtest/prompts_m6_fm_llm_${PROMPT_SUFFIX}.json"
done

echo "=== Collect artifacts for upload ==="
cd "$ROOT"
mkdir -p logs/crisis_backtest
cp -r paper/results/run_fm_vs_persistence_crisis logs/crisis_backtest/
echo "PRE2022 CRISIS RUN DONE (horizons=$TRAIN_HORIZONS)"
