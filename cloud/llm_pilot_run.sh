#!/usr/bin/env bash
# Run one paired M6/M7 LLM pilot against the pre-2022 FM checkpoint.
# Invoked on Vast through cloud/train.sh; the wrapper uploads logs/results.
set -euo pipefail

ROOT="$PWD"
PY="$ROOT/.venv/bin/python"
export PATH="$HOME/.local/bin:$PATH"
if [ ! -x "$PY" ]; then
  (cd "$ROOT" && uv sync --frozen)
fi
test -x "$PY" || { echo "FATAL: venv bootstrap failed" >&2; exit 1; }

"$PY" cloud/preflight.py --phase host \
  --train-cmd "bash cloud/llm_pilot_run.sh"

export DEXPOSURE_FM_PI_MIN="$PREFLIGHT_PI_MIN"
export DEXPOSURE_AUTHORITATIVE=1
export DEXPOSURE_DATA_PATH="$ROOT/data/historical-network_week_2020-03-30.json"
export DEXPOSURE_META_PATH="$ROOT/data/meta_df.csv"

export DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1
export PYTHONPATH="$ROOT:$ROOT/paper:$ROOT/cloud/train_src"
export LD_LIBRARY_PATH="$(echo "$ROOT"/.venv/lib/python3.12/site-packages/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH:-}"

test -n "${HF_TOKEN:-}" || { echo "FATAL: HF_TOKEN is required" >&2; exit 1; }
test -n "${OPENROUTER_API_KEY:-}" || {
  echo "FATAL: OPENROUTER_API_KEY is required for a real LLM pilot" >&2
  exit 1
}

REMOTE_RUN="runs/20260805_225404/checkpoints/pre2022"
PILOT_CKPT_DIR="$ROOT/checkpoints/pilot_pre2022"
mkdir -p "$PILOT_CKPT_DIR" "$ROOT/logs/llm_pilot"

copy_partial_results() {
  for result_dir in "$ROOT"/paper/results/llm_eval_*; do
    test -d "$result_dir" || continue
    cp -r "$result_dir" "$ROOT/logs/llm_pilot/"
  done
}

echo "=== [1/4] Download and verify pre-2022 FM checkpoints ==="
REMOTE_RUN="$REMOTE_RUN" PILOT_CKPT_DIR="$PILOT_CKPT_DIR" HF_TOKEN="$HF_TOKEN" \
  "$PY" - <<'PY'
import hashlib
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

repo = "losdwind/graph-dexposure-ckpt"
remote_run = os.environ["REMOTE_RUN"]
target = Path(os.environ["PILOT_CKPT_DIR"])
files = (
    "dexposure-fm-h4.pt",
    "feature_schema.json",
)
for name in files:
    cached = hf_hub_download(
        repo_id=repo,
        filename=f"{remote_run}/{name}",
        repo_type="model",
        token=os.environ["HF_TOKEN"],
    )
    destination = target / name
    destination.write_bytes(Path(cached).read_bytes())
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(f"{name}: {destination.stat().st_size} bytes sha256={digest}")

h4 = target / "dexposure-fm-h4.pt"
schema = target / "feature_schema.json"
if not h4.is_file() or h4.stat().st_size == 0:
    raise SystemExit("FATAL: downloaded h=4 checkpoint is missing or empty")
if not schema.is_file() or schema.stat().st_size == 0:
    raise SystemExit("FATAL: downloaded feature schema is missing or empty")
PY

echo "=== [2/4] Start FM API ==="
export DEXPOSURE_FM_CKPT_DIR="$PILOT_CKPT_DIR"
export DEXPOSURE_DATA_DIR="$ROOT/data"
cd "$ROOT/paper"
"$PY" -m uvicorn dexposure_agent.serve:app --host 127.0.0.1 --port 8000 \
  > "$ROOT/logs/llm_pilot/serve.log" 2>&1 &
SERVE_PID=$!
trap 'copy_partial_results; kill "$SERVE_PID" 2>/dev/null || true' EXIT
ok=0
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then ok=1; break; fi
  sleep 5
done
test "$ok" = 1 || { echo "FATAL: FM API failed to start"; tail -100 "$ROOT/logs/llm_pilot/serve.log"; exit 1; }
curl -fsS http://127.0.0.1:8000/health; echo

echo "=== [3/4] Run one actual M6 decision stream ==="
# One crisis window is the cost-controlled pilot. Three consistency calls
# freeze the decision budget while --no-judge isolates decision quality/FIR.
# M7 is deliberately not sampled independently. The strict replay below
# derives the gate condition from these exact primary tickets, so any M6/M7
# difference is attributable to the deterministic gate rather than LLM drift.
OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  "$PY" experiments/llm_eval_b5.py \
    --method m6_fm_llm \
    --test-split 2022-04~2022-07 \
    --consistency-runs 3 --no-judge

RUN_DIR="$(find "$ROOT/paper/results" -maxdepth 1 -type d -name 'llm_eval_*' -print | sort | tail -1)"
test -n "$RUN_DIR" && test -f "$RUN_DIR/comparison.json" || {
  echo "FATAL: LLM result directory not found" >&2
  exit 1
}
cp -r "$RUN_DIR" "$ROOT/logs/llm_pilot/"

echo "=== [4/4] Replay fixed tickets across budgets ==="
RAW="$RUN_DIR/raw_m6_fm_llm.json"
SUMMARY="$RUN_DIR/summary_m6_fm_llm.json"
REPLAY_OUT="$ROOT/logs/llm_pilot/strict_replay"
"$PY" scripts/replay_coverage_intervention_frontier.py \
  --raw "$RAW" --summary "$SUMMARY" --output-dir "$REPLAY_OUT" \
  --budgets 1,3,5,7 --bootstrap 20000 --seed 20260806

cd "$ROOT"
echo "LLM PILOT RUN DONE: $(basename "$RUN_DIR")"
