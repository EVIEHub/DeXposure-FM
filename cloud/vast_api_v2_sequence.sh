#!/usr/bin/env bash
# Train and recover h=1,4,8,12 sequentially, one Vast instance at a time.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
for command in vastai jq uvx shasum curl python3; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "ERROR: $command is required for the v2 sequence." >&2
    exit 1
  }
done

API_TIMEOUT_SECONDS="${VAST_API_TIMEOUT_SECONDS:-120}"
TRANSFER_TIMEOUT_SECONDS="${VAST_TRANSFER_TIMEOUT_SECONDS:-7200}"
case "$API_TIMEOUT_SECONDS:$TRANSFER_TIMEOUT_SECONDS" in
  *[!0-9:]*) echo "ERROR: sequence network timeouts must be whole seconds." >&2; exit 1 ;;
esac
[ "$API_TIMEOUT_SECONDS" -gt 0 ] && [ "$TRANSFER_TIMEOUT_SECONDS" -gt 0 ] || {
  echo "ERROR: sequence network timeouts must be positive." >&2
  exit 1
}

run_bounded() {
  local seconds="${1:?timeout is required}"
  shift
  python3 - "$seconds" "$@" <<'PY'
import os
import signal
import subprocess
import sys

process = subprocess.Popen(sys.argv[2:], start_new_session=True)

def stop_group() -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

def relay(signum: int, _frame: object) -> None:
    stop_group()
    raise SystemExit(128 + signum)

signal.signal(signal.SIGINT, relay)
signal.signal(signal.SIGTERM, relay)
try:
    status = process.wait(timeout=int(sys.argv[1]))
except subprocess.TimeoutExpired:
    stop_group()
    raise SystemExit(124)
raise SystemExit(status if status >= 0 else 128 - status)
PY
}

LOCK_DIR="/tmp/dexposure-v2-vast-sequence.lock"
LOCK_OWNER="$$:$(date +%s):$RANDOM"
LOCK_PARENT_START="$(ps -o lstart= -p "$$" | sed -E 's/^[[:space:]]+//')"
while ! mkdir "$LOCK_DIR" 2>/dev/null; do
  lock_pid="$(head -n1 "$LOCK_DIR/pid" 2>/dev/null || true)"
  lock_start="$(cat "$LOCK_DIR/start" 2>/dev/null || true)"
  current_lock_start=""
  if [[ "$lock_pid" =~ ^[1-9][0-9]*$ ]]; then
    current_lock_start="$(ps -o lstart= -p "$lock_pid" 2>/dev/null | sed -E 's/^[[:space:]]+//' || true)"
  fi
  if [ -n "$lock_start" ] && [ "$current_lock_start" = "$lock_start" ]; then
    echo "ERROR: another v2 sequence process $lock_pid is active; refusing duplicate instances." >&2
    exit 1
  fi
  if [ ! -s "$LOCK_DIR/owner" ] || [ -z "$lock_start" ]; then
    lock_mtime="$(stat -c %Y "$LOCK_DIR" 2>/dev/null || stat -f %m "$LOCK_DIR" 2>/dev/null || true)"
    lock_now="$(date +%s)"
    case "$lock_mtime" in
      *[!0-9]*|'') echo "ERROR: cannot verify ownership of the v2 sequence lock." >&2; exit 1 ;;
    esac
    if [ $((lock_now - lock_mtime)) -lt 30 ]; then
      sleep 1
      continue
    fi
  fi
  stale_lock="$LOCK_DIR.stale.$$.$RANDOM"
  if mv "$LOCK_DIR" "$stale_lock" 2>/dev/null; then
    rm -rf "$stale_lock"
  fi
done
printf '%s\n' "$$" >"$LOCK_DIR/pid"
printf '%s\n' "$LOCK_PARENT_START" >"$LOCK_DIR/start"
printf '%s\n' "$LOCK_OWNER" >"$LOCK_DIR/owner"
cleanup_sequence_lock() {
  if [ "$(cat "$LOCK_DIR/owner" 2>/dev/null || true)" = "$LOCK_OWNER" ]; then
    rm -f "$LOCK_DIR/pid" "$LOCK_DIR/start" "$LOCK_DIR/owner"
    rmdir "$LOCK_DIR" 2>/dev/null || true
  fi
}
handle_sequence_signal() {
  trap - INT TERM
  trap - EXIT
  cleanup_sequence_lock
  exit 130
}
trap cleanup_sequence_lock EXIT
trap handle_sequence_signal INT TERM

SEQUENCE_TAG="${V2_SEQUENCE_TAG:-$(date -u +%Y%m%d_%H%M%S)_v2}"
case "$SEQUENCE_TAG" in
  *[!A-Za-z0-9._-]*) echo "ERROR: V2_SEQUENCE_TAG contains unsafe characters." >&2; exit 1 ;;
esac
SEQUENCE_ROOT="$ROOT_DIR/outputs/v2_checkpoint_rerun/$SEQUENCE_TAG"
mkdir -p "$SEQUENCE_ROOT/attempts" "$SEQUENCE_ROOT/controller_logs" \
  "$SEQUENCE_ROOT/verified"

SOURCE_SHA="$(python3 cloud/preflight.py --print-source-digest)"
SEQUENCE_SOURCE_FILE="$SEQUENCE_ROOT/source_sha256.txt"
if [ -s "$SEQUENCE_SOURCE_FILE" ] && [ "$(awk 'NR == 1 {print; exit}' "$SEQUENCE_SOURCE_FILE")" != "$SOURCE_SHA" ]; then
  echo "ERROR: sequence $SEQUENCE_TAG belongs to a different source digest; refusing mixed checkpoints." >&2
  exit 1
fi
printf '%s\n' "$SOURCE_SHA" >"$SEQUENCE_SOURCE_FILE"
ATTESTATION_DIR="$ROOT_DIR/tmp/v2_source_attestations"
mkdir -p "$ATTESTATION_DIR"
ATTESTATION_FILE="$ATTESTATION_DIR/$SOURCE_SHA.txt"
printf '%s\n' "$SOURCE_SHA" >"$ATTESTATION_FILE"
ATTESTATION_SHA="$(shasum -a 256 "$ATTESTATION_FILE" | awk '{print $1}')"
export PREFLIGHT_SOURCE_ATTESTATION="$ATTESTATION_FILE"
export PREFLIGHT_SOURCE_ATTESTATION_SHA256="$ATTESTATION_SHA"

echo "V2_SEQUENCE_TAG=$SEQUENCE_TAG"
echo "V2_SOURCE_SHA256=$SOURCE_SHA"
echo "The sequence advances only after each checkpoint is downloaded, verified, and its instance is destroyed."

write_verified_receipt() {
  local path="$1" run_tag="$2" horizon="$3" source_sha="$4"
  local model_sha="$5" credit_before="$6" vast_cost="$7" credit_after="$8"
  local instance_id="$9" destroyed="${10}"
  local temp="${path}.tmp.$$"
  printf 'VERIFIED=1\nDESTROYED=%s\nINSTANCE_ID=%s\nRUN_TAG=%s\nHORIZON=%s\nSOURCE_SHA256=%s\nMODEL_SHA256=%s\nCREDIT_BEFORE=%s\nVAST_COST=%s\nCREDIT_AFTER=%s\n' \
    "$destroyed" "$instance_id" "$run_tag" "$horizon" "$source_sha" \
    "$model_sha" "$credit_before" "$vast_cost" "$credit_after" >"$temp"
  mv "$temp" "$path"
}

read_receipt_field() {
  local path="$1" key="$2"
  awk -F= -v key="$key" '
    $1 == key { count++; value = substr($0, index($0, "=") + 1) }
    END { if (count == 1) print value; else exit 1 }
  ' "$path"
}

validate_verified_receipt() {
  local path="$1" expected_run_tag="$2" expected_horizon="$3"
  local expected_source="$4" expected_credit_before="$5"
  local expected_instance_id="$6" require_destroyed="$7"

  awk -F= '
    NF != 2 { exit 1 }
    $1 !~ /^(VERIFIED|DESTROYED|INSTANCE_ID|RUN_TAG|HORIZON|SOURCE_SHA256|MODEL_SHA256|CREDIT_BEFORE|VAST_COST|CREDIT_AFTER)$/ { exit 1 }
  ' "$path" || {
    echo "ERROR: h=$expected_horizon verified receipt contains malformed or unknown fields." >&2
    return 1
  }
  verified_status="$(read_receipt_field "$path" VERIFIED)" || return 1
  verified_destroyed="$(read_receipt_field "$path" DESTROYED)" || return 1
  verified_instance_id="$(read_receipt_field "$path" INSTANCE_ID)" || return 1
  verified_run_tag="$(read_receipt_field "$path" RUN_TAG)" || return 1
  verified_horizon="$(read_receipt_field "$path" HORIZON)" || return 1
  verified_source="$(read_receipt_field "$path" SOURCE_SHA256)" || return 1
  verified_model_sha="$(read_receipt_field "$path" MODEL_SHA256)" || return 1
  verified_credit_before="$(read_receipt_field "$path" CREDIT_BEFORE)" || return 1
  verified_cost="$(read_receipt_field "$path" VAST_COST)" || return 1
  verified_credit_after="$(read_receipt_field "$path" CREDIT_AFTER)" || return 1

  [ "$verified_status" = 1 ] && [ "$verified_run_tag" = "$expected_run_tag" ] && \
    [ "$verified_horizon" = "$expected_horizon" ] && \
    [ "$verified_source" = "$expected_source" ] || return 1
  case "$verified_destroyed" in
    0|1) ;;
    *) return 1 ;;
  esac
  [ "$require_destroyed" != 1 ] || [ "$verified_destroyed" = 1 ] || return 1
  [[ "$verified_instance_id" =~ ^[1-9][0-9]*$ ]] || return 1
  [ -z "$expected_instance_id" ] || \
    [ "$verified_instance_id" = "$expected_instance_id" ] || return 1
  [[ "$verified_model_sha" =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ "$verified_credit_before" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || return 1
  [ -z "$expected_credit_before" ] || \
    [ "$verified_credit_before" = "$expected_credit_before" ] || return 1
  if [ "$verified_cost" = pending ]; then
    [ "$verified_credit_after" = pending ] || return 1
  else
    [[ "$verified_cost" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || return 1
    [[ "$verified_credit_after" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || return 1
    python3 - "$verified_cost" <<'PY'
from decimal import Decimal
import sys

raise SystemExit(0 if Decimal(sys.argv[1]) >= 0 else 1)
PY
  fi
}

write_attempt() {
  local path="$1" state="$2" run_tag="$3" horizon="$4" source_sha="$5"
  local instance_id="$6" credit_before="$7"
  local temp="${path}.tmp.$$"
  printf 'STATE=%s\nRUN_TAG=%s\nHORIZON=%s\nSOURCE_SHA256=%s\nINSTANCE_ID=%s\nCREDIT_BEFORE=%s\n' \
    "$state" "$run_tag" "$horizon" "$source_sha" \
    "$instance_id" "$credit_before" >"$temp"
  mv "$temp" "$path"
}

read_attempt_field() {
  local path="$1" key="$2"
  awk -F= -v key="$key" '
    $1 == key { count++; value = substr($0, index($0, "=") + 1) }
    END { if (count == 1) print value; else exit 1 }
  ' "$path"
}

validate_attempt() {
  local path="$1" expected_horizon="$2"
  awk -F= '
    NF != 2 { exit 1 }
    $1 !~ /^(STATE|RUN_TAG|HORIZON|SOURCE_SHA256|INSTANCE_ID|CREDIT_BEFORE)$/ { exit 1 }
  ' "$path" || {
    echo "ERROR: h=$expected_horizon attempt contains malformed or unknown fields." >&2
    return 1
  }
  attempt_state="$(read_attempt_field "$path" STATE)" || return 1
  attempt_run_tag="$(read_attempt_field "$path" RUN_TAG)" || return 1
  attempt_horizon="$(read_attempt_field "$path" HORIZON)" || return 1
  attempt_source="$(read_attempt_field "$path" SOURCE_SHA256)" || return 1
  attempt_instance_id="$(read_attempt_field "$path" INSTANCE_ID)" || return 1
  attempt_credit_before="$(read_attempt_field "$path" CREDIT_BEFORE)" || return 1
  case "$attempt_state" in
    pending|active|complete) ;;
    *) echo "ERROR: h=$expected_horizon attempt has an invalid state." >&2; return 1 ;;
  esac
  [ "$attempt_horizon" = "$expected_horizon" ] || {
    echo "ERROR: h=$expected_horizon attempt records horizon $attempt_horizon." >&2
    return 1
  }
  [ "$attempt_source" = "$SOURCE_SHA" ] || {
    echo "ERROR: h=$expected_horizon attempt belongs to a different source digest." >&2
    return 1
  }
  case "$attempt_run_tag" in
    "${SEQUENCE_TAG}_h${expected_horizon}_"*) ;;
    *) echo "ERROR: h=$expected_horizon attempt has a run tag outside this sequence." >&2; return 1 ;;
  esac
  case "$attempt_run_tag" in
    *[!A-Za-z0-9._-]*) echo "ERROR: h=$expected_horizon attempt has an unsafe run tag." >&2; return 1 ;;
  esac
  [[ "$attempt_credit_before" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || {
    echo "ERROR: h=$expected_horizon attempt has an invalid credit value." >&2
    return 1
  }
  if [ "$attempt_state" = pending ]; then
    [ "$attempt_instance_id" = pending ] || {
      echo "ERROR: h=$expected_horizon pending attempt has an invalid instance ID." >&2
      return 1
    }
  else
    [[ "$attempt_instance_id" =~ ^[1-9][0-9]*$ ]] || {
      echo "ERROR: h=$expected_horizon attempt has no numeric instance ID." >&2
      return 1
    }
  fi
}

read_vast_instances() {
  local response
  response="$(run_bounded "$API_TIMEOUT_SECONDS" vastai show instances-v1 --all --raw)" || return 1
  jq -e '.success == true and (.instances | type == "array")' \
    >/dev/null <<<"$response" || return 1
  jq -c '.instances' <<<"$response"
}

BILLING_SETTLE_ATTEMPTS="${VAST_BILLING_SETTLE_ATTEMPTS:-20}"
BILLING_SETTLE_INTERVAL_SECONDS="${VAST_BILLING_SETTLE_INTERVAL_SECONDS:-30}"
case "$BILLING_SETTLE_ATTEMPTS:$BILLING_SETTLE_INTERVAL_SECONDS" in
  *[!0-9:]*) echo "ERROR: Vast billing-settlement limits must be whole numbers." >&2; exit 1 ;;
esac
[ "$BILLING_SETTLE_ATTEMPTS" -gt 0 ] && [ "$BILLING_SETTLE_INTERVAL_SECONDS" -gt 0 ] || {
  echo "ERROR: Vast billing-settlement limits must be positive." >&2
  exit 1
}
PENDING_RECONCILE_SAMPLES="${VAST_PENDING_RECONCILE_SAMPLES:-3}"
PENDING_RECONCILE_INTERVAL_SECONDS="${VAST_PENDING_RECONCILE_INTERVAL_SECONDS:-5}"
case "$PENDING_RECONCILE_SAMPLES:$PENDING_RECONCILE_INTERVAL_SECONDS" in
  *[!0-9:]*) echo "ERROR: pending-reconciliation limits must be whole numbers." >&2; exit 1 ;;
esac
[ "$PENDING_RECONCILE_SAMPLES" -ge 3 ] && \
  [ "$PENDING_RECONCILE_INTERVAL_SECONDS" -gt 0 ] || {
  echo "ERROR: pending reconciliation requires at least three positive-interval samples." >&2
  exit 1
}
SETTLED_COST=""
SETTLED_CREDIT_AFTER=""
settle_vast_cost() {
  local instance_id="$1" run_tag="$2" settle_attempt="" payload="" user_payload="" observed_credit="" observed_cost=""
  local expected_source="instance-$instance_id" expected_label="dexposure-v2-$run_tag"
  for settle_attempt in $(seq 1 "$BILLING_SETTLE_ATTEMPTS"); do
    payload="$(run_bounded "$API_TIMEOUT_SECONDS" vastai show invoices-v1 --charges --charge-type instance \
      --latest-first --limit 100 --raw)" || return 1
    jq -e '.success == true and (.results | type == "array")' \
      >/dev/null <<<"$payload" || return 1
    if jq -e --arg source "$expected_source" --arg label "$expected_label" \
        '.results[] | select(.source == $source and (.metadata.label // "") != $label)' \
        >/dev/null <<<"$payload"; then
      echo "ERROR: Vast billing source $expected_source has a different instance label." >&2
      return 1
    fi
    observed_cost="$(jq -er --arg source "$expected_source" --arg label "$expected_label" \
      '[.results[] | select(.source == $source and (.metadata.label // "") == $label) | .amount] | add // 0' \
      <<<"$payload")" || return 1
    if python3 - "$observed_cost" <<'PY'
from decimal import Decimal
import sys

raise SystemExit(0 if Decimal(sys.argv[1]) > 0 else 1)
PY
    then
      user_payload="$(run_bounded "$API_TIMEOUT_SECONDS" vastai show user --raw)" || return 1
      observed_credit="$(jq -er '.credit | numbers' <<<"$user_payload")" || return 1
      SETTLED_COST="$observed_cost"
      SETTLED_CREDIT_AFTER="$observed_credit"
      return 0
    fi
    echo "Vast billing is still pending after query $settle_attempt/$BILLING_SETTLE_ATTEMPTS; no next instance may start."
    [ "$settle_attempt" -eq "$BILLING_SETTLE_ATTEMPTS" ] || \
      sleep "$BILLING_SETTLE_INTERVAL_SECONDS"
  done
  return 1
}

probe_hf_run_prefix() {
  local run_tag="$1" token="${HF_TOKEN:-}"
  if [ -z "$token" ] && [ -f "$ROOT_DIR/.env" ]; then
    token="$(awk -F= '$1 == "HF_TOKEN" {sub(/^[^=]*=/, ""); print; exit}' "$ROOT_DIR/.env")"
  fi
  token="${token#\"}"
  token="${token%\"}"
  token="${token#\'}"
  token="${token%\'}"
  if [ -z "$token" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    token="$(<"$HOME/.cache/huggingface/token")"
  fi
  [ -n "$token" ] || {
    echo "ERROR: HF_TOKEN is required to reconcile a pending attempt." >&2
    return 1
  }
  run_bounded "$API_TIMEOUT_SECONDS" curl --connect-timeout 15 --max-time "$API_TIMEOUT_SECONDS" \
    -sS -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer $token" \
    "https://huggingface.co/api/models/losdwind/graph-dexposure-ckpt/tree/main/runs/$run_tag?recursive=true&expand=false"
}

validate_downloaded_run() {
  local run_root="$1" run_tag="$2" horizon="$3"
  local model_dir="$run_root/checkpoints/main2025_v2_h${horizon}"
  local train_dir="$run_root/checkpoints/main2025_v2_h${horizon}_train/finetuned"
  local log_dir="$run_root/logs" required_path
  for required_path in \
    "$model_dir/dexposure-fm-h${horizon}.pt" "$model_dir/feature_schema.json" \
    "$model_dir/task1_metrics.json" "$model_dir/run_config.json" \
    "$model_dir/SHA256SUMS" "$train_dir/best_model_h${horizon}.pt" \
    "$train_dir/feature_schema.json" "$train_dir/experiment_results.json" \
    "$train_dir/data_quality.json" "$train_dir/metrics.json" \
    "$train_dir/all_results.json" "$log_dir/run_identity.env" \
    "$log_dir/local_run_status.env" "$log_dir/train_${run_tag}.log" \
    "$log_dir/postflight_${run_tag}.log" "$log_dir/controller_${run_tag}.log" \
    "$log_dir/artifact_manifest_${run_tag}.sha256"; do
    [ -s "$required_path" ] || return 1
  done
  (
    cd "$run_root" && shasum -a 256 -c "logs/artifact_manifest_${run_tag}.sha256"
  ) >/dev/null 2>&1
}

destroy_receipted_instance() {
  local instance_id="$1" run_tag="$2"
  local expected_label="dexposure-v2-$run_tag" live_json observed_label
  live_json="$(read_vast_instances)" || return 1
  if ! jq -e --arg id "$instance_id" '.[] | select((.id | tostring) == $id)' \
      >/dev/null <<<"$live_json"; then
    echo "Confirmed receipted Vast instance id=$instance_id is already absent"
    return 0
  fi
  observed_label="$(jq -r --arg id "$instance_id" \
    '.[] | select((.id | tostring) == $id) | (.label // "")' <<<"$live_json")"
  [ "$observed_label" = "$expected_label" ] || {
    echo "ERROR: refusing to destroy instance id=$instance_id because label $observed_label does not equal $expected_label." >&2
    return 1
  }
  for _ in 1 2 3; do
    if run_bounded "$API_TIMEOUT_SECONDS" vastai destroy instance "$instance_id" -y >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
  for _ in $(seq 1 30); do
    live_json="$(read_vast_instances)" || { sleep 2; continue; }
    if ! jq -e --arg id "$instance_id" '.[] | select((.id | tostring) == $id)' \
         >/dev/null <<<"$live_json"; then
      echo "Confirmed receipted Vast instance id=$instance_id is absent"
      return 0
    fi
    sleep 2
  done
  return 1
}

PREVIOUS_VERIFIED_COST=""
for horizon in 1 4 8 12; do
  verified_file="$SEQUENCE_ROOT/verified/h${horizon}.env"
  attempt_file="$SEQUENCE_ROOT/attempts/h${horizon}.env"
  attempt_exists=0
  if [ -e "$attempt_file" ]; then
    validate_attempt "$attempt_file" "$horizon" || {
      echo "ERROR: refusing malformed h=$horizon attempt mapping $attempt_file." >&2
      exit 1
    }
    attempt_exists=1
  fi
  if [ -s "$verified_file" ]; then
    run_tag="$(read_receipt_field "$verified_file" RUN_TAG)" || {
      echo "ERROR: h=$horizon verified receipt has no unique run tag." >&2
      exit 1
    }
    expected_receipt_credit=""
    expected_receipt_instance=""
    if [ "$attempt_exists" = 1 ]; then
      expected_receipt_credit="$attempt_credit_before"
      expected_receipt_instance="$attempt_instance_id"
    fi
    validate_verified_receipt "$verified_file" "$run_tag" "$horizon" \
      "$SOURCE_SHA" "$expected_receipt_credit" "$expected_receipt_instance" 0 || {
      echo "ERROR: h=$horizon verified receipt failed exact field validation." >&2
      exit 1
    }
    credit_before="$verified_credit_before"
    case "$run_tag" in
      "${SEQUENCE_TAG}_h${horizon}_"*) ;;
      *) echo "ERROR: h=$horizon verified marker has a run tag outside this sequence." >&2; exit 1 ;;
    esac
    case "$run_tag" in
      *[!A-Za-z0-9._-]*) echo "ERROR: h=$horizon verified marker has an unsafe run tag." >&2; exit 1 ;;
    esac
    if [ "$attempt_exists" = 1 ]; then
      [ "$attempt_run_tag" = "$run_tag" ] && \
        [ "$attempt_instance_id" = "$verified_instance_id" ] && \
        [ "$attempt_credit_before" = "$credit_before" ] || {
          echo "ERROR: h=$horizon attempt and verified receipt identify different work." >&2
          exit 1
        }
    fi
    verified_run_root="$ROOT_DIR/outputs/v2_checkpoint_rerun/$run_tag/hf_download/runs/$run_tag"
    validate_downloaded_run "$verified_run_root" "$run_tag" "$horizon" || {
      echo "ERROR: h=$horizon downloaded run failed manifest or required-artifact validation; its instance was not destroyed." >&2
      exit 1
    }
    release_root="$verified_run_root/checkpoints/main2025_v2_h${horizon}"
    (
      cd "$release_root"
      shasum -a 256 -c SHA256SUMS
    )
    observed_model_sha="$(shasum -a 256 "$release_root/dexposure-fm-h${horizon}.pt" | awk '{print $1}')"
    test "$observed_model_sha" = "$verified_model_sha" || {
      echo "ERROR: h=$horizon local model no longer matches its verified receipt." >&2
      exit 1
    }
    PREVIOUS_VERIFIED_COST="$verified_cost"
    credit_after="$verified_credit_after"
    destroy_receipted_instance "$verified_instance_id" "$run_tag" || {
      echo "ERROR: h=$horizon checkpoint is verified, but Vast instance $verified_instance_id was not confirmed destroyed." >&2
      exit 1
    }
    if [ "$verified_destroyed" != 1 ]; then
      verified_destroyed=1
      write_verified_receipt "$verified_file" "$run_tag" "$horizon" "$SOURCE_SHA" \
        "$verified_model_sha" "$credit_before" "$PREVIOUS_VERIFIED_COST" "$credit_after" \
        "$verified_instance_id" "$verified_destroyed"
      validate_verified_receipt "$verified_file" "$run_tag" "$horizon" \
        "$SOURCE_SHA" "$credit_before" "$verified_instance_id" 1 || {
        echo "ERROR: h=$horizon reconciled receipt failed exact field validation." >&2
        exit 1
      }
    fi
    if ! settle_vast_cost "$verified_instance_id" "$run_tag"; then
      echo "ERROR: Vast has not published a positive exact-instance cost for verified h=$horizon; no new instance was created." >&2
      exit 1
    fi
    credit_after="$SETTLED_CREDIT_AFTER"
    PREVIOUS_VERIFIED_COST="$SETTLED_COST"
    write_verified_receipt "$verified_file" "$run_tag" "$horizon" "$SOURCE_SHA" \
      "$verified_model_sha" "$credit_before" "$PREVIOUS_VERIFIED_COST" "$credit_after" \
      "$verified_instance_id" 1
    validate_verified_receipt "$verified_file" "$run_tag" "$horizon" \
      "$SOURCE_SHA" "$credit_before" "$verified_instance_id" 1 || {
      echo "ERROR: h=$horizon settled receipt failed exact field validation." >&2
      exit 1
    }
    write_attempt "$attempt_file" complete "$run_tag" "$horizon" "$SOURCE_SHA" \
      "$verified_instance_id" "$credit_before"
    echo "Skipping already verified h=$horizon run=$run_tag cost=$PREVIOUS_VERIFIED_COST"
    continue
  fi
  available_credit=""
  attach_existing=0
  attach_instance_id=pending
  if [ "$attempt_exists" = 1 ]; then
    run_tag="$attempt_run_tag"
    credit_before="$attempt_credit_before"
    attach_existing=1
    attach_instance_id="$attempt_instance_id"
    echo "Resuming recorded Vast run $run_tag instance=$attach_instance_id"
  else
    run_tag="${SEQUENCE_TAG}_h${horizon}_$(date -u +%Y%m%dT%H%M%SZ)_p$$"
    user_payload="$(run_bounded "$API_TIMEOUT_SECONDS" vastai show user --raw)"
    available_credit="$(jq -er '.credit | numbers' <<<"$user_payload")"
    credit_before="$available_credit"
  fi
  if [ "$attempt_exists" = 1 ] && [ "$attempt_state" = pending ]; then
    pending_label="dexposure-v2-$run_tag"
    pending_clear_samples=0
    pending_reconciled_instance=0
    for pending_sample in $(seq 1 "$PENDING_RECONCILE_SAMPLES"); do
      pending_live_json="$(read_vast_instances)" || {
        echo "ERROR: cannot reconcile pending h=$horizon against the Vast instance list." >&2
        exit 1
      }
      pending_match_count="$(jq --arg label "$pending_label" \
        '[.[] | select((.label // "") == $label)] | length' <<<"$pending_live_json")"
      if [ "$pending_match_count" = 1 ]; then
        attach_instance_id="$(jq -r --arg label "$pending_label" \
          '.[] | select((.label // "") == $label) | .id' <<<"$pending_live_json")"
        [[ "$attach_instance_id" =~ ^[1-9][0-9]*$ ]] || {
          echo "ERROR: pending h=$horizon matched an instance with a malformed ID." >&2
          exit 1
        }
        attach_existing=1
        pending_reconciled_instance=1
        write_attempt "$attempt_file" active "$run_tag" "$horizon" "$SOURCE_SHA" \
          "$attach_instance_id" "$credit_before"
        echo "Reconciled pending Vast run $run_tag to instance=$attach_instance_id"
        break
      elif [ "$pending_match_count" = 0 ]; then
        hf_prefix_status="$(probe_hf_run_prefix "$run_tag")" || exit 1
        case "$hf_prefix_status" in
          404) pending_clear_samples=$((pending_clear_samples + 1)) ;;
          200)
            echo "ERROR: pending run $run_tag has a Hugging Face prefix but no recorded numeric instance ID; refusing a new instance." >&2
            exit 1
            ;;
          *)
            echo "ERROR: Hugging Face run-prefix check returned HTTP $hf_prefix_status for pending run $run_tag." >&2
            exit 1
            ;;
        esac
      else
        echo "ERROR: pending run $run_tag matches $pending_match_count Vast instances; refusing ambiguous attach." >&2
        exit 1
      fi
      if [ "$pending_sample" -lt "$PENDING_RECONCILE_SAMPLES" ]; then
        echo "Pending run $run_tag is clear in sample $pending_sample/$PENDING_RECONCILE_SAMPLES; waiting before another check."
        sleep "$PENDING_RECONCILE_INTERVAL_SECONDS"
      fi
    done
    if [ "$pending_reconciled_instance" = 0 ]; then
      [ "$pending_clear_samples" = "$PENDING_RECONCILE_SAMPLES" ] || {
        echo "ERROR: pending run $run_tag did not complete its stable no-instance reconciliation." >&2
        exit 1
      }
      attach_existing=0
      user_payload="$(run_bounded "$API_TIMEOUT_SECONDS" vastai show user --raw)"
      available_credit="$(jq -er '.credit | numbers' <<<"$user_payload")"
      credit_before="$available_credit"
      write_attempt "$attempt_file" pending "$run_tag" "$horizon" "$SOURCE_SHA" \
        pending "$credit_before"
      echo "Pending run $run_tag had no Vast instance or Hugging Face prefix in $PENDING_RECONCILE_SAMPLES stable samples; starting its first controller attempt"
    fi
  fi
  controller_log="$SEQUENCE_ROOT/controller_logs/${run_tag}.log"
  if [ "$attach_existing" = 0 ] && [ -n "$PREVIOUS_VERIFIED_COST" ]; then
    budget_ok="$(python3 - "$available_credit" "$PREVIOUS_VERIFIED_COST" <<'PY'
from decimal import Decimal
import sys

print(int(Decimal(sys.argv[1]) >= Decimal(sys.argv[2])))
PY
)"
    if [ "$budget_ok" != 1 ]; then
      echo "ERROR: remaining Vast credit $available_credit is below the last verified horizon cost $PREVIOUS_VERIFIED_COST; no new instance was created." >&2
      exit 1
    fi
  fi
  if [ "$attempt_exists" = 0 ]; then
    write_attempt "$attempt_file" pending "$run_tag" "$horizon" "$SOURCE_SHA" \
      "$attach_instance_id" "$credit_before"
  fi
  echo "Starting verified Vast run $run_tag"
  set +e
  VAST_RUN_TAG="$run_tag" VAST_EXPECTED_HORIZON="$horizon" \
    VAST_ATTACH_EXISTING="$attach_existing" \
    VAST_ATTACH_INSTANCE_ID="$attach_instance_id" VAST_ATTEMPT_FILE="$attempt_file" \
    VAST_SUCCESS_RECEIPT="$verified_file" VAST_CREDIT_BEFORE="$credit_before" \
    VAST_LOCAL_CONTROLLER_LOG="$controller_log" \
    bash cloud/train.sh bash cloud/main2025_v2_horizon_run.sh "$horizon" \
    2>&1 | tee -a "$controller_log"
  pipeline_status=("${PIPESTATUS[@]}")
  run_status=${pipeline_status[0]}
  tee_status=${pipeline_status[1]}
  set -e
  if [ "$run_status" -ne 0 ]; then
    if [ -s "$verified_file" ]; then
      echo "ERROR: h=$horizon controller returned $run_status after writing a verified receipt; rerun the same sequence tag to audit without retraining." >&2
    else
      echo "ERROR: h=$horizon failed with status $run_status; later horizons were not started." >&2
    fi
    exit "$run_status"
  fi
  test -s "$verified_file" || {
    echo "ERROR: h=$horizon controller returned success without a verified receipt." >&2
    exit 1
  }
  if [ "$tee_status" -ne 0 ]; then
    echo "WARNING: local tee failed with status $tee_status; the remote uploaded log remains authoritative." >&2
  fi
  expected_receipt_instance=""
  if [ "$attach_existing" = 1 ]; then
    expected_receipt_instance="$attach_instance_id"
  fi
  validate_verified_receipt "$verified_file" "$run_tag" "$horizon" \
    "$SOURCE_SHA" "$credit_before" "$expected_receipt_instance" 1 || {
    echo "ERROR: h=$horizon controller receipt failed exact final-field validation." >&2
    exit 1
  }
  verified_run_root="$ROOT_DIR/outputs/v2_checkpoint_rerun/$run_tag/hf_download/runs/$run_tag"
  validate_downloaded_run "$verified_run_root" "$run_tag" "$horizon" || {
    echo "ERROR: h=$horizon controller returned success without a valid local downloaded bundle." >&2
    exit 1
  }
  release_root="$verified_run_root/checkpoints/main2025_v2_h${horizon}"
  (
    cd "$release_root"
    shasum -a 256 -c SHA256SUMS
  ) || {
    echo "ERROR: h=$horizon local release hashes failed after the controller returned." >&2
    exit 1
  }
  observed_model_sha="$(shasum -a 256 "$release_root/dexposure-fm-h${horizon}.pt" | awk '{print $1}')"
  [ "$observed_model_sha" = "$verified_model_sha" ] || {
    echo "ERROR: h=$horizon local model differs from the controller receipt." >&2
    exit 1
  }
  destroy_receipted_instance "$verified_instance_id" "$run_tag" || {
    echo "ERROR: h=$horizon controller returned success, but instance $verified_instance_id is still present." >&2
    exit 1
  }
  if ! settle_vast_cost "$verified_instance_id" "$run_tag"; then
    write_verified_receipt "$verified_file" "$run_tag" "$horizon" "$SOURCE_SHA" \
      "$verified_model_sha" "$credit_before" pending pending \
      "$verified_instance_id" 1
    validate_verified_receipt "$verified_file" "$run_tag" "$horizon" \
      "$SOURCE_SHA" "$credit_before" "$verified_instance_id" 1 || {
      echo "ERROR: h=$horizon pending-billing receipt failed exact field validation." >&2
      exit 1
    }
    write_attempt "$attempt_file" complete "$run_tag" "$horizon" "$SOURCE_SHA" \
      "$verified_instance_id" "$credit_before"
    echo "ERROR: h=$horizon is verified and its instance is destroyed, but Vast billing has not published a positive cost. Later horizons were not started." >&2
    exit 1
  fi
  credit_after="$SETTLED_CREDIT_AFTER"
  PREVIOUS_VERIFIED_COST="$SETTLED_COST"
  write_verified_receipt "$verified_file" "$run_tag" "$horizon" "$SOURCE_SHA" \
    "$verified_model_sha" "$credit_before" "$PREVIOUS_VERIFIED_COST" "$credit_after" \
    "$verified_instance_id" 1
  validate_verified_receipt "$verified_file" "$run_tag" "$horizon" \
    "$SOURCE_SHA" "$credit_before" "$verified_instance_id" 1 || {
    echo "ERROR: h=$horizon settled receipt failed exact field validation." >&2
    exit 1
  }
  write_attempt "$attempt_file" complete "$run_tag" "$horizon" "$SOURCE_SHA" \
    "$verified_instance_id" "$credit_before"
  echo "Verified h=$horizon Vast cost=$PREVIOUS_VERIFIED_COST remaining_credit=$credit_after"
done

FINAL_ROOT="$SEQUENCE_ROOT/final"
MODEL_ROOT="$FINAL_ROOT/checkpoints/main2025_v2"
TASK1_ROOT="$FINAL_ROOT/task1"
LOG_ROOT="$FINAL_ROOT/logs"
mkdir -p "$MODEL_ROOT" "$TASK1_ROOT" "$LOG_ROOT"

for horizon in 1 4 8 12; do
  verified_file="$SEQUENCE_ROOT/verified/h${horizon}.env"
  run_tag="$(awk -F= '$1 == "RUN_TAG" {print $2}' "$verified_file")"
  download_root="$ROOT_DIR/outputs/v2_checkpoint_rerun/$run_tag/hf_download/runs/$run_tag"
  release_root="$download_root/checkpoints/main2025_v2_h${horizon}"
  train_root="$download_root/checkpoints/main2025_v2_h${horizon}_train/finetuned"
  artifact_manifest="$download_root/logs/artifact_manifest_${run_tag}.sha256"
  test -s "$artifact_manifest" || {
    echo "ERROR: h=$horizon aggregation source has no artifact manifest." >&2
    exit 1
  }
  (
    cd "$download_root"
    shasum -a 256 -c "logs/artifact_manifest_${run_tag}.sha256"
  ) || {
    echo "ERROR: h=$horizon aggregation source failed its artifact manifest." >&2
    exit 1
  }
  (
    cd "$release_root"
    shasum -a 256 -c SHA256SUMS
  )
  receipt_model_sha="$(awk -F= '$1 == "MODEL_SHA256" {print $2}' "$verified_file")"
  observed_model_sha="$(shasum -a 256 "$release_root/dexposure-fm-h${horizon}.pt" | awk '{print $1}')"
  test "$observed_model_sha" = "$receipt_model_sha" || {
    echo "ERROR: h=$horizon aggregation source differs from its verified model SHA-256." >&2
    exit 1
  }
  "$ROOT_DIR/.venv/bin/python" cloud/verify_v2_checkpoint.py \
    --checkpoint "$release_root/dexposure-fm-h${horizon}.pt" \
    --metrics "$release_root/task1_metrics.json" \
    --config "$release_root/run_config.json" \
    --manifest cloud/preflight_manifest.json --horizon "$horizon" \
    --source-sha256 "$SOURCE_SHA"
  cp "$release_root/dexposure-fm-h${horizon}.pt" "$MODEL_ROOT/"
  mkdir -p "$TASK1_ROOT/h${horizon}" "$LOG_ROOT/h${horizon}"
  cp "$release_root/task1_metrics.json" "$TASK1_ROOT/h${horizon}/"
  cp "$release_root/run_config.json" "$TASK1_ROOT/h${horizon}/"
  cp "$train_root/experiment_results.json" "$TASK1_ROOT/h${horizon}/"
  cp "$train_root/data_quality.json" "$TASK1_ROOT/h${horizon}/"
  cp "$train_root/metrics.json" "$TASK1_ROOT/h${horizon}/"
  cp "$train_root/all_results.json" "$TASK1_ROOT/h${horizon}/"
  cp -R "$download_root/logs/." "$LOG_ROOT/h${horizon}/"
  cp "$SEQUENCE_ROOT/controller_logs/${run_tag}.log" "$LOG_ROOT/h${horizon}/controller_local.log"
  if [ "$horizon" = 1 ]; then
    cp "$release_root/feature_schema.json" "$MODEL_ROOT/feature_schema.json"
  else
    cmp -s "$release_root/feature_schema.json" "$MODEL_ROOT/feature_schema.json" || {
      echo "ERROR: feature schema differs at h=$horizon." >&2
      exit 1
    }
  fi
done

(
  cd "$MODEL_ROOT"
  shasum -a 256 dexposure-fm-h1.pt dexposure-fm-h4.pt \
    dexposure-fm-h8.pt dexposure-fm-h12.pt > SHA256SUMS
  shasum -a 256 -c SHA256SUMS
)

SEQUENCE_TAG="$SEQUENCE_TAG" SOURCE_SHA="$SOURCE_SHA" \
  "$ROOT_DIR/.venv/bin/python" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

root = Path("outputs/v2_checkpoint_rerun") / os.environ["SEQUENCE_TAG"] / "final"
results = {}
run_configs = {}
for horizon in (1, 4, 8, 12):
    metrics = json.loads(
        (root / "task1" / f"h{horizon}" / "task1_metrics.json").read_text()
    )
    results[f"h{horizon}"] = metrics[f"h{horizon}"]
    run_configs[f"h{horizon}"] = json.loads(
        (root / "task1" / f"h{horizon}" / "run_config.json").read_text()
    )
payload = {
    "artifact_status": "complete",
    "model": "DeXposure-FM",
    "results": results,
    "experiment": {
        "forecast_horizons_weeks": [1, 4, 8, 12],
        "holdout_start": "2025-01-01",
        "validation_weeks": 24,
        "epochs": 20,
        "seed": 42,
        "network_snapshots": 283,
        "source_sha256": os.environ["SOURCE_SHA"],
        "sequence_tag": os.environ["SEQUENCE_TAG"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    },
    "per_horizon_run_config": run_configs,
}
(root / "task1_metrics_all_horizons.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
PY

(
  cd "$FINAL_ROOT"
  find . -type f ! -path './SHA256SUMS' -print | LC_ALL=C sort | \
    while IFS= read -r artifact; do
      shasum -a 256 "$artifact"
    done >SHA256SUMS
  shasum -a 256 -c SHA256SUMS
)

HF_TOKEN_VALUE="${HF_TOKEN:-}"
if [ -z "$HF_TOKEN_VALUE" ] && [ -f "$ROOT_DIR/.env" ]; then
  HF_TOKEN_VALUE="$(awk -F= '$1 == "HF_TOKEN" {sub(/^[^=]*=/, ""); print; exit}' "$ROOT_DIR/.env")"
fi
HF_TOKEN_VALUE="${HF_TOKEN_VALUE#\"}"
HF_TOKEN_VALUE="${HF_TOKEN_VALUE%\"}"
HF_TOKEN_VALUE="${HF_TOKEN_VALUE#\'}"
HF_TOKEN_VALUE="${HF_TOKEN_VALUE%\'}"
if [ -z "$HF_TOKEN_VALUE" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  HF_TOKEN_VALUE="$(<"$HOME/.cache/huggingface/token")"
fi
test -n "$HF_TOKEN_VALUE" || { echo "ERROR: HF_TOKEN unavailable for final package upload." >&2; exit 1; }
export HF_TOKEN="$HF_TOKEN_VALUE"
unset HF_TOKEN_VALUE

HF_REPO="losdwind/graph-dexposure-ckpt"
HF_FINAL_PATH="v2_reconstructions/$SEQUENCE_TAG"
run_bounded "$TRANSFER_TIMEOUT_SECONDS" uvx --from 'huggingface_hub[cli]' hf upload "$HF_REPO" \
  "$FINAL_ROOT" "$HF_FINAL_PATH" --repo-type model \
  --commit-message "Complete v2 reconstruction $SEQUENCE_TAG"

ROUNDTRIP_ROOT="$SEQUENCE_ROOT/hf_roundtrip"
run_bounded "$TRANSFER_TIMEOUT_SECONDS" uvx --from 'huggingface_hub[cli]' hf download "$HF_REPO" \
  --repo-type model --include "$HF_FINAL_PATH/**" --local-dir "$ROUNDTRIP_ROOT" \
  --force-download
(
  cd "$ROUNDTRIP_ROOT/$HF_FINAL_PATH"
  shasum -a 256 -c SHA256SUMS
)

LIVE_JSON="$(read_vast_instances)" || {
  echo "ERROR: final Vast instance list was malformed; absence is not confirmed." >&2
  exit 1
}
LIVE_V2="$(jq '[.[] | select(((.label // "") | startswith("dexposure-v2-")))] | length' <<<"$LIVE_JSON")"
test "$LIVE_V2" = 0 || {
  echo "ERROR: at least one DeXposure v2 Vast instance is still listed." >&2
  exit 1
}

echo "V2_LOCAL_FINAL=$FINAL_ROOT"
echo "V2_HF_ROUNDTRIP=$ROUNDTRIP_ROOT/$HF_FINAL_PATH"
echo "V2_HF_PATH=https://huggingface.co/$HF_REPO/tree/main/$HF_FINAL_PATH"
echo "V2_SEQUENCE_COMPLETE=1"
