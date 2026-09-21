#!/usr/bin/env bash
# Direct Vast.ai API controller for one verified DeXposure-FM v2 horizon.
set -euo pipefail

TRAIN_CMD="${1:?training command is required}"
SOURCE_SHA="${2:?source digest is required}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

for command in vastai jq rsync ssh uvx curl python3; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "ERROR: $command is required for the Vast API controller." >&2
    exit 1
  }
done
test -n "${HF_TOKEN:-}" || { echo "ERROR: HF_TOKEN is required." >&2; exit 1; }
SSH_OVER_HTTPS="${VAST_SSH_OVER_HTTPS:-0}"
case "$SSH_OVER_HTTPS" in 0|1) ;; *) echo "ERROR: VAST_SSH_OVER_HTTPS must be 0 or 1." >&2; exit 1 ;; esac
CLOUDFLARED_BIN="${VAST_CLOUDFLARED_BIN:-}"
if [ "$SSH_OVER_HTTPS" = 1 ]; then
  test -f "$CLOUDFLARED_BIN" && test -x "$CLOUDFLARED_BIN" || {
    echo "ERROR: VAST_CLOUDFLARED_BIN must name a local executable." >&2; exit 1;
  }
  bash -n cloud/vast_https_bootstrap.sh
fi

TRAIN_TIMEOUT_SECONDS="${VAST_TRAIN_TIMEOUT_SECONDS:-21600}"
API_TIMEOUT_SECONDS="${VAST_API_TIMEOUT_SECONDS:-120}"
TRANSFER_TIMEOUT_SECONDS="${VAST_TRANSFER_TIMEOUT_SECONDS:-7200}"
MIN_DISK_BW="${VAST_MIN_DISK_BW:-1000}"
MAX_DPH="${VAST_MAX_DPH:-0.65}"
case "$TRAIN_TIMEOUT_SECONDS:$API_TIMEOUT_SECONDS:$TRANSFER_TIMEOUT_SECONDS" in
  *[!0-9:]*) echo "ERROR: Vast controller timeouts must be whole seconds." >&2; exit 1 ;;
esac
[ "$TRAIN_TIMEOUT_SECONDS" -gt 0 ] && [ "$API_TIMEOUT_SECONDS" -gt 0 ] && \
  [ "$TRANSFER_TIMEOUT_SECONDS" -gt 0 ] || {
  echo "ERROR: Vast controller timeouts must be positive." >&2
  exit 1
}
jq -en --arg min_disk_bw "$MIN_DISK_BW" --arg max_dph "$MAX_DPH" \
  'try ((($min_disk_bw | tonumber) > 0) and (($max_dph | tonumber) > 0)) catch false' \
  >/dev/null || {
  echo "ERROR: VAST_MIN_DISK_BW and VAST_MAX_DPH must be positive numbers." >&2
  exit 1
}

# macOS does not ship GNU timeout. This small wrapper gives every local network
# command a total deadline and kills its whole process group if the deadline or
# the controller signal arrives.
run_bounded() {
  local seconds="${1:?timeout is required}"
  shift
  python3 - "$seconds" "$@" <<'PY'
import os
import re
import signal
import subprocess
import sys
import threading

seconds = int(sys.argv[1])
command = sys.argv[2:]
process = subprocess.Popen(command, start_new_session=True, stderr=subprocess.PIPE)

def relay_stderr() -> None:
    for line in iter(process.stderr.readline, b""):
        line = re.sub(rb"(?i)(api_key=|hf_)[A-Za-z0-9_-]+", rb"\1[REDACTED]", line)
        sys.stderr.buffer.write(line)
        sys.stderr.buffer.flush()

stderr_thread = threading.Thread(target=relay_stderr, daemon=True)
stderr_thread.start()

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
    status = process.wait(timeout=seconds)
except subprocess.TimeoutExpired:
    stop_group()
    raise SystemExit(124)
stderr_thread.join(timeout=2)
raise SystemExit(status if status >= 0 else 128 - status)
PY
}

list_instances() {
  local payload
  payload="$(run_bounded "$API_TIMEOUT_SECONDS" vastai show instances-v1 --all --raw)" || return 1
  jq -e '.success == true and (.instances | type == "array")' \
    >/dev/null <<<"$payload" || return 1
  jq -c '.instances' <<<"$payload"
}

BINDING_JSON="$(jq -c --arg command "$TRAIN_CMD" \
  '.bindings | to_entries[] | select(.value.command == $command)' \
  cloud/preflight_manifest.json)"
test -n "$BINDING_JSON" || { echo "ERROR: no preflight binding for $TRAIN_CMD" >&2; exit 1; }
BINDING_NAME="$(jq -r '.key' <<<"$BINDING_JSON")"
HORIZON_COUNT="$(jq -r '.value.horizons | length' <<<"$BINDING_JSON")"
BATCH_MODE=0
INPUT_FETCH_CMD=""
BATCH_HORIZONS="4 8 12"
if [ "$BINDING_NAME" = main2025_v2_remaining_checkpoints ] || [ "$BINDING_NAME" = main2025_v2_repeat20 ] || [ "$BINDING_NAME" = main2025_fm_tuning ] || [ "$BINDING_NAME" = main2025_fm_eval_paper ] || [ "$BINDING_NAME" = main2025_fm_sequential ]; then
  BATCH_MODE=1
  HORIZON=remaining
  EXPECTED_COMMAND="bash cloud/main2025_v2_remaining_run.sh"
  if [ "$BINDING_NAME" = main2025_v2_repeat20 ]; then
    EXPECTED_COMMAND="$EXPECTED_COMMAND --all"
    BATCH_HORIZONS="1 4 8 12"
  fi
  if [ "$BINDING_NAME" = main2025_fm_tuning ]; then
    EXPECTED_COMMAND="bash cloud/fm_tune_run.sh"
    BATCH_HORIZONS="1 4 8 12"
    jq -e --argjson hours "$TRAIN_TIMEOUT_SECONDS" --argjson price "$MAX_DPH" \
      '$hours <= .max_train_seconds and $price <= .max_offer_dph' cloud/fm_tune_plan.json >/dev/null || {
      echo "ERROR: tuning runtime or offer price exceeds the reviewed plan." >&2
      exit 1
    }
  fi
  if [ "$BINDING_NAME" = main2025_fm_sequential ]; then
    EXPECTED_COMMAND="bash cloud/fm_sequential_run.sh"
    BATCH_HORIZONS="1 4 8 12"
    jq -e --argjson hours "$TRAIN_TIMEOUT_SECONDS" --argjson price "$MAX_DPH" \
      '$hours <= .max_train_seconds and $price <= .max_offer_dph' cloud/fm_sequential_plan.json >/dev/null || {
      echo "ERROR: sequential runtime or offer price exceeds the reviewed plan." >&2
      exit 1
    }
  fi
  if [ "$BINDING_NAME" = main2025_fm_eval_paper ]; then
    EXPECTED_COMMAND="bash cloud/fm_eval_paper_run.sh"
    BATCH_HORIZONS="1 4 8 12"
    jq -e --argjson hours "$TRAIN_TIMEOUT_SECONDS" --argjson price "$MAX_DPH" \
      '$hours <= .max_train_seconds and $price <= .max_offer_dph' cloud/fm_eval_paper_plan.json >/dev/null || {
      echo "ERROR: paper-eval runtime or offer price exceeds the reviewed plan." >&2
      exit 1
    }
  fi
  test "$TRAIN_CMD" = "$EXPECTED_COMMAND" || {
    echo "ERROR: use the exact v2 remaining-horizon command: $EXPECTED_COMMAND" >&2
    exit 1
  }
  if [ "$BATCH_HORIZONS" = "1 4 8 12" ]; then
    jq -e '.value.horizons == [1, 4, 8, 12]' >/dev/null <<<"$BINDING_JSON" || exit 1
  else
    test "$HORIZON_COUNT" = 3 && \
    jq -e '.value.horizons == [4, 8, 12]' >/dev/null <<<"$BINDING_JSON" || {
    echo "ERROR: the remaining-horizon binding must declare exactly [4,8,12]." >&2
    exit 1
  }
  fi
  INPUT_FETCH_CMD="$(jq -r '.value.input_fetch.command // empty' <<<"$BINDING_JSON")"
  test "$INPUT_FETCH_CMD" = "bash cloud/fetch_v2_inputs_from_hf.sh" || {
    echo "ERROR: the remaining-horizon binding must declare the verified Hugging Face input fetch command." >&2
    exit 1
  }
else
  test "$HORIZON_COUNT" = 1 || {
    echo "ERROR: the safe Vast controller accepts one horizon, except for the exact [4,8,12] remaining binding." >&2
    exit 1
  }
  HORIZON="$(jq -r '.value.horizons[0]' <<<"$BINDING_JSON")"
  EXPECTED_COMMAND="bash cloud/main2025_v2_horizon_run.sh $HORIZON"
  if [ "$BINDING_NAME" = main2025_v2_h12_extended ]; then
    EXPECTED_COMMAND="bash cloud/main2025_v2_horizon_run.sh 12 --extended"
    INPUT_FETCH_CMD="bash cloud/fetch_v2_inputs_from_hf.sh"
    test "$HORIZON" = 12 || exit 1
  fi
  test "$TRAIN_CMD" = "$EXPECTED_COMMAND" || {
    echo "ERROR: use the v2 single-horizon command: $EXPECTED_COMMAND" >&2
    exit 1
  }
fi
if [ -n "${VAST_EXPECTED_HORIZON:-}" ] && [ "$VAST_EXPECTED_HORIZON" != "$HORIZON" ]; then
  echo "ERROR: expected horizon $VAST_EXPECTED_HORIZON but binding selected $HORIZON." >&2
  exit 1
fi

RELEASE_REL="checkpoints/main2025_v2_h${HORIZON}"
TRAIN_REL="checkpoints/main2025_v2_h${HORIZON}_train"
MODEL_SHA256_H1=""
MODEL_SHA256_H4=""
MODEL_SHA256_H8=""
MODEL_SHA256_H12=""

RUN_TAG="${VAST_RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)_h${HORIZON}}"
case "$RUN_TAG" in
  *[!A-Za-z0-9._-]*) echo "ERROR: VAST_RUN_TAG contains unsafe characters." >&2; exit 1 ;;
esac
HF_REPO="losdwind/graph-dexposure-ckpt"
INSTANCE_LABEL="dexposure-v2-$RUN_TAG"
ATTACH_EXISTING="${VAST_ATTACH_EXISTING:-0}"
case "$ATTACH_EXISTING" in
  0|1) ;;
  *) echo "ERROR: VAST_ATTACH_EXISTING must be 0 or 1." >&2; exit 1 ;;
esac

REPO_JSON="$(run_bounded "$API_TIMEOUT_SECONDS" curl --connect-timeout 15 --max-time "$API_TIMEOUT_SECONDS" -fsS -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/models/$HF_REPO")" || {
  echo "ERROR: cannot read the Hugging Face artifact repository." >&2
  exit 1
}
jq -e '.private == true' >/dev/null <<<"$REPO_JSON" || {
  echo "ERROR: $HF_REPO is not private; refusing to upload reconstructed checkpoints." >&2
  exit 1
}

TREE_PROBE="/tmp/dexposure-hf-tree-${RUN_TAG}-$$.json"
TREE_STATUS="$(run_bounded "$API_TIMEOUT_SECONDS" curl --connect-timeout 15 --max-time "$API_TIMEOUT_SECONDS" -sS -o "$TREE_PROBE" -w '%{http_code}' \
  -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/models/$HF_REPO/tree/main/runs/$RUN_TAG?recursive=true&expand=false")" || {
  echo "ERROR: cannot check whether Hugging Face run tag $RUN_TAG already exists." >&2
  exit 1
}
rm -f "$TREE_PROBE"
case "$TREE_STATUS" in
  404)
    [ "$ATTACH_EXISTING" = 0 ] || {
      echo "ERROR: attach run $RUN_TAG has no Hugging Face run prefix." >&2
      exit 1
    }
    ;;
  200)
    [ "$ATTACH_EXISTING" = 1 ] || {
      echo "ERROR: Hugging Face run tag $RUN_TAG already exists; refusing to mix attempts." >&2
      exit 1
    }
    ;;
  *) echo "ERROR: Hugging Face run-tag check returned HTTP $TREE_STATUS." >&2; exit 1 ;;
esac

STATE_ROOT="$ROOT_DIR/outputs/v2_checkpoint_rerun/$RUN_TAG"
CONTROLLER_DIR="$STATE_ROOT/controller"
mkdir -p "$CONTROLLER_DIR"
CONTROLLER_STATE="$CONTROLLER_DIR/preflight.env"
printf 'RUN_TAG=%s\nHORIZON=%s\nSOURCE_SHA256=%s\n' \
  "$RUN_TAG" "$HORIZON" "$SOURCE_SHA" >"$CONTROLLER_STATE"
run_bounded "$TRANSFER_TIMEOUT_SECONDS" uvx --from 'huggingface_hub[cli]' hf upload "$HF_REPO" \
  "$CONTROLLER_STATE" "controller_preflights/$RUN_TAG.env" --repo-type model \
  --commit-message "Vast controller write preflight $RUN_TAG" >/dev/null
echo "Hugging Face private/write preflight passed for $HF_REPO"

CREATE_LOCK_DIR="/tmp/dexposure-v2-vast-create.lock"
CREATE_LOCK_HELD=0
CREATE_LOCK_TOKEN="$$-$(date +%s)-$RANDOM-$RANDOM"
CREATE_LOCK_PROCESS_START="$(ps -o lstart= -p "$$" | sed -E 's/^[[:space:]]+//')"
create_lock_mtime() {
  local value=""
  value="$(stat -c %Y "$1" 2>/dev/null || true)"
  case "$value" in *[!0-9]*|'') ;; *) printf '%s\n' "$value"; return 0 ;; esac
  value="$(stat -f %m "$1" 2>/dev/null || true)"
  case "$value" in *[!0-9]*|'') return 1 ;; *) printf '%s\n' "$value" ;; esac
}
create_lock_is_ours() {
  local owner="$CREATE_LOCK_DIR/owner"
  [ -s "$owner" ] && \
    [ "$(awk -F= '$1 == "TOKEN" {print $2}' "$owner")" = "$CREATE_LOCK_TOKEN" ] && \
    [ "$(awk -F= '$1 == "PID" {print $2}' "$owner")" = "$$" ] && \
    [ "$(awk -F= '$1 == "START" {print $2}' "$owner")" = "$CREATE_LOCK_PROCESS_START" ]
}
release_create_lock() {
  if [ "$CREATE_LOCK_HELD" -eq 1 ] && create_lock_is_ours; then
    rm -f "$CREATE_LOCK_DIR/owner"
    rmdir "$CREATE_LOCK_DIR" 2>/dev/null || true
    CREATE_LOCK_HELD=0
  fi
}
handle_create_lock_signal() {
  trap - EXIT INT TERM
  release_create_lock
  exit 130
}
acquire_create_lock() {
  local attempt owner_pid owner_start current_start lock_mtime now stale_path
  for attempt in 1 2 3 4; do
    if mkdir "$CREATE_LOCK_DIR" 2>/dev/null; then
      (
        umask 077
        printf 'TOKEN=%s\nPID=%s\nSTART=%s\n' \
          "$CREATE_LOCK_TOKEN" "$$" "$CREATE_LOCK_PROCESS_START" \
          >"$CREATE_LOCK_DIR/owner"
      )
      CREATE_LOCK_HELD=1
      return 0
    fi

    owner_pid="$(awk -F= '$1 == "PID" {print $2}' "$CREATE_LOCK_DIR/owner" 2>/dev/null || true)"
    owner_start="$(awk -F= '$1 == "START" {print $2}' "$CREATE_LOCK_DIR/owner" 2>/dev/null || true)"
    case "$owner_pid" in
      *[!0-9]*|'') owner_pid="" ;;
    esac
    if [ -n "$owner_pid" ] && kill -0 "$owner_pid" 2>/dev/null; then
      current_start="$(ps -o lstart= -p "$owner_pid" 2>/dev/null | sed -E 's/^[[:space:]]+//' || true)"
      if [ -n "$current_start" ] && [ "$current_start" = "$owner_start" ]; then
        echo "ERROR: another local Vast create controller pid=$owner_pid is active." >&2
        return 1
      fi
    elif [ -z "$owner_pid" ]; then
      lock_mtime="$(create_lock_mtime "$CREATE_LOCK_DIR" || true)"
      now="$(date +%s)"
      if [ -z "$lock_mtime" ] || [ "$lock_mtime" -gt "$now" ] || \
          [ $((now - lock_mtime)) -lt 30 ]; then
        echo "ERROR: another local Vast create controller is initializing its lock." >&2
        return 1
      fi
    fi

    stale_path="${CREATE_LOCK_DIR}.stale.${CREATE_LOCK_TOKEN}.${attempt}"
    if mv "$CREATE_LOCK_DIR" "$stale_path" 2>/dev/null; then
      rm -f "$stale_path/owner" "$stale_path/pid"
      rmdir "$stale_path" 2>/dev/null || \
        echo "WARNING: stale create lock retained unexpected files at $stale_path" >&2
    fi
  done
  echo "ERROR: could not acquire the local Vast create lock." >&2
  return 1
}
if [ "$ATTACH_EXISTING" = 0 ]; then
  trap release_create_lock EXIT
  trap handle_create_lock_signal INT TERM
  acquire_create_lock
fi

confirm_no_live_v2_instances() {
  local clean_streak=0 candidate="" live_v2=0 round
  for round in $(seq 1 30); do
    if ! candidate="$(list_instances 2>/dev/null)"; then
      clean_streak=0
      echo "WARNING: Vast instance list failed during the pre-create guard." >&2
    elif ! jq -e 'type == "array"' >/dev/null 2>&1 <<<"$candidate"; then
      clean_streak=0
      echo "WARNING: Vast instance list was malformed during the pre-create guard." >&2
    else
      live_v2="$(jq '[.[] | select(((.label // "") | startswith("dexposure-v2-")))] | length' <<<"$candidate")"
      if [ "$live_v2" -ne 0 ]; then
        jq -r '.[] | select(((.label // "") | startswith("dexposure-v2-"))) | "id=\(.id) label=\(.label) status=\(.actual_status)"' <<<"$candidate" >&2
        echo "ERROR: an existing DeXposure v2 Vast instance is present; refusing a duplicate." >&2
        return 1
      fi
      clean_streak=$((clean_streak + 1))
      if [ "$clean_streak" -ge 3 ]; then
        LIVE_JSON="$candidate"
        return 0
      fi
    fi
    [ "$round" -eq 30 ] || sleep 2
  done
  echo "ERROR: three consecutive valid empty Vast instance samples were not obtained; refusing to create." >&2
  return 1
}

if [ "$ATTACH_EXISTING" = 0 ]; then
  confirm_no_live_v2_instances || exit 1
  LIVE_V2=0
  ATTACH_MATCHES=0
else
  LIVE_JSON="$(list_instances)"
  jq -e 'type == "array"' >/dev/null <<<"$LIVE_JSON" || {
    echo "ERROR: Vast instance list returned an unexpected payload; refusing to attach." >&2
    exit 1
  }
  LIVE_V2="$(jq '[.[] | select(((.label // "") | startswith("dexposure-v2-")))] | length' <<<"$LIVE_JSON")"
  ATTACH_MATCHES="$(jq --arg label "$INSTANCE_LABEL" '[.[] | select((.label // "") == $label)] | length' <<<"$LIVE_JSON")"
  test "$LIVE_V2" = "$ATTACH_MATCHES" || {
    echo "ERROR: another DeXposure v2 run is live; refusing an ambiguous attach." >&2
    exit 1
  }
  [ "$ATTACH_MATCHES" -le 1 ] || {
    echo "ERROR: multiple live instances match attach label $INSTANCE_LABEL." >&2
    exit 1
  }
fi

OFFER_ID=""
if [ "$ATTACH_EXISTING" = 0 ]; then
  QUERY="${VAST_OFFER_QUERY:-num_gpus=1 gpu_ram>=40 gpu_ram<=48 reliability>0.99 disk_space>=80 cpu_ram>=32 direct_port_count>=1 disk_bw>=$MIN_DISK_BW dph_total<=$MAX_DPH}"
  OFFER_ORDER="${VAST_OFFER_ORDER:-dlperf_usd-}"
  EXCLUDE_GEO="${VAST_EXCLUDE_GEO:-}"
  OFFERS_JSON="$(run_bounded "$API_TIMEOUT_SECONDS" vastai search offers "$QUERY" --raw --limit 50 -o "$OFFER_ORDER")"
  SELECTED_OFFER="$(jq -c --arg min_disk_bw "$MIN_DISK_BW" --arg max_dph "$MAX_DPH" --arg exclude_geo "$EXCLUDE_GEO" '[.[] | select(
    (
      (.gpu_name == "A40") or (.gpu_name == "L40S") or
      (.gpu_name == "RTX A6000") or (.gpu_name == "RTX 6000Ada") or
      ((.gpu_name // "") | startswith("A100"))
    ) and
    (.num_gpus == 1) and
    ((.gpu_ram | type) == "number" and .gpu_ram >= 40000 and .gpu_ram <= 49152) and
    ((.reliability | type) == "number" and .reliability > 0.995) and
    (.static_ip == true) and
    ((.inet_down | type) == "number" and .inet_down >= 500) and
    ((.inet_up | type) == "number" and .inet_up >= 500) and
    ((.disk_bw | type) == "number" and .disk_bw >= ($min_disk_bw | tonumber)) and
    ((.disk_space | type) == "number" and .disk_space >= 80) and
    ((.cpu_ram | type) == "number" and .cpu_ram >= 32000) and
    ((.direct_port_count | type) == "number" and .direct_port_count >= 1) and
    ((.dph_total | type) == "number" and .dph_total <= ($max_dph | tonumber)) and
    (.rentable == true) and (.verification == "verified") and
    ($exclude_geo == "" or ((.geolocation // "") | test($exclude_geo) | not))
  )][0] // empty' <<<"$OFFERS_JSON")"
  OFFER_ID="$(jq -r '.id // empty' <<<"$SELECTED_OFFER")"
  test -n "$OFFER_ID" || { echo "ERROR: no compatible 40-48 GB GPU offer at or below the price ceiling." >&2; exit 1; }
  jq -r --arg min_disk_bw "$MIN_DISK_BW" --arg max_dph "$MAX_DPH" '"Selected Vast offer id=\(.id) gpu=\(.gpu_name) vram_mib=\(.gpu_ram) location=\(.geolocation) price_per_hour=\(.dph_total) disk_bw=\(.disk_bw) min_disk_bw=\($min_disk_bw) max_price_per_hour=\($max_dph) dlperf_per_dollar=\(.dlperf_per_dphtotal) reliability=\(.reliability)"' <<<"$SELECTED_OFFER"
fi

INSTANCE_ID=""
REMOTE_STARTED=0
CONTROLLER_HEARTBEAT_PID=""
CONTROLLER_PARENT_PID="$$"
CONTROLLER_PARENT_START="$(ps -o lstart= -p "$$" | sed -E 's/^[[:space:]]+//')"
write_controller_heartbeat() {
  "${SSH[@]}" "tmp=/root/vast_workdir/logs/.controller_heartbeat.env.\$\$; printf 'INSTANCE_ID=%s\\nRUN_TAG=%s\\nHORIZON=%s\\nSOURCE_SHA256=%s\\nCONTROLLER_TIMESTAMP=%s\\n' '$INSTANCE_ID' '$RUN_TAG' '$HORIZON' '$SOURCE_SHA' \"\$(date +%s)\" >\"\$tmp\" && mv \"\$tmp\" /root/vast_workdir/logs/controller_heartbeat.env" \
    >/dev/null 2>&1
}
start_controller_heartbeat() {
  write_controller_heartbeat || return 1
  (
    trap 'exit 0' INT TERM
    while true; do
      sleep 60
      current_parent_start="$(ps -o lstart= -p "$CONTROLLER_PARENT_PID" 2>/dev/null | sed -E 's/^[[:space:]]+//' || true)"
      [ "$current_parent_start" = "$CONTROLLER_PARENT_START" ] || exit 0
      write_controller_heartbeat || true
    done
  ) &
  CONTROLLER_HEARTBEAT_PID=$!
}
stop_controller_heartbeat() {
  if [ -n "$CONTROLLER_HEARTBEAT_PID" ]; then
    kill "$CONTROLLER_HEARTBEAT_PID" >/dev/null 2>&1 || true
    wait "$CONTROLLER_HEARTBEAT_PID" >/dev/null 2>&1 || true
    CONTROLLER_HEARTBEAT_PID=""
  fi
}
destroy_instance() {
  local id="${1:-$INSTANCE_ID}"
  test -n "$id" || return 0
  local before_destroy="" pre_absent_streak=0 round
  for round in 1 2 3; do
    if ! before_destroy="$(list_instances 2>/dev/null)"; then
      pre_absent_streak=0
      echo "WARNING: Vast instance list failed during the pre-destroy check for id=$id." >&2
    elif ! jq -e 'type == "array"' >/dev/null 2>&1 <<<"$before_destroy"; then
      pre_absent_streak=0
      echo "WARNING: Vast instance list was malformed during the pre-destroy check for id=$id." >&2
    elif jq -e --arg id "$id" '.[] | select((.id | tostring) == $id)' >/dev/null <<<"$before_destroy"; then
      pre_absent_streak=0
    else
      pre_absent_streak=$((pre_absent_streak + 1))
      if [ "$pre_absent_streak" -ge 3 ]; then
        echo "Confirmed Vast instance id=$id is already absent in three consecutive valid samples"
        if [ "$INSTANCE_ID" = "$id" ]; then
          INSTANCE_ID=""
        fi
        return 0
      fi
    fi
    [ "$round" -eq 3 ] || sleep 2
  done
  echo "Destroying Vast instance id=$id"
  local destroy_submitted=0
  for _ in 1 2 3; do
    if run_bounded "$API_TIMEOUT_SECONDS" vastai destroy instance "$id" -y >/dev/null 2>&1; then
      destroy_submitted=1
      break
    fi
    echo "WARNING: Vast destroy request failed for id=$id; retrying." >&2
    sleep 2
  done
  if [ "$destroy_submitted" -ne 1 ]; then
    echo "ERROR: could not submit Vast destroy request for id=$id." >&2
    return 1
  fi
  local absent_streak=0 instances=""
  for round in $(seq 1 30); do
    if ! instances="$(list_instances 2>/dev/null)"; then
      absent_streak=0
      echo "WARNING: Vast instance list failed while confirming destroy for id=$id." >&2
    elif ! jq -e 'type == "array"' >/dev/null 2>&1 <<<"$instances"; then
      absent_streak=0
      echo "WARNING: Vast instance list was malformed while confirming destroy for id=$id." >&2
    elif jq -e --arg id "$id" '.[] | select((.id | tostring) == $id)' >/dev/null <<<"$instances"; then
      absent_streak=0
    else
      absent_streak=$((absent_streak + 1))
      if [ "$absent_streak" -ge 3 ]; then
        echo "Confirmed Vast instance id=$id is absent in three consecutive valid samples"
        if [ "$INSTANCE_ID" = "$id" ]; then
          INSTANCE_ID=""
        fi
        return 0
      fi
    fi
    [ "$round" -eq 30 ] || sleep 2
  done
  echo "ERROR: Vast instance id=$id did not remain absent for three consecutive valid samples after destroy." >&2
  return 1
}

stop_instance_preserving_disk() {
  local id="${1:-$INSTANCE_ID}"
  test -n "$id" || return 0
  echo "Stopping Vast instance id=$id and retaining its disk"
  local stop_submitted=0
  for _ in 1 2 3; do
    if run_bounded "$API_TIMEOUT_SECONDS" vastai stop instance "$id" --raw >/dev/null 2>&1; then
      stop_submitted=1
      break
    fi
    echo "WARNING: Vast stop request failed for id=$id; retrying." >&2
    sleep 2
  done
  if [ "$stop_submitted" -ne 1 ]; then
    echo "ERROR: could not submit Vast stop request for id=$id; disk state is unchanged." >&2
    return 1
  fi
  local stopped_streak=0 instances="" status="" match_count=0 round
  for round in $(seq 1 30); do
    if ! instances="$(list_instances 2>/dev/null)"; then
      stopped_streak=0
      echo "WARNING: Vast instance list failed while confirming stop for id=$id." >&2
    elif ! jq -e 'type == "array"' >/dev/null 2>&1 <<<"$instances"; then
      stopped_streak=0
      echo "WARNING: Vast instance list was malformed while confirming stop for id=$id." >&2
    else
      match_count="$(jq --arg id "$id" '[.[] | select((.id | tostring) == $id)] | length' <<<"$instances")"
      if [ "$match_count" -eq 1 ]; then
        status="$(jq -r --arg id "$id" '.[] | select((.id | tostring) == $id) | (.actual_status // .status // "unknown")' <<<"$instances")"
        case "$status" in
          stopped|exited)
            stopped_streak=$((stopped_streak + 1))
            if [ "$stopped_streak" -ge 3 ]; then
              echo "Confirmed Vast instance id=$id status=$status in three consecutive valid samples; disk retained"
              return 0
            fi
            ;;
          *) stopped_streak=0 ;;
        esac
      else
        stopped_streak=0
        if [ "$match_count" -eq 0 ]; then
          echo "WARNING: Vast instance id=$id is absent after stop; retained-disk status is not proven." >&2
        else
          echo "WARNING: Vast instance id=$id appeared more than once while confirming stop." >&2
        fi
      fi
    fi
    [ "$round" -eq 30 ] || sleep 2
  done
  echo "ERROR: Vast instance id=$id did not remain stopped/exited for three consecutive valid samples." >&2
  return 1
}

publish_watchdog_recovery_evidence() {
  local evidence_level="$1" model_sha="$2"
  [ "$ATTACH_INSTANCE_ABSENT" -eq 0 ] || return 0
  local authorization_live=""
  if authorization_live="$(list_instances 2>/dev/null)" && \
      ! jq -e --arg id "$INSTANCE_ID" '.[] | select((.id | tostring) == $id)' \
        >/dev/null <<<"$authorization_live"; then
    echo "Vast instance id=$INSTANCE_ID is already absent; no remote authorization is needed."
    return 0
  fi
  case "$evidence_level" in
    complete_bundle|failure_inventory) ;;
    *) echo "ERROR: invalid watchdog recovery evidence level: $evidence_level" >&2; return 1 ;;
  esac
  case "$model_sha" in
    none) ;;
    *[!0-9a-f]*|'') echo "ERROR: invalid watchdog model SHA-256." >&2; return 1 ;;
    *) [ "${#model_sha}" -eq 64 ] || { echo "ERROR: invalid watchdog model SHA-256 length." >&2; return 1; } ;;
  esac

  local authorization authorization_tmp remote_tmp roundtrip published_roundtrip
  authorization="$CONTROLLER_DIR/watchdog_destroy_authorized.env"
  authorization_tmp="${authorization}.tmp.$$"
  remote_tmp="/root/vast_workdir/logs/.watchdog_destroy_authorized.env.$$"
  roundtrip="${authorization}.roundtrip.$$"
  printf 'LOCAL_RECOVERY_VERIFIED=1\nHF_ROUNDTRIP_VERIFIED=0\nINSTANCE_ID=%s\nRUN_TAG=%s\nHORIZON=%s\nSOURCE_SHA256=%s\nEVIDENCE_LEVEL=%s\nMODEL_SHA256=%s\nSNAPSHOT_SHA256=none\n' \
    "$INSTANCE_ID" "$RUN_TAG" "$HORIZON" "$SOURCE_SHA" "$evidence_level" "$model_sha" \
    >"$authorization_tmp"
  mv "$authorization_tmp" "$authorization"
  run_bounded "$TRANSFER_TIMEOUT_SECONDS" rsync -az -e "$SSH_TRANSPORT" "$authorization" \
    "root@$SSH_HOST:$remote_tmp" || return 1
  "${SSH[@]}" "cat $remote_tmp" >"$roundtrip" || return 1
  if ! cmp -s "$authorization" "$roundtrip"; then
    rm -f "$roundtrip"
    echo "ERROR: remote temporary watchdog recovery evidence failed its round-trip comparison." >&2
    return 1
  fi
  rm -f "$roundtrip"
  "${SSH[@]}" "mv $remote_tmp /root/vast_workdir/logs/watchdog_destroy_authorized.env" || return 1
  published_roundtrip="${authorization}.published.$$"
  "${SSH[@]}" 'cat /root/vast_workdir/logs/watchdog_destroy_authorized.env' \
    >"$published_roundtrip" || return 1
  if ! cmp -s "$authorization" "$published_roundtrip"; then
    rm -f "$published_roundtrip"
    echo "ERROR: published watchdog recovery evidence failed its round-trip comparison." >&2
    return 1
  fi
  rm -f "$published_roundtrip"
  echo "Published stop-preserving watchdog evidence after local recovery level=$evidence_level"
}

cleanup_before_handoff() {
  stop_controller_heartbeat
  release_create_lock
  if [ -n "$INSTANCE_ID" ] && [ "$REMOTE_STARTED" -eq 0 ]; then
    if [ "$BATCH_MODE" -eq 1 ]; then
      stop_instance_preserving_disk "$INSTANCE_ID" || true
    else
      destroy_instance "$INSTANCE_ID" || true
    fi
  fi
}
handle_controller_signal() {
  trap - INT TERM
  cleanup_before_handoff
  exit 130
}
trap cleanup_before_handoff EXIT
trap handle_controller_signal INT TERM

write_success_receipt() {
  local destroyed="$1" completed_instance_id="$2" model_sha="$3"
  [ -n "${VAST_SUCCESS_RECEIPT:-}" ] || return 0
  mkdir -p "$(dirname "$VAST_SUCCESS_RECEIPT")"
  local receipt_tmp="${VAST_SUCCESS_RECEIPT}.tmp.$$"
  if [ "$BATCH_MODE" -eq 1 ]; then
    for model_sha in "$MODEL_SHA256_H4" "$MODEL_SHA256_H8" "$MODEL_SHA256_H12"; do
      case "$model_sha" in *[!0-9a-f]*|'') return 1 ;; esac
      [ "${#model_sha}" -eq 64 ] || return 1
    done
    printf 'VERIFIED=1\nDESTROYED=%s\nINSTANCE_ID=%s\nRUN_TAG=%s\nHORIZON=remaining\nHORIZONS=4,8,12\nSOURCE_SHA256=%s\nMODEL_SHA256_H4=%s\nMODEL_SHA256_H8=%s\nMODEL_SHA256_H12=%s\nCREDIT_BEFORE=%s\nVAST_COST=pending\nCREDIT_AFTER=pending\n' \
      "$destroyed" "$completed_instance_id" "$RUN_TAG" "$SOURCE_SHA" \
      "$MODEL_SHA256_H4" "$MODEL_SHA256_H8" "$MODEL_SHA256_H12" \
      "${VAST_CREDIT_BEFORE:-unknown}" >"$receipt_tmp"
  else
    printf 'VERIFIED=1\nDESTROYED=%s\nINSTANCE_ID=%s\nRUN_TAG=%s\nHORIZON=%s\nSOURCE_SHA256=%s\nMODEL_SHA256=%s\nCREDIT_BEFORE=%s\nVAST_COST=pending\nCREDIT_AFTER=pending\n' \
      "$destroyed" "$completed_instance_id" "$RUN_TAG" "$HORIZON" "$SOURCE_SHA" \
      "$model_sha" "${VAST_CREDIT_BEFORE:-unknown}" >"$receipt_tmp"
  fi
  if [ "$BATCH_HORIZONS" = "1 4 8 12" ]; then
    [[ "$MODEL_SHA256_H1" =~ ^[0-9a-f]{64}$ ]] || return 1
    # Override the legacy three-horizon receipt field with the exact batch.
    sed 's/^HORIZONS=4,8,12$/HORIZONS=1,4,8,12/' "$receipt_tmp" >"${receipt_tmp}.all"
    mv "${receipt_tmp}.all" "$receipt_tmp"
    printf 'MODEL_SHA256_H1=%s\n' "$MODEL_SHA256_H1" >>"$receipt_tmp"
  fi
  mv "$receipt_tmp" "$VAST_SUCCESS_RECEIPT"
}

# The remote watchdog can only stop and retain the disk. Instance destruction
# remains below in this local controller after download, SHA-256, provenance,
# and metrics verification.
WATCHDOG_SECONDS="${VAST_WATCHDOG_SECONDS:-3600}"
WATCHDOG_POLL_SECONDS="${VAST_WATCHDOG_POLL_SECONDS:-60}"
WATCHDOG_REMOTE_GRACE_SECONDS="${VAST_WATCHDOG_REMOTE_GRACE_SECONDS:-300}"
WATCHDOG_CONTROLLER_STALE_SECONDS="${VAST_WATCHDOG_CONTROLLER_STALE_SECONDS:-600}"
WATCHDOG_FINALIZATION_MARGIN_SECONDS="${VAST_WATCHDOG_FINALIZATION_MARGIN_SECONDS:-3600}"
WATCHDOG_RUNNER_STALE_SECONDS="${VAST_WATCHDOG_RUNNER_STALE_SECONDS:-$((TRAIN_TIMEOUT_SECONDS + WATCHDOG_FINALIZATION_MARGIN_SECONDS))}"
WATCHDOG_BOOT_GRACE_SECONDS="${VAST_WATCHDOG_BOOT_GRACE_SECONDS:-1200}"
WATCHDOG_EMERGENCY_PHASE_MAX_SECONDS="${VAST_WATCHDOG_EMERGENCY_PHASE_MAX_SECONDS:-10800}"
case "$WATCHDOG_SECONDS:$WATCHDOG_POLL_SECONDS:$WATCHDOG_REMOTE_GRACE_SECONDS:$WATCHDOG_CONTROLLER_STALE_SECONDS:$WATCHDOG_FINALIZATION_MARGIN_SECONDS:$WATCHDOG_RUNNER_STALE_SECONDS:$WATCHDOG_BOOT_GRACE_SECONDS:$WATCHDOG_EMERGENCY_PHASE_MAX_SECONDS" in
  *[!0-9:]*) echo "ERROR: Vast watchdog durations must be whole seconds." >&2; exit 1 ;;
esac
[ "$WATCHDOG_SECONDS" -gt 0 ] && [ "$WATCHDOG_POLL_SECONDS" -gt 0 ] && \
  [ "$WATCHDOG_REMOTE_GRACE_SECONDS" -ge "$WATCHDOG_POLL_SECONDS" ] && \
  [ "$WATCHDOG_CONTROLLER_STALE_SECONDS" -ge $((WATCHDOG_POLL_SECONDS * 2)) ] && \
  [ "$WATCHDOG_FINALIZATION_MARGIN_SECONDS" -gt 0 ] && \
  [ "$WATCHDOG_RUNNER_STALE_SECONDS" -ge $((TRAIN_TIMEOUT_SECONDS + WATCHDOG_FINALIZATION_MARGIN_SECONDS)) ] && \
  [ "$WATCHDOG_BOOT_GRACE_SECONDS" -ge $((WATCHDOG_POLL_SECONDS * 2)) ] && \
  [ "$WATCHDOG_EMERGENCY_PHASE_MAX_SECONDS" -gt 0 ] || {
  echo "ERROR: Vast watchdog durations are outside the safe range." >&2
  exit 1
}
WATCHDOG_ARGUMENTS="$WATCHDOG_SECONDS $WATCHDOG_POLL_SECONDS $WATCHDOG_REMOTE_GRACE_SECONDS $WATCHDOG_CONTROLLER_STALE_SECONDS $WATCHDOG_RUNNER_STALE_SECONDS"
ONSTART="mkdir -p /root/vast_workdir/logs; BOOT_TMP=/root/vast_workdir/logs/.watchdog_boot_started_at.\$\$; date +%s >\"\$BOOT_TMP\"; mv \"\$BOOT_TMP\" /root/vast_workdir/logs/watchdog_boot_started_at; env | grep _ >> /etc/environment; (export PATH=/root/.local/bin:\$PATH; for _ in \$(seq 1 360); do if [ -s /root/vast_workdir/cloud/vast_api_watchdog.sh ] && [ -s /root/vast_workdir/logs/watchdog_instance_id ]; then INSTANCE_SELF=\$(head -n1 /root/vast_workdir/logs/watchdog_instance_id); case \"\$INSTANCE_SELF\" in *[!0-9]*|'') ;; *) while true; do bash /root/vast_workdir/cloud/vast_api_watchdog.sh $WATCHDOG_ARGUMENTS \"\$INSTANCE_SELF\" $RUN_TAG $HORIZON $SOURCE_SHA $WATCHDOG_BOOT_GRACE_SECONDS $WATCHDOG_EMERGENCY_PHASE_MAX_SECONDS; sleep 30; done ;; esac; fi; sleep 10; done; mkdir -p /root/vast_workdir/logs; INSTANCE_SELF=\${CONTAINER_ID:-\${VAST_CONTAINERLABEL#C.}}; while true; do REMOTE_NOW=\$(date +%s); MARKER_TS=; for MARKER in /root/vast_workdir/logs/controller_heartbeat.env /root/vast_workdir/logs/controller_bootstrap.env; do CANDIDATE_TS=\$(awk -F= '\$1 == \"CONTROLLER_TIMESTAMP\" {print \$2}' \"\$MARKER\" 2>/dev/null || true); case \"\$CANDIDATE_TS\" in *[!0-9]*|'') ;; *) if [ \"\$CANDIDATE_TS\" -le \"\$REMOTE_NOW\" ] && [ \$((REMOTE_NOW - CANDIDATE_TS)) -lt 900 ]; then MARKER_TS=\$CANDIDATE_TS; fi ;; esac; done; if [ -n \"\$MARKER_TS\" ]; then sleep 60; continue; fi; VAST_BIN=\$(command -v vastai 2>/dev/null || true); if [ -z \"\$VAST_BIN\" ]; then timeout --kill-after=30 300 bash -c 'curl --connect-timeout 15 --max-time 240 -fsSL https://vast.ai/install.sh | bash' || true; VAST_BIN=\$(command -v vastai 2>/dev/null || true); fi; case \"\$INSTANCE_SELF\" in *[!0-9]*|'') ;; *) if [ -n \"\$VAST_BIN\" ]; then timeout --kill-after=30 120 \"\$VAST_BIN\" stop instance \"\$INSTANCE_SELF\" --raw >/dev/null 2>&1 || true; fi ;; esac; sleep 60; done) >/tmp/vast-emergency-stop.log 2>&1 &"
ATTACH_INSTANCE_ABSENT=0
if [ "$ATTACH_EXISTING" = 1 ]; then
  if [ "$ATTACH_MATCHES" = 1 ]; then
    INSTANCE_ID="$(jq -r --arg label "$INSTANCE_LABEL" '.[] | select((.label // "") == $label) | .id' <<<"$LIVE_JSON")"
    ATTACH_STATUS="$(jq -r --arg label "$INSTANCE_LABEL" '.[] | select((.label // "") == $label) | (.actual_status // .status // "unknown")' <<<"$LIVE_JSON")"
    REMOTE_STARTED=1
    echo "Attached to existing Vast instance id=$INSTANCE_ID run=$RUN_TAG horizon=$HORIZON"
    case "$ATTACH_STATUS" in
      stopped|exited)
        if ! run_bounded "$API_TIMEOUT_SECONDS" python3 cloud/vast_control_api.py start "$INSTANCE_ID" "$RUN_TAG"; then
          echo "ERROR: exact-label stopped instance id=$INSTANCE_ID could not be started; preserving it." >&2
          exit 1
        fi
        echo "Requested start of exact-label Vast instance id=$INSTANCE_ID; waiting for API running state."
        ;;
    esac
    if [ "$BATCH_MODE" -eq 1 ]; then
      # The instance may now be billable, but no matching runner has been
      # proven yet. Early setup failures must stop it and retain its disk.
      REMOTE_STARTED=0
    fi
  else
    INSTANCE_ID="${VAST_ATTACH_INSTANCE_ID:-}"
    [[ "$INSTANCE_ID" =~ ^[0-9]+$ ]] || {
      echo "ERROR: attach run has no live instance and no recorded numeric instance ID." >&2
      exit 1
    }
    ATTACH_INSTANCE_ABSENT=1
    REMOTE_STARTED=1
    echo "Recorded Vast instance id=$INSTANCE_ID is absent; recovering run=$RUN_TAG from Hugging Face."
  fi
else
  if [ "$SSH_OVER_HTTPS" = 1 ]; then
    CF_FILE="${VAST_CF_QUICK_RESPONSE_FILE:-}"
    if [ "${VAST_CF_QUICK_PROVISION:-0}" = 1 ]; then
      CF_FILE="${CF_FILE:-/tmp/dexposure_cf_quick_${RUN_TAG}.json}"
      umask 077
      run_bounded 30 curl -fsS --connect-timeout 15 --max-time 25 -X POST \
        https://api.trycloudflare.com/tunnel -o "$CF_FILE"
      chmod 600 "$CF_FILE"
      python3 - "$CF_FILE" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
host = ((payload.get("result") or {}).get("hostname") or "")
if payload.get("success") is not True or not str(host).endswith(".trycloudflare.com"):
    raise SystemExit("quick tunnel provision failed")
print(f"Provisioned Quick Tunnel hostname={host}")
PY
    fi
    if [ -n "$CF_FILE" ]; then
      test -f "$CF_FILE" || {
        echo "ERROR: VAST_CF_QUICK_RESPONSE_FILE is missing." >&2
        exit 1
      }
      encoded="$(base64 < "$CF_FILE" | tr -d '\n')"
      [[ "$encoded" =~ ^[A-Za-z0-9+/]+=*$ ]] || {
        echo "ERROR: Quick Tunnel payload is not base64." >&2
        exit 1
      }
      ONSTART+=$'\n'
      ONSTART+="umask 077
python3 - <<'DEXPOSURE_PROVISION_RESPONSE'
import base64
from pathlib import Path
path = Path('/root/.dexposure_cf_quick_response')
path.write_bytes(base64.b64decode('${encoded}'))
path.chmod(0o600)
DEXPOSURE_PROVISION_RESPONSE
"
    fi
    printf -v HTTPS_BOOTSTRAP_Q '%q' "$(<cloud/vast_https_bootstrap.sh)"
    ONSTART+=$'\n'
    ONSTART+="nohup bash -c $HTTPS_BOOTSTRAP_Q -- $RUN_TAG $SOURCE_SHA >/proc/1/fd/1 2>&1 </dev/null &"
  fi
  set +e
  CREATE_JSON="$(run_bounded "$API_TIMEOUT_SECONDS" vastai create instance "$OFFER_ID" \
    --image vastai/base:0.0.2 --disk 80 --label "$INSTANCE_LABEL" \
    --ssh --direct --cancel-unavail --onstart-cmd "$ONSTART" --raw)"
  CREATE_STATUS=$?
  set -e
  INSTANCE_ID="$(jq -r '.new_contract // empty' <<<"$CREATE_JSON" 2>/dev/null || true)"
  if [ -z "$INSTANCE_ID" ]; then
    echo "Vast create response did not contain an instance ID; reconciling by exact label." >&2
    MATCHING_IDS=""
    MATCHING_COUNT=0
    for _ in $(seq 1 30); do
      sleep 5
      if ! RECONCILE_JSON="$(list_instances 2>/dev/null)" || \
         ! jq -e 'type == "array"' >/dev/null <<<"$RECONCILE_JSON"; then
        echo "WARNING: Vast create reconciliation query failed; retrying exact label $INSTANCE_LABEL." >&2
        continue
      fi
      MATCHING_IDS="$(jq -r --arg label "$INSTANCE_LABEL" '.[] | select((.label // "") == $label) | .id' <<<"$RECONCILE_JSON")"
      MATCHING_COUNT="$(awk 'NF {count++} END {print count + 0}' <<<"$MATCHING_IDS")"
      if [ "$MATCHING_COUNT" -gt 0 ]; then
        break
      fi
    done
    if [ "$MATCHING_COUNT" = 1 ]; then
      INSTANCE_ID="$MATCHING_IDS"
    elif [ "$MATCHING_COUNT" -gt 1 ]; then
      echo "ERROR: create reconciliation found $MATCHING_COUNT instances with label $INSTANCE_LABEL; ownership is ambiguous, so all are preserved." >&2
      exit 1
    else
      echo "ERROR: Vast create remained ambiguous for 150 seconds (status=$CREATE_STATUS, label=$INSTANCE_LABEL): $CREATE_JSON" >&2
      echo "ERROR: no second create will be attempted; a delayed instance still has the emergency watchdog." >&2
      exit 1
    fi
  fi
  echo "Created Vast instance id=$INSTANCE_ID run=$RUN_TAG horizon=$HORIZON"

  # The local lock closes the common race. This account-level arbitration also
  # catches controllers started from another machine. A losing controller only
  # destroys its own new instance.
  ARBITRATION_STABLE=0
  for _ in $(seq 1 24); do
    sleep 5
    ARBITRATION_JSON="$(list_instances)" || continue
    jq -e 'type == "array"' >/dev/null <<<"$ARBITRATION_JSON" || continue
    ARBITRATION_IDS="$(jq -r '.[] | select(((.label // "") | startswith("dexposure-v2-"))) | .id' <<<"$ARBITRATION_JSON")"
    ARBITRATION_COUNT="$(awk 'NF {count++} END {print count + 0}' <<<"$ARBITRATION_IDS")"
    ARBITRATION_WINNER="$(awk 'NF && ($1 + 0 < winner || winner == 0) {winner=$1 + 0} END {print winner + 0}' <<<"$ARBITRATION_IDS")"
    if ! grep -qx "$INSTANCE_ID" <<<"$ARBITRATION_IDS"; then
      echo "ERROR: newly created instance id=$INSTANCE_ID disappeared during arbitration." >&2
      INSTANCE_ID=""
      exit 1
    fi
    if [ "$ARBITRATION_COUNT" -gt 1 ] && [ "$INSTANCE_ID" != "$ARBITRATION_WINNER" ]; then
      echo "ERROR: concurrent Vast create detected; this controller lost arbitration." >&2
      if [ "$BATCH_MODE" -eq 1 ]; then
        stop_instance_preserving_disk "$INSTANCE_ID"
        REMOTE_STARTED=1
      else
        destroy_instance "$INSTANCE_ID"
      fi
      exit 1
    fi
    if [ "$ARBITRATION_COUNT" = 1 ]; then
      ARBITRATION_STABLE=$((ARBITRATION_STABLE + 1))
      [ "$ARBITRATION_STABLE" -ge 3 ] && break
    else
      ARBITRATION_STABLE=0
    fi
  done
  if [ "$ARBITRATION_STABLE" -lt 3 ]; then
    echo "ERROR: account-level single-instance arbitration did not stabilize." >&2
    if [ "$BATCH_MODE" -eq 1 ]; then
      stop_instance_preserving_disk "$INSTANCE_ID"
      REMOTE_STARTED=1
    else
      destroy_instance "$INSTANCE_ID"
    fi
    exit 1
  fi
  release_create_lock
fi
printf 'INSTANCE_ID=%s\n' "$INSTANCE_ID" >>"$CONTROLLER_STATE"
if [ -n "${VAST_ATTEMPT_FILE:-}" ]; then
  ATTEMPT_TMP="${VAST_ATTEMPT_FILE}.tmp.$$"
  printf 'STATE=active\nRUN_TAG=%s\nHORIZON=%s\nSOURCE_SHA256=%s\nINSTANCE_ID=%s\nCREDIT_BEFORE=%s\n' \
    "$RUN_TAG" "$HORIZON" "$SOURCE_SHA" "$INSTANCE_ID" "${VAST_CREDIT_BEFORE:-unknown}" >"$ATTEMPT_TMP"
  mv "$ATTEMPT_TMP" "$VAST_ATTEMPT_FILE"
fi
run_bounded "$TRANSFER_TIMEOUT_SECONDS" uvx --from 'huggingface_hub[cli]' hf upload "$HF_REPO" \
  "$CONTROLLER_STATE" "runs/$RUN_TAG/controller/preflight.env" --repo-type model \
  --commit-message "Vast controller bind instance $RUN_TAG" >/dev/null

SSH=()
SSH_TRANSPORT=""
SSH_HOST=""
SSH_PORT=""
REMOTE_PID="0"
INITIAL_REMOTE_DEAD_COUNT=0
NEEDS_REMOTE_START=0

if [ "$ATTACH_INSTANCE_ABSENT" -eq 1 ]; then
  INITIAL_REMOTE_DEAD_COUNT=3
else
  INSTANCE_JSON=""
  STATUS=""
  for _ in $(seq 1 90); do
    INSTANCE_JSON="$(run_bounded "$API_TIMEOUT_SECONDS" vastai show instance "$INSTANCE_ID" --raw 2>/dev/null || true)"
    STATUS="$(jq -r '.instances.actual_status // .actual_status // empty' <<<"$INSTANCE_JSON" 2>/dev/null || true)"
    SSH_HOST="$(jq -r '.instances.ssh_host // .ssh_host // empty' <<<"$INSTANCE_JSON" 2>/dev/null || true)"
    SSH_PORT="$(jq -r '.instances.ssh_port // .ssh_port // empty' <<<"$INSTANCE_JSON" 2>/dev/null || true)"
    if [ "$STATUS" = running ] && [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ] && [ "$SSH_PORT" != null ]; then
      break
    fi
    sleep 10
  done
  test "$STATUS" = running || { echo "ERROR: Vast instance did not reach running state." >&2; exit 1; }
  test -n "$SSH_HOST" && test -n "$SSH_PORT" || { echo "ERROR: Vast SSH endpoint is unavailable." >&2; exit 1; }

  SSH_OPTIONS=(-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -o ConnectTimeout=15 -p "$SSH_PORT")
  if [ "$SSH_OVER_HTTPS" = 1 ]; then
    HTTPS_READY=0
    HTTPS_DEADLINE=$(( $(date +%s) + 720 ))
    while [ "$(date +%s)" -lt "$HTTPS_DEADLINE" ]; do
      if run_bounded 60 python3 cloud/vast_control_api.py discover "$INSTANCE_ID" "$RUN_TAG" \
          --source-sha "$SOURCE_SHA" --output "$CONTROLLER_DIR" --client "$CLOUDFLARED_BIN"; then
        HTTPS_READY=1
        break
      fi
      sleep 5
    done
    [ "$HTTPS_READY" = 1 ] || { echo "ERROR: HTTPS SSH bootstrap failed; preserving instance disk." >&2; exit 1; }
    SSH_OPTIONS=(-F "$CONTROLLER_DIR/https_ssh.config" -o StrictHostKeyChecking=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -o ConnectTimeout=15 -p "$SSH_PORT")
    if [ -f "$HOME/.ssh/id_ed25519.pub" ]; then
      run_bounded 30 vastai attach ssh "$INSTANCE_ID" "$HOME/.ssh/id_ed25519.pub" --raw >/dev/null || true
    fi
  fi
  SSH=(ssh "${SSH_OPTIONS[@]}" "root@$SSH_HOST")
  printf -v SSH_TRANSPORT '%q ' ssh "${SSH_OPTIONS[@]}"
  REMOTE_PID_PROBE_COMMAND='cd /root/vast_workdir 2>/dev/null || exit 0; if [ -s logs/remote_runner.pid ]; then read -r pid expected_start phase <logs/remote_runner.pid; case "$pid" in *[!0-9]*|"") echo invalid; exit ;; esac; case "$expected_start" in *[!0-9]*|"") echo invalid; exit ;; esac; if [ "$phase" != running ]; then printf "dead:%s\n" "$pid"; exit; fi; current_start=$(awk '\''{print $22}'\'' "/proc/$pid/stat" 2>/dev/null || true); state=$(awk '\''{print $3}'\'' "/proc/$pid/stat" 2>/dev/null || true); if kill -0 "$pid" 2>/dev/null && [ "$current_start" = "$expected_start" ] && [ "$state" != Z ]; then printf "alive:%s\n" "$pid"; else printf "dead:%s\n" "$pid"; fi; else echo absent; fi'
  for _ in $(seq 1 30); do
    if "${SSH[@]}" true >/dev/null 2>&1; then
      break
    fi
    sleep 5
  done
  "${SSH[@]}" true >/dev/null
  "${SSH[@]}" "mkdir -p /root/vast_workdir/logs; tmp=/root/vast_workdir/logs/.controller_bootstrap.env.\$\$; printf 'INSTANCE_ID=%s\\nRUN_TAG=%s\\nHORIZON=%s\\nSOURCE_SHA256=%s\\nCONTROLLER_TIMESTAMP=%s\\n' '$INSTANCE_ID' '$RUN_TAG' '$HORIZON' '$SOURCE_SHA' \"\$(date +%s)\" >\"\$tmp\" && mv \"\$tmp\" /root/vast_workdir/logs/controller_bootstrap.env" >/dev/null

  # Check only that the non-interactive Vast CLI is installed and executable.
  # The injected instance credential is intentionally not required to read the
  # account-level instance list or user record. Exact-id stop is the watchdog's
  # only remote lifecycle mutation.
  if ! "${SSH[@]}" 'export PATH=/root/.local/bin:$PATH; VAST_BIN=$(command -v vastai 2>/dev/null || true); if [ -z "$VAST_BIN" ]; then timeout --kill-after=30 300 bash -c '\''set -o pipefail; curl --connect-timeout 15 --max-time 240 -fsSL https://vast.ai/install.sh | bash'\'' >/tmp/vast-install-controller.log 2>&1 || exit; VAST_BIN=$(command -v vastai 2>/dev/null || true); fi; [ -n "$VAST_BIN" ] || exit 127; timeout --kill-after=30 120 "$VAST_BIN" --help >/tmp/vast-cli-help-controller.log 2>&1'; then
    echo "ERROR: emergency watchdog Vast CLI readiness check failed." >&2
    exit 1
  fi
  echo "Emergency watchdog Vast CLI readiness passed; account-list credential access was not tested."

  GPU_LINES="$("${SSH[@]}" "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")"
  GPU_COUNT="$(awk 'NF {count++} END {print count + 0}' <<<"$GPU_LINES")"
  test "$GPU_COUNT" = 1 || {
    echo "ERROR: expected exactly one allocated GPU, but nvidia-smi reported $GPU_COUNT." >&2
    exit 1
  }
  GPU_LINE="$GPU_LINES"
  echo "Actual Vast GPU: $GPU_LINE"
  case "$GPU_LINE" in
    *RTX\ 3090*|*RTX\ 4090*|*A10*|*A40*|*A100*|*V100*|*L40S*|*RTX\ A5000*|*RTX\ A6000*|*RTX\ 6000\ Ada*) ;;
    *) echo "ERROR: actual GPU is outside the compatible allowlist: $GPU_LINE" >&2; exit 1 ;;
  esac
  GPU_MEMORY_MIB="$(sed -E 's/.*,[[:space:]]*([0-9]+).*/\1/' <<<"$GPU_LINE")"
  test "$GPU_MEMORY_MIB" -ge 40000 || { echo "ERROR: actual GPU has less than 40 GB." >&2; exit 1; }
  test "$GPU_MEMORY_MIB" -le 49152 || { echo "ERROR: actual GPU exceeds 48 GiB." >&2; exit 1; }
  "${SSH[@]}" 'for command in timeout flock tar find sort setsid seq; do command -v "$command" >/dev/null 2>&1 || exit 1; done; timeout --kill-after=1 1 true; mkdir -p /root/vast_workdir/logs' || {
    echo "ERROR: remote host lacks a bounded recovery command; refusing paid training." >&2
    exit 1
  }
  start_controller_heartbeat || {
    echo "ERROR: local controller heartbeat could not be written; refusing paid training." >&2
    exit 1
  }

  if [ "$ATTACH_EXISTING" = 1 ]; then
    RUN_IDENTITY=""
    PID_PROBE="unknown"
    ATTACH_ABSENT_CHECKS=0
    for _ in $(seq 1 12); do
      RUN_IDENTITY="$("${SSH[@]}" 'test -s /root/vast_workdir/logs/run_identity.env && cat /root/vast_workdir/logs/run_identity.env || true')"
      PID_PROBE="$("${SSH[@]}" "$REMOTE_PID_PROBE_COMMAND")"
      if [ -n "$RUN_IDENTITY" ] || [ "$PID_PROBE" != absent ]; then
        break
      fi
      ATTACH_ABSENT_CHECKS=$((ATTACH_ABSENT_CHECKS + 1))
      sleep 5
    done
    if [ -n "$RUN_IDENTITY" ]; then
      [ "$(awk -F= '$1 == "RUN_TAG" {print $2}' <<<"$RUN_IDENTITY")" = "$RUN_TAG" ] && \
      [ "$(awk -F= '$1 == "HORIZON" {print $2}' <<<"$RUN_IDENTITY")" = "$HORIZON" ] && \
      [ "$(awk -F= '$1 == "SOURCE_SHA256" {print $2}' <<<"$RUN_IDENTITY")" = "$SOURCE_SHA" ] && \
      [ "$(awk -F= '$1 == "INSTANCE_ID" {print $2}' <<<"$RUN_IDENTITY")" = "$INSTANCE_ID" ] || {
        echo "ERROR: live instance run identity does not match the recorded attempt; preserving it." >&2
        exit 1
      }
      case "$PID_PROBE" in
        alive:*)
          REMOTE_PID="${PID_PROBE#alive:}"
          REMOTE_STARTED=1
          echo "Attached to the existing runner pid=$REMOTE_PID state=alive."
          ;;
        dead:*)
          REMOTE_PID="${PID_PROBE#dead:}"
          NEEDS_REMOTE_START=1
          echo "Attached instance has a dead runner pid=$REMOTE_PID; relaunching on the same disk."
          ;;
        *) echo "ERROR: recorded run identity exists but its runner PID is unavailable; preserving instance id=$INSTANCE_ID." >&2; exit 1 ;;
      esac
    elif [ "$PID_PROBE" = absent ] && [ "$ATTACH_ABSENT_CHECKS" -ge 12 ]; then
      if "${SSH[@]}" "{ find /root/vast_workdir/checkpoints/main2025_v2_h${HORIZON} /root/vast_workdir/checkpoints/main2025_v2_h${HORIZON}_train -type f -size +0c 2>/dev/null; find /root/vast_workdir/logs -type f \( -name 'controller_${RUN_TAG}.log' -o -name 'train_${RUN_TAG}.log' -o -name 'postflight_${RUN_TAG}.log' -o -name 'artifact_manifest_${RUN_TAG}.sha256' -o -name 'local_run_status.env' \) -size +0c 2>/dev/null; } | grep -q ."; then
        echo "ERROR: recorded instance has artifacts but no run identity; preserving it for manual recovery." >&2
        exit 1
      fi
      NEEDS_REMOTE_START=1
      echo "The recorded instance has no runner; resuming setup on the same instance."
    else
      echo "ERROR: runner PID exists without a matching run identity; preserving instance id=$INSTANCE_ID." >&2
      exit 1
    fi
  else
    NEEDS_REMOTE_START=1
  fi

  if [ "$NEEDS_REMOTE_START" -eq 1 ]; then
    "${SSH[@]}" "mkdir -p /root/vast_workdir/cloud /root/vast_workdir/data /root/vast_workdir/checkpoints /root/vast_workdir/logs"
    "${SSH[@]}" 'if [ -s /root/vast_workdir/logs/watchdog.pid ]; then read -r pid expected_start phase </root/vast_workdir/logs/watchdog.pid; case "$pid" in *[!0-9]*|"") ;; *) kill "$pid" 2>/dev/null || true ;; esac; fi; rm -rf /tmp/dexposure-vast-watchdog.lock'
    run_bounded "$TRANSFER_TIMEOUT_SECONDS" rsync -az -e "$SSH_TRANSPORT" \
      cloud/vast_api_watchdog.sh cloud/vast_api_emergency_snapshot.sh \
      "root@$SSH_HOST:/root/vast_workdir/cloud/"
    printf '%s\n' "$INSTANCE_ID" | "${SSH[@]}" 'cat > /root/vast_workdir/logs/watchdog_instance_id'
    "${SSH[@]}" "cd /root/vast_workdir || exit 1; setsid -f bash -c 'while true; do bash cloud/vast_api_watchdog.sh $WATCHDOG_ARGUMENTS $INSTANCE_ID $RUN_TAG $HORIZON $SOURCE_SHA $WATCHDOG_BOOT_GRACE_SECONDS $WATCHDOG_EMERGENCY_PHASE_MAX_SECONDS; sleep 30; done' >/tmp/vast-emergency-destroy.log 2>&1 </dev/null"
    WATCHDOG_READY=0
    for _ in $(seq 1 12); do
      if "${SSH[@]}" 'read -r pid expected_start phase </root/vast_workdir/logs/watchdog.pid 2>/dev/null || exit 1; case "$pid:$expected_start" in *[!0-9:]*) exit 1 ;; esac; [ "$phase" = running ] || exit 1; current_start=$(awk '\''{print $22}'\'' "/proc/$pid/stat" 2>/dev/null || true); state=$(awk '\''{print $3}'\'' "/proc/$pid/stat" 2>/dev/null || true); kill -0 "$pid" 2>/dev/null && [ "$current_start" = "$expected_start" ] && [ "$state" != Z ]'; then
        WATCHDOG_READY=1
        break
      fi
      sleep 5
    done
    [ "$WATCHDOG_READY" -eq 1 ] || {
      echo "ERROR: remote emergency watchdog is not alive; refusing to start paid training." >&2
      exit 1
    }
    # Both modes sync the repository exactly once after the emergency watchdog
    # is confirmed alive. Batch inputs then come from private Hugging Face.
    run_bounded "$TRANSFER_TIMEOUT_SECONDS" rsync -az --delete --exclude=.git --exclude-from=.vastignore \
      -e "$SSH_TRANSPORT" ./ "root@$SSH_HOST:/root/vast_workdir/"
    if [ -z "$INPUT_FETCH_CMD" ]; then
      run_bounded "$TRANSFER_TIMEOUT_SECONDS" rsync -az -e "$SSH_TRANSPORT" \
        data/historical-network_week_2020-03-30.json \
        "root@$SSH_HOST:/root/vast_workdir/data/historical-network_week_2020-03-30.json"
      run_bounded "$TRANSFER_TIMEOUT_SECONDS" rsync -az -e "$SSH_TRANSPORT" \
        data/historical-network_week_2025-07-01.json \
        "root@$SSH_HOST:/root/vast_workdir/data/historical-network_week_2025-07-01.json"
      run_bounded "$TRANSFER_TIMEOUT_SECONDS" rsync -az -e "$SSH_TRANSPORT" cloud/upload/data/meta_df.csv \
        "root@$SSH_HOST:/root/vast_workdir/data/meta_df.csv"
      run_bounded "$TRANSFER_TIMEOUT_SECONDS" rsync -az -e "$SSH_TRANSPORT" cloud/upload/graphpfn-v1.ckpt \
        "root@$SSH_HOST:/root/vast_workdir/checkpoints/graphpfn-v1.ckpt"
    fi

    START_LEASE_RESULT="$("${SSH[@]}" 'cd /root/vast_workdir && if [ -d logs/watchdog_destroy.lock ]; then echo teardown; elif mkdir logs/runner_start.lock 2>/dev/null; then date +%s >logs/runner_start.lock/created_at; if [ -d logs/watchdog_destroy.lock ]; then rm -f logs/runner_start.lock/created_at; rmdir logs/runner_start.lock 2>/dev/null || true; echo teardown; else echo acquired; fi; else echo held; fi')"
    START_LEASE_ACQUIRED=0
    case "$START_LEASE_RESULT" in
      acquired) START_LEASE_ACQUIRED=1 ;;
      held)
        [ "$BATCH_MODE" -eq 1 ] || REMOTE_STARTED=1
        echo "Another controller owns the remote runner start lease; attaching without launching a second runner."
        for _ in $(seq 1 24); do
          RUN_IDENTITY="$("${SSH[@]}" 'test -s /root/vast_workdir/logs/run_identity.env && cat /root/vast_workdir/logs/run_identity.env || true')"
          PID_PROBE="$("${SSH[@]}" "$REMOTE_PID_PROBE_COMMAND")"
          if [ -n "$RUN_IDENTITY" ]; then
            break
          fi
          sleep 5
        done
        [ "$(awk -F= '$1 == "RUN_TAG" {print $2}' <<<"$RUN_IDENTITY")" = "$RUN_TAG" ] && \
        [ "$(awk -F= '$1 == "HORIZON" {print $2}' <<<"$RUN_IDENTITY")" = "$HORIZON" ] && \
        [ "$(awk -F= '$1 == "SOURCE_SHA256" {print $2}' <<<"$RUN_IDENTITY")" = "$SOURCE_SHA" ] && \
        [ "$(awk -F= '$1 == "INSTANCE_ID" {print $2}' <<<"$RUN_IDENTITY")" = "$INSTANCE_ID" ] || {
          echo "ERROR: start-lease owner did not produce the matching run identity; preserving instance id=$INSTANCE_ID." >&2
          exit 1
        }
        case "$PID_PROBE" in
          alive:*) REMOTE_PID="${PID_PROBE#alive:}" ;;
          dead:*) REMOTE_PID="${PID_PROBE#dead:}"; INITIAL_REMOTE_DEAD_COUNT=1 ;;
          *) echo "ERROR: start-lease owner produced no valid runner PID; preserving instance id=$INSTANCE_ID." >&2; exit 1 ;;
        esac
        REMOTE_STARTED=1
        ;;
      teardown)
        echo "ERROR: watchdog teardown lease is active; no runner was started." >&2
        exit 1
        ;;
      *) echo "ERROR: invalid remote start-lease response: $START_LEASE_RESULT" >&2; exit 1 ;;
    esac

    if [ "$START_LEASE_ACQUIRED" -eq 1 ]; then
      printf '%s' "$HF_TOKEN" | "${SSH[@]}" "umask 077; cat > /root/.hf_token"
      printf -v TRAIN_CMD_Q '%q' "$TRAIN_CMD"
      printf -v SOURCE_SHA_Q '%q' "$SOURCE_SHA"
      printf -v RUN_TAG_Q '%q' "$RUN_TAG"
      printf -v HORIZON_Q '%q' "$HORIZON"
      printf -v INSTANCE_ID_Q '%q' "$INSTANCE_ID"
      printf -v TRAIN_TIMEOUT_Q '%q' "$TRAIN_TIMEOUT_SECONDS"
      printf -v INPUT_FETCH_CMD_Q '%q' "$INPUT_FETCH_CMD"
      # Enter the handed-off state before asking SSH to fork. If the SSH response is
      # lost after the fork, the EXIT trap must not destroy a possibly running job.
      REMOTE_STARTED=1
      set +e
      REMOTE_PID="$("${SSH[@]}" "cd /root/vast_workdir || exit 1; setsid nohup bash cloud/vast_api_remote_run.sh $TRAIN_CMD_Q $SOURCE_SHA_Q $RUN_TAG_Q $HORIZON_Q $INSTANCE_ID_Q $TRAIN_TIMEOUT_Q $INPUT_FETCH_CMD_Q >logs/controller_${RUN_TAG}.log 2>&1 </dev/null & echo \$!")"
      HANDOFF_STATUS=$?
      set -e
      if [ "$HANDOFF_STATUS" -ne 0 ] || ! [[ "$REMOTE_PID" =~ ^[0-9]+$ ]]; then
        echo "Remote handoff response was ambiguous; reconciling remote_runner.pid without destroying the instance." >&2
        REMOTE_PID=""
        HANDOFF_ABSENT_CHECKS=0
        for _ in $(seq 1 12); do
          if PID_PROBE="$("${SSH[@]}" "$REMOTE_PID_PROBE_COMMAND" 2>/dev/null)"; then
            case "$PID_PROBE" in
              alive:*) REMOTE_PID="${PID_PROBE#alive:}"; break ;;
              dead:*) REMOTE_PID="${PID_PROBE#dead:}"; INITIAL_REMOTE_DEAD_COUNT=1; break ;;
              absent) HANDOFF_ABSENT_CHECKS=$((HANDOFF_ABSENT_CHECKS + 1)) ;;
            esac
          fi
          sleep 5
        done
        if [ -z "$REMOTE_PID" ]; then
          if [ "$HANDOFF_ABSENT_CHECKS" -ge 3 ]; then
            "${SSH[@]}" 'rm -f /root/vast_workdir/logs/runner_start.lock/created_at; rmdir /root/vast_workdir/logs/runner_start.lock 2>/dev/null || true' || true
            REMOTE_STARTED=0
            echo "ERROR: remote handoff failed and the runner PID file was confirmed absent." >&2
          else
            echo "ERROR: remote handoff state remains unknown; preserving instance id=$INSTANCE_ID for recovery." >&2
          fi
          exit 1
        fi
      fi
      sleep 3
      if ! PID_PROBE="$("${SSH[@]}" "$REMOTE_PID_PROBE_COMMAND" 2>/dev/null)"; then
        echo "Remote runner liveness is temporarily unknown; completion-marker polling will reconcile it." >&2
      else
        case "$PID_PROBE" in
          alive:*) REMOTE_PID="${PID_PROBE#alive:}" ;;
          dead:*)
            REMOTE_PID="${PID_PROBE#dead:}"
            INITIAL_REMOTE_DEAD_COUNT=1
            echo "Remote runner exited quickly; completion-marker polling will recover its status and artifacts." >&2
            ;;
          *) echo "Remote runner identity is not available yet; completion-marker polling will reconcile it." >&2 ;;
        esac
      fi
      echo "Remote runner was handed off with pid=$REMOTE_PID; SSH loss will not terminate an active runner."
    fi
  fi
fi
echo "VAST_RUN_TAG=$RUN_TAG"

remote_runner_progress_fresh() {
  local payload="" progress_timestamp="" remote_now=""
  [ "$ATTACH_INSTANCE_ABSENT" -eq 0 ] || return 2
  payload="$("${SSH[@]}" 'test -s /root/vast_workdir/logs/runner_progress.env || exit 3; cat /root/vast_workdir/logs/runner_progress.env; printf "REMOTE_NOW=%s\n" "$(date +%s)"' 2>/dev/null)" || return 2
  awk -F= '
    NF != 2 { exit 1 }
    $1 !~ /^(INSTANCE_ID|RUN_TAG|HORIZON|SOURCE_SHA256|PHASE|PROGRESS_TIMESTAMP|REMOTE_NOW)$/ { exit 1 }
    { seen[$1]++ }
    END {
      if (NR != 7) exit 1
      for (key in seen) if (seen[key] != 1) exit 1
    }
  ' <<<"$payload" || return 1
  [ "$(awk -F= '$1 == "INSTANCE_ID" {print $2}' <<<"$payload")" = "$INSTANCE_ID" ] && \
    [ "$(awk -F= '$1 == "RUN_TAG" {print $2}' <<<"$payload")" = "$RUN_TAG" ] && \
    [ "$(awk -F= '$1 == "HORIZON" {print $2}' <<<"$payload")" = "$HORIZON" ] && \
    [ "$(awk -F= '$1 == "SOURCE_SHA256" {print $2}' <<<"$payload")" = "$SOURCE_SHA" ] || return 1
  progress_timestamp="$(awk -F= '$1 == "PROGRESS_TIMESTAMP" {print $2}' <<<"$payload")"
  remote_now="$(awk -F= '$1 == "REMOTE_NOW" {print $2}' <<<"$payload")"
  case "$progress_timestamp:$remote_now" in *[!0-9:]*) return 1 ;; esac
  [ "$progress_timestamp" -le "$remote_now" ] && \
    [ $((remote_now - progress_timestamp)) -le "$WATCHDOG_RUNNER_STALE_SECONDS" ]
}

STATUS_REMOTE="runs/$RUN_TAG/logs/run_status.env"
STATUS_ROOT="$STATE_ROOT/status"
mkdir -p "$STATUS_ROOT"
STATUS_FILE="$STATUS_ROOT/$STATUS_REMOTE"
MARKER_MATCHED=0
REMOTE_DEAD_COUNT="$INITIAL_REMOTE_DEAD_COUNT"
CONTROLLER_HEARTBEAT_RELINQUISHED=0
for attempt in $(seq 1 2100); do
  if run_bounded "$API_TIMEOUT_SECONDS" uvx --from 'huggingface_hub[cli]' hf download "$HF_REPO" \
      --repo-type model --include "$STATUS_REMOTE" --local-dir "$STATUS_ROOT" \
      --force-download >/dev/null 2>&1 && [ -s "$STATUS_FILE" ]; then
    MARKER_INSTANCE="$(awk -F= '$1 == "INSTANCE_ID" {print $2}' "$STATUS_FILE")"
    MARKER_HORIZON="$(awk -F= '$1 == "HORIZON" {print $2}' "$STATUS_FILE")"
    MARKER_SOURCE="$(awk -F= '$1 == "SOURCE_SHA256" {print $2}' "$STATUS_FILE")"
    if [ "$MARKER_INSTANCE" = "$INSTANCE_ID" ] && \
       [ "$MARKER_HORIZON" = "$HORIZON" ] && \
       [ "$MARKER_SOURCE" = "$SOURCE_SHA" ]; then
      MARKER_MATCHED=1
      break
    fi
    echo "Ignoring a completion marker that does not bind instance=$INSTANCE_ID h=$HORIZON source=$SOURCE_SHA." >&2
  fi
  if [ "$ATTACH_INSTANCE_ABSENT" -eq 1 ]; then
    echo "Recorded instance is already absent; switching immediately to Hugging Face recovery."
    break
  fi
  if [ $((attempt % 5)) -eq 0 ]; then
    CURRENT_STATUS="api-error"
    if CURRENT_INSTANCE_JSON="$(run_bounded "$API_TIMEOUT_SECONDS" vastai show instance "$INSTANCE_ID" --raw 2>/dev/null)" && \
       jq -e 'type == "object"' >/dev/null <<<"$CURRENT_INSTANCE_JSON"; then
      CURRENT_STATUS="$(jq -r '.instances.actual_status // .actual_status // "not-listed"' <<<"$CURRENT_INSTANCE_JSON")"
    fi
    echo "Waiting for h=$HORIZON completion marker; instance=$INSTANCE_ID status=$CURRENT_STATUS minutes=$attempt"
    if [ "$CURRENT_STATUS" = running ]; then
      if PID_PROBE="$("${SSH[@]}" "$REMOTE_PID_PROBE_COMMAND" 2>/dev/null)"; then
        case "$PID_PROBE" in
          alive:*)
            REMOTE_PID="${PID_PROBE#alive:}"
            REMOTE_DEAD_COUNT=0
            if [ "$CONTROLLER_HEARTBEAT_RELINQUISHED" -eq 0 ]; then
              set +e
              remote_runner_progress_fresh
              PROGRESS_STATUS=$?
              set -e
              if [ "$PROGRESS_STATUS" -eq 1 ]; then
                echo "Runner progress exceeded $WATCHDOG_RUNNER_STALE_SECONDS seconds; the controller is handing recovery to the emergency watchdog." >&2
                stop_controller_heartbeat
                CONTROLLER_HEARTBEAT_RELINQUISHED=1
              elif [ "$PROGRESS_STATUS" -eq 2 ]; then
                echo "Runner progress could not be read; this check does not declare a stall." >&2
              fi
            fi
            ;;
          dead:*|absent)
            [ "$PID_PROBE" = absent ] || REMOTE_PID="${PID_PROBE#dead:}"
            REMOTE_DEAD_COUNT=$((REMOTE_DEAD_COUNT + 1))
            echo "Remote runner was confirmed inactive on check $REMOTE_DEAD_COUNT/3." >&2
            ;;
          *) echo "Remote runner identity was malformed; this check does not count as a dead runner." >&2 ;;
        esac
      else
        echo "SSH transport was unavailable; this check does not count as a dead runner." >&2
      fi
    elif [ "$CURRENT_STATUS" != api-error ]; then
      REMOTE_DEAD_COUNT=$((REMOTE_DEAD_COUNT + 1))
    else
      echo "Vast status API was unavailable; this check does not count as a dead runner." >&2
    fi
    if [ "$REMOTE_DEAD_COUNT" -ge 3 ]; then
      echo "Remote runner is no longer alive and no matching marker arrived." >&2
      break
    fi
  fi
  sleep 60
done

RECOVERED_BUNDLE=""
RECOVERED_TRAIN_ROOT=""
RECOVERED_LOG_ROOT=""
RECOVERED_FAILURE_ROOT=""
EMERGENCY_RECOVERED_ROOT=""
EMERGENCY_RECOVERED_SHA=""
EMERGENCY_RECOVERY_RECEIPT=""
RECOVERY_REMOTE_INSPECTED=0
validate_complete_bundle() {
  local run_root="$1"
  local model_dir="$run_root/$RELEASE_REL"
  local train_dir="$run_root/$TRAIN_REL/finetuned"
  local log_dir="$run_root/logs"
  local identity_file="$log_dir/run_identity.env"
  local local_status_file="$log_dir/local_run_status.env"
  local artifact_manifest="$log_dir/artifact_manifest_${RUN_TAG}.sha256"
  local required_path input_status train_status post_status upload_status marker_status horizon config_file
  local -a required_paths=(
    "$model_dir/feature_schema.json" "$model_dir/task1_metrics.json"
    "$model_dir/SHA256SUMS" "$train_dir/feature_schema.json"
    "$train_dir/experiment_results.json" "$train_dir/data_quality.json"
    "$train_dir/metrics.json" "$train_dir/all_results.json"
    "$log_dir/train_${RUN_TAG}.log" "$log_dir/postflight_${RUN_TAG}.log"
    "$log_dir/controller_${RUN_TAG}.log" "$identity_file"
    "$log_dir/upload_release_${RUN_TAG}.log"
    "$log_dir/upload_train_${RUN_TAG}.log"
    "$local_status_file" "$artifact_manifest"
  )
  if [ "$BATCH_MODE" -eq 1 ]; then
    required_paths+=(
      "$log_dir/input_transport_status.env"
      "$log_dir/input_manifest.sha256"
      "$log_dir/input_download_${RUN_TAG}.log"
    )
    for horizon in $BATCH_HORIZONS; do
      required_paths+=(
        "$model_dir/dexposure-fm-h${horizon}.pt"
        "$model_dir/run_config_h${horizon}.json"
        "$train_dir/best_model_h${horizon}.pt"
      )
    done
  else
    required_paths+=(
      "$model_dir/dexposure-fm-h${HORIZON}.pt"
      "$model_dir/run_config.json"
      "$train_dir/best_model_h${HORIZON}.pt"
    )
  fi

  for required_path in "${required_paths[@]}"; do
    [ -s "$required_path" ] || return 1
  done

  for identity_file in "$identity_file" "$local_status_file"; do
    [ "$(awk -F= '$1 == "RUN_TAG" {print $2}' "$identity_file")" = "$RUN_TAG" ] || return 1
    [ "$(awk -F= '$1 == "HORIZON" {print $2}' "$identity_file")" = "$HORIZON" ] || return 1
    [ "$(awk -F= '$1 == "SOURCE_SHA256" {print $2}' "$identity_file")" = "$SOURCE_SHA" ] || return 1
    [ "$(awk -F= '$1 == "INSTANCE_ID" {print $2}' "$identity_file")" = "$INSTANCE_ID" ] || return 1
  done
  input_status="$(awk -F= '$1 == "INPUT_STATUS" {print $2}' "$local_status_file")"
  train_status="$(awk -F= '$1 == "TRAIN_STATUS" {print $2}' "$local_status_file")"
  post_status="$(awk -F= '$1 == "POST_STATUS" {print $2}' "$local_status_file")"
  upload_status="$(awk -F= '$1 == "UPLOAD_STATUS" {print $2}' "$local_status_file")"
  marker_status="$(awk -F= '$1 == "MARKER_STATUS" {print $2}' "$local_status_file")"
  [[ "$input_status" =~ ^[0-9]+$ ]] && [[ "$train_status" =~ ^[0-9]+$ ]] && \
    [[ "$post_status" =~ ^[0-9]+$ ]] && \
    [[ "$upload_status" =~ ^[0-9]+$ ]] && [[ "$marker_status" =~ ^[0-9]+$ ]] || return 1
  [ "$input_status" -eq 0 ] && [ "$train_status" -eq 0 ] && [ "$post_status" -eq 0 ] && \
    [ "$upload_status" -eq 0 ] && [ "$marker_status" -eq 0 ] || return 1

  local observed_paths="$CONTROLLER_DIR/manifest_paths_observed.$$"
  local expected_paths="$CONTROLLER_DIR/manifest_paths_expected.$$"
  awk 'NF == 2 && length($1) == 64 && $1 ~ /^[0-9a-f]+$/ {print $2; next} {exit 1}' \
    "$artifact_manifest" >"$observed_paths" || {
    rm -f "$observed_paths" "$expected_paths"
    return 1
  }
  if [ "$BATCH_MODE" -eq 1 ]; then
    cat >"$expected_paths" <<EOF
checkpoints/main2025_v2_hremaining/dexposure-fm-h4.pt
checkpoints/main2025_v2_hremaining/dexposure-fm-h8.pt
checkpoints/main2025_v2_hremaining/dexposure-fm-h12.pt
checkpoints/main2025_v2_hremaining/feature_schema.json
checkpoints/main2025_v2_hremaining/task1_metrics.json
checkpoints/main2025_v2_hremaining/run_config_h4.json
checkpoints/main2025_v2_hremaining/run_config_h8.json
checkpoints/main2025_v2_hremaining/run_config_h12.json
checkpoints/main2025_v2_hremaining/SHA256SUMS
checkpoints/main2025_v2_hremaining_train/finetuned/best_model_h4.pt
checkpoints/main2025_v2_hremaining_train/finetuned/best_model_h8.pt
checkpoints/main2025_v2_hremaining_train/finetuned/best_model_h12.pt
checkpoints/main2025_v2_hremaining_train/finetuned/feature_schema.json
checkpoints/main2025_v2_hremaining_train/finetuned/experiment_results.json
checkpoints/main2025_v2_hremaining_train/finetuned/data_quality.json
checkpoints/main2025_v2_hremaining_train/finetuned/metrics.json
checkpoints/main2025_v2_hremaining_train/finetuned/all_results.json
logs/run_identity.env
logs/input_transport_status.env
logs/input_manifest.sha256
logs/input_download_${RUN_TAG}.log
logs/train_${RUN_TAG}.log
logs/postflight_${RUN_TAG}.log
logs/controller_${RUN_TAG}.log
logs/upload_release_${RUN_TAG}.log
logs/upload_train_${RUN_TAG}.log
EOF
  else
    cat >"$expected_paths" <<EOF
checkpoints/main2025_v2_h${HORIZON}/dexposure-fm-h${HORIZON}.pt
checkpoints/main2025_v2_h${HORIZON}/feature_schema.json
checkpoints/main2025_v2_h${HORIZON}/task1_metrics.json
checkpoints/main2025_v2_h${HORIZON}/run_config.json
checkpoints/main2025_v2_h${HORIZON}/SHA256SUMS
checkpoints/main2025_v2_h${HORIZON}_train/finetuned/best_model_h${HORIZON}.pt
checkpoints/main2025_v2_h${HORIZON}_train/finetuned/feature_schema.json
checkpoints/main2025_v2_h${HORIZON}_train/finetuned/experiment_results.json
checkpoints/main2025_v2_h${HORIZON}_train/finetuned/data_quality.json
checkpoints/main2025_v2_h${HORIZON}_train/finetuned/metrics.json
checkpoints/main2025_v2_h${HORIZON}_train/finetuned/all_results.json
logs/run_identity.env
logs/train_${RUN_TAG}.log
logs/postflight_${RUN_TAG}.log
logs/controller_${RUN_TAG}.log
logs/upload_release_${RUN_TAG}.log
logs/upload_train_${RUN_TAG}.log
EOF
  fi
  if [ "$BATCH_HORIZONS" = "1 4 8 12" ]; then
    printf '%s\n' \
      "$RELEASE_REL/dexposure-fm-h1.pt" \
      "$RELEASE_REL/run_config_h1.json" \
      "$TRAIN_REL/finetuned/best_model_h1.pt" >>"$expected_paths"
  fi
  if [ "$BINDING_NAME" = main2025_fm_tuning ] || [ "$BINDING_NAME" = main2025_fm_sequential ]; then
    printf '%s\n' "$TRAIN_REL/tuning_manifest.sha256" >>"$expected_paths"
  fi
  if [ "$BINDING_NAME" = main2025_fm_eval_paper ]; then
    printf '%s\n' "$TRAIN_REL/eval_manifest.sha256" >>"$expected_paths"
  fi
  if ! cmp -s "$expected_paths" "$observed_paths"; then
    rm -f "$observed_paths" "$expected_paths"
    return 1
  fi
  rm -f "$observed_paths" "$expected_paths"

  (
    cd "$run_root" && shasum -a 256 -c "logs/artifact_manifest_${RUN_TAG}.sha256"
  ) >/dev/null 2>&1 || return 1
  (
    cd "$model_dir" && shasum -a 256 -c SHA256SUMS
  ) >/dev/null 2>&1 || return 1
  if [ "$BINDING_NAME" = main2025_fm_tuning ]; then
    "$ROOT_DIR/.venv/bin/python" cloud/train_src/tune_fm.py \
      --verify "$run_root" --source-sha256 "$SOURCE_SHA" || return 1
  fi
  if [ "$BINDING_NAME" = main2025_fm_sequential ]; then
    "$ROOT_DIR/.venv/bin/python" cloud/train_src/sequential_fm.py \
      --verify "$run_root" --source-sha256 "$SOURCE_SHA" || return 1
  fi
  if [ "$BINDING_NAME" = main2025_fm_eval_paper ]; then
    "$ROOT_DIR/.venv/bin/python" cloud/train_src/eval_paper.py \
      --verify "$run_root" --source-sha256 "$SOURCE_SHA" || return 1
  fi
  if [ "$BATCH_MODE" -eq 1 ]; then
    local input_status_expected="$CONTROLLER_DIR/input_transport_status_expected.$$"
    local input_manifest_expected="$CONTROLLER_DIR/input_manifest_expected.$$"
    local input_revision input_repo input_files local_input host_input
    input_revision="$(jq -r '.value.input_fetch.revision // empty' <<<"$BINDING_JSON")"
    input_repo="$(jq -r '.value.input_fetch.repo // empty' <<<"$BINDING_JSON")"
    input_files="$(jq -r '.value.input_fetch.files | length' <<<"$BINDING_JSON")"
    printf 'STATUS=verified\nREPO=%s\nREVISION=%s\nFILES=%s\n' \
      "$input_repo" "$input_revision" "$input_files" >"$input_status_expected"
    jq -r '.value.input_fetch.files[] | "\(.sha256)  \(.host_path)"' \
      <<<"$BINDING_JSON" >"$input_manifest_expected"
    if ! cmp -s "$input_status_expected" "$log_dir/input_transport_status.env" || \
       ! cmp -s "$input_manifest_expected" "$log_dir/input_manifest.sha256"; then
      rm -f "$input_status_expected" "$input_manifest_expected"
      return 1
    fi
    rm -f "$input_status_expected" "$input_manifest_expected"
    while read -r local_input host_input; do
      [ -n "$local_input" ] && [ -n "$host_input" ] || return 1
      mkdir -p "$run_root/$(dirname "$host_input")"
      if [ ! -e "$run_root/$host_input" ]; then
        ln "$ROOT_DIR/$local_input" "$run_root/$host_input" || return 1
      fi
    done <<'EOF'
data/historical-network_week_2020-03-30.json data/historical-network_week_2020-03-30.json
data/historical-network_week_2025-07-01.json data/historical-network_week_2025-07-01.json
cloud/upload/data/meta_df.csv data/meta_df.csv
cloud/upload/graphpfn-v1.ckpt checkpoints/graphpfn-v1.ckpt
EOF
    (
      cd "$run_root" && shasum -a 256 -c logs/input_manifest.sha256
    ) >/dev/null 2>&1 || return 1
    for horizon in $BATCH_HORIZONS; do
      config_file="$model_dir/run_config_h${horizon}.json"
      cmp -s "$model_dir/dexposure-fm-h${horizon}.pt" \
        "$train_dir/best_model_h${horizon}.pt" || return 1
      if [ "$BINDING_NAME" != main2025_fm_eval_paper ]; then
        "$ROOT_DIR/.venv/bin/python" cloud/verify_v2_checkpoint.py \
          --checkpoint "$model_dir/dexposure-fm-h${horizon}.pt" \
          --metrics "$model_dir/task1_metrics.json" \
          --config "$config_file" \
          --manifest cloud/preflight_manifest.json --horizon "$horizon" \
          --epochs "$(jq -r '.value.epochs' <<<"$BINDING_JSON")" \
          --source-sha256 "$SOURCE_SHA" >/dev/null 2>&1 || return 1
      fi
    done
  else
    cmp -s "$model_dir/dexposure-fm-h${HORIZON}.pt" \
      "$train_dir/best_model_h${HORIZON}.pt" || return 1
    "$ROOT_DIR/.venv/bin/python" cloud/verify_v2_checkpoint.py \
      --checkpoint "$model_dir/dexposure-fm-h${HORIZON}.pt" \
      --metrics "$model_dir/task1_metrics.json" \
      --config "$model_dir/run_config.json" \
      --manifest cloud/preflight_manifest.json --horizon "$HORIZON" \
      --epochs "$(jq -r '.value.epochs' <<<"$BINDING_JSON")" \
      --source-sha256 "$SOURCE_SHA" >/dev/null 2>&1 || return 1
  fi
}

capture_verified_failure_snapshot() {
  local recovery="$1"
  [ "$ATTACH_INSTANCE_ABSENT" -eq 0 ] || return 1

  local snapshot manifest_name roots_name remote_manifest remote_roots
  snapshot="$(mktemp -d "$recovery/failure_snapshot.XXXXXX")"
  manifest_name="failure_evidence_${RUN_TAG}.sha256"
  roots_name="failure_evidence_${RUN_TAG}.roots"
  remote_manifest="/tmp/$manifest_name"
  remote_roots="/tmp/$roots_name"

  create_remote_manifest() {
    "${SSH[@]}" bash -s -- "$HORIZON" "$RUN_TAG" <<'REMOTE'
set -euo pipefail
horizon="$1"
run_tag="$2"
cd /root/vast_workdir
read -r pid expected_start phase <logs/remote_runner.pid 2>/dev/null || exit 74
case "$pid" in
  *[!0-9]*|'') exit 74 ;;
esac
case "$expected_start" in
  *[!0-9]*|'') exit 74 ;;
esac
if [ "$phase" = running ] && kill -0 "$pid" 2>/dev/null; then
  current_start="$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || true)"
  state="$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)"
  if [ "$current_start" = "$expected_start" ] && [ "$state" != Z ]; then
    exit 75
  fi
fi

manifest="/tmp/failure_evidence_${run_tag}.sha256"
roots_file="/tmp/failure_evidence_${run_tag}.roots"
paths_file="/tmp/failure_evidence_${run_tag}.paths"
manifest_tmp="${manifest}.tmp.$$"
roots_tmp="${roots_file}.tmp.$$"
paths_tmp="${paths_file}.tmp.$$"
roots=(logs)
for path in \
  "checkpoints/main2025_v2_h${horizon}" \
  "checkpoints/main2025_v2_h${horizon}_train"; do
  [ ! -d "$path" ] || roots+=("$path")
done
printf '%s\n' "${roots[@]}" >"$roots_tmp"
find "${roots[@]}" -type f \
  ! -path 'logs/controller_heartbeat.env' \
  ! -path 'logs/.controller_heartbeat.env.*' \
  ! -path 'logs/runner_progress.env' \
  ! -path 'logs/.runner_progress.env.*' \
  ! -path 'logs/watchdog.pid' \
  ! -path 'logs/watchdog_destroy_authorized.env' \
  ! -path 'logs/watchdog_destroy.lock/*' \
  ! -path 'logs/watchdog_recovery.lock/*' \
  ! -path 'logs/emergency_snapshot.lock' \
  -print | LC_ALL=C sort >"$paths_tmp"
[ -s "$paths_tmp" ]
: >"$manifest_tmp"
while IFS= read -r path; do
  shasum -a 256 "$path" >>"$manifest_tmp"
done <"$paths_tmp"
mv "$roots_tmp" "$roots_file"
mv "$paths_tmp" "$paths_file"
mv "$manifest_tmp" "$manifest"
REMOTE
  }

  # Generate the inventory only after the runner PID has exited. Fetch every
  # ordinary file under the advertised roots, then regenerate the inventory.
  # The two inventories must match so a surviving uploader cannot race rsync.
  create_remote_manifest || return 1
  run_bounded "$TRANSFER_TIMEOUT_SECONDS" rsync -az -e "$SSH_TRANSPORT" \
    "root@$SSH_HOST:$remote_manifest" "$snapshot/$manifest_name.before" || return 1
  run_bounded "$TRANSFER_TIMEOUT_SECONDS" rsync -az -e "$SSH_TRANSPORT" \
    "root@$SSH_HOST:$remote_roots" "$snapshot/$roots_name.before" || return 1

  local root
  while IFS= read -r root; do
    case "$root" in
      logs|"checkpoints/main2025_v2_h${HORIZON}"|"checkpoints/main2025_v2_h${HORIZON}_train") ;;
      *) echo "ERROR: remote failure inventory advertised an unexpected root: $root" >&2; return 1 ;;
    esac
    mkdir -p "$snapshot/$root"
    run_bounded "$TRANSFER_TIMEOUT_SECONDS" rsync -az --delete \
      --exclude='controller_heartbeat.env' --exclude='.controller_heartbeat.env.*' \
      --exclude='runner_progress.env' --exclude='.runner_progress.env.*' \
      --exclude='watchdog.pid' --exclude='watchdog_destroy_authorized.env' \
      --exclude='watchdog_destroy.lock/' --exclude='watchdog_recovery.lock/' \
      --exclude='emergency_snapshot.lock' -e "$SSH_TRANSPORT" \
      "root@$SSH_HOST:/root/vast_workdir/$root/" "$snapshot/$root/" || return 1
  done <"$snapshot/$roots_name.before"

  create_remote_manifest || return 1
  run_bounded "$TRANSFER_TIMEOUT_SECONDS" rsync -az -e "$SSH_TRANSPORT" \
    "root@$SSH_HOST:$remote_manifest" "$snapshot/$manifest_name.after" || return 1
  run_bounded "$TRANSFER_TIMEOUT_SECONDS" rsync -az -e "$SSH_TRANSPORT" \
    "root@$SSH_HOST:$remote_roots" "$snapshot/$roots_name.after" || return 1
  cmp -s "$snapshot/$manifest_name.before" "$snapshot/$manifest_name.after" || return 1
  cmp -s "$snapshot/$roots_name.before" "$snapshot/$roots_name.after" || return 1

  local local_paths expected_paths
  local_paths="$snapshot/local_paths.txt"
  expected_paths="$snapshot/expected_paths.txt"
  (
    cd "$snapshot"
    while IFS= read -r root; do
      find "$root" -type f -print
    done <"$roots_name.after" | LC_ALL=C sort
  ) >"$local_paths"
  sed -E 's/^[0-9a-fA-F]{64}[[:space:]]+//' \
    "$snapshot/$manifest_name.after" | LC_ALL=C sort >"$expected_paths"
  cmp -s "$expected_paths" "$local_paths" || return 1
  (
    cd "$snapshot"
    shasum -a 256 -c "$manifest_name.after"
  ) >/dev/null 2>&1 || return 1

  local identity status failure_log bound_file train_status post_status upload_status marker_status field
  identity="$snapshot/logs/run_identity.env"
  status="$snapshot/logs/local_run_status.env"
  failure_log="$snapshot/logs/failure_${RUN_TAG}.log"
  test -s "$identity" && test -s "$status" && \
    test -s "$failure_log" && \
    test -s "$snapshot/logs/controller_${RUN_TAG}.log" || return 1
  for bound_file in "$identity" "$status" "$failure_log"; do
    [ "$(awk -F= '$1 == "RUN_TAG" {print $2}' "$bound_file")" = "$RUN_TAG" ] && \
      [ "$(awk -F= '$1 == "HORIZON" {print $2}' "$bound_file")" = "$HORIZON" ] && \
      [ "$(awk -F= '$1 == "SOURCE_SHA256" {print $2}' "$bound_file")" = "$SOURCE_SHA" ] && \
      [ "$(awk -F= '$1 == "INSTANCE_ID" {print $2}' "$bound_file")" = "$INSTANCE_ID" ] || return 1
  done
  train_status="$(awk -F= '$1 == "TRAIN_STATUS" {print $2}' "$status")"
  post_status="$(awk -F= '$1 == "POST_STATUS" {print $2}' "$status")"
  upload_status="$(awk -F= '$1 == "UPLOAD_STATUS" {print $2}' "$status")"
  marker_status="$(awk -F= '$1 == "MARKER_STATUS" {print $2}' "$status")"
  [[ "$train_status" =~ ^[0-9]+$ ]] && [[ "$post_status" =~ ^[0-9]+$ ]] && \
    [[ "$upload_status" =~ ^[0-9]+$ ]] && [[ "$marker_status" =~ ^[0-9]+$ ]] || return 1
  [ "$train_status" -ne 0 ] || [ "$post_status" -ne 0 ] || \
    [ "$upload_status" -ne 0 ] || [ "$marker_status" -ne 0 ] || return 1
  for field in TRAIN_STATUS POST_STATUS UPLOAD_STATUS MARKER_STATUS INPUT_STATUS; do
    [ "$(awk -F= -v key="$field" '$1 == key {print $2}' "$failure_log")" = \
      "$(awk -F= -v key="$field" '$1 == key {print $2}' "$status")" ] || return 1
  done

  RECOVERED_FAILURE_ROOT="$snapshot"
  echo "Verified the complete remote failure inventory at $snapshot"
  return 0
}

validate_emergency_authorization() {
  local authorization="$1" expected_snapshot_sha="$2"
  [ -s "$authorization" ] || return 1
  awk -F= '
    NF != 2 { exit 1 }
    $1 !~ /^(LOCAL_RECOVERY_VERIFIED|HF_ROUNDTRIP_VERIFIED|INSTANCE_ID|RUN_TAG|HORIZON|SOURCE_SHA256|EVIDENCE_LEVEL|MODEL_SHA256|SNAPSHOT_SHA256)$/ { exit 1 }
    { seen[$1]++ }
    END {
      if (NR != 9) exit 1
      for (key in seen) if (seen[key] != 1) exit 1
    }
  ' "$authorization" || return 1
  [ "$(awk -F= '$1 == "LOCAL_RECOVERY_VERIFIED" {print $2}' "$authorization")" = 0 ] && \
    [ "$(awk -F= '$1 == "HF_ROUNDTRIP_VERIFIED" {print $2}' "$authorization")" = 1 ] && \
    [ "$(awk -F= '$1 == "INSTANCE_ID" {print $2}' "$authorization")" = "$INSTANCE_ID" ] && \
    [ "$(awk -F= '$1 == "RUN_TAG" {print $2}' "$authorization")" = "$RUN_TAG" ] && \
    [ "$(awk -F= '$1 == "HORIZON" {print $2}' "$authorization")" = "$HORIZON" ] && \
    [ "$(awk -F= '$1 == "SOURCE_SHA256" {print $2}' "$authorization")" = "$SOURCE_SHA" ] && \
    [ "$(awk -F= '$1 == "EVIDENCE_LEVEL" {print $2}' "$authorization")" = remote_snapshot ] && \
    [ "$(awk -F= '$1 == "SNAPSHOT_SHA256" {print $2}' "$authorization")" = "$expected_snapshot_sha" ]
}

validate_emergency_failure_snapshot() {
  local run_root="$1" identity="$1/logs/run_identity.env"
  local status="$1/logs/local_run_status.env"
  local recovery_status="$1/logs/emergency_snapshot_status.env"
  local failure_log="$1/logs/failure_${RUN_TAG}.log"
  local bound_file=""
  local train_status="" post_status="" upload_status="" marker_status="" field=""
  test -s "$identity" && test -s "$status" && test -s "$recovery_status" && \
    test -s "$failure_log" && \
    test -s "$run_root/logs/controller_${RUN_TAG}.log" || return 1
  for bound_file in "$identity" "$status" "$recovery_status" "$failure_log"; do
    [ "$(awk -F= '$1 == "RUN_TAG" {print $2}' "$bound_file")" = "$RUN_TAG" ] && \
      [ "$(awk -F= '$1 == "HORIZON" {print $2}' "$bound_file")" = "$HORIZON" ] && \
      [ "$(awk -F= '$1 == "SOURCE_SHA256" {print $2}' "$bound_file")" = "$SOURCE_SHA" ] && \
      [ "$(awk -F= '$1 == "INSTANCE_ID" {print $2}' "$bound_file")" = "$INSTANCE_ID" ] || return 1
  done
  train_status="$(awk -F= '$1 == "TRAIN_STATUS" {print $2}' "$status")"
  post_status="$(awk -F= '$1 == "POST_STATUS" {print $2}' "$status")"
  upload_status="$(awk -F= '$1 == "UPLOAD_STATUS" {print $2}' "$status")"
  marker_status="$(awk -F= '$1 == "MARKER_STATUS" {print $2}' "$status")"
  [[ "$train_status" =~ ^[0-9]+$ ]] && [[ "$post_status" =~ ^[0-9]+$ ]] && \
    [[ "$upload_status" =~ ^[0-9]+$ ]] && [[ "$marker_status" =~ ^[0-9]+$ ]] || return 1
  [ "$train_status" -ne 0 ] || [ "$post_status" -ne 0 ] || \
    [ "$upload_status" -ne 0 ] || [ "$marker_status" -ne 0 ] || return 1
  for field in TRAIN_STATUS POST_STATUS UPLOAD_STATUS MARKER_STATUS INPUT_STATUS; do
    [ "$(awk -F= -v key="$field" '$1 == key {print $2}' "$failure_log")" = \
      "$(awk -F= -v key="$field" '$1 == key {print $2}' "$status")" ] || return 1
  done
}

recover_hf_emergency_snapshot() {
  local recovery="$1"
  local emergency="$recovery/hf/runs/$RUN_TAG/emergency"
  local archive="$emergency/artifact_snapshot.tar"
  local checksum="$emergency/artifact_snapshot.sha256"
  local authorization="$emergency/watchdog_destroy_authorized.env"
  local advertised_sha="" attempt_root="" destination="" receipt="" receipt_tmp="" snapshot_sha=""
  test -s "$archive" && test -s "$checksum" && test -s "$authorization" || return 1
  advertised_sha="$(awk 'NF == 2 && $1 ~ /^[0-9a-f]{64}$/ && $2 == "artifact_snapshot.tar" {print $1}' "$checksum")"
  [ "${#advertised_sha}" -eq 64 ] || return 1

  if [ "$EMERGENCY_RECOVERED_SHA" = "$advertised_sha" ] && \
      [ -d "$EMERGENCY_RECOVERED_ROOT" ] && \
      [ -s "$EMERGENCY_RECOVERY_RECEIPT" ]; then
    snapshot_sha="$(jq -er '.snapshot_sha256 | select(test("^[0-9a-f]{64}$"))' \
      "$EMERGENCY_RECOVERY_RECEIPT")" || return 1
    [ "$snapshot_sha" = "$advertised_sha" ] || return 1
    [ "$(jq -r '.destination' "$EMERGENCY_RECOVERY_RECEIPT")" = \
      "$EMERGENCY_RECOVERED_ROOT" ] || return 1
    validate_emergency_authorization "$authorization" "$snapshot_sha" || return 1
    return 0
  fi

  # Each controller process publishes into a fresh attempt directory. If the
  # controller dies after extraction but before its receipt rename, the next
  # controller uses a new directory instead of getting stuck on the old one.
  attempt_root="$(mktemp -d "$recovery/.emergency_recovery_${advertised_sha}.XXXXXX")" || return 1
  destination="$attempt_root/recovered"
  receipt="$attempt_root/receipt.json"
  receipt_tmp="${receipt}.tmp.$$"
  "$ROOT_DIR/.venv/bin/python" cloud/recover_emergency_snapshot.py \
    --archive "$archive" --checksum "$checksum" --destination "$destination" \
    >"$receipt_tmp" || { rm -f "$receipt_tmp"; return 1; }
  mv "$receipt_tmp" "$receipt" || return 1
  [ -s "$receipt" ] || return 1
  snapshot_sha="$(jq -er '.snapshot_sha256 | select(test("^[0-9a-f]{64}$"))' "$receipt")" || return 1
  [ "$snapshot_sha" = "$advertised_sha" ] || return 1
  [ "$(jq -r '.destination' "$receipt")" = "$destination" ] || return 1
  validate_emergency_authorization "$authorization" "$snapshot_sha" || return 1
  EMERGENCY_RECOVERED_ROOT="$destination"
  EMERGENCY_RECOVERED_SHA="$snapshot_sha"
  EMERGENCY_RECOVERY_RECEIPT="$receipt"
  echo "Verified and safely extracted Hugging Face emergency snapshot sha256=$snapshot_sha"
}

direct_recovery() {
  local required="${1:-failure_evidence}"
  local recovery="$STATE_ROOT/direct_recovery"
  mkdir -p "$recovery/logs" "$recovery/checkpoints" "$recovery/hf"
  for recovery_attempt in $(seq 1 10); do
    run_bounded "$TRANSFER_TIMEOUT_SECONDS" uvx --from 'huggingface_hub[cli]' hf download "$HF_REPO" \
      --repo-type model --include "runs/$RUN_TAG/**" --local-dir "$recovery/hf" \
      --force-download >/dev/null 2>&1 || true
    if recover_hf_emergency_snapshot "$recovery"; then
      if [ "$required" = complete_bundle ] && \
          validate_complete_bundle "$EMERGENCY_RECOVERED_ROOT"; then
        RECOVERED_BUNDLE="$EMERGENCY_RECOVERED_ROOT/checkpoints/main2025_v2_h${HORIZON}"
        RECOVERED_TRAIN_ROOT="$EMERGENCY_RECOVERED_ROOT/checkpoints/main2025_v2_h${HORIZON}_train/finetuned"
        RECOVERED_LOG_ROOT="$EMERGENCY_RECOVERED_ROOT/logs"
        return 0
      elif [ "$required" = failure_evidence ] && \
          validate_emergency_failure_snapshot "$EMERGENCY_RECOVERED_ROOT"; then
        RECOVERED_FAILURE_ROOT="$EMERGENCY_RECOVERED_ROOT"
        return 0
      fi
    fi
    if [ "$ATTACH_INSTANCE_ABSENT" -eq 0 ]; then
      recovery_live="$(list_instances 2>/dev/null || true)"
      if [ -n "$recovery_live" ] && \
          ! jq -e --arg id "$INSTANCE_ID" '.[] | select((.id | tostring) == $id)' \
            >/dev/null <<<"$recovery_live"; then
        ATTACH_INSTANCE_ABSENT=1
        echo "Vast instance id=$INSTANCE_ID became absent; continuing with Hugging Face recovery only."
      fi
    fi
    if [ "$ATTACH_INSTANCE_ABSENT" -eq 0 ]; then
      if "${SSH[@]}" 'cd /root/vast_workdir && for path in logs checkpoints; do if [ -e "$path" ]; then find "$path" -maxdepth 6 -type f -print >/dev/null; fi; done' \
          >/dev/null 2>&1; then
        RECOVERY_REMOTE_INSPECTED=1
      fi
      run_bounded "$TRANSFER_TIMEOUT_SECONDS" rsync -az -e "$SSH_TRANSPORT" "root@$SSH_HOST:/root/vast_workdir/logs/" "$recovery/logs/" || true
      run_bounded "$TRANSFER_TIMEOUT_SECONDS" rsync -az -e "$SSH_TRANSPORT" \
        "root@$SSH_HOST:/root/vast_workdir/checkpoints/main2025_v2_h${HORIZON}/" \
        "$recovery/checkpoints/main2025_v2_h${HORIZON}/" || true
      run_bounded "$TRANSFER_TIMEOUT_SECONDS" rsync -az -e "$SSH_TRANSPORT" \
        "root@$SSH_HOST:/root/vast_workdir/checkpoints/main2025_v2_h${HORIZON}_train/" \
        "$recovery/checkpoints/main2025_v2_h${HORIZON}_train/" || true
    fi
    if [ "$required" = complete_bundle ]; then
      while IFS= read -r run_root; do
        [ -n "$run_root" ] || continue
        if validate_complete_bundle "$run_root"; then
          RECOVERED_BUNDLE="$run_root/checkpoints/main2025_v2_h${HORIZON}"
          RECOVERED_TRAIN_ROOT="$run_root/checkpoints/main2025_v2_h${HORIZON}_train/finetuned"
          RECOVERED_LOG_ROOT="$run_root/logs"
          return 0
        fi
      done <<EOF
$recovery
$recovery/hf/runs/$RUN_TAG
$STATE_ROOT/hf_download/runs/$RUN_TAG
$EMERGENCY_RECOVERED_ROOT
EOF
    elif [ -n "$EMERGENCY_RECOVERED_ROOT" ] && \
        validate_emergency_failure_snapshot "$EMERGENCY_RECOVERED_ROOT"; then
      RECOVERED_FAILURE_ROOT="$EMERGENCY_RECOVERED_ROOT"
      return 0
    elif [ "$ATTACH_INSTANCE_ABSENT" -eq 0 ] && \
        capture_verified_failure_snapshot "$recovery"; then
      return 0
    fi
    [ "$recovery_attempt" -eq 10 ] || sleep 30
  done
  return 1
}

recover_failure_and_destroy() {
  local required="${1:-failure_evidence}"
  if direct_recovery "$required"; then
    if [ "$BATCH_MODE" -eq 1 ]; then
      if stop_instance_preserving_disk "$INSTANCE_ID"; then
        return 0
      fi
    else
      if ! publish_watchdog_recovery_evidence failure_inventory none; then
        echo "WARNING: local failure evidence is verified, but remote watchdog evidence could not be published." >&2
      fi
      if destroy_instance "$INSTANCE_ID"; then
        return 0
      fi
    fi
    echo "ERROR: recovery succeeded, but the lifecycle action for instance id=$INSTANCE_ID was not confirmed." >&2
    return 1
  fi
  if [ "$BATCH_MODE" -eq 1 ]; then
    stop_instance_preserving_disk "$INSTANCE_ID" || true
    echo "ERROR: required recovery level '$required' was not reached; instance id=$INSTANCE_ID was stopped with its disk retained for manual recovery." >&2
    return 1
  fi
  echo "ERROR: required recovery level '$required' was not reached; preserving instance id=$INSTANCE_ID for the emergency watchdog." >&2
  return 1
}

finalize_recovered_success() {
  [ -n "$RECOVERED_BUNDLE" ] && [ -n "$RECOVERED_TRAIN_ROOT" ] && \
    [ -n "$RECOVERED_LOG_ROOT" ] || return 1
  echo "A complete checkpoint bundle was recovered; re-uploading it for a clean Hugging Face round trip."
  run_bounded "$TRANSFER_TIMEOUT_SECONDS" uvx --from 'huggingface_hub[cli]' hf upload "$HF_REPO" \
    "$RECOVERED_BUNDLE" "runs/$RUN_TAG/checkpoints/main2025_v2_h${HORIZON}" \
    --repo-type model --commit-message "Vast recovered checkpoint h=$HORIZON $RUN_TAG" \
    >/dev/null || return 1
  run_bounded "$TRANSFER_TIMEOUT_SECONDS" uvx --from 'huggingface_hub[cli]' hf upload "$HF_REPO" \
    "$RECOVERED_TRAIN_ROOT" "runs/$RUN_TAG/checkpoints/main2025_v2_h${HORIZON}_train/finetuned" \
    --repo-type model --commit-message "Vast recovered training artifacts h=$HORIZON $RUN_TAG" \
    >/dev/null || return 1
  run_bounded "$TRANSFER_TIMEOUT_SECONDS" uvx --from 'huggingface_hub[cli]' hf upload "$HF_REPO" \
    "$RECOVERED_LOG_ROOT" "runs/$RUN_TAG/logs" --repo-type model \
    --commit-message "Vast recovered logs h=$HORIZON $RUN_TAG" >/dev/null || return 1

  local recovery_record="$CONTROLLER_DIR/recovery_status.env"
  printf 'RECOVERY_STATUS=controller_verified_complete_bundle\nINSTANCE_ID=%s\nRUN_TAG=%s\nHORIZON=%s\nSOURCE_SHA256=%s\n' \
    "$INSTANCE_ID" "$RUN_TAG" "$HORIZON" "$SOURCE_SHA" >"$recovery_record"
  run_bounded "$TRANSFER_TIMEOUT_SECONDS" uvx --from 'huggingface_hub[cli]' hf upload "$HF_REPO" \
    "$recovery_record" "runs/$RUN_TAG/controller/recovery_status.env" --repo-type model \
    --commit-message "Vast recovery attestation h=$HORIZON $RUN_TAG" >/dev/null || return 1

  local download_root="$STATE_ROOT/hf_download"
  run_bounded "$TRANSFER_TIMEOUT_SECONDS" uvx --from 'huggingface_hub[cli]' hf download "$HF_REPO" --repo-type model \
    --include "runs/$RUN_TAG/**" --local-dir "$download_root" --force-download \
    >/dev/null || return 1
  local run_root="$download_root/runs/$RUN_TAG"
  local model_dir="$run_root/checkpoints/main2025_v2_h${HORIZON}"
  validate_complete_bundle "$run_root" || return 1

  local model_sha completed_instance_id
  if [ "$BATCH_MODE" -eq 1 ]; then
    if [ "$BATCH_HORIZONS" = "1 4 8 12" ]; then
      MODEL_SHA256_H1="$(shasum -a 256 "$model_dir/dexposure-fm-h1.pt" | awk '{print $1}')"
    fi
    MODEL_SHA256_H4="$(shasum -a 256 "$model_dir/dexposure-fm-h4.pt" | awk '{print $1}')"
    MODEL_SHA256_H8="$(shasum -a 256 "$model_dir/dexposure-fm-h8.pt" | awk '{print $1}')"
    MODEL_SHA256_H12="$(shasum -a 256 "$model_dir/dexposure-fm-h12.pt" | awk '{print $1}')"
    model_sha=multiple
  else
    model_sha="$(shasum -a 256 "$model_dir/dexposure-fm-h${HORIZON}.pt" | awk '{print $1}')"
  fi
  completed_instance_id="$INSTANCE_ID"
  write_success_receipt 0 "$completed_instance_id" "$model_sha" || return 1
  destroy_instance "$completed_instance_id" || return 1
  write_success_receipt 1 "$completed_instance_id" "$model_sha" || return 1
  stop_controller_heartbeat
  echo "VAST_LOCAL_MODEL_DIR=$model_dir"
  echo "VAST_INSTANCE_DESTROYED=1"
  echo "VAST API H=$HORIZON RECOVERY, DOWNLOAD, AND VERIFICATION DONE"
  return 0
}

finish_from_complete_recovery() {
  if direct_recovery complete_bundle && finalize_recovered_success; then
    trap - EXIT INT TERM
    exit 0
  fi
  return 1
}

stop_batch_after_terminal_failure() {
  [ "$BATCH_MODE" -eq 1 ] || return 0
  stop_instance_preserving_disk "$INSTANCE_ID" || true
}

trusted_recovered_training_failure() {
  local status_file train_status post_status upload_status marker_status
  for status_file in \
    "$STATE_ROOT/direct_recovery/logs/local_run_status.env" \
    "$STATE_ROOT/direct_recovery/hf/runs/$RUN_TAG/logs/local_run_status.env" \
    "$STATE_ROOT/direct_recovery"/.emergency_recovery_*/recovered/logs/local_run_status.env \
    "$STATE_ROOT/hf_download/runs/$RUN_TAG/logs/local_run_status.env"; do
    [ -s "$status_file" ] || continue
    [ "$(awk -F= '$1 == "RUN_TAG" {print $2}' "$status_file")" = "$RUN_TAG" ] || continue
    [ "$(awk -F= '$1 == "HORIZON" {print $2}' "$status_file")" = "$HORIZON" ] || continue
    [ "$(awk -F= '$1 == "SOURCE_SHA256" {print $2}' "$status_file")" = "$SOURCE_SHA" ] || continue
    [ "$(awk -F= '$1 == "INSTANCE_ID" {print $2}' "$status_file")" = "$INSTANCE_ID" ] || continue
    train_status="$(awk -F= '$1 == "TRAIN_STATUS" {print $2}' "$status_file")"
    post_status="$(awk -F= '$1 == "POST_STATUS" {print $2}' "$status_file")"
    upload_status="$(awk -F= '$1 == "UPLOAD_STATUS" {print $2}' "$status_file")"
    marker_status="$(awk -F= '$1 == "MARKER_STATUS" {print $2}' "$status_file")"
    if [[ "$train_status" =~ ^[0-9]+$ ]] && \
       [[ "$post_status" =~ ^[0-9]+$ ]] && \
       [[ "$upload_status" =~ ^[0-9]+$ ]] && \
       [[ "$marker_status" =~ ^[0-9]+$ ]] && \
       { [ "$train_status" -ne 0 ] || [ "$post_status" -ne 0 ] || \
         [ "$upload_status" -ne 0 ] || [ "$marker_status" -ne 0 ]; }; then
      return 0
    fi
  done
  return 1
}

if [ "$MARKER_MATCHED" -ne 1 ]; then
  echo "ERROR: no matching Hugging Face completion marker for $RUN_TAG." >&2
  if ! finish_from_complete_recovery; then
    if [ -n "$RECOVERED_BUNDLE" ]; then
    echo "ERROR: the complete recovered bundle could not pass a clean Hugging Face round trip; preserving instance id=$INSTANCE_ID." >&2
      exit 1
    fi
  fi
  FINAL_STATUS="api-error"
  if FINAL_INSTANCE_JSON="$(run_bounded "$API_TIMEOUT_SECONDS" vastai show instance "$INSTANCE_ID" --raw 2>/dev/null)" && \
     jq -e 'type == "object"' >/dev/null <<<"$FINAL_INSTANCE_JSON"; then
    FINAL_STATUS="$(jq -r '.instances.actual_status // .actual_status // "not-listed"' <<<"$FINAL_INSTANCE_JSON")"
  fi
  if [ "$FINAL_STATUS" = running ] && [ "$REMOTE_DEAD_COUNT" -lt 3 ]; then
    direct_recovery failure_evidence || true
    echo "ERROR: instance id=$INSTANCE_ID is still running; the controller will not destroy an active training process." >&2
  elif [ "$FINAL_STATUS" = api-error ]; then
    direct_recovery failure_evidence || true
    stop_batch_after_terminal_failure
    echo "ERROR: final Vast status is unknown; the controller will not destroy an unverified instance." >&2
  elif trusted_recovered_training_failure; then
    recover_failure_and_destroy failure_evidence || true
  else
    stop_batch_after_terminal_failure
    echo "ERROR: runner is not active, but neither a verified complete bundle nor a trusted training-failure status was recovered; preserving instance id=$INSTANCE_ID." >&2
  fi
  exit 1
fi

TRAIN_STATUS="$(awk -F= '$1 == "TRAIN_STATUS" {print $2}' "$STATUS_FILE")"
POST_STATUS="$(awk -F= '$1 == "POST_STATUS" {print $2}' "$STATUS_FILE")"
UPLOAD_STATUS="$(awk -F= '$1 == "UPLOAD_STATUS" {print $2}' "$STATUS_FILE")"
MARKER_STATUS="$(awk -F= '$1 == "MARKER_STATUS" {print $2}' "$STATUS_FILE")"
if ! [[ "$TRAIN_STATUS" =~ ^[0-9]+$ ]] || \
   ! [[ "$POST_STATUS" =~ ^[0-9]+$ ]] || \
   ! [[ "$UPLOAD_STATUS" =~ ^[0-9]+$ ]] || \
   ! [[ "$MARKER_STATUS" =~ ^[0-9]+$ ]]; then
  finish_from_complete_recovery || true
  stop_batch_after_terminal_failure
  echo "ERROR: completion marker contains empty or malformed status fields; preserving any unverified remote artifact." >&2
  exit 1
fi

DOWNLOAD_ROOT="$STATE_ROOT/hf_download"
DOWNLOAD_OK=0
for attempt in $(seq 1 10); do
  if run_bounded "$TRANSFER_TIMEOUT_SECONDS" uvx --from 'huggingface_hub[cli]' hf download "$HF_REPO" \
      --repo-type model --include "runs/$RUN_TAG/**" --local-dir "$DOWNLOAD_ROOT"; then
    DOWNLOAD_OK=1
    break
  fi
  echo "Hugging Face download attempt $attempt failed; retrying." >&2
  sleep 60
done
if [ "$DOWNLOAD_OK" -ne 1 ]; then
  finish_from_complete_recovery || true
  stop_batch_after_terminal_failure
  echo "ERROR: Hugging Face round-trip download failed; recovered evidence, if available, is in $STATE_ROOT/direct_recovery." >&2
  exit 1
fi

if [ "$TRAIN_STATUS" != 0 ]; then
  recover_failure_and_destroy failure_evidence || true
  echo "ERROR: remote statuses train=$TRAIN_STATUS post=$POST_STATUS upload=$UPLOAD_STATUS; failure evidence is in $STATE_ROOT." >&2
  exit 1
fi
if [ "$POST_STATUS" != 0 ] || [ "$UPLOAD_STATUS" != 0 ] || [ "$MARKER_STATUS" != 0 ]; then
  if ! finish_from_complete_recovery; then
    recover_failure_and_destroy failure_evidence || true
  fi
  echo "ERROR: training finished but final status was post=$POST_STATUS upload=$UPLOAD_STATUS marker=$MARKER_STATUS; a verified checkpoint bundle is required before destroy." >&2
  exit 1
fi

MODEL_DIR="$DOWNLOAD_ROOT/runs/$RUN_TAG/$RELEASE_REL"
RUN_ROOT="$DOWNLOAD_ROOT/runs/$RUN_TAG"
METRICS_FILE="$MODEL_DIR/task1_metrics.json"
test -s "$METRICS_FILE" || { finish_from_complete_recovery || true; stop_batch_after_terminal_failure; echo "ERROR: downloaded Task I metrics are missing." >&2; exit 1; }
if [ "$BATCH_MODE" -eq 1 ]; then
  for horizon in $BATCH_HORIZONS; do
    test -s "$MODEL_DIR/dexposure-fm-h${horizon}.pt" || {
      finish_from_complete_recovery || true
      stop_batch_after_terminal_failure
      echo "ERROR: downloaded h=$horizon checkpoint is missing." >&2
      exit 1
    }
    test -s "$MODEL_DIR/run_config_h${horizon}.json" || {
      finish_from_complete_recovery || true
      stop_batch_after_terminal_failure
      echo "ERROR: downloaded h=$horizon run config is missing." >&2
      exit 1
    }
  done
else
  MODEL_FILE="$MODEL_DIR/dexposure-fm-h${HORIZON}.pt"
  CONFIG_FILE="$MODEL_DIR/run_config.json"
  test -s "$MODEL_FILE" || { finish_from_complete_recovery || true; stop_batch_after_terminal_failure; echo "ERROR: downloaded checkpoint is missing." >&2; exit 1; }
  test -s "$CONFIG_FILE" || { finish_from_complete_recovery || true; stop_batch_after_terminal_failure; echo "ERROR: downloaded run config is missing." >&2; exit 1; }
fi
if ! validate_complete_bundle "$RUN_ROOT"; then
  finish_from_complete_recovery || true
  stop_batch_after_terminal_failure
  echo "ERROR: downloaded h=$HORIZON checkpoint, training JSON, logs, or hashes failed verification." >&2
  exit 1
fi
[ -z "${VAST_LOCAL_CONTROLLER_LOG:-}" ] || [ -s "$VAST_LOCAL_CONTROLLER_LOG" ] || {
  stop_batch_after_terminal_failure
  echo "ERROR: local controller log is missing or empty; refusing to destroy the verified instance." >&2
  exit 1
}
if [ "$BATCH_MODE" -eq 1 ]; then
  if [ "$BATCH_HORIZONS" = "1 4 8 12" ]; then
    MODEL_SHA256_H1="$(shasum -a 256 "$MODEL_DIR/dexposure-fm-h1.pt" | awk '{print $1}')"
  fi
  MODEL_SHA256_H4="$(shasum -a 256 "$MODEL_DIR/dexposure-fm-h4.pt" | awk '{print $1}')"
  MODEL_SHA256_H8="$(shasum -a 256 "$MODEL_DIR/dexposure-fm-h8.pt" | awk '{print $1}')"
  MODEL_SHA256_H12="$(shasum -a 256 "$MODEL_DIR/dexposure-fm-h12.pt" | awk '{print $1}')"
  MODEL_SHA256=multiple
else
  MODEL_SHA256="$(shasum -a 256 "$MODEL_FILE" | awk '{print $1}')"
fi

COMPLETED_INSTANCE_ID="$INSTANCE_ID"
write_success_receipt 0 "$COMPLETED_INSTANCE_ID" "$MODEL_SHA256"
destroy_instance "$COMPLETED_INSTANCE_ID"
write_success_receipt 1 "$COMPLETED_INSTANCE_ID" "$MODEL_SHA256"
stop_controller_heartbeat
trap - EXIT INT TERM
echo "VAST_LOCAL_MODEL_DIR=$MODEL_DIR"
echo "VAST_INSTANCE_DESTROYED=1"
echo "VAST API H=$HORIZON TRAINING, DOWNLOAD, AND VERIFICATION DONE"
