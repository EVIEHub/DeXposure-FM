#!/usr/bin/env bash
# Detached Vast worker. Normal instance teardown belongs to the local controller.
set -uo pipefail

TRAIN_CMD="${1:?training command is required}"
SOURCE_SHA="${2:?source digest is required}"
RUN_TAG="${3:?run tag is required}"
HORIZON="${4:?horizon is required}"
INSTANCE_ID="${5:?instance ID is required}"
TRAIN_TIMEOUT_SECONDS="${6:?training timeout is required}"
INPUT_FETCH_CMD="${7:-}"
ROOT="$PWD"
HF_REPO=losdwind/graph-dexposure-ckpt
export PATH="$HOME/.local/bin:$PATH"
export PREFLIGHT_SOURCE_SHA256="$SOURCE_SHA"

BATCH_MODE=0
case "$HORIZON" in
  1|4|8|12)
    [ -z "$INPUT_FETCH_CMD" ] || { [ "$HORIZON" = 12 ] &&
      [ "$TRAIN_CMD" = "bash cloud/main2025_v2_horizon_run.sh 12 --extended" ] &&
      [ "$INPUT_FETCH_CMD" = "bash cloud/fetch_v2_inputs_from_hf.sh" ]; } || {
      echo "Single-horizon runners do not accept a remote input-fetch command." >&2
      exit 2
    }
    ;;
  remaining)
    BATCH_MODE=1
    { [ "$TRAIN_CMD" = "bash cloud/main2025_v2_remaining_run.sh" ] ||
      [ "$TRAIN_CMD" = "bash cloud/main2025_v2_remaining_run.sh --all" ] ||
      [ "$TRAIN_CMD" = "bash cloud/fm_tune_run.sh" ] ||
      [ "$TRAIN_CMD" = "bash cloud/fm_eval_paper_run.sh" ] ||
      [ "$TRAIN_CMD" = "bash cloud/fm_sequential_run.sh" ]; } && \
      [ "$INPUT_FETCH_CMD" = "bash cloud/fetch_v2_inputs_from_hf.sh" ] || {
      echo "The remaining-horizon runner requires the exact training and input-fetch commands." >&2
      exit 2
    }
    ;;
  *) echo "Unsupported horizon token: $HORIZON" >&2; exit 2 ;;
esac
RELEASE_REL="checkpoints/main2025_v2_h${HORIZON}"
TRAIN_REL="checkpoints/main2025_v2_h${HORIZON}_train"

case "$TRAIN_TIMEOUT_SECONDS" in
  *[!0-9]*|'') echo "Training timeout must be a positive whole number of seconds." >&2; exit 2 ;;
esac
[ "$TRAIN_TIMEOUT_SECONDS" -gt 0 ] || {
  echo "Training timeout must be positive." >&2
  exit 2
}

mkdir -p checkpoints logs
RUNNER_PID_FILE=logs/remote_runner.pid
RUNNER_STARTTIME=""
LIVE_UPLOADER_PID=""
PROGRESS_MONITOR_PID=""
START_LEASE_DIR=logs/runner_start.lock
REMOTE_FINALIZED=0
TRAIN_STATUS=125
POST_STATUS=125
UPLOAD_STATUS=125
MARKER_STATUS=125
INPUT_STATUS=125
LOCAL_STATUS=logs/local_run_status.env
FAILURE_LOG="logs/failure_${RUN_TAG}.log"

write_local_status() {
  local status_tmp="${LOCAL_STATUS}.tmp.$$"
  cat >"$status_tmp" <<EOF
RUN_TAG=$RUN_TAG
HORIZON=$HORIZON
SOURCE_SHA256=$SOURCE_SHA
INSTANCE_ID=$INSTANCE_ID
INPUT_STATUS=$INPUT_STATUS
TRAIN_STATUS=$TRAIN_STATUS
POST_STATUS=$POST_STATUS
UPLOAD_STATUS=$UPLOAD_STATUS
MARKER_STATUS=$MARKER_STATUS
EOF
  mv "$status_tmp" "$LOCAL_STATUS"
}

write_failure_log() {
  local reason="$1" failure_tmp="${FAILURE_LOG}.tmp.$$"
  printf 'RUN_TAG=%s\nHORIZON=%s\nSOURCE_SHA256=%s\nINSTANCE_ID=%s\nINPUT_STATUS=%s\nTRAIN_STATUS=%s\nPOST_STATUS=%s\nUPLOAD_STATUS=%s\nMARKER_STATUS=%s\nREASON=%s\n' \
    "$RUN_TAG" "$HORIZON" "$SOURCE_SHA" "$INSTANCE_ID" "$INPUT_STATUS" "$TRAIN_STATUS" \
    "$POST_STATUS" "$UPLOAD_STATUS" "$MARKER_STATUS" "$reason" >"$failure_tmp"
  mv "$failure_tmp" "$FAILURE_LOG"
}

write_runner_progress() {
  local phase="$1" progress_tmp="logs/.runner_progress.env.${BASHPID:-$$}"
  printf 'INSTANCE_ID=%s\nRUN_TAG=%s\nHORIZON=%s\nSOURCE_SHA256=%s\nPHASE=%s\nPROGRESS_TIMESTAMP=%s\n' \
    "$INSTANCE_ID" "$RUN_TAG" "$HORIZON" "$SOURCE_SHA" "$phase" "$(date +%s)" \
    >"$progress_tmp"
  mv "$progress_tmp" logs/runner_progress.env
}

stop_live_uploader() {
  if [ -n "$LIVE_UPLOADER_PID" ]; then
    kill -TERM -- "-$LIVE_UPLOADER_PID" >/dev/null 2>&1 || true
    for _ in $(seq 1 90); do
      kill -0 "$LIVE_UPLOADER_PID" >/dev/null 2>&1 || break
      sleep 1
    done
    kill -KILL -- "-$LIVE_UPLOADER_PID" >/dev/null 2>&1 || true
    wait "$LIVE_UPLOADER_PID" >/dev/null 2>&1 || true
    LIVE_UPLOADER_PID=""
  fi
}

stop_progress_monitor() {
  if [ -n "$PROGRESS_MONITOR_PID" ]; then
    kill "$PROGRESS_MONITOR_PID" >/dev/null 2>&1 || true
    wait "$PROGRESS_MONITOR_PID" >/dev/null 2>&1 || true
    PROGRESS_MONITOR_PID=""
  fi
}

cleanup_remote() {
  local recorded_pid="" recorded_start="" recorded_phase="" runner_pid_tmp=""
  stop_live_uploader
  stop_progress_monitor
  read -r recorded_pid recorded_start recorded_phase <"$RUNNER_PID_FILE" 2>/dev/null || true
  if [ -n "$RUNNER_STARTTIME" ] && [ "$recorded_pid" = "$$" ] && \
      [ "$recorded_start" = "$RUNNER_STARTTIME" ]; then
    runner_pid_tmp="${RUNNER_PID_FILE}.tmp.$$"
    printf '%s %s exited\n' "$$" "$RUNNER_STARTTIME" >"$runner_pid_tmp"
    mv "$runner_pid_tmp" "$RUNNER_PID_FILE"
  fi
  rm -f "$START_LEASE_DIR/created_at"
  rmdir "$START_LEASE_DIR" >/dev/null 2>&1 || true
}

handle_remote_signal() {
  trap - INT TERM
  TRAIN_STATUS=143
  POST_STATUS=143
  UPLOAD_STATUS=143
  MARKER_STATUS=143
  exit 143
}

finalize_remote_exit() {
  local exit_status=$?
  trap - EXIT INT TERM
  stop_live_uploader
  stop_progress_monitor
  if [ "$REMOTE_FINALIZED" -ne 1 ]; then
    write_local_status || true
    write_failure_log unexpected_exit || true
    while true; do
      write_runner_progress emergency_recovery || true
      if timeout --signal=TERM --kill-after=60 10800 \
          bash cloud/vast_api_emergency_snapshot.sh \
          "$SOURCE_SHA" "$RUN_TAG" "$HORIZON" "$INSTANCE_ID" unexpected_exit; then
        REMOTE_FINALIZED=1
        break
      fi
      echo "Emergency round trip failed; the instance and artifacts remain preserved. Retrying in 300 seconds." >&2
      write_runner_progress emergency_retry || true
      sleep 300
    done
  fi
  cleanup_remote
  exit "$exit_status"
}

# Install recovery traps before writing the runner identity, package setup, or
# host preflight. Any later exit must first create a byte-verified HF snapshot.
trap finalize_remote_exit EXIT
trap handle_remote_signal INT TERM

cat >logs/run_identity.env <<EOF
RUN_TAG=$RUN_TAG
HORIZON=$HORIZON
SOURCE_SHA256=$SOURCE_SHA
INSTANCE_ID=$INSTANCE_ID
EOF
RUNNER_STARTTIME="$(awk '{print $22}' "/proc/$$/stat")"
case "$RUNNER_STARTTIME" in
  *[!0-9]*|'') echo "Cannot bind the remote runner PID to its Linux start time." >&2; exit 1 ;;
esac
RUNNER_PID_TMP="${RUNNER_PID_FILE}.tmp.$$"
printf '%s %s running\n' "$$" "$RUNNER_STARTTIME" >"$RUNNER_PID_TMP"
mv "$RUNNER_PID_TMP" "$RUNNER_PID_FILE"
write_runner_progress runner_started

# Keep this mode-0600 file for the emergency watchdog. It is on the ephemeral
# instance only and is never included in rsync or the recovery archive.
if [ -s /root/.hf_token ]; then
  export HF_TOKEN="$(cat /root/.hf_token)"
else
  echo "The controller did not provide the private Hugging Face token." >&2
  exit 70
fi

SETUP_STATUS=0
write_runner_progress setup_uv
command -v uv >/dev/null 2>&1 || \
  timeout --signal=TERM --kill-after=30 300 bash -c \
    'curl --connect-timeout 15 --max-time 240 -LsSf https://astral.sh/uv/install.sh | sh' || \
  SETUP_STATUS=$?
if [ "$SETUP_STATUS" -eq 0 ]; then
  timeout --signal=TERM --kill-after=60 1800 uv sync --frozen || SETUP_STATUS=$?
fi
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  SETUP_STATUS=1
fi

if [ "$SETUP_STATUS" -eq 0 ] && [ -n "$INPUT_FETCH_CMD" ]; then
  write_runner_progress input_download
  set +e
  timeout --signal=TERM --kill-after=300 10800 \
    bash -c "$INPUT_FETCH_CMD" 2>&1 | tee "logs/input_download_${RUN_TAG}.log"
  INPUT_PIPELINE_STATUS=("${PIPESTATUS[@]}")
  INPUT_STATUS=${INPUT_PIPELINE_STATUS[0]}
  if [ "${INPUT_PIPELINE_STATUS[1]}" -ne 0 ] && [ "$INPUT_STATUS" -eq 0 ]; then
    INPUT_STATUS=${INPUT_PIPELINE_STATUS[1]}
  fi
  set +e
  if [ "$INPUT_STATUS" -ne 0 ]; then
    SETUP_STATUS="$INPUT_STATUS"
  fi
elif [ -z "$INPUT_FETCH_CMD" ]; then
  INPUT_STATUS=0
fi

HOST_STATUS="$SETUP_STATUS"
if [ "$SETUP_STATUS" -eq 0 ]; then
  write_runner_progress host_preflight
  timeout --signal=TERM --kill-after=60 1800 \
    "$PY" cloud/preflight.py --phase host --train-cmd "$TRAIN_CMD"
  HOST_STATUS=$?
fi

if [ "$HOST_STATUS" -eq 0 ]; then
  write_runner_progress training_start
  setsid timeout --signal=TERM --kill-after=60 86400 \
    uvx --from 'huggingface_hub[cli]' hf upload "$HF_REPO" . "runs/$RUN_TAG/live" \
    --repo-type model \
    --include 'logs/**' \
    --include "$RELEASE_REL/**" \
    --include "$TRAIN_REL/**/*.partial.pt" \
    --include "$TRAIN_REL/**/best_model_h*.pt" \
    --include "$TRAIN_REL/**/*.json" \
    --every 5 \
    --commit-message "Vast live recovery h=$HORIZON $RUN_TAG" \
    >"logs/live_uploader_${RUN_TAG}.log" 2>&1 &
  LIVE_UPLOADER_PID=$!

  # This monitor advances only when the training log or a target checkpoint
  # changes. A merely alive but frozen Python process does not refresh it.
  (
    trap 'exit 0' INT TERM
    last_signature=""
    while true; do
      sleep 60
      current_start="$(awk '{print $22}' "/proc/$$/stat" 2>/dev/null || true)"
      [ "$current_start" = "$RUNNER_STARTTIME" ] || exit 0
      signature="$(
        find "logs/train_${RUN_TAG}.log" \
          "$RELEASE_REL" \
          "$TRAIN_REL" \
          -type f -printf '%p:%s:%T@\n' 2>/dev/null | LC_ALL=C sort | \
          shasum -a 256 | awk '{print $1}'
      )"
      if [ -n "$signature" ] && [ "$signature" != "$last_signature" ]; then
        write_runner_progress training
        last_signature="$signature"
      fi
    done
  ) &
  PROGRESS_MONITOR_PID=$!

  set +e
  timeout --signal=TERM --kill-after=300 "$TRAIN_TIMEOUT_SECONDS" \
    bash -c "$TRAIN_CMD" 2>&1 | tee "logs/train_${RUN_TAG}.log"
  PIPELINE_STATUS=("${PIPESTATUS[@]}")
  TRAIN_STATUS=${PIPELINE_STATUS[0]}
  if [ "${PIPELINE_STATUS[1]}" -ne 0 ]; then
    echo "Training log tee failed with status ${PIPELINE_STATUS[1]}." >&2
    if [ "$TRAIN_STATUS" -eq 0 ]; then
      TRAIN_STATUS=${PIPELINE_STATUS[1]}
    fi
  fi
  set -e
else
  TRAIN_STATUS="$HOST_STATUS"
fi

stop_live_uploader
stop_progress_monitor
write_runner_progress postflight

set +e
if [ "$SETUP_STATUS" -eq 0 ]; then
  timeout --signal=TERM --kill-after=60 1800 \
    "$PY" cloud/preflight.py --phase post --train-cmd "$TRAIN_CMD" \
    2>&1 | tee "logs/postflight_${RUN_TAG}.log"
  POST_PIPELINE_STATUS=("${PIPESTATUS[@]}")
  POST_STATUS=${POST_PIPELINE_STATUS[0]}
  if [ "${POST_PIPELINE_STATUS[1]}" -ne 0 ] && [ "$POST_STATUS" -eq 0 ]; then
    POST_STATUS=${POST_PIPELINE_STATUS[1]}
  fi
else
  POST_STATUS="$HOST_STATUS"
fi

UPLOAD_STATUS=0
if [ "$TRAIN_STATUS" -ne 0 ] || [ "$POST_STATUS" -ne 0 ]; then
  write_failure_log pre_upload_failure || true
fi
sanitize_upload_log() {
  local upload_log="$1" sanitizer="$PY"
  if [ ! -x "$sanitizer" ]; then
    sanitizer="$(command -v python3 2>/dev/null || true)"
  fi
  [ -x "$sanitizer" ] || return 1
  UPLOAD_LOG="$upload_log" "$sanitizer" - <<'PY'
import os
import re
from pathlib import Path

path = Path(os.environ["UPLOAD_LOG"])
data = path.read_bytes()
token = os.environ.get("HF_TOKEN", "").encode()
if token:
    data = data.replace(token, b"[REDACTED]")
data = re.sub(rb"hf_[A-Za-z0-9]+", b"[REDACTED]", data)
tmp = path.with_name(path.name + ".redacted")
tmp.write_bytes(data)
tmp.replace(path)
PY
}

upload_with_retries() {
  local local_path="$1" remote_path="$2" commit_message="$3" upload_log="$4"
  local attempt attempt_status=1
  shift 4
  : >"$upload_log" || return 1
  if [ ! -e "$local_path" ]; then
    printf 'Upload source is missing: %s\n' "$local_path" >>"$upload_log"
    return 1
  fi
  for attempt in 1 2 3; do
    printf 'ATTEMPT=%s STARTED_AT=%s SOURCE=%s DESTINATION=%s\n' \
      "$attempt" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$local_path" "$remote_path" \
      >>"$upload_log"
    timeout --signal=TERM --kill-after=60 1800 \
      uvx --from 'huggingface_hub[cli]' hf upload "$HF_REPO" \
      "$local_path" "runs/$RUN_TAG/$remote_path" --repo-type model \
      --commit-message "$commit_message" "$@" >>"$upload_log" 2>&1
    attempt_status=$?
    if ! sanitize_upload_log "$upload_log"; then
      printf 'Upload log redaction failed.\n' >"$upload_log"
      return 1
    fi
    printf 'ATTEMPT=%s STATUS=%s FINISHED_AT=%s\n' \
      "$attempt" "$attempt_status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >>"$upload_log"
    if [ "$attempt_status" -eq 0 ]; then
      return 0
    fi
    [ "$attempt" -eq 3 ] || sleep 15
  done
  return "$attempt_status"
}

complete_emergency_roundtrip() {
  local reason="$1"
  while true; do
    write_runner_progress emergency_recovery
    if timeout --signal=TERM --kill-after=60 10800 \
        bash cloud/vast_api_emergency_snapshot.sh \
        "$SOURCE_SHA" "$RUN_TAG" "$HORIZON" "$INSTANCE_ID" "$reason" \
        >"/tmp/emergency-roundtrip-${RUN_TAG}.log" 2>&1; then
      return 0
    fi
    echo "Emergency round trip failed; the instance and artifacts remain preserved. Retrying in 300 seconds." >&2
    write_runner_progress emergency_retry
    sleep 300
  done
}

RELEASE_UPLOAD_LOG="logs/upload_release_${RUN_TAG}.log"
TRAIN_UPLOAD_LOG="logs/upload_train_${RUN_TAG}.log"
CONTROLLER_LOG="logs/controller_${RUN_TAG}.log"
write_runner_progress uploading_checkpoints
if ! upload_with_retries "$RELEASE_REL" "$RELEASE_REL" \
    "Vast release h=$HORIZON $RUN_TAG" "$RELEASE_UPLOAD_LOG"; then
  UPLOAD_STATUS=1
fi
if ! upload_with_retries "$TRAIN_REL" "$TRAIN_REL" \
    "Vast training artifacts h=$HORIZON $RUN_TAG" "$TRAIN_UPLOAD_LOG"; then
  UPLOAD_STATUS=1
fi

echo "VAST_RUN_TAG=$RUN_TAG"
echo "VAST_HORIZON=$HORIZON"
echo "VAST_TRAIN_STATUS=$TRAIN_STATUS"
echo "VAST_POST_STATUS=$POST_STATUS"
echo "VAST_CHECKPOINT_UPLOAD_STATUS=$UPLOAD_STATUS"
write_local_status || UPLOAD_STATUS=1

# The controller log is now immutable. Its hash can therefore be included in
# the artifact manifest without later runner output invalidating the bundle.
sync "$CONTROLLER_LOG" "$RELEASE_UPLOAD_LOG" "$TRAIN_UPLOAD_LOG" "$LOCAL_STATUS" \
  >/dev/null 2>&1 || true
exec >"/tmp/controller_after_freeze_${RUN_TAG}.log" 2>&1

ARTIFACT_MANIFEST="logs/artifact_manifest_${RUN_TAG}.sha256"
if [ "$TRAIN_STATUS" -eq 0 ] && [ "$POST_STATUS" -eq 0 ]; then
  if [ "$BATCH_MODE" -eq 1 ]; then
    shasum -a 256 \
      "$RELEASE_REL/dexposure-fm-h4.pt" \
      "$RELEASE_REL/dexposure-fm-h8.pt" \
      "$RELEASE_REL/dexposure-fm-h12.pt" \
      "$RELEASE_REL/feature_schema.json" \
      "$RELEASE_REL/task1_metrics.json" \
      "$RELEASE_REL/run_config_h4.json" \
      "$RELEASE_REL/run_config_h8.json" \
      "$RELEASE_REL/run_config_h12.json" \
      "$RELEASE_REL/SHA256SUMS" \
      "$TRAIN_REL/finetuned/best_model_h4.pt" \
      "$TRAIN_REL/finetuned/best_model_h8.pt" \
      "$TRAIN_REL/finetuned/best_model_h12.pt" \
      "$TRAIN_REL/finetuned/feature_schema.json" \
      "$TRAIN_REL/finetuned/experiment_results.json" \
      "$TRAIN_REL/finetuned/data_quality.json" \
      "$TRAIN_REL/finetuned/metrics.json" \
      "$TRAIN_REL/finetuned/all_results.json" \
      logs/run_identity.env logs/input_transport_status.env \
      logs/input_manifest.sha256 "logs/input_download_${RUN_TAG}.log" \
      "logs/train_${RUN_TAG}.log" "logs/postflight_${RUN_TAG}.log" \
      "logs/controller_${RUN_TAG}.log" \
      "logs/upload_release_${RUN_TAG}.log" \
      "logs/upload_train_${RUN_TAG}.log" \
      >"$ARTIFACT_MANIFEST" || POST_STATUS=1
  else
    shasum -a 256 \
      "$RELEASE_REL/dexposure-fm-h${HORIZON}.pt" \
      "$RELEASE_REL/feature_schema.json" \
      "$RELEASE_REL/task1_metrics.json" \
      "$RELEASE_REL/run_config.json" \
      "$RELEASE_REL/SHA256SUMS" \
      "$TRAIN_REL/finetuned/best_model_h${HORIZON}.pt" \
      "$TRAIN_REL/finetuned/feature_schema.json" \
      "$TRAIN_REL/finetuned/experiment_results.json" \
      "$TRAIN_REL/finetuned/data_quality.json" \
      "$TRAIN_REL/finetuned/metrics.json" \
      "$TRAIN_REL/finetuned/all_results.json" \
      logs/run_identity.env "logs/train_${RUN_TAG}.log" \
      "logs/postflight_${RUN_TAG}.log" "logs/controller_${RUN_TAG}.log" \
      "logs/upload_release_${RUN_TAG}.log" \
      "logs/upload_train_${RUN_TAG}.log" \
      >"$ARTIFACT_MANIFEST" || POST_STATUS=1
  fi
fi

if { [ "$TRAIN_CMD" = "bash cloud/main2025_v2_remaining_run.sh --all" ] ||
     [ "$TRAIN_CMD" = "bash cloud/fm_tune_run.sh" ] ||
     [ "$TRAIN_CMD" = "bash cloud/fm_eval_paper_run.sh" ] ||
      [ "$TRAIN_CMD" = "bash cloud/fm_sequential_run.sh" ]; } &&
   [ "$TRAIN_STATUS" -eq 0 ] && [ "$POST_STATUS" -eq 0 ]; then
  shasum -a 256 "$RELEASE_REL/dexposure-fm-h1.pt" \
    "$RELEASE_REL/run_config_h1.json" "$TRAIN_REL/finetuned/best_model_h1.pt" \
    >>"$ARTIFACT_MANIFEST" || POST_STATUS=1
fi

if { [ "$TRAIN_CMD" = "bash cloud/fm_tune_run.sh" ] ||
     [ "$TRAIN_CMD" = "bash cloud/fm_sequential_run.sh" ]; } &&
   [ "$TRAIN_STATUS" -eq 0 ] && [ "$POST_STATUS" -eq 0 ]; then
  shasum -a 256 "$TRAIN_REL/tuning_manifest.sha256" >>"$ARTIFACT_MANIFEST" || POST_STATUS=1
fi

if [ "$TRAIN_CMD" = "bash cloud/fm_eval_paper_run.sh" ] &&
   [ "$TRAIN_STATUS" -eq 0 ] && [ "$POST_STATUS" -eq 0 ]; then
  shasum -a 256 "$TRAIN_REL/eval_manifest.sha256" >>"$ARTIFACT_MANIFEST" || POST_STATUS=1
fi

write_runner_progress uploading_logs
if ! upload_with_retries logs logs "Vast logs h=$HORIZON $RUN_TAG" \
    /tmp/hf-upload-logs.log \
    --exclude 'run_status.env' \
    --exclude 'controller_heartbeat.env' --exclude 'runner_progress.env'; then
  UPLOAD_STATUS=1
fi

# The marker file may advertise zero only before its own successful upload. If
# that upload fails there is no matching marker, and the emergency snapshot
# records the nonzero result locally.
MARKER_STATUS=0
if ! write_local_status; then
  UPLOAD_STATUS=1
  write_local_status || true
fi
if ! upload_with_retries "$LOCAL_STATUS" logs/local_run_status.env \
    "Vast local status h=$HORIZON $RUN_TAG" /tmp/hf-upload-local-status.log; then
  UPLOAD_STATUS=1
  write_local_status || true
fi

# A failure marker must not race the controller against an unfinished recovery
# upload. Preserve and round-trip the failure evidence before publishing it.
if [ "$TRAIN_STATUS" -ne 0 ] || [ "$POST_STATUS" -ne 0 ] || \
    [ "$UPLOAD_STATUS" -ne 0 ]; then
  write_local_status || true
  write_failure_log pre_marker_failure || true
  complete_emergency_roundtrip terminal_status_before_marker
fi

# This is the last normal Hugging Face operation. A successful marker is safe
# only because all earlier status fields are already in its payload.
upload_with_retries "$LOCAL_STATUS" logs/run_status.env \
  "Vast completion marker h=$HORIZON $RUN_TAG" /tmp/hf-upload-marker.log
MARKER_STATUS=$?
if [ "$MARKER_STATUS" -ne 0 ]; then
  write_local_status || true
  # The marker failed, so publish the truthful failure status if Hugging Face
  # becomes reachable again. Never create a completion marker in this branch.
  if ! upload_with_retries "$LOCAL_STATUS" logs/local_run_status.env \
      "Vast marker failure status h=$HORIZON $RUN_TAG" \
      /tmp/hf-upload-local-status-after-marker-failure.log; then
    UPLOAD_STATUS=1
  fi
  write_local_status || true
  write_failure_log marker_upload_failed || true
  complete_emergency_roundtrip marker_upload_failed
fi
REMOTE_FINALIZED=1
write_local_status || true
set -e

if [ "$TRAIN_STATUS" -ne 0 ] || [ "$POST_STATUS" -ne 0 ] || \
    [ "$UPLOAD_STATUS" -ne 0 ] || [ "$MARKER_STATUS" -ne 0 ]; then
  write_failure_log terminal_status || true
fi

[ "$TRAIN_STATUS" -eq 0 ] || exit "$TRAIN_STATUS"
[ "$POST_STATUS" -eq 0 ] || exit "$POST_STATUS"
[ "$UPLOAD_STATUS" -eq 0 ] || exit 1
[ "$MARKER_STATUS" -eq 0 ] || exit 1
