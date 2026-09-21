#!/usr/bin/env bash
# Emergency Vast lifecycle guard. Every remote action is an exact-id stop that
# retains the instance disk. Only the local controller may destroy an instance.
set -u

NO_RUNNER_TIMEOUT_SECONDS="${1:?no-runner timeout is required}"
POLL_SECONDS="${2:?watchdog poll interval is required}"
REMOTE_RECOVERY_GRACE_SECONDS="${3:?remote recovery grace is required}"
CONTROLLER_STALE_SECONDS="${4:?controller heartbeat timeout is required}"
RUNNER_STALE_SECONDS="${5:?runner progress timeout is required}"
INSTANCE_ID="${6:?Vast instance ID is required}"
RUN_TAG="${7:?run tag is required}"
HORIZON="${8:?horizon is required}"
SOURCE_SHA="${9:?source digest is required}"
BOOT_GRACE_SECONDS="${VAST_WATCHDOG_BOOT_GRACE_SECONDS:-${10:-1200}}"
EMERGENCY_PHASE_MAX_SECONDS="${VAST_WATCHDOG_EMERGENCY_PHASE_MAX_SECONDS:-${11:-10800}}"

case "$NO_RUNNER_TIMEOUT_SECONDS:$POLL_SECONDS:$REMOTE_RECOVERY_GRACE_SECONDS:$CONTROLLER_STALE_SECONDS:$RUNNER_STALE_SECONDS:$INSTANCE_ID" in
  *[!0-9:]*) exit 2 ;;
esac
case "$RUN_TAG" in *[!A-Za-z0-9._-]*|'') exit 2 ;; esac
case "$HORIZON" in 1|4|8|12|remaining) ;; *) exit 2 ;; esac
case "$SOURCE_SHA" in *[!0-9a-f]*|'') exit 2 ;; esac
[ "${#SOURCE_SHA}" -eq 64 ] || exit 2

WORKDIR="${VAST_WATCHDOG_WORKDIR:-/root/vast_workdir}"
LOG_DIR="$WORKDIR/logs"
RUNNER_PID_FILE="$LOG_DIR/remote_runner.pid"
RUNNER_PROGRESS_FILE="$LOG_DIR/runner_progress.env"
CONTROLLER_HEARTBEAT_FILE="$LOG_DIR/controller_heartbeat.env"
START_LEASE_DIR="$LOG_DIR/runner_start.lock"
RECOVERY_AUTH_FILE="$LOG_DIR/watchdog_destroy_authorized.env"
WATCHDOG_LOCK="${VAST_WATCHDOG_LOCK:-/tmp/dexposure-vast-watchdog.lock}"
START_LEASE_STALE_SECONDS=1800
LOCK_INCOMPLETE_STALE_SECONDS=30
EMERGENCY_SNAPSHOT_TIMEOUT_SECONDS="${VAST_WATCHDOG_EMERGENCY_SNAPSHOT_TIMEOUT_SECONDS:-2400}"
VAST_CLI_TIMEOUT_SECONDS="${VAST_WATCHDOG_CLI_TIMEOUT_SECONDS:-90}"
VAST_INSTALL_TIMEOUT_SECONDS="${VAST_WATCHDOG_INSTALL_TIMEOUT_SECONDS:-180}"
EMERGENCY_PHASE_START_FILE="/tmp/dexposure-${INSTANCE_ID}-${RUN_TAG}-emergency.started"
WATCHDOG_TEST_DECISION="${VAST_WATCHDOG_TEST_DECISION:-0}"
case "$WATCHDOG_TEST_DECISION" in 0|1) ;; *) exit 2 ;; esac
case "$EMERGENCY_SNAPSHOT_TIMEOUT_SECONDS:$EMERGENCY_PHASE_MAX_SECONDS:$BOOT_GRACE_SECONDS:$VAST_CLI_TIMEOUT_SECONDS:$VAST_INSTALL_TIMEOUT_SECONDS" in
  *[!0-9:]*) exit 2 ;;
esac
[ "$EMERGENCY_SNAPSHOT_TIMEOUT_SECONDS" -gt 0 ] && \
  [ "$EMERGENCY_PHASE_MAX_SECONDS" -gt 0 ] && \
  [ "$BOOT_GRACE_SECONDS" -gt 0 ] && \
  [ "$VAST_CLI_TIMEOUT_SECONDS" -gt 0 ] && \
  [ "$VAST_INSTALL_TIMEOUT_SECONDS" -gt 0 ] || exit 2

mkdir -p "$LOG_DIR"
WATCHDOG_STARTTIME="${VAST_WATCHDOG_SELF_STARTTIME:-$(awk '{print $22}' "/proc/$$/stat" 2>/dev/null || true)}"
case "$WATCHDOG_STARTTIME" in *[!0-9]*|'') exit 2 ;; esac

process_alive() {
  local pid="$1" expected_start="$2" current_start="" state=""
  case "$pid:$expected_start" in *[!0-9:]*) return 1 ;; esac
  kill -0 "$pid" 2>/dev/null || return 1
  current_start="$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || true)"
  state="$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)"
  [ "$current_start" = "$expected_start" ] && [ "$state" != Z ]
}

acquire_watchdog_lock() {
  local owner_pid="" owner_start="" lock_mtime="" now=""
  while ! mkdir "$WATCHDOG_LOCK" 2>/dev/null; do
    read -r owner_pid owner_start <"$WATCHDOG_LOCK/owner" 2>/dev/null || true
    if process_alive "$owner_pid" "$owner_start"; then
      return 1
    fi
    case "$owner_pid:$owner_start" in
      *[!0-9:]*|:|*:)
      lock_mtime="$(stat -c %Y "$WATCHDOG_LOCK" 2>/dev/null || stat -f %m "$WATCHDOG_LOCK" 2>/dev/null || true)"
      now="$(date +%s)"
      case "$lock_mtime" in *[!0-9]*|'') sleep 2; continue ;; esac
      if [ $((now - lock_mtime)) -lt "$LOCK_INCOMPLETE_STALE_SECONDS" ]; then
        sleep 2
        continue
      fi
        ;;
    esac
    rm -f "$WATCHDOG_LOCK/owner"
    rmdir "$WATCHDOG_LOCK" 2>/dev/null || return 1
  done
  # A crash can leave this file empty or partial. The mtime grace above treats
  # that state as owned briefly, then safely reclaims the known single file.
  printf '%s %s\n' "$$" "$WATCHDOG_STARTTIME" >"$WATCHDOG_LOCK/owner" || return 1
}

if ! acquire_watchdog_lock; then
  exit 0
fi
printf '%s %s running\n' "$$" "$WATCHDOG_STARTTIME" >"$LOG_DIR/watchdog.pid"

cleanup_watchdog() {
  local owner_pid="" owner_start=""
  read -r owner_pid owner_start <"$WATCHDOG_LOCK/owner" 2>/dev/null || true
  if [ "$owner_pid" = "$$" ] && [ "$owner_start" = "$WATCHDOG_STARTTIME" ]; then
    rm -f "$WATCHDOG_LOCK/owner"
    rmdir "$WATCHDOG_LOCK" 2>/dev/null || true
  fi
}
handle_watchdog_signal() {
  trap - EXIT INT TERM
  cleanup_watchdog
  exit 130
}
trap cleanup_watchdog EXIT
trap handle_watchdog_signal INT TERM

runner_record() {
  RUNNER_PID=""
  RUNNER_EXPECTED_START=""
  RUNNER_PHASE=""
  [ -s "$RUNNER_PID_FILE" ] || return 1
  read -r RUNNER_PID RUNNER_EXPECTED_START RUNNER_PHASE <"$RUNNER_PID_FILE" || return 1
  case "$RUNNER_PID:$RUNNER_EXPECTED_START" in *[!0-9:]*) return 1 ;; esac
}

runner_alive() {
  runner_record || return 1
  [ "$RUNNER_PHASE" = running ] || return 1
  process_alive "$RUNNER_PID" "$RUNNER_EXPECTED_START"
}

field_value() {
  local file="$1" key="$2"
  awk -F= -v key="$key" '
    $1 == key { count++; value = substr($0, index($0, "=") + 1) }
    END { if (count == 1) print value; else exit 1 }
  ' "$file"
}

validate_bound_env() {
  local file="$1" allowed="$2" field_count="$3"
  [ -s "$file" ] || return 1
  awk -F= -v allowed="$allowed" -v field_count="$field_count" '
    NF != 2 { exit 1 }
    index("|" allowed "|", "|" $1 "|") == 0 { exit 1 }
    { seen[$1]++ }
    END {
      if (NR != field_count) exit 1
      for (key in seen) if (seen[key] != 1) exit 1
    }
  ' "$file" || return 1
  [ "$(field_value "$file" INSTANCE_ID 2>/dev/null)" = "$INSTANCE_ID" ] && \
    [ "$(field_value "$file" RUN_TAG 2>/dev/null)" = "$RUN_TAG" ] && \
    [ "$(field_value "$file" HORIZON 2>/dev/null)" = "$HORIZON" ] && \
    [ "$(field_value "$file" SOURCE_SHA256 2>/dev/null)" = "$SOURCE_SHA" ]
}

timestamp_fresh() {
  local timestamp="$1" maximum_age="$2" now="$(date +%s)"
  case "$timestamp" in *[!0-9]*|'') return 1 ;; esac
  [ "$timestamp" -le "$now" ] && [ $((now - timestamp)) -le "$maximum_age" ]
}

controller_alive() {
  validate_bound_env "$CONTROLLER_HEARTBEAT_FILE" \
    'INSTANCE_ID|RUN_TAG|HORIZON|SOURCE_SHA256|CONTROLLER_TIMESTAMP' 5 || return 1
  timestamp_fresh "$(field_value "$CONTROLLER_HEARTBEAT_FILE" CONTROLLER_TIMESTAMP 2>/dev/null)" \
    "$CONTROLLER_STALE_SECONDS"
}

boot_grace_active() {
  local boot_file="$LOG_DIR/watchdog_boot_started_at" boot_timestamp=""
  [ -s "$boot_file" ] || return 1
  boot_timestamp="$(head -n1 "$boot_file" 2>/dev/null || true)"
  timestamp_fresh "$boot_timestamp" "$BOOT_GRACE_SECONDS"
}

load_runner_progress() {
  RUNNER_PROGRESS_PHASE=""
  RUNNER_PROGRESS_TIMESTAMP=""
  validate_bound_env "$RUNNER_PROGRESS_FILE" \
    'INSTANCE_ID|RUN_TAG|HORIZON|SOURCE_SHA256|PHASE|PROGRESS_TIMESTAMP' 6 || return 1
  RUNNER_PROGRESS_PHASE="$(field_value "$RUNNER_PROGRESS_FILE" PHASE 2>/dev/null)"
  case "$RUNNER_PROGRESS_PHASE" in *[!A-Za-z0-9._-]*|'') return 1 ;; esac
  RUNNER_PROGRESS_TIMESTAMP="$(field_value "$RUNNER_PROGRESS_FILE" PROGRESS_TIMESTAMP 2>/dev/null)"
  case "$RUNNER_PROGRESS_TIMESTAMP" in *[!0-9]*|'') return 1 ;; esac
}

runner_progress_fresh() {
  load_runner_progress || return 1
  timestamp_fresh "$RUNNER_PROGRESS_TIMESTAMP" "$RUNNER_STALE_SECONDS"
}

runner_progress_expired() {
  local now="$(date +%s)"
  load_runner_progress || return 1
  [ "$RUNNER_PROGRESS_TIMESTAMP" -le "$now" ] && \
    [ $((now - RUNNER_PROGRESS_TIMESTAMP)) -gt "$RUNNER_STALE_SECONDS" ]
}

emergency_retry_expired() {
  local started_at="" now="$(date +%s)" marker_tmp=""
  load_runner_progress || {
    rm -f "$EMERGENCY_PHASE_START_FILE"
    return 1
  }
  case "$RUNNER_PROGRESS_PHASE" in
    emergency_recovery|emergency_retry) ;;
    *)
      rm -f "$EMERGENCY_PHASE_START_FILE"
      return 1
      ;;
  esac
  started_at="$(head -n1 "$EMERGENCY_PHASE_START_FILE" 2>/dev/null || true)"
  case "$started_at" in
    *[!0-9]*|'')
      marker_tmp="${EMERGENCY_PHASE_START_FILE}.tmp.$$"
      printf '%s\n' "$now" >"$marker_tmp" || return 1
      mv "$marker_tmp" "$EMERGENCY_PHASE_START_FILE" || return 1
      return 1
      ;;
  esac
  [ "$started_at" -le "$now" ] || return 0
  [ $((now - started_at)) -ge "$EMERGENCY_PHASE_MAX_SECONDS" ]
}

start_lease_active() {
  local created_at="" lease_mtime="" now="$(date +%s)"
  [ -d "$START_LEASE_DIR" ] || return 1
  created_at="$(head -n1 "$START_LEASE_DIR/created_at" 2>/dev/null || true)"
  case "$created_at" in
    *[!0-9]*|'')
      lease_mtime="$(stat -c %Y "$START_LEASE_DIR" 2>/dev/null || stat -f %m "$START_LEASE_DIR" 2>/dev/null || true)"
      case "$lease_mtime" in *[!0-9]*|'') return 0 ;; esac
      created_at="$lease_mtime"
      ;;
  esac
  if [ "$created_at" -le "$now" ] && \
      [ $((now - created_at)) -lt "$START_LEASE_STALE_SECONDS" ]; then
    return 0
  fi
  rm -f "$START_LEASE_DIR/created_at"
  rmdir "$START_LEASE_DIR" 2>/dev/null || return 0
  echo "Removed a stale runner start lease."
  return 1
}

run_identity_matches() {
  local identity="$LOG_DIR/run_identity.env"
  validate_bound_env "$identity" 'RUN_TAG|HORIZON|SOURCE_SHA256|INSTANCE_ID' 4
}

recovery_authorized() {
  local local_verified="" hf_verified="" evidence_level="" model_sha="" snapshot_sha=""
  validate_bound_env "$RECOVERY_AUTH_FILE" \
    'LOCAL_RECOVERY_VERIFIED|HF_ROUNDTRIP_VERIFIED|INSTANCE_ID|RUN_TAG|HORIZON|SOURCE_SHA256|EVIDENCE_LEVEL|MODEL_SHA256|SNAPSHOT_SHA256' \
    9 || return 1
  local_verified="$(field_value "$RECOVERY_AUTH_FILE" LOCAL_RECOVERY_VERIFIED 2>/dev/null)"
  hf_verified="$(field_value "$RECOVERY_AUTH_FILE" HF_ROUNDTRIP_VERIFIED 2>/dev/null)"
  evidence_level="$(field_value "$RECOVERY_AUTH_FILE" EVIDENCE_LEVEL 2>/dev/null)"
  model_sha="$(field_value "$RECOVERY_AUTH_FILE" MODEL_SHA256 2>/dev/null)"
  snapshot_sha="$(field_value "$RECOVERY_AUTH_FILE" SNAPSHOT_SHA256 2>/dev/null)"
  case "$model_sha" in
    none) ;;
    *[!0-9a-f]*|'') return 1 ;;
    *) [ "${#model_sha}" -eq 64 ] || return 1 ;;
  esac
  if [ "$local_verified:$hf_verified" = 1:0 ]; then
    case "$evidence_level" in complete_bundle|failure_inventory) ;; *) return 1 ;; esac
    [ "$snapshot_sha" = none ]
  elif [ "$local_verified:$hf_verified" = 0:1 ]; then
    [ "$evidence_level" = remote_snapshot ] && \
      [ "${#snapshot_sha}" -eq 64 ] && [[ "$snapshot_sha" != *[!0-9a-f]* ]]
  else
    return 1
  fi
}

never_started_clean() {
  [ ! -e "$RUNNER_PID_FILE" ] && [ ! -s "$LOG_DIR/run_identity.env" ] && \
    [ ! -s "$LOG_DIR/controller_${RUN_TAG}.log" ] && \
    [ ! -s "$LOG_DIR/train_${RUN_TAG}.log" ] || return 1
  ! find \
    "$WORKDIR/checkpoints/main2025_v2_h${HORIZON}" \
    "$WORKDIR/checkpoints/main2025_v2_h${HORIZON}_train" \
    -type f -size +0c -print -quit 2>/dev/null | grep -q .
}

create_emergency_snapshot() {
  local reason="$1"
  [ -s "$WORKDIR/cloud/vast_api_emergency_snapshot.sh" ] || return 1
  timeout --signal=TERM --kill-after=30 "$EMERGENCY_SNAPSHOT_TIMEOUT_SECONDS" \
    bash "$WORKDIR/cloud/vast_api_emergency_snapshot.sh" \
    "$SOURCE_SHA" "$RUN_TAG" "$HORIZON" "$INSTANCE_ID" "$reason"
}

NO_RUNNER_DEADLINE=$(( $(date +%s) + NO_RUNNER_TIMEOUT_SECONDS ))
local_recovery_stop_ready() {
  recovery_authorized || return 1
  [ "$(field_value "$RECOVERY_AUTH_FILE" LOCAL_RECOVERY_VERIFIED 2>/dev/null)" = 1 ]
}

remote_snapshot_stop_ready() {
  recovery_authorized || return 1
  [ "$(field_value "$RECOVERY_AUTH_FILE" LOCAL_RECOVERY_VERIFIED 2>/dev/null)" = 0 ] || return 1
  [ "$(field_value "$RECOVERY_AUTH_FILE" HF_ROUNDTRIP_VERIFIED 2>/dev/null)" = 1 ] || return 1
  local authorization_mtime="" now="$(date +%s)"
  authorization_mtime="$(stat -c %Y "$RECOVERY_AUTH_FILE" 2>/dev/null || stat -f %m "$RECOVERY_AUTH_FILE" 2>/dev/null || true)"
  case "$authorization_mtime" in *[!0-9]*|'') return 1 ;; esac
  [ "$authorization_mtime" -le "$now" ] && \
    [ $((now - authorization_mtime)) -ge "$REMOTE_RECOVERY_GRACE_SECONDS" ]
}

test_decision() {
  [ "$WATCHDOG_TEST_DECISION" = 1 ] || return 1
  echo "WATCHDOG_DECISION=$1"
  exit 0
}

ACTION=""
ACTION_REASON=""
while [ -z "$ACTION" ]; do
  if boot_grace_active; then
    test_decision boot_grace_active || true
    echo "Instance boot grace is active; watchdog actions are deferred."
    sleep "$POLL_SECONDS"
    continue
  fi
  if start_lease_active; then
    test_decision start_lease_active || true
    echo "Remote runner start is leased; emergency action deferred."
    sleep "$POLL_SECONDS"
    continue
  fi
  # Billing protection is independent of controller liveness. Once a runner has
  # spent the full emergency-recovery allowance, stop the instance and retain
  # its disk even if the local controller process is still alive.
  if emergency_retry_expired; then
    test_decision emergency_retry_stalled || true
    ACTION=stop
    ACTION_REASON=emergency_retry_stalled
    break
  fi
  if runner_progress_expired; then
    test_decision runner_progress_deadline || true
    ACTION=stop
    ACTION_REASON=runner_progress_deadline
    break
  fi
  # A remote-only HF round trip is enough to stop billing. It never authorizes
  # this remote process to destroy the disk.
  if remote_snapshot_stop_ready; then
    test_decision remote_snapshot_stop_ready || true
    ACTION=stop
    ACTION_REASON=remote_snapshot_verified_no_local_recovery
    break
  fi
  if controller_alive; then
    test_decision controller_alive || true
    echo "Local controller heartbeat is current; emergency watchdog deferred."
    sleep "$POLL_SECONDS"
    continue
  fi
  if local_recovery_stop_ready; then
    test_decision local_recovery_stop_ready || true
    ACTION=stop
    ACTION_REASON=local_recovery_verified_controller_absent
    break
  fi
  if runner_alive; then
    if runner_progress_fresh; then
      test_decision runner_progress_fresh || true
      echo "Remote runner made recent progress; emergency stop deferred."
    else
      if [ "$WATCHDOG_TEST_DECISION" = 1 ]; then
        echo "WATCHDOG_DECISION=runner_stalled"
        exit 0
      fi
      ACTION=stop
      ACTION_REASON=runner_progress_stale
      break
    fi
    sleep "$POLL_SECONDS"
    continue
  fi

  # A valid remote snapshot may still be inside its stop grace period.
  # Rebuilding it would refresh the authorization mtime forever.
  if recovery_authorized; then
    test_decision authorization_grace || true
    echo "Verified remote recovery is waiting for the controller grace period."
    sleep "$POLL_SECONDS"
    continue
  fi

  # A dead, identified runner gets one bounded recovery attempt. A failed or
  # stalled HF round trip falls back to stop, which preserves the instance disk.
  if run_identity_matches; then
    test_decision emergency_snapshot_required || true
    if create_emergency_snapshot watchdog_dead_runner; then
      continue
    fi
    ACTION=stop
    ACTION_REASON=emergency_snapshot_failed
    break
  fi

  if [ "$(date +%s)" -ge "$NO_RUNNER_DEADLINE" ]; then
    if never_started_clean; then
      ACTION_REASON=runner_never_started_clean
    else
      ACTION_REASON=runner_start_unreliable
    fi
    test_decision "$ACTION_REASON" || true
    ACTION=stop
    break
  fi

  test_decision preserve_unverified || true
  echo "No verified runner identity yet; emergency watchdog is waiting before stop."
  sleep "$POLL_SECONDS"
done

if ! command -v vastai >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
fi
ensure_vast_cli() {
  command -v vastai >/dev/null 2>&1 && return 0
  timeout --signal=TERM --kill-after=15 "$VAST_INSTALL_TIMEOUT_SECONDS" \
    bash -c 'set -o pipefail; curl -fsSL --connect-timeout 15 --max-time 120 https://vast.ai/install.sh | bash' \
    >/dev/null 2>&1 || return 1
  command -v vastai >/dev/null 2>&1
}

stop_should_defer() {
  local reason="$1"
  boot_grace_active && return 0
  start_lease_active && return 0
  case "$reason" in
    emergency_retry_stalled)
      if runner_alive && ! emergency_retry_expired; then
        return 0
      fi
      return 1
      ;;
    remote_snapshot_verified_no_local_recovery)
      remote_snapshot_stop_ready || return 0
      return 1
      ;;
    local_recovery_verified_controller_absent)
      controller_alive && return 0
      local_recovery_stop_ready || return 0
      return 1
      ;;
    runner_progress_deadline)
      runner_progress_expired || return 0
      return 1
      ;;
  esac
  controller_alive && return 0
  if runner_alive; then
    emergency_retry_expired && return 1
    runner_progress_fresh && return 0
  fi
  return 1
}

stop_instance_preserving_disk() {
  local reason="$1" stop_response=""
  while true; do
    if stop_should_defer "$reason"; then
      echo "Stop cancelled because controller or runner activity resumed."
      return 75
    fi
    if ! ensure_vast_cli; then
      echo "Vast CLI installation timed out; retrying bounded stop setup." >&2
      sleep "$POLL_SECONDS"
      continue
    fi
    # This is intentionally adjacent to the API mutation. A controller that
    # resumes during CLI setup cancels the stop before any remote state changes.
    if stop_should_defer "$reason"; then
      echo "Stop cancelled because controller or runner activity resumed."
      return 75
    fi
    echo "Requesting Vast stop for exact instance id=$INSTANCE_ID reason=$reason; disk is retained."
    stop_response="$(timeout --signal=TERM --kill-after=15 "$VAST_CLI_TIMEOUT_SECONDS" \
      vastai stop instance "$INSTANCE_ID" 2>&1)" || stop_response=""
    if grep -Fxq "stopping instance $INSTANCE_ID." <<<"$stop_response"; then
      echo "Vast stop request accepted for instance id=$INSTANCE_ID reason=$reason."
      return 0
    fi
    echo "Vast stop request failed or timed out; retrying after a fresh heartbeat check." >&2
    sleep "$POLL_SECONDS"
  done
}

stop_instance_preserving_disk "$ACTION_REASON"
action_status=$?
# Status 75 means activity resumed after the decision but before the API call.
# Exit cleanly so the detached supervisor starts a fresh full-state evaluation.
[ "$action_status" -eq 0 ] || [ "$action_status" -eq 75 ]
