#!/usr/bin/env bash
# Preserve one stopped Vast run in the private Hugging Face repository.
# The caller must stop every process that can write logs/checkpoints first.
set -euo pipefail

SOURCE_SHA="${1:?source digest is required}"
RUN_TAG="${2:?run tag is required}"
HORIZON="${3:?horizon is required}"
INSTANCE_ID="${4:?instance ID is required}"
REASON="${5:?recovery reason is required}"

case "$RUN_TAG:$REASON" in
  *[!A-Za-z0-9._:-]*|'') exit 2 ;;
esac
case "$INSTANCE_ID" in *[!0-9]*|'') exit 2 ;; esac
case "$HORIZON" in 1|4|8|12|remaining) ;; *) exit 2 ;; esac
case "$SOURCE_SHA" in *[!0-9a-f]*|'') exit 2 ;; esac
[ "${#SOURCE_SHA}" -eq 64 ] || exit 2

ROOT=/root/vast_workdir
cd "$ROOT"
mkdir -p logs

HF_REPO=losdwind/graph-dexposure-ckpt
if [ -z "${HF_TOKEN:-}" ] && [ -s /root/.hf_token ]; then
  export HF_TOKEN="$(cat /root/.hf_token)"
fi
[ -n "${HF_TOKEN:-}" ] || {
  echo "Emergency recovery has no Hugging Face token; instance is preserved." >&2
  exit 70
}

for command in flock tar find sort shasum timeout; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "Emergency recovery requires command: $command" >&2
    exit 71
  }
done
if ! command -v uvx >/dev/null 2>&1; then
  timeout --foreground 300 bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uvx >/dev/null 2>&1 || exit 71

# flock releases automatically if this process is killed. The path may remain,
# but unlike a mkdir lock it cannot become permanently stale.
exec 9>logs/emergency_snapshot.lock
flock -w 600 9 || exit 72

STATUS_FILE=logs/local_run_status.env
if [ ! -s "$STATUS_FILE" ]; then
  status_tmp="${STATUS_FILE}.tmp.$$"
  printf 'RUN_TAG=%s\nHORIZON=%s\nSOURCE_SHA256=%s\nINSTANCE_ID=%s\nINPUT_STATUS=124\nTRAIN_STATUS=124\nPOST_STATUS=124\nUPLOAD_STATUS=124\nMARKER_STATUS=124\n' \
    "$RUN_TAG" "$HORIZON" "$SOURCE_SHA" "$INSTANCE_ID" >"$status_tmp"
  mv "$status_tmp" "$STATUS_FILE"
fi

RECOVERY_STATUS=logs/emergency_snapshot_status.env
recovery_status_tmp="${RECOVERY_STATUS}.tmp.$$"
printf 'RUN_TAG=%s\nHORIZON=%s\nSOURCE_SHA256=%s\nINSTANCE_ID=%s\nREASON=%s\n' \
  "$RUN_TAG" "$HORIZON" "$SOURCE_SHA" "$INSTANCE_ID" "$REASON" \
  >"$recovery_status_tmp"
mv "$recovery_status_tmp" "$RECOVERY_STATUS"

emergency_root="$(mktemp -d "/tmp/dexposure-emergency-${RUN_TAG}.XXXXXX")"
cleanup_emergency() {
  rm -rf "$emergency_root"
}
trap cleanup_emergency EXIT INT TERM
archive="$emergency_root/artifact_snapshot.tar"
checksum="$emergency_root/artifact_snapshot.sha256"
paths="$emergency_root/artifact_paths.nul"
roundtrip="$emergency_root/roundtrip"
mkdir -p "$roundtrip"

roots=(logs)
for output_root in \
  "checkpoints/main2025_v2_h${HORIZON}" \
  "checkpoints/main2025_v2_h${HORIZON}_train"; do
  [ ! -d "$output_root" ] || roots+=("$output_root")
done

# Archive ordinary files only. Operational files can change independently of
# model artifacts and are deliberately excluded from the recovery contract.
find "${roots[@]}" -type f \
  ! -path 'logs/remote_runner.pid' \
  ! -path 'logs/watchdog.pid' \
  ! -path 'logs/watchdog_instance_id' \
  ! -path 'logs/watchdog_destroy_authorized.env' \
  ! -path 'logs/controller_heartbeat.env' \
  ! -path 'logs/.controller_heartbeat.env.*' \
  ! -path 'logs/runner_progress.env' \
  ! -path 'logs/.runner_progress.env.*' \
  ! -path 'logs/emergency_snapshot.lock' \
  -print0 | sort -z >"$paths"
[ -s "$paths" ] || exit 73
tar --create --file="$archive" --no-recursion --null --files-from="$paths"
snapshot_sha="$(shasum -a 256 "$archive" | awk '{print $1}')"
printf '%s  artifact_snapshot.tar\n' "$snapshot_sha" >"$checksum"

retry_hf() {
  local attempt
  for attempt in 1 2 3; do
    if timeout --foreground 600 "$@"; then
      return 0
    fi
    echo "Hugging Face emergency operation attempt $attempt/3 failed." >&2
    [ "$attempt" -eq 3 ] || sleep 60
  done
  return 1
}

retry_hf uvx --from 'huggingface_hub[cli]' hf upload "$HF_REPO" \
  "$archive" "runs/$RUN_TAG/emergency/artifact_snapshot.tar" --repo-type model \
  --commit-message "Vast emergency artifact snapshot h=$HORIZON $RUN_TAG" >/dev/null
retry_hf uvx --from 'huggingface_hub[cli]' hf upload "$HF_REPO" \
  "$checksum" "runs/$RUN_TAG/emergency/artifact_snapshot.sha256" --repo-type model \
  --commit-message "Vast emergency artifact checksum h=$HORIZON $RUN_TAG" >/dev/null
retry_hf uvx --from 'huggingface_hub[cli]' hf download "$HF_REPO" --repo-type model \
  --include "runs/$RUN_TAG/emergency/artifact_snapshot.tar" \
  --include "runs/$RUN_TAG/emergency/artifact_snapshot.sha256" \
  --local-dir "$roundtrip" --force-download >/dev/null

downloaded_root="$roundtrip/runs/$RUN_TAG/emergency"
test -s "$downloaded_root/artifact_snapshot.tar"
test -s "$downloaded_root/artifact_snapshot.sha256"
[ "$(shasum -a 256 "$downloaded_root/artifact_snapshot.tar" | awk '{print $1}')" = "$snapshot_sha" ]
cmp -s "$checksum" "$downloaded_root/artifact_snapshot.sha256"

model_sha=none
model_file="checkpoints/main2025_v2_h${HORIZON}/dexposure-fm-h${HORIZON}.pt"
if [ -s "$model_file" ]; then
  model_sha="$(shasum -a 256 "$model_file" | awk '{print $1}')"
fi
authorization="$emergency_root/watchdog_destroy_authorized.env"
authorization_tmp="${authorization}.tmp.$$"
printf 'LOCAL_RECOVERY_VERIFIED=0\nHF_ROUNDTRIP_VERIFIED=1\nINSTANCE_ID=%s\nRUN_TAG=%s\nHORIZON=%s\nSOURCE_SHA256=%s\nEVIDENCE_LEVEL=remote_snapshot\nMODEL_SHA256=%s\nSNAPSHOT_SHA256=%s\n' \
  "$INSTANCE_ID" "$RUN_TAG" "$HORIZON" "$SOURCE_SHA" "$model_sha" "$snapshot_sha" \
  >"$authorization_tmp"
mv "$authorization_tmp" "$authorization"
retry_hf uvx --from 'huggingface_hub[cli]' hf upload "$HF_REPO" \
  "$authorization" "runs/$RUN_TAG/emergency/watchdog_destroy_authorized.env" \
  --repo-type model --commit-message "Vast emergency recovery proof h=$HORIZON $RUN_TAG" \
  >/dev/null
retry_hf uvx --from 'huggingface_hub[cli]' hf download "$HF_REPO" --repo-type model \
  --include "runs/$RUN_TAG/emergency/watchdog_destroy_authorized.env" \
  --local-dir "$roundtrip" --force-download >/dev/null
cmp -s "$authorization" "$downloaded_root/watchdog_destroy_authorized.env"

# Publish the watchdog-visible authorization last. At this point the archive,
# checksum, and authorization have all completed a private HF round trip.
published_authorization=logs/watchdog_destroy_authorized.env
published_tmp="${published_authorization}.tmp.$$"
cp "$authorization" "$published_tmp"
mv "$published_tmp" "$published_authorization"

echo "Emergency Hugging Face round trip verified sha256=$snapshot_sha reason=$REASON"
