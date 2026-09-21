from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = ROOT / "cloud" / "vast_api_train.sh"
RUNNER_PATH = ROOT / "cloud" / "vast_api_remote_run.sh"


def _source(path: Path) -> str:
    return path.read_text()


def _block(source: str, start: str, end: str) -> str:
    return source[source.index(start) : source.index(end, source.index(start))]


def _function(source: str, name: str) -> str:
    start = source.index(f"{name}() {{")
    end = source.index("\n}\n", start) + len("\n}\n")
    return source[start:end]


def _run_bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_lifecycle_function(
    function_source: str,
    invocation: str,
    responses: list[str],
    *,
    default_response: str = "[]",
) -> subprocess.CompletedProcess[str]:
    script = r'''set -u
INSTANCE_ID=42
API_TIMEOUT_SECONDS=1
LIST_CALL_FILE="$(mktemp)"
COMMAND_FILE="$(mktemp)"
printf '0\n' >"$LIST_CALL_FILE"
RESPONSES=(__RESPONSES__)
DEFAULT_RESPONSE=__DEFAULT_RESPONSE__
list_instances() {
  local call response
  call="$(cat "$LIST_CALL_FILE")"
  call=$((call + 1))
  printf '%s\n' "$call" >"$LIST_CALL_FILE"
  if [ "$call" -le "${#RESPONSES[@]}" ]; then
    response="${RESPONSES[$((call - 1))]}"
  else
    response="$DEFAULT_RESPONSE"
  fi
  if [ "$response" = FAIL ]; then
    return 1
  fi
  printf '%s\n' "$response"
}
run_bounded() {
  shift
  printf '%s\n' "$*" >>"$COMMAND_FILE"
  return 0
}
sleep() { :; }
__FUNCTION__
__INVOCATION__
function_status=$?
printf 'status=%s calls=%s commands=%s instance=<%s>\n' \
  "$function_status" "$(cat "$LIST_CALL_FILE")" "$(wc -l <"$COMMAND_FILE" | tr -d ' ')" "$INSTANCE_ID"
rm -f "$LIST_CALL_FILE" "$COMMAND_FILE"
'''
    script = script.replace(
        "__RESPONSES__", " ".join(shlex.quote(value) for value in responses)
    )
    script = script.replace("__DEFAULT_RESPONSE__", shlex.quote(default_response))
    script = script.replace("__FUNCTION__", function_source)
    script = script.replace("__INVOCATION__", invocation)
    return _run_bash(script)


def test_vast_shells_and_generated_onstart_are_valid_bash() -> None:
    subprocess.run(
        ["bash", "-n", str(CONTROLLER_PATH), str(RUNNER_PATH)],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        [
            "bash",
            "-c",
            """
WATCHDOG_ARGUMENTS='3600 60 300 600 7200'
WATCHDOG_BOOT_GRACE_SECONDS=1200
WATCHDOG_EMERGENCY_PHASE_MAX_SECONDS=10800
RUN_TAG=test
HORIZON=1
SOURCE_SHA=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
eval "$(grep '^ONSTART=' cloud/vast_api_train.sh)"
bash -n <<<"$ONSTART"
""",
        ],
        check=True,
        cwd=ROOT,
    )


def test_bounded_network_command_redacts_sdk_credential_urls() -> None:
    function = _function(_source(CONTROLLER_PATH), "run_bounded")
    result = _run_bash(function + "\nrun_bounded 5 python3 -c " + shlex.quote(
        "import sys; sys.stderr.write('https://example.test/?api_key=DUMMY_TOKEN_123\\n'); print('output stays readable')"
    ))
    assert result.returncode == 0, result.stderr
    assert "DUMMY_TOKEN_123" not in result.stderr
    assert "api_key=[REDACTED]" in result.stderr
    assert "output stays readable" in result.stdout


def test_training_has_a_controller_supplied_hard_deadline() -> None:
    controller = _source(CONTROLLER_PATH)
    runner = _source(RUNNER_PATH)

    assert 'TRAIN_TIMEOUT_SECONDS="${VAST_TRAIN_TIMEOUT_SECONDS:-21600}"' in controller
    assert "printf -v TRAIN_TIMEOUT_Q '%q' \"$TRAIN_TIMEOUT_SECONDS\"" in controller
    handoff = next(
        line for line in controller.splitlines() if "vast_api_remote_run.sh" in line
    )
    assert "$INSTANCE_ID_Q $TRAIN_TIMEOUT_Q" in handoff
    assert 'TRAIN_TIMEOUT_SECONDS="${6:?training timeout is required}"' in runner
    assert 'timeout --signal=TERM --kill-after=300 "$TRAIN_TIMEOUT_SECONDS"' in runner


def test_quiet_runner_deadline_cannot_preempt_training_cap() -> None:
    controller = _source(CONTROLLER_PATH)
    assert (
        'WATCHDOG_RUNNER_STALE_SECONDS="${VAST_WATCHDOG_RUNNER_STALE_SECONDS:-'
        '$((TRAIN_TIMEOUT_SECONDS + WATCHDOG_FINALIZATION_MARGIN_SECONDS))}"'
        in controller
    )
    assert (
        '[ "$WATCHDOG_RUNNER_STALE_SECONDS" -ge '
        '$((TRAIN_TIMEOUT_SECONDS + WATCHDOG_FINALIZATION_MARGIN_SECONDS)) ]'
        in controller
    )


def test_runner_installs_recovery_trap_before_identity_and_setup() -> None:
    runner = _source(RUNNER_PATH)
    trap = runner.index("trap finalize_remote_exit EXIT")
    assert trap < runner.index("cat >logs/run_identity.env")
    assert trap < runner.index("uv sync --frozen")
    assert "write_failure_log unexpected_exit" in runner


def test_runner_network_operations_are_bounded() -> None:
    runner = _source(RUNNER_PATH)
    uvx_calls = runner.count("uvx --from 'huggingface_hub[cli]'")
    bounded_uvx_calls = len(
        re.findall(
            r"(?:setsid )?timeout --signal=TERM --kill-after=\d+ \d+ \\\n\s+uvx --from 'huggingface_hub\[cli\]'",
            runner,
        )
    )
    assert uvx_calls == bounded_uvx_calls
    assert "curl --connect-timeout 15 --max-time 240" in runner
    assert "timeout --signal=TERM --kill-after=60 10800" in runner


def test_precreation_hf_probe_does_not_claim_the_run_prefix() -> None:
    controller = _source(CONTROLLER_PATH)
    probe = controller.index('"controller_preflights/$RUN_TAG.env"')
    create = controller.index('vastai create instance "$OFFER_ID"')
    bound = controller.index('"runs/$RUN_TAG/controller/preflight.env"')

    assert probe < create < bound
    assert "runs/$RUN_TAG/controller/preflight.env" not in controller[:create]


def test_create_lock_is_owned_and_stale_reclaim_is_atomic() -> None:
    controller = _source(CONTROLLER_PATH)
    lock = _block(controller, 'CREATE_LOCK_DIR="', "LIVE_JSON=")

    assert "CREATE_LOCK_TOKEN=" in lock
    assert "CREATE_LOCK_PROCESS_START=" in lock
    assert "create_lock_is_ours" in lock
    assert 'mv "$CREATE_LOCK_DIR" "$stale_path"' in lock
    assert 'rm -rf "$CREATE_LOCK_DIR"' not in lock
    assert lock.index("trap handle_create_lock_signal INT TERM") < lock.rindex(
        "\n  acquire_create_lock\n"
    )
    assert "trap - EXIT INT TERM" in lock
    assert "exit 130" in lock
    assert "(\n        umask 077" in lock


def test_precreate_guard_requires_three_consecutive_valid_empty_samples() -> None:
    controller = _source(CONTROLLER_PATH)
    guard = _function(controller, "confirm_no_live_v2_instances")

    recovered = _run_lifecycle_function(
        guard,
        "confirm_no_live_v2_instances",
        ["[]", "FAIL", "[]", "not-json", "[]", "[]", "[]"],
    )
    assert recovered.returncode == 0, recovered.stderr
    assert "status=0 calls=7 commands=0" in recovered.stdout

    collision = _run_lifecycle_function(
        guard,
        "confirm_no_live_v2_instances",
        [
            "[]",
            '[{"id":99,"label":"dexposure-v2-other","actual_status":"running"}]',
        ],
    )
    assert collision.returncode == 0
    assert "status=1 calls=2 commands=0" in collision.stdout
    assert "refusing a duplicate" in collision.stderr

    lock_call = controller.index("  acquire_create_lock\n")
    guard_call = controller.index("  confirm_no_live_v2_instances || exit 1")
    create_call = controller.index('vastai create instance "$OFFER_ID"')
    release_after_create = controller.index("  release_create_lock\n", create_call)
    assert lock_call < guard_call < create_call < release_after_create
    assert controller.count('vastai create instance "$OFFER_ID"') == 1


def test_destroy_requires_three_consecutive_valid_absence_samples() -> None:
    destroy = _function(_source(CONTROLLER_PATH), "destroy_instance")

    already_absent = _run_lifecycle_function(
        destroy,
        "destroy_instance 42",
        ["[]", "[]", "[]"],
    )
    assert already_absent.returncode == 0, already_absent.stderr
    assert "status=0 calls=3 commands=0 instance=<>" in already_absent.stdout
    assert "three consecutive valid samples" in already_absent.stdout

    reset_then_absent = _run_lifecycle_function(
        destroy,
        "destroy_instance 42",
        [
            '[{"id":42}]',
            '[{"id":42}]',
            '[{"id":42}]',
            "[]",
            "FAIL",
            "[]",
            "not-json",
            "[]",
            '[{"id":"42"}]',
            "[]",
            "[]",
            "[]",
        ],
    )
    assert reset_then_absent.returncode == 0, reset_then_absent.stderr
    assert "status=0 calls=12 commands=1 instance=<>" in reset_then_absent.stdout

    never_stable = _run_lifecycle_function(
        destroy,
        "destroy_instance 42",
        ['[{"id":42}]', '[{"id":42}]', '[{"id":42}]'],
        default_response='[{"id":42}]',
    )
    assert never_stable.returncode == 0
    assert "status=1 calls=33 commands=1 instance=<42>" in never_stable.stdout
    assert "did not remain absent" in never_stable.stderr


def test_stop_requires_three_consecutive_exact_id_stopped_samples() -> None:
    stop = _function(_source(CONTROLLER_PATH), "stop_instance_preserving_disk")

    reset_then_stopped = _run_lifecycle_function(
        stop,
        "stop_instance_preserving_disk 42",
        [
            '[{"id":42,"actual_status":"stopped"}]',
            "FAIL",
            '[{"id":42,"actual_status":"stopped"}]',
            "not-json",
            '[{"id":42,"actual_status":"stopped"}]',
            "[]",
            '[{"id":42,"actual_status":"stopped"}]',
            '[{"id":42,"actual_status":"running"}]',
            '[{"id":42,"actual_status":"stopped"}]',
            '[{"id":42,"actual_status":"exited"}]',
            '[{"id":42,"actual_status":"stopped"}]',
        ],
    )
    assert reset_then_stopped.returncode == 0, reset_then_stopped.stderr
    assert "status=0 calls=11 commands=1 instance=<42>" in reset_then_stopped.stdout
    assert "three consecutive valid samples" in reset_then_stopped.stdout

    absent_is_not_stopped = _run_lifecycle_function(
        stop,
        "stop_instance_preserving_disk 42",
        [],
    )
    assert absent_is_not_stopped.returncode == 0
    assert "status=1 calls=30 commands=1 instance=<42>" in absent_is_not_stopped.stdout
    assert "did not remain stopped/exited" in absent_is_not_stopped.stderr


def test_stopped_exact_label_instance_is_started_with_a_deadline() -> None:
    controller = _source(CONTROLLER_PATH)
    attach = _block(controller, 'if [ "$ATTACH_EXISTING" = 1 ]; then', 'else\n  if [ "$SSH_OVER_HTTPS" = 1 ]; then')

    assert "stopped|exited" in attach
    assert 'run_bounded "$API_TIMEOUT_SECONDS" python3 cloud/vast_control_api.py start "$INSTANCE_ID" "$RUN_TAG"' in attach
    assert "Started exact-label" not in attach
    assert "could not be started; preserving it" in attach


def test_batch_attach_is_stopped_on_setup_failure_until_runner_is_proven() -> None:
    controller = _source(CONTROLLER_PATH)
    attach = _block(controller, 'if [ "$ATTACH_EXISTING" = 1 ]; then', 'else\n  if [ "$SSH_OVER_HTTPS" = 1 ]; then')
    started = attach.index('cloud/vast_control_api.py start "$INSTANCE_ID"')
    unowned = attach.index("REMOTE_STARTED=0", started)
    assert started < unowned

    existing_runner = _block(
        controller,
        '    if [ -n "$RUN_IDENTITY" ]; then',
        '    elif [ "$PID_PROBE" = absent ]',
    )
    assert existing_runner.index('case "$PID_PROBE" in') < existing_runner.rindex(
        "REMOTE_STARTED=1"
    )

    held_lease = _block(controller, "      held)", "      teardown)")
    assert held_lease.index('case "$PID_PROBE" in') < held_lease.rindex(
        "REMOTE_STARTED=1"
    )

    handoff = _block(
        controller,
        '    if [ "$START_LEASE_ACQUIRED" -eq 1 ]; then',
        "      HANDOFF_STATUS=$?",
    )
    assert handoff.index("REMOTE_STARTED=1") < handoff.index(
        "vast_api_remote_run.sh"
    )

    cleanup = _block(controller, "cleanup_before_handoff()", "handle_controller_signal()")
    batch_cleanup = _block(cleanup, 'if [ "$BATCH_MODE" -eq 1 ]; then', "    else")
    assert "stop_instance_preserving_disk" in batch_cleanup
    assert "destroy_instance" not in batch_cleanup


def test_onstart_fallback_waits_for_controller_and_only_stops() -> None:
    controller = _source(CONTROLLER_PATH)
    onstart = next(line for line in controller.splitlines() if line.startswith("ONSTART="))

    assert "controller_bootstrap.env" in onstart
    assert "controller_heartbeat.env" in onstart
    assert "CONTROLLER_TIMESTAMP" in onstart
    assert 'timeout --kill-after=30 120 \\"\\$VAST_BIN\\" stop instance' in onstart
    assert "destroy instance" not in onstart
    assert "curl --connect-timeout 15 --max-time 240" in onstart
    onstart_path = "export PATH=/root/.local/bin:\\$PATH"
    assert onstart_path in onstart
    assert onstart.index(onstart_path) < onstart.index("command -v vastai")
    assert onstart.index("watchdog_boot_started_at") < onstart.index(
        "vast_api_watchdog.sh"
    )
    assert "$WATCHDOG_BOOT_GRACE_SECONDS $WATCHDOG_EMERGENCY_PHASE_MAX_SECONDS" in onstart

    bootstrap = controller.index(".controller_bootstrap.env")
    repository_sync = controller.index(
        "rsync -az --delete --exclude=.git --exclude-from=.vastignore"
    )
    assert bootstrap < repository_sync


def test_noninteractive_watchdog_probe_checks_cli_without_account_read() -> None:
    controller = _source(CONTROLLER_PATH)
    probe = _block(
        controller,
        "# Check only that the non-interactive Vast CLI",
        "GPU_LINE=",
    )
    path_export = "export PATH=/root/.local/bin:$PATH"
    first_lookup = probe.index("command -v vastai")
    install = probe.index("https://vast.ai/install.sh")
    install_log = probe.index(">/tmp/vast-install-controller.log 2>&1")
    install_failure = probe.index("|| exit", install_log)
    second_lookup = probe.index("command -v vastai", first_lookup + 1)
    readiness_probe = probe.index('"$VAST_BIN" --help')

    assert path_export in probe
    assert (
        probe.index(path_export)
        < first_lookup
        < install
        < install_log
        < install_failure
        < second_lookup
        < readiness_probe
    )
    assert "set -o pipefail; curl" in probe
    assert '[ -n "$VAST_BIN" ] || exit 127' in probe
    assert "show instances-v1" not in probe
    assert "show user" not in probe
    assert "credential access was not tested" in probe


def test_failure_statuses_require_bound_concrete_evidence() -> None:
    controller = _source(CONTROLLER_PATH)
    runner = _source(RUNNER_PATH)

    assert 'FAILURE_LOG="logs/failure_${RUN_TAG}.log"' in runner
    assert "write_failure_log pre_upload_failure" in runner
    assert "write_failure_log terminal_status" in runner
    for field in ("TRAIN_STATUS", "POST_STATUS", "UPLOAD_STATUS"):
        assert field in runner

    capture = _block(
        controller,
        "capture_verified_failure_snapshot()",
        "validate_emergency_authorization()",
    )
    emergency = _block(
        controller,
        "validate_emergency_failure_snapshot()",
        "recover_hf_emergency_snapshot()",
    )
    for validator in (capture, emergency):
        assert 'failure_${RUN_TAG}.log' in validator
        assert 'controller_${RUN_TAG}.log' in validator
        assert 'for field in TRAIN_STATUS POST_STATUS UPLOAD_STATUS' in validator


def test_emergency_recovery_is_fresh_idempotent_and_precedes_ssh() -> None:
    controller = _source(CONTROLLER_PATH)
    recovery = _block(controller, "recover_hf_emergency_snapshot()", "recover_failure_and_destroy()")
    direct = recovery[recovery.index("direct_recovery()") :]

    assert 'mktemp -d "$recovery/.emergency_recovery_${advertised_sha}.XXXXXX"' in recovery
    assert 'destination="$attempt_root/recovered"' in recovery
    assert 'snapshot_sha256 | select(test("^[0-9a-f]{64}$"))' in recovery
    assert direct.index('hf download "$HF_REPO"') < direct.index(
        'recover_hf_emergency_snapshot "$recovery"'
    )
    assert direct.index('recover_hf_emergency_snapshot "$recovery"') < direct.index(
        "list_instances"
    )
    assert direct.index("list_instances") < direct.index('${SSH[@]}')
    assert "became absent; continuing with Hugging Face recovery only" in direct


def test_local_api_and_transfer_commands_use_deadlines() -> None:
    controller = _source(CONTROLLER_PATH)
    for line in controller.splitlines():
        if re.search(r"\bvastai (?:show|search|create|destroy|start)\b", line):
            assert "run_bounded" in line
        if re.search(r"\b(?:uvx --from|rsync -az)\b", line):
            assert "run_bounded" in line
