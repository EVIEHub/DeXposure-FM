from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "cloud" / "vast_api_train.sh"
RUNNER = ROOT / "cloud" / "vast_api_remote_run.sh"
EMERGENCY = ROOT / "cloud" / "vast_api_emergency_snapshot.sh"
WATCHDOG = ROOT / "cloud" / "vast_api_watchdog.sh"
MANIFEST = ROOT / "cloud" / "preflight_manifest.json"
SOURCE_SHA = "a" * 64


def source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def block(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    return text[begin : text.index(end, begin)]


def test_emergency_tar_sets_positional_options_before_the_file_list() -> None:
    emergency = source(EMERGENCY)
    assert (
        'tar --create --file="$archive" --no-recursion --null --files-from="$paths"'
        in emergency
    )


def test_remaining_binding_is_the_only_three_horizon_controller_exception() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    binding = manifest["bindings"]["main2025_v2_remaining_checkpoints"]
    controller = source(CONTROLLER)

    assert binding["command"] == "bash cloud/main2025_v2_remaining_run.sh"
    assert binding["horizons"] == [4, 8, 12]
    assert binding["input_fetch"]["command"] == "bash cloud/fetch_v2_inputs_from_hf.sh"
    assert 'if [ "$BINDING_NAME" = main2025_v2_remaining_checkpoints ]' in controller
    assert "jq -e '.value.horizons == [4, 8, 12]'" in controller
    assert 'HORIZON="$(jq -r \'.value.horizons[0]\'' in controller
    assert "HORIZON_COUNT\" = 1" in controller


def test_attach_relaunches_dead_runner_on_same_disk() -> None:
    controller = source(CONTROLLER)
    attach = block(
        controller,
        'echo "Attached to the existing runner pid=$REMOTE_PID state=alive."',
        'echo "The recorded instance has no runner; resuming setup on the same instance."',
    )
    assert "NEEDS_REMOTE_START=1" in attach
    assert "relaunching on the same disk" in attach
    assert "INITIAL_REMOTE_DEAD_COUNT=1" not in attach
    setup = block(controller, 'if [ "$NEEDS_REMOTE_START" -eq 1 ]; then', 'START_LEASE_RESULT=')
    assert "rm -rf /tmp/dexposure-vast-watchdog.lock" in setup


def test_remaining_syncs_code_once_and_skips_all_four_large_input_rsyncs() -> None:
    controller = source(CONTROLLER)
    setup = block(controller, 'if [ "$NEEDS_REMOTE_START" -eq 1 ]; then', 'START_LEASE_RESULT=')

    assert setup.count(
        "rsync -az --delete --exclude=.git --exclude-from=.vastignore"
    ) == 1
    batch_guard = setup.index('if [ -z "$INPUT_FETCH_CMD" ]; then')
    for path in (
        "data/historical-network_week_2020-03-30.json",
        "data/historical-network_week_2025-07-01.json",
        "cloud/upload/data/meta_df.csv",
        "cloud/upload/graphpfn-v1.ckpt",
    ):
        assert setup.index(path) > batch_guard

    watchdog_sync = setup.index("cloud/vast_api_watchdog.sh cloud/vast_api_emergency_snapshot.sh")
    watchdog_ready = setup.index('WATCHDOG_READY=1')
    code_sync = setup.index(
        "rsync -az --delete --exclude=.git --exclude-from=.vastignore"
    )
    assert watchdog_sync < watchdog_ready < code_sync < batch_guard


def test_runner_fetches_pinned_inputs_before_host_preflight_and_training() -> None:
    runner = source(RUNNER)
    input_download = runner.index("write_runner_progress input_download")
    fetch = runner.index('bash -c "$INPUT_FETCH_CMD"', input_download)
    host_preflight = runner.index("write_runner_progress host_preflight")
    training = runner.index('bash -c "$TRAIN_CMD"')

    assert input_download < fetch < host_preflight < training
    assert runner.count('bash -c "$TRAIN_CMD"') == 1
    assert 'INPUT_FETCH_CMD="${7:-}"' in runner
    assert 'INPUT_STATUS=$INPUT_STATUS' in runner
    assert 'MARKER_STATUS=$MARKER_STATUS' in runner


def test_runner_rejects_fake_batch_and_single_horizon_fetches() -> None:
    common = [SOURCE_SHA, "test-run", "123", "10"]
    fake_batch = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "bash cloud/main2025_v2_horizon_run.sh 4",
            common[0],
            common[1],
            "remaining",
            common[2],
            common[3],
            "bash cloud/fetch_v2_inputs_from_hf.sh",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    single_with_fetch = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "bash cloud/main2025_v2_horizon_run.sh 4",
            common[0],
            common[1],
            "4",
            common[2],
            common[3],
            "bash cloud/fetch_v2_inputs_from_hf.sh",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert fake_batch.returncode == 2
    assert single_with_fetch.returncode == 2


def test_batch_manifest_and_controller_cover_every_required_artifact() -> None:
    runner = source(RUNNER)
    controller = source(CONTROLLER)
    required = (
        "dexposure-fm-h4.pt",
        "dexposure-fm-h8.pt",
        "dexposure-fm-h12.pt",
        "best_model_h4.pt",
        "best_model_h8.pt",
        "best_model_h12.pt",
        "run_config_h4.json",
        "run_config_h8.json",
        "run_config_h12.json",
        "feature_schema.json",
        "task1_metrics.json",
        "SHA256SUMS",
        "input_transport_status.env",
        "input_manifest.sha256",
        "input_download_${RUN_TAG}.log",
        "train_${RUN_TAG}.log",
        "postflight_${RUN_TAG}.log",
        "controller_${RUN_TAG}.log",
        "upload_release_${RUN_TAG}.log",
        "upload_train_${RUN_TAG}.log",
    )
    artifact_builder = block(
        runner, 'ARTIFACT_MANIFEST="', "write_runner_progress uploading_logs"
    )
    validator = block(controller, "validate_complete_bundle()", "capture_verified_failure_snapshot()")

    for name in required:
        assert name in artifact_builder
        assert name in validator
    for horizon in (4, 8, 12):
        assert f'--horizon "$horizon"' in validator
    assert 'shasum -a 256 -c logs/input_manifest.sha256' in validator
    assert 'REVISION=%s\\nFILES=%s\\n' in validator


def test_final_uploads_retry_three_times_and_freeze_controller_before_manifest() -> None:
    runner = source(RUNNER)
    retry = block(runner, "upload_with_retries()", 'RELEASE_UPLOAD_LOG="')
    final_uploads = block(runner, 'RELEASE_UPLOAD_LOG="', "set -e")

    assert "for attempt in 1 2 3" in retry
    assert 'timeout --signal=TERM --kill-after=60 1800' in retry
    assert 'sanitize_upload_log "$upload_log"' in retry
    for operation in (
        'upload_with_retries "$RELEASE_REL"',
        'upload_with_retries "$TRAIN_REL"',
        "upload_with_retries logs logs",
        'upload_with_retries "$LOCAL_STATUS" logs/local_run_status.env',
        'upload_with_retries "$LOCAL_STATUS" logs/run_status.env',
    ):
        assert operation in final_uploads

    release_upload = final_uploads.index('upload_with_retries "$RELEASE_REL"')
    train_upload = final_uploads.index('upload_with_retries "$TRAIN_REL"')
    status_write = final_uploads.index("write_local_status", train_upload)
    freeze = final_uploads.index('exec >"/tmp/controller_after_freeze_${RUN_TAG}.log"')
    manifest = final_uploads.index('ARTIFACT_MANIFEST="')
    logs_upload = final_uploads.index("upload_with_retries logs logs")
    local_status_upload = final_uploads.index(
        'upload_with_retries "$LOCAL_STATUS" logs/local_run_status.env'
    )
    marker_upload = final_uploads.index(
        'upload_with_retries "$LOCAL_STATUS" logs/run_status.env'
    )
    assert release_upload < train_upload < status_write < freeze < manifest
    assert manifest < logs_upload < local_status_upload < marker_upload
    assert "Never create a completion marker in this branch" in final_uploads
    assert "complete_emergency_roundtrip terminal_status_before_marker" in final_uploads
    assert final_uploads.index("terminal_status_before_marker") < marker_upload
    assert final_uploads.index("marker_upload_failed", marker_upload) > marker_upload
    assert final_uploads.index("REMOTE_FINALIZED=1", marker_upload) > marker_upload
    assert "normal_exit" not in final_uploads


def test_batch_success_receipt_has_three_model_hashes_and_then_one_destroy() -> None:
    controller = source(CONTROLLER)
    receipt = block(controller, "write_success_receipt()", "# The remote watchdog")
    final_start = controller.rindex(
        'if [ "$BATCH_MODE" -eq 1 ]; then\n  if [ "$BATCH_HORIZONS"'
    )
    final = controller[final_start:]

    for horizon in (4, 8, 12):
        assert f"MODEL_SHA256_H{horizon}" in receipt
        assert f'dexposure-fm-h{horizon}.pt' in final
    assert "HORIZONS=4,8,12" in receipt
    assert final.count('destroy_instance "$COMPLETED_INSTANCE_ID"') == 1
    assert final.index("write_success_receipt 0") < final.index("destroy_instance")
    assert final.index("destroy_instance") < final.index("write_success_receipt 1")
    assert "publish_watchdog_recovery_evidence complete_bundle" not in final


def test_batch_failure_stops_and_never_uses_failure_destroy_branch() -> None:
    controller = source(CONTROLLER)
    recovery = block(controller, "recover_failure_and_destroy()", "finalize_recovered_success()")
    cleanup = block(controller, "cleanup_before_handoff()", "handle_controller_signal()")

    assert 'if [ "$BATCH_MODE" -eq 1 ]; then' in recovery
    assert 'stop_instance_preserving_disk "$INSTANCE_ID"' in recovery
    assert recovery.index('if [ "$BATCH_MODE" -eq 1 ]; then') < recovery.index(
        'if destroy_instance "$INSTANCE_ID"'
    )
    assert 'stop_instance_preserving_disk "$INSTANCE_ID"' in cleanup


def test_remaining_token_and_rtx_6000_ada_are_accepted_by_all_guards(tmp_path: Path) -> None:
    controller = source(CONTROLLER)
    assert '(.gpu_name == "RTX 6000Ada")' in controller
    assert "*RTX\\ 6000\\ Ada*" in controller
    assert '(.num_gpus == 1)' in controller
    assert '.reliability > 0.995' in controller
    assert '.static_ip == true' in controller
    assert '.inet_down >= 500' in controller
    assert '.inet_up >= 500' in controller
    assert 'VAST_MIN_DISK_BW:-1000' in controller
    assert 'VAST_MAX_DPH:-0.65' in controller
    assert '.disk_bw >= ($min_disk_bw | tonumber)' in controller
    assert '.dph_total <= ($max_dph | tonumber)' in controller
    assert 'VAST_MIN_DISK_BW and VAST_MAX_DPH must be positive numbers' in controller
    assert 'test "$GPU_COUNT" = 1' in controller
    assert "1|4|8|12|remaining" in source(EMERGENCY)
    assert "1|4|8|12|remaining" in source(WATCHDOG)

    environment = os.environ.copy()
    environment.update(
        {
            "VAST_WATCHDOG_WORKDIR": str(tmp_path / "work"),
            "VAST_WATCHDOG_LOCK": str(tmp_path / "watchdog.lock"),
            "VAST_WATCHDOG_TEST_DECISION": "1",
            "VAST_WATCHDOG_SELF_STARTTIME": "1",
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(WATCHDOG),
            "0",
            "1",
            "0",
            "2",
            "2",
            "123",
            "remaining-test",
            "remaining",
            SOURCE_SHA,
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "WATCHDOG_DECISION=runner_never_started_clean" in result.stdout


def test_all_batch_lifecycle_shells_parse() -> None:
    subprocess.run(
        ["bash", "-n", str(CONTROLLER), str(RUNNER), str(EMERGENCY), str(WATCHDOG)],
        cwd=ROOT,
        check=True,
    )


def test_four_horizon_repeat_requires_h1_artifacts_and_keeps_legacy_binding() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "cloud"))
    import preflight
    import pytest

    manifest = json.loads(source(MANIFEST))
    command = "bash cloud/main2025_v2_remaining_run.sh --all"
    name, binding = preflight._validate_binding(manifest, command)
    assert name == "main2025_v2_repeat20"
    assert binding["horizons"] == [1, 4, 8, 12]
    assert binding["epochs"] == 20 and binding["seed"] == 42
    assert manifest["bindings"]["main2025_v2_remaining_checkpoints"]["horizons"] == [4, 8, 12]
    manifest["outputs"][name] = [
        item for item in manifest["outputs"][name]
        if not item["path"].endswith("dexposure-fm-h1.pt")
    ]
    with pytest.raises(preflight.PreflightError, match="output paths"):
        preflight._validate_binding(manifest, command)
    controller = source(CONTROLLER)
    assert controller.count("for horizon in $BATCH_HORIZONS; do") == 3
    assert 'MODEL_SHA256_H1=%s' in controller
    assert '"$RELEASE_REL/dexposure-fm-h1.pt"' in source(RUNNER)
