from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cloud" / "train_src"))

from dexposure_fm.network_statistics import (  # noqa: E402
    compute_all_network_statistics,
    degree_assortativity,
    sector_connectivity_matrix,
)


def test_empty_graph_assortativity_is_undefined():
    edge_index = np.empty((2, 0), dtype=np.int64)
    assert np.isnan(degree_assortativity(edge_index, num_nodes=2))


def test_network_statistics_rejects_out_of_range_edges():
    with pytest.raises(ValueError, match="out-of-range"):
        compute_all_network_statistics(
            np.array([[0], [2]], dtype=np.int64), num_nodes=2
        )


def test_sector_matrix_rejects_missing_weights():
    with pytest.raises(ValueError, match="edge_weights length"):
        sector_connectivity_matrix(
            np.array([[0], [1]], dtype=np.int64),
            np.array([], dtype=float),
            ["A", "B"],
            ["A", "B"],
        )


def test_scenario_events_do_not_snap_to_an_unrelated_split():
    source = (ROOT / "cloud" / "train_src" / "run_full_experiment.py").read_text()
    assert "find_nearest_date" not in source
    assert "if event.event_date not in label_map" in source


def test_per_horizon_encoder_reset_requires_an_exact_state_match():
    source = (ROOT / "cloud" / "train_src" / "run_full_experiment.py").read_text()
    assert "encoder.load_state_dict(base_encoder_state, strict=True)" in source


def test_cloud_manifest_binds_primary_to_main_2025_h4():
    import json

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    primary = manifest["bindings"][manifest["primary"]["binding"]]
    assert primary["command"] == "bash cloud/main2025_run.sh"
    assert primary["horizons"] == [4]
    assert primary["holdout_start"] == "2025-01-01"
    assert primary["val_weeks"] == 24
    assert primary["epochs"] == 20
    assert primary["seed"] == 42
    assert primary.get("requires_pi_min") is not True


def test_cloud_manifest_binds_v2_reconstruction_to_all_horizons():
    import json

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    binding = manifest["bindings"]["main2025_v2_all_horizons"]
    assert binding["command"] == "bash cloud/main2025_v2_run.sh"
    assert binding["horizons"] == [1, 4, 8, 12]
    assert binding["holdout_start"] == "2025-01-01"
    assert binding["val_weeks"] == 24
    assert binding["epochs"] == 20
    assert binding["seed"] == 42
    outputs = {item["path"] for item in manifest["outputs"]["main2025_v2_all_horizons"]}
    for horizon in (1, 4, 8, 12):
        assert f"checkpoints/main2025_v2/dexposure-fm-h{horizon}.pt" in outputs
        assert f"checkpoints/main2025_v2_train/finetuned/best_model_h{horizon}.pt" in outputs
    for artifact in (
        "output/task2_model_based_v2/exp1_forward_risk.json",
        "output/task2_model_based_v2/exp2_predictive_contagion.json",
        "output/task2_model_based_v2/exp3_early_warning.json",
        "output/task2_model_based_v2/figures/fig_contagion_advantage.pdf",
        "output/task2_model_based_v2/figures/fig_spillover_matrix_example.pdf",
    ):
        assert artifact in outputs

    script = (ROOT / binding["script"]).read_text()
    assert "--mode all" in script
    assert "archive/code/run_task2_model_based.py" in script
    assert "make_spillover_matrix_figure.py" in script
    assert "early v2 Task I checkpoint push" in script


def test_cloud_manifest_binds_v2_h12_checkpoint_reconstruction():
    import json

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    binding = manifest["bindings"]["main2025_v2_h12"]
    assert binding["command"] == "bash cloud/main2025_v2_h12_run.sh"
    assert binding["horizons"] == [12]
    assert binding["holdout_start"] == "2025-01-01"
    assert binding["val_weeks"] == 24
    assert binding["epochs"] == 20
    assert binding["seed"] == 42

    outputs = {item["path"] for item in manifest["outputs"]["main2025_v2_h12"]}
    assert "checkpoints/main2025_v2_h12/dexposure-fm-h12.pt" in outputs
    assert "checkpoints/main2025_v2_h12_train/finetuned/best_model_h12.pt" in outputs
    assert "checkpoints/main2025_v2_h12/SHA256SUMS" in outputs

    script = (ROOT / binding["script"]).read_text()
    assert "TRAIN_HORIZONS=12" in script
    assert "--mode dexposure-fm" in script
    assert "--horizons \"$TRAIN_HORIZONS\"" in script
    assert "*Blackwell*" in script
    assert "GPU_MEMORY_MIB\" -gt 49152" in script


def test_cloud_manifest_binds_v2_four_checkpoint_reconstruction():
    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    binding = manifest["bindings"]["main2025_v2_checkpoints"]
    assert binding["command"] == "bash cloud/main2025_v2_checkpoints_run.sh"
    assert binding["horizons"] == [1, 4, 8, 12]

    script = (ROOT / binding["script"]).read_text()
    assert "TRAIN_HORIZONS=1,4,8,12" in script
    assert "--mode dexposure-fm" in script
    assert 'GPU_SELECTOR="${CUDA_VISIBLE_DEVICES:-0}"' in script
    assert "*Blackwell*" in script
    assert "GPU_MEMORY_MIB\" -gt 49152" in script
    for horizon in (1, 4, 8, 12):
        assert f'dexposure-fm-h${{h}}.pt' in script
        assert any(
            item["path"] == f"checkpoints/main2025_v2_rerun/dexposure-fm-h{horizon}.pt"
            for item in manifest["outputs"]["main2025_v2_checkpoints"]
        )


def test_cloud_manifest_binds_v2_remaining_checkpoints_without_h1():
    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    name = "main2025_v2_remaining_checkpoints"
    binding = manifest["bindings"][name]

    assert binding["command"] == "bash cloud/main2025_v2_remaining_run.sh"
    assert binding["script"] == "cloud/main2025_v2_remaining_run.sh"
    assert binding["horizons"] == [4, 8, 12]
    assert binding["holdout_start"] == "2025-01-01"
    assert binding["val_weeks"] == 24
    assert binding["epochs"] == 20
    assert binding["seed"] == 42

    fetch = binding["input_fetch"]
    assert fetch["command"] == "bash cloud/fetch_v2_inputs_from_hf.sh"
    assert fetch["script"] == "cloud/fetch_v2_inputs_from_hf.sh"
    assert fetch["repo"] == "losdwind/graph-dexposure-ckpt"
    assert fetch["revision"] == "cd8fcf4264054b58109e8a50c35cd69f6b2ff72a"
    assert [item["name"] for item in fetch["files"]] == [
        "network_data",
        "spillover_data",
        "metadata",
        "base_checkpoint",
    ]
    assert all(
        item["remote_path"].startswith("v2_inputs/main2025/")
        for item in fetch["files"]
    )

    release_root = "checkpoints/main2025_v2_hremaining"
    train_root = "checkpoints/main2025_v2_hremaining_train/finetuned"
    output_paths = {item["path"] for item in manifest["outputs"][name]}
    for horizon in (4, 8, 12):
        assert f"{release_root}/dexposure-fm-h{horizon}.pt" in output_paths
        assert f"{release_root}/run_config_h{horizon}.json" in output_paths
        assert f"{train_root}/best_model_h{horizon}.pt" in output_paths
    assert not any("dexposure-fm-h1.pt" in path for path in output_paths)
    assert f"{release_root}/feature_schema.json" in output_paths
    assert f"{release_root}/task1_metrics.json" in output_paths
    assert f"{release_root}/SHA256SUMS" in output_paths
    assert "logs/input_transport_status.env" not in output_paths
    assert "logs/input_manifest.sha256" not in output_paths

    json_contract = binding["json_contract"]
    assert set(json_contract["required_files"]) <= output_paths
    assert json_contract["minimum_count"] == len(json_contract["required_files"])
    assert json_contract["minimum_count"] == 10

    script = (ROOT / binding["script"]).read_text()
    assert script.count("cloud/train_src/run_full_experiment.py") == 1
    assert "TRAIN_HORIZONS=4,8,12" in script
    assert 'TRAIN_ROOT="checkpoints/main2025_v2_hremaining_train"' in script
    assert 'RELEASE_ROOT="checkpoints/main2025_v2_hremaining"' in script
    assert "for h in $HORIZON_LIST; do" in script
    assert "best_model_h1.pt" not in script
    assert 'if [ "$TRAIN_HORIZONS" = 1,4,8,12 ]; then' in script
    assert "shasum -a 256 -c SHA256SUMS" in script

    fetch_script = (ROOT / fetch["script"]).read_text()
    assert fetch["revision"] in fetch_script
    assert "resume_download=True" in fetch_script
    assert "os.replace(temporary, target)" in fetch_script
    assert "HF_V2_INPUT_MAX_ATTEMPTS" in fetch_script
    assert 'ROOT / "logs" / "input_transport_status.env"' in fetch_script
    assert 'ROOT / "logs" / "input_manifest.sha256"' in fetch_script
    assert '"STATUS=verified"' in fetch_script
    assert 'f"FILES={len(INPUTS)}"' in fetch_script
    for item in fetch["files"]:
        assert item["remote_path"] in fetch_script
        assert item["host_path"] in fetch_script
        assert str(item["size_bytes"]) in fetch_script.replace("_", "")
        assert item["sha256"] in fetch_script


def test_cloud_preflight_rejects_h1_in_v2_remaining_binding():
    sys.path.insert(0, str(ROOT / "cloud"))
    import preflight

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    name, binding = preflight._validate_binding(
        manifest, "bash cloud/main2025_v2_remaining_run.sh"
    )
    assert name == "main2025_v2_remaining_checkpoints"
    assert binding["horizons"] == [4, 8, 12]

    bad_manifest = json.loads(json.dumps(manifest))
    bad_manifest["bindings"][name]["horizons"] = [1, 4, 8, 12]
    with pytest.raises(preflight.PreflightError, match=r"declare \[4,8,12\]"):
        preflight._validate_binding(
            bad_manifest, "bash cloud/main2025_v2_remaining_run.sh"
        )


def test_cloud_preflight_verifies_remaining_input_transport_receipts(tmp_path):
    sys.path.insert(0, str(ROOT / "cloud"))
    import preflight

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    binding = manifest["bindings"]["main2025_v2_remaining_checkpoints"]
    fetch = binding["input_fetch"]
    logs = tmp_path / "logs"
    logs.mkdir()
    status = logs / "input_transport_status.env"
    status.write_text(
        "STATUS=verified\n"
        f"REPO={fetch['repo']}\n"
        f"REVISION={fetch['revision']}\n"
        "FILES=4\n",
        encoding="utf-8",
    )
    manifest_path = logs / "input_manifest.sha256"
    manifest_path.write_text(
        "".join(
            f"{item['sha256']}  {item['host_path']}\n" for item in fetch["files"]
        ),
        encoding="utf-8",
    )
    preflight._verify_remaining_input_transport(tmp_path, binding)

    status.write_text("STATUS=unverified\n", encoding="utf-8")
    with pytest.raises(preflight.PreflightError, match="transport status is invalid"):
        preflight._verify_remaining_input_transport(tmp_path, binding)


def test_cloud_manifest_binds_safe_single_horizon_v2_sequence():
    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    script = (ROOT / "cloud" / "main2025_v2_horizon_run.sh").read_text()
    sequence = (ROOT / "cloud" / "vast_api_v2_sequence.sh").read_text()
    for horizon in (1, 4, 8, 12):
        name = f"main2025_v2_h{horizon}_checkpoint"
        binding = manifest["bindings"][name]
        assert binding["command"] == f"bash cloud/main2025_v2_horizon_run.sh {horizon}"
        assert binding["horizons"] == [horizon]
        output_paths = {item["path"] for item in manifest["outputs"][name]}
        assert f"checkpoints/main2025_v2_h{horizon}/dexposure-fm-h{horizon}.pt" in output_paths
        assert f"checkpoints/main2025_v2_h{horizon}/SHA256SUMS" in output_paths
        assert f'cloud/main2025_v2_horizon_run.sh "$horizon"' in sequence
    assert 'case "$HORIZON" in' in script
    assert "1|4|8|12" in script
    assert "--holdout-start 2025-01-01" in script
    assert "EPOCHS=20" in script
    assert "PATIENCE=3" in script
    assert '--val-weeks 24 --epochs "$EPOCHS" --patience "$PATIENCE" --seed 42' in script
    assert 'shasum -a 256 -c SHA256SUMS' in script


def test_v2_sequence_persists_validated_attempt_identity_for_safe_resume():
    sequence = (ROOT / "cloud" / "vast_api_v2_sequence.sh").read_text()

    assert '"$SEQUENCE_ROOT/attempts"' in sequence
    assert 'local temp="${path}.tmp.$$"' in sequence
    assert 'mv "$temp" "$path"' in sequence
    assert 'validate_attempt "$attempt_file" "$horizon"' in sequence
    assert 'write_attempt "$attempt_file" pending' in sequence
    assert 'run_tag="$attempt_run_tag"' in sequence
    assert 'attach_instance_id="$attempt_instance_id"' in sequence
    assert 'VAST_ATTACH_EXISTING="$attach_existing"' in sequence
    assert 'VAST_ATTACH_INSTANCE_ID="$attach_instance_id"' in sequence
    assert 'VAST_ATTEMPT_FILE="$attempt_file"' in sequence
    assert 'write_attempt "$attempt_file" complete' in sequence
    assert 'pending_label="dexposure-v2-$run_tag"' in sequence
    assert 'if [ "$pending_match_count" = 1 ]' in sequence
    assert 'elif [ "$pending_match_count" = 0 ]' in sequence
    assert 'hf_prefix_status="$(probe_hf_run_prefix "$run_tag")"' in sequence
    assert "has a Hugging Face prefix but no recorded numeric instance ID" in sequence
    assert 'if [ "$attach_existing" = 0 ] && [ -n "$PREVIOUS_VERIFIED_COST" ]' in sequence
    assert "attempt has a run tag outside this sequence" in sequence
    assert "attempt belongs to a different source digest" in sequence
    assert "attempt has an invalid credit value" in sequence
    assert "attempt has no numeric instance ID" in sequence
    assert sequence.index('write_attempt "$attempt_file" pending') < sequence.index(
        "bash cloud/train.sh bash cloud/main2025_v2_horizon_run.sh"
    )
    assert "vastai show instances-v1 --all --raw" in sequence
    assert '.success == true and (.instances | type == "array")' in sequence
    assert "vastai show instances --raw" not in sequence
    assert 'artifact_manifest_${run_tag}.sha256' in sequence
    assert 'cp "$train_root/metrics.json"' in sequence
    assert 'validate_downloaded_run "$verified_run_root" "$run_tag" "$horizon"' in sequence
    assert sequence.index(
        'validate_downloaded_run "$verified_run_root" "$run_tag" "$horizon"'
    ) < sequence.index('destroy_receipted_instance "$verified_instance_id"')


def test_cloud_source_digest_covers_executable_imports_and_sync_policy():
    import json

    sys.path.insert(0, str(ROOT / "cloud"))
    import preflight

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    source = manifest["source"]
    assert ".vastignore" in source["paths"]
    assert "archive/code/**/*.py" in source["paths"]
    assert "DeXposure_FM_V2/scripts/make_spillover_matrix_figure.py" in source["paths"]
    assert "paper/experiments/**/*.py" in source["paths"]
    assert "paper/scripts/**/*.py" in source["paths"]
    expanded = {
        path.relative_to(ROOT).as_posix()
        for path in preflight._expand_source_paths(ROOT, source["paths"])
    }
    for relative in (
        "paper/experiments/exp_logger.py",
        "paper/experiments/methods.py",
        "paper/experiments/exceptions.py",
        "paper/scripts/build_paired_frontier.py",
        "archive/code/run_task2_model_based.py",
        "DeXposure_FM_V2/scripts/make_spillover_matrix_figure.py",
        "cloud/vast_api_watchdog.sh",
        "cloud/main2025_v2_remaining_run.sh",
        "cloud/fetch_v2_inputs_from_hf.sh",
        ".vastignore",
    ):
        assert relative in expanded

    vastignore = (ROOT / ".vastignore").read_text().splitlines()
    for generated in ("logs/", "output/", "outputs/", "paper/results/", "results/"):
        assert generated in vastignore

    for path in (ROOT / "cloud").glob("*"):
        if path.suffix not in {".sh", ".yaml"}:
            continue
        assert "uv sync --frozen || uv sync" not in path.read_text()


def test_fm_service_bindings_require_bounded_pi_min(monkeypatch):
    import json

    sys.path.insert(0, str(ROOT / "cloud"))
    import preflight

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    for name in ("pre2022_crisis_h4", "pre2022_rq1", "llm_pilot"):
        assert manifest["bindings"][name]["requires_pi_min"] is True
    monkeypatch.setenv("PREFLIGHT_PI_MIN", "0.25")
    _, binding = preflight._validate_binding(
        manifest, "bash cloud/pre2022_run.sh"
    )
    assert binding["requires_pi_min"] is True
    monkeypatch.setenv("PREFLIGHT_PI_MIN", "nan")
    with pytest.raises(preflight.PreflightError, match="finite float"):
        preflight._validate_binding(manifest, "bash cloud/pre2022_run.sh")


def test_cloud_preflight_rejects_direct_all_horizon_command():
    import json

    sys.path.insert(0, str(ROOT / "cloud"))
    import preflight

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    with pytest.raises(preflight.PreflightError, match="allowlisted"):
        preflight._binding_for_command(
            manifest,
            "python cloud/train_src/run_full_experiment.py --horizons 1,4,8,12",
        )


def test_cloud_preflight_disallows_independent_m7_sampling(monkeypatch):
    import json

    sys.path.insert(0, str(ROOT / "cloud"))
    import preflight

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    bad = dict(manifest["bindings"]["llm_pilot"])
    bad["command"] = "bash cloud/llm_pilot_run.sh --method m7_fm_llm_gated"
    bad["enabled"] = True
    manifest["bindings"]["bad_m7"] = bad
    monkeypatch.setenv("PREFLIGHT_PI_MIN", "0.5")
    with pytest.raises(preflight.PreflightError, match="independent M7"):
        preflight._validate_binding(
            manifest,
            "bash cloud/llm_pilot_run.sh --method m7_fm_llm_gated",
        )


def test_cloud_scripts_use_vast_only_and_destroy_after_local_verification():
    train = (ROOT / "cloud" / "train.sh").read_text()
    assert "cloud/preflight.py" in train
    assert "exec bash cloud/vast_api_train.sh" in train
    assert "sky launch" not in train
    assert "CLOUD_BACKEND" not in train
    assert "train.yaml" not in train
    assert not (ROOT / "cloud" / "train.yaml").exists()
    vast_local = (ROOT / "cloud" / "vast_api_train.sh").read_text()
    vast_remote = (ROOT / "cloud" / "vast_api_remote_run.sh").read_text()
    vast_watchdog = (ROOT / "cloud" / "vast_api_watchdog.sh").read_text()
    assert "vastai destroy instance" in vast_local
    assert "vastai destroy instance" not in vast_remote
    assert "vastai destroy instance" not in vast_watchdog
    assert "VAST_WATCHDOG_SECONDS:-3600" in vast_local
    assert "VAST_WATCHDOG_REMOTE_GRACE_SECONDS:-300" in vast_local
    assert "VAST_WATCHDOG_CONTROLLER_STALE_SECONDS:-600" in vast_local
    assert "TRAIN_TIMEOUT_SECONDS + WATCHDOG_FINALIZATION_MARGIN_SECONDS" in vast_local
    assert "VAST_WATCHDOG_BOOT_GRACE_SECONDS:-1200" in vast_local
    assert "VAST_WATCHDOG_EMERGENCY_PHASE_MAX_SECONDS:-10800" in vast_local
    assert "emergency action deferred" in vast_watchdog
    assert "while true" in vast_watchdog
    assert "WATCHDOG_DECISION=runner_stalled" in vast_watchdog
    assert "WATCHDOG_DECISION=remote_snapshot_stop_ready" not in vast_watchdog
    assert "remote_snapshot_stop_ready" in vast_watchdog
    assert "runner_progress_deadline" in vast_watchdog
    assert "runner_progress_fresh" in vast_watchdog
    assert "controller_alive" in vast_watchdog
    assert "HF_ROUNDTRIP_VERIFIED" in vast_watchdog
    assert "never_started_clean" in vast_watchdog
    assert "vastai show instances-v1 --all --raw" not in vast_watchdog
    assert "vastai show user" not in vast_watchdog
    assert "local_recovery_stop_ready" in vast_watchdog
    assert 'PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"' in (
        ROOT / "cloud" / "main2025_v2_horizon_run.sh"
    ).read_text()
    assert "gpu_ram>=40 gpu_ram<=48" in vast_local
    assert "setsid nohup bash cloud/vast_api_remote_run.sh" in vast_local
    assert "cloud/vast_api_watchdog.sh" in vast_local
    assert "runner_start.lock" in vast_local
    assert "Another controller owns the remote runner start lease" in vast_local
    assert vast_local.index("cloud/vast_api_watchdog.sh cloud/vast_api_emergency_snapshot.sh") < vast_local.index(
        'rsync -az --delete --exclude=.git --exclude-from=.vastignore'
    )
    assert "--exclude-from=.vastignore" in vast_local
    vast_ignore = (ROOT / ".vastignore").read_text().splitlines()
    assert ".env" in vast_ignore
    assert "*.pem" in vast_ignore
    assert "*.key" in vast_ignore
    assert ".private == true" in vast_local
    assert 'CURRENT_STATUS="api-error"' in vast_local
    assert "this check does not count as a dead runner" in vast_local
    assert "will not destroy an active training process" in vast_local
    assert "complete_bundle" in vast_local
    assert "capture_verified_failure_snapshot" in vast_local
    assert "failure_evidence_${RUN_TAG}.sha256" in vast_local
    assert 'find "${roots[@]}" -type f' in vast_local
    assert 'cmp -s "$snapshot/$manifest_name.before" "$snapshot/$manifest_name.after"' in vast_local
    assert 'cmp -s "$expected_paths" "$local_paths"' in vast_local
    assert 'shasum -a 256 -c "$manifest_name.after"' in vast_local
    assert "-name 'train_*.log'" not in vast_local
    assert "VAST_SUCCESS_RECEIPT" in vast_local
    assert "DESTROYED=%s" in vast_local
    assert "MODEL_SHA256" in vast_local
    assert "for _ in $(seq 1 30)" in vast_local
    assert "handle_controller_signal" in vast_local
    assert "Remote handoff response was ambiguous" in vast_local
    assert "exit 130" in vast_local
    assert "run_status.env" in vast_local
    assert "run_status.env" in vast_remote
    assert 'PIPELINE_STATUS=("${PIPESTATUS[@]}")' in vast_remote
    assert "remote_runner.pid" in vast_remote
    assert "vast_api_emergency_snapshot.sh" in vast_remote
    assert "Emergency round trip failed" in vast_remote
    assert "artifact_snapshot.tar" in (
        ROOT / "cloud" / "vast_api_emergency_snapshot.sh"
    ).read_text()
    assert "RUNNER_STARTTIME" in vast_remote
    assert "running\\n" in vast_remote
    assert "exited\\n" in vast_remote
    assert "handle_remote_signal" in vast_remote
    assert vast_remote.index("trap finalize_remote_exit EXIT") < vast_remote.index(
        "uv sync --frozen"
    )
    assert 'hf download "$HF_REPO"' in vast_local
    assert "shasum -a 256 -c SHA256SUMS" in vast_local
    assert "REMOTE_PID_PROBE_COMMAND" in vast_local
    assert 'current_start=$(awk' in vast_local
    assert '[ "$state" != Z ]' in vast_local
    assert "ownership is ambiguous, so all are preserved" in vast_local
    assert "cloud/verify_v2_checkpoint.py" in vast_local
    assert 'validate_complete_bundle "$RUN_ROOT"' in vast_local
    assert 'artifact_manifest_${RUN_TAG}.sha256' in vast_local
    assert 'artifact_manifest_${RUN_TAG}.sha256' in vast_remote
    assert 'postflight_${RUN_TAG}.log' in vast_remote
    assert "in_progress_best_model_h${HORIZON}_train" not in vast_remote
    assert "**/*.partial.pt" in vast_remote
    download_call = vast_local.rindex("hf download")
    bundle_check = vast_local.rindex('validate_complete_bundle "$RUN_ROOT"')
    receipt_call = vast_local.rindex("write_success_receipt 0")
    destroy_call = vast_local.rindex('destroy_instance "$COMPLETED_INSTANCE_ID"')
    final_receipt_call = vast_local.rindex("write_success_receipt 1")
    assert download_call < bundle_check < receipt_call < destroy_call
    assert destroy_call < final_receipt_call

    sequence = (ROOT / "cloud" / "vast_api_v2_sequence.sh").read_text()
    assert "VAST_SUCCESS_RECEIPT" in sequence
    assert "MODEL_SHA256" in sequence
    assert "handle_sequence_signal" in sequence
    assert 'shasum -a 256 -c SHA256SUMS' in sequence
    assert "aggregation source differs from its verified model SHA-256" in sequence
    assert "settle_vast_cost" in sequence
    assert "Vast billing has not published a positive cost" in sequence


def test_training_persists_partial_best_model_and_skips_unused_gpu_history():
    source = (ROOT / "cloud" / "train_src" / "run_full_experiment.py").read_text()
    assert '"artifact_status": "in_progress"' in source
    assert '"artifact_status": "complete"' in source
    assert "os.replace(partial_tmp, partial_path)" in source
    assert "track_temporal_embeddings = config.smooth_loss_weight > 0.0" in source


@pytest.mark.parametrize(
    "entrypoint",
    ("bootstrap_and_run.sh", "launch_verified_gpu.sh"),
)
def test_legacy_gpu_entrypoints_are_retired_fail_closed_shims(entrypoint):
    source = (ROOT / "cloud" / entrypoint).read_text()
    assert "cloud/train.sh" in source
    assert "retired" in source
    assert "sky launch" not in source
    assert "sky exec" not in source
    assert "train_nosync.yaml" not in source


def test_training_bindings_require_only_raw_data_and_base_checkpoint():
    import json

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    assert "pre2022_smoke_2week" not in manifest["bindings"]
    assert "pre2022_smoke_2week" not in manifest["outputs"]
    assert set(manifest["inputs"]) == {"local"}
    network_data = manifest["inputs"]["local"][0]
    assert network_data["path"] == "data/historical-network_week_2020-03-30.json"
    assert network_data["sha256"] == "aa330bbb8fbf99719fc85d49625d7df7bd68f2b042f5806ded080cec99bad3f8"
    for binding in manifest["bindings"].values():
        assert binding["required_inputs"] == ["local"]
    assert not (ROOT / "cloud" / "train.yaml").exists()
    assert not (ROOT / "cloud" / "train_nosync.yaml").exists()


def test_rq1_binding_requires_explicit_opt_in_with_source_attestation(monkeypatch, tmp_path):
    import hashlib
    import json

    sys.path.insert(0, str(ROOT / "cloud"))
    import preflight

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    digest = preflight.source_digest(ROOT, manifest)
    attestation = tmp_path / "source-digest.txt"
    attestation.write_text(digest + "\n")
    monkeypatch.setenv("PREFLIGHT_RQ1", "1")
    monkeypatch.setenv("PREFLIGHT_PI_MIN", "0.5")
    monkeypatch.setenv("PREFLIGHT_SOURCE_ATTESTATION", str(attestation))
    monkeypatch.setenv(
        "PREFLIGHT_SOURCE_ATTESTATION_SHA256",
        hashlib.sha256(attestation.read_bytes()).hexdigest(),
    )
    preflight._validate_source_state(ROOT, manifest, "local")
    name, binding = preflight._validate_binding(
        manifest, "bash cloud/pre2022_run.sh --rq1-all-horizons"
    )
    assert name == "pre2022_rq1"
    assert binding["horizons"] == [1, 4, 8, 12]

    attestation.write_text("0" * 64 + "\n")
    monkeypatch.setenv(
        "PREFLIGHT_SOURCE_ATTESTATION_SHA256",
        hashlib.sha256(attestation.read_bytes()).hexdigest(),
    )
    with pytest.raises(preflight.PreflightError, match="does not bind"):
        preflight._validate_source_state(ROOT, manifest, "local")


def test_legacy_llm_pilot_is_blocked_before_cloud_launch():
    import json

    sys.path.insert(0, str(ROOT / "cloud"))
    import preflight

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    assert manifest["bindings"]["llm_pilot"]["enabled"] is False
    with pytest.raises(preflight.PreflightError, match="disabled"):
        preflight._validate_binding(manifest, "bash cloud/llm_pilot_run.sh")


def test_cloud_output_contracts_name_exact_json_artifacts_and_minimum_counts():
    import json

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    crisis_json = {
        "logs/crisis_backtest/b1_crisis.json",
        "logs/crisis_backtest/b5_crisis.json",
        "logs/crisis_backtest/prompts_m6_fm_llm_2022-04__2022-07.json",
        "logs/crisis_backtest/prompts_m6_fm_llm_2022-10__2023-01.json",
        "logs/crisis_backtest/prompts_m6_fm_llm_2023-01__2023-05.json",
    }
    training_json = {
        "checkpoints/pre2022_train/finetuned/feature_schema.json",
        "checkpoints/pre2022_train/finetuned/experiment_results.json",
        "checkpoints/pre2022_train/finetuned/data_quality.json",
        "checkpoints/pre2022_train/finetuned/metrics.json",
        "checkpoints/pre2022_train/finetuned/all_results.json",
    }
    for name in ("pre2022_crisis_h4", "pre2022_rq1"):
        output_paths = {item["path"] for item in manifest["outputs"][name]}
        contract = manifest["bindings"][name]["json_contract"]
        assert set(contract["required_files"]) == crisis_json | training_json
        assert crisis_json | training_json <= output_paths
        assert contract["minimum_count"] >= len(crisis_json | training_json)

    main = manifest["bindings"]["main2025_h4"]["json_contract"]
    main_paths = {item["path"] for item in manifest["outputs"]["main2025_h4"]}
    assert set(main["required_files"]) <= main_paths
    assert "checkpoints/main2025_train/finetuned/best_model_h4.pt" in main_paths
    assert "checkpoints/main2025/dexposure-fm-h4.pt" in main_paths
    assert main["minimum_count"] >= len(main["required_files"])

    main_script = (ROOT / "cloud" / "main2025_run.sh").read_text()
    crisis_script = (ROOT / "cloud" / "pre2022_run.sh").read_text()
    assert 'src="checkpoints/main2025_train/finetuned/best_model_h4.pt"' in main_script
    assert 'src="checkpoints/pre2022_train/finetuned/best_model_h${h}.pt"' in crisis_script
    assert "find checkpoints/main2025_train" not in main_script
    assert "find checkpoints/pre2022_train" not in crisis_script
    assert "rm -rf checkpoints/pre2022_train" not in crisis_script
    assert "rm -f checkpoints/graphpfn-v1.ckpt" not in crisis_script
    assert "paper/results/llm_eval_*" not in crisis_script


def test_crisis_output_contract_requires_complete_aligned_horizon_grid():
    sys.path.insert(0, str(ROOT / "cloud"))
    import preflight

    horizons = [1, 4, 8, 12]
    rows = [
        {
            "crisis": crisis,
            "method": method,
            "prediction_horizon": horizon,
            "label_horizon": horizon,
        }
        for crisis in ("terra_luna", "ftx", "svb")
        for method in ("m5_fm_rules", "m1_persistence_rules")
        for horizon in horizons
    ]
    preflight._verify_crisis_horizon_rows(rows, horizons)

    mismatched = [dict(row) for row in rows]
    mismatched[0]["label_horizon"] = 4
    with pytest.raises(preflight.PreflightError, match="horizon mismatch"):
        preflight._verify_crisis_horizon_rows(mismatched, horizons)

    with pytest.raises(preflight.PreflightError, match="grid is incomplete"):
        preflight._verify_crisis_horizon_rows(rows[:-1], horizons)


def test_cloud_json_verifier_fails_when_required_artifact_is_absent(tmp_path, monkeypatch):
    import json

    sys.path.insert(0, str(ROOT / "cloud"))
    import preflight

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    monkeypatch.setattr(
        preflight,
        "_import_verifiers",
        lambda *_args: (lambda *_values: "hash", lambda *_values: None, lambda value: (value, 0)),
    )
    with pytest.raises(preflight.PreflightError, match="required JSON artifact is missing"):
        preflight._verify_json_artifacts(tmp_path, "pre2022_crisis_h4", manifest)


def test_cloud_json_verifier_binds_raw_outputs_and_frozen_evidence_to_candidates(tmp_path):
    sys.path.insert(0, str(ROOT / "cloud"))
    sys.path.insert(0, str(ROOT / "paper"))
    import preflight
    from experiments.b5_decision_quality import CandidateSet, evidence_hash

    candidates = CandidateSet.from_rows(
        [
            {
                "protocol": f"P{index}",
                "exposure": 10 - index,
                "scenario_max_loss": index,
                "scenario_sum_loss": index,
            }
            for index in range(10)
        ]
    )
    system_prompt = "system"
    user_prompt = "user"
    bundle = {
        "evidence_sha256": evidence_hash(system_prompt, user_prompt, candidates),
        "candidate_set_sha256": candidates.candidate_set_sha256,
        "candidate_membership_sha256": candidates.candidate_membership_sha256,
        "candidate_set": candidates.as_dicts(),
    }
    output = {
        "risk_level": "elevated",
        "rationale": "Seven ranked candidates.",
        "target_protocols": [
            {
                "protocol": f"P{index}",
                "risk_score": 1.0 - index / 10,
                "action": "Investigate",
                "reason": "Visible candidate evidence.",
            }
            for index in range(7)
        ],
    }
    logs = tmp_path / "logs"
    logs.mkdir()
    raw_path = logs / "raw_m6_fm_llm.json"
    raw_path.write_text(
        json.dumps(
            [
                {
                    "method": "m6_fm_llm",
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "evidence_bundle": bundle,
                    "raw_llm_outputs": [output],
                }
            ]
        )
    )
    frozen_path = logs / "frozen_evidence.json"
    frozen_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "evidence": {
                            **bundle,
                            "system_prompt": system_prompt,
                            "user_prompt": user_prompt,
                        }
                    }
                ]
            }
        )
    )
    manifest = {
        "bindings": {
            "test": {
                "json_contract": {
                    "required_files": [
                        "logs/raw_m6_fm_llm.json",
                        "logs/frozen_evidence.json",
                    ],
                    "minimum_count": 2,
                }
            }
        },
        "verifiers": {
            "evidence": "module:evidence_hash",
            "candidate": "module:validate_decision_output",
            "gate": "module:apply_action_gate",
        },
    }
    summary = preflight._verify_json_artifacts(tmp_path, "test", manifest)
    assert summary["candidates"] == 1
    assert summary["gates"] == 1

    tampered = json.loads(raw_path.read_text())
    tampered[0]["evidence_bundle"]["candidate_membership_sha256"] = "0" * 64
    raw_path.write_text(json.dumps(tampered))
    with pytest.raises(preflight.PreflightError, match="candidate-membership hash mismatch"):
        preflight._verify_json_artifacts(tmp_path, "test", manifest)
