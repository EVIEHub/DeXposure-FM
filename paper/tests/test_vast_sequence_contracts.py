from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_PATH = ROOT / "cloud" / "vast_api_v2_sequence.sh"


def _function_source(name: str) -> str:
    lines = SEQUENCE_PATH.read_text().splitlines()
    start = lines.index(f"{name}() {{")
    end = next(index for index in range(start + 1, len(lines)) if lines[index] == "}")
    return "\n".join(lines[start : end + 1])


def _run_bash(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", source],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_sequence_shell_syntax_and_horizon_order():
    result = subprocess.run(
        ["bash", "-n", str(SEQUENCE_PATH)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "for horizon in 1 4 8 12; do" in SEQUENCE_PATH.read_text()


def test_sequence_lock_cleanup_is_owner_bound_and_signal_safe():
    source = SEQUENCE_PATH.read_text()
    assert 'LOCK_OWNER="$$:$(date +%s):$RANDOM"' in source
    assert 'mv "$LOCK_DIR" "$stale_lock"' in source
    assert 'rm -rf "$LOCK_DIR"' not in source
    cleanup = source[source.index("cleanup_sequence_lock() {") :]
    assert cleanup.index('cat "$LOCK_DIR/owner"') < cleanup.index(
        'rm -f "$LOCK_DIR/pid"'
    )
    signal_handler = source[
        source.index("handle_sequence_signal() {") : source.index(
            "trap cleanup_sequence_lock EXIT"
        )
    ]
    assert signal_handler.index("trap - EXIT") < signal_handler.index(
        "cleanup_sequence_lock"
    )
    assert source.index('stat -c %Y "$LOCK_DIR"') < source.index(
        'stat -f %m "$LOCK_DIR"'
    )


def test_sequence_cleanup_does_not_remove_another_owner(tmp_path):
    lock = tmp_path / "sequence.lock"
    lock.mkdir()
    (lock / "pid").write_text("321\n")
    (lock / "start").write_text("different start\n")
    (lock / "owner").write_text("new-owner\n")
    cleanup = _function_source("cleanup_sequence_lock")
    harness = f"""
set -euo pipefail
LOCK_DIR={shlex.quote(str(lock))}
LOCK_OWNER=old-owner
{cleanup}
cleanup_sequence_lock
[ -d "$LOCK_DIR" ]
[ "$(cat "$LOCK_DIR/owner")" = new-owner ]
"""
    result = _run_bash(harness)
    assert result.returncode == 0, result.stderr


def test_exact_instance_invoice_charge_is_selected_without_account_delta():
    settle = _function_source("settle_vast_cost")
    payload = shlex.quote(
        json.dumps(
            {
                "success": True,
                "results": [
                    {
                        "source": "instance-999",
                        "metadata": {"label": "dexposure-v2-run-a"},
                        "amount": 90,
                    },
                    {
                        "source": "instance-123",
                        "metadata": {"label": "dexposure-v2-run-a"},
                        "amount": 0.75,
                    },
                ],
            }
        )
    )
    harness = f"""
set -euo pipefail
BILLING_SETTLE_ATTEMPTS=1
BILLING_SETTLE_INTERVAL_SECONDS=1
API_TIMEOUT_SECONDS=1
SETTLED_COST=
SETTLED_CREDIT_AFTER=
run_bounded() {{ shift; "$@"; }}
vastai() {{
  invoice_command="show invoices-v1 --charges --charge-type instance"
  invoice_command="$invoice_command --latest-first --limit 100 --raw"
  case "$*" in
    "$invoice_command") printf '%s\\n' {payload} ;;
    "show user --raw") printf '%s\\n' '{{"credit":9.25}}' ;;
    *) return 88 ;;
  esac
}}
{settle}
settle_vast_cost 123 run-a
[ "$SETTLED_COST" = 0.75 ]
[ "$SETTLED_CREDIT_AFTER" = 9.25 ]
"""
    result = _run_bash(harness)
    assert result.returncode == 0, result.stderr
    assert "credit_before" not in settle
    assert 'expected_source="instance-$instance_id"' in settle
    assert 'expected_label="dexposure-v2-$run_tag"' in settle


def test_invoice_charge_with_wrong_run_label_fails_closed():
    settle = _function_source("settle_vast_cost")
    payload = shlex.quote(
        json.dumps(
            {
                "success": True,
                "results": [
                    {
                        "source": "instance-123",
                        "metadata": {"label": "another-run"},
                        "amount": 1.25,
                    }
                ],
            }
        )
    )
    harness = f"""
set -euo pipefail
BILLING_SETTLE_ATTEMPTS=1
BILLING_SETTLE_INTERVAL_SECONDS=1
API_TIMEOUT_SECONDS=1
SETTLED_COST=
SETTLED_CREDIT_AFTER=
run_bounded() {{ shift; "$@"; }}
vastai() {{
  invoice_command="show invoices-v1 --charges --charge-type instance"
  invoice_command="$invoice_command --latest-first --limit 100 --raw"
  case "$*" in
    "$invoice_command") printf '%s\\n' {payload} ;;
    *) return 88 ;;
  esac
}}
{settle}
if settle_vast_cost 123 run-a; then
  exit 90
fi
"""
    result = _run_bash(harness)
    assert result.returncode == 0, result.stderr
    assert "different instance label" in result.stderr


def test_final_controller_receipt_requires_exact_fields_and_destroyed_state(tmp_path):
    receipt = tmp_path / "receipt.env"
    valid_receipt = (
        "VERIFIED=1\n"
        "DESTROYED=1\n"
        "INSTANCE_ID=123\n"
        "RUN_TAG=sequence_h1_run\n"
        "HORIZON=1\n"
        f"SOURCE_SHA256={'a' * 64}\n"
        f"MODEL_SHA256={'b' * 64}\n"
        "CREDIT_BEFORE=10.0\n"
        "VAST_COST=pending\n"
        "CREDIT_AFTER=pending\n"
    )
    receipt.write_text(valid_receipt)
    functions = "\n".join(
        (
            _function_source("read_receipt_field"),
            _function_source("validate_verified_receipt"),
        )
    )
    quoted_receipt = shlex.quote(str(receipt))
    harness = f"""
set -euo pipefail
{functions}
validate_verified_receipt {quoted_receipt} sequence_h1_run 1 {'a' * 64} 10.0 123 1
[ "$verified_model_sha" = {'b' * 64} ]
"""
    result = _run_bash(harness)
    assert result.returncode == 0, result.stderr

    receipt.write_text(valid_receipt.replace("DESTROYED=1", "DESTROYED=0"))
    result = _run_bash(harness)
    assert result.returncode != 0

    receipt.write_text(valid_receipt + "UNKNOWN_FIELD=1\n")
    result = _run_bash(harness)
    assert result.returncode != 0


def test_pending_attempt_needs_three_stable_vast_and_hf_samples():
    source = SEQUENCE_PATH.read_text()
    assert 'PENDING_RECONCILE_SAMPLES="${VAST_PENDING_RECONCILE_SAMPLES:-3}"' in source
    assert '[ "$PENDING_RECONCILE_SAMPLES" -ge 3 ]' in source
    loop = source[
        source.index('for pending_sample in $(seq 1 "$PENDING_RECONCILE_SAMPLES")') :
        source.index('controller_log="$SEQUENCE_ROOT/controller_logs/${run_tag}.log"')
    ]
    assert 'pending_live_json="$(read_vast_instances)"' in loop
    assert 'hf_prefix_status="$(probe_hf_run_prefix "$run_tag")"' in loop
    assert 'pending_clear_samples=$((pending_clear_samples + 1))' in loop
    assert '[ "$pending_clear_samples" = "$PENDING_RECONCILE_SAMPLES" ]' in loop
    assert 'sleep "$PENDING_RECONCILE_INTERVAL_SECONDS"' in loop


def test_controller_success_is_revalidated_before_billing_can_advance():
    source = SEQUENCE_PATH.read_text()
    final_receipt = source.index(
        "controller receipt failed exact final-field validation"
    )
    local_bundle = source.index(
        "controller returned success without a valid local downloaded bundle"
    )
    exact_absence = source.index(
        "controller returned success, but instance "
        "$verified_instance_id is still present"
    )
    settle = source.index(
        'settle_vast_cost "$verified_instance_id" "$run_tag"', exact_absence
    )
    assert final_receipt < local_bundle < exact_absence < settle


def test_sequence_network_and_resume_destroy_calls_are_bounded():
    source = SEQUENCE_PATH.read_text()
    for line in source.splitlines():
        if re.search(r"\bvastai (?:show|destroy|create|start|stop|search)\b", line):
            assert "run_bounded" in line
        if "uvx --from 'huggingface_hub[cli]'" in line:
            assert "run_bounded" in line
    probe = _function_source("probe_hf_run_prefix")
    assert 'run_bounded "$API_TIMEOUT_SECONDS" curl' in probe
    assert '--connect-timeout 15 --max-time "$API_TIMEOUT_SECONDS"' in probe
    assert '--force-download' in source[source.index('ROUNDTRIP_ROOT=') :]
