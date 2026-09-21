from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
WATCHDOG = ROOT / "cloud" / "vast_api_watchdog.sh"
SOURCE_SHA = "a" * 64


def watchdog_command(*, remote_grace_seconds: int = 0) -> list[str]:
    return [
        "bash",
        str(WATCHDOG),
        "0",
        "1",
        str(remote_grace_seconds),
        "2",
        "2",
        "123",
        "test-run",
        "1",
        SOURCE_SHA,
    ]


def watchdog_environment(
    workdir: Path, lock: Path, *, decision_only: bool = True
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "VAST_WATCHDOG_WORKDIR": str(workdir),
            "VAST_WATCHDOG_LOCK": str(lock),
            "VAST_WATCHDOG_TEST_DECISION": "1" if decision_only else "0",
            "VAST_WATCHDOG_SELF_STARTTIME": "1",
        }
    )
    return environment


def run_watchdog(
    workdir: Path,
    lock: Path,
    *,
    decision_only: bool = True,
    remote_grace_seconds: int = 0,
    extra_environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = watchdog_environment(workdir, lock, decision_only=decision_only)
    environment.update(extra_environment or {})
    return subprocess.run(
        watchdog_command(remote_grace_seconds=remote_grace_seconds),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
    )


def write_bound_file(path: Path, extra: str) -> None:
    path.write_text(
        "INSTANCE_ID=123\n"
        "RUN_TAG=test-run\n"
        "HORIZON=1\n"
        f"SOURCE_SHA256={SOURCE_SHA}\n"
        f"{extra}",
        encoding="ascii",
    )


def test_current_controller_heartbeat_blocks_teardown(tmp_path: Path) -> None:
    logs = tmp_path / "work" / "logs"
    logs.mkdir(parents=True)
    write_bound_file(
        logs / "controller_heartbeat.env",
        f"CONTROLLER_TIMESTAMP={int(time.time())}\n",
    )

    result = run_watchdog(tmp_path / "work", tmp_path / "watchdog.lock")

    assert result.returncode == 0, result.stderr
    assert "WATCHDOG_DECISION=controller_alive" in result.stdout


def test_boot_grace_precedes_controller_and_teardown_decisions(tmp_path: Path) -> None:
    logs = tmp_path / "work" / "logs"
    logs.mkdir(parents=True)
    (logs / "watchdog_boot_started_at").write_text(
        f"{int(time.time())}\n", encoding="ascii"
    )
    write_bound_file(
        logs / "watchdog_destroy_authorized.env",
        "LOCAL_RECOVERY_VERIFIED=1\n"
        "HF_ROUNDTRIP_VERIFIED=0\n"
        "EVIDENCE_LEVEL=failure_inventory\n"
        "MODEL_SHA256=none\n"
        "SNAPSHOT_SHA256=none\n",
    )

    result = run_watchdog(tmp_path / "work", tmp_path / "watchdog.lock")

    assert result.returncode == 0, result.stderr
    assert "WATCHDOG_DECISION=boot_grace_active" in result.stdout


def test_recent_malformed_start_lease_is_preserved(tmp_path: Path) -> None:
    lease = tmp_path / "work" / "logs" / "runner_start.lock"
    lease.mkdir(parents=True)
    (lease / "created_at").write_text("broken\n", encoding="ascii")

    result = run_watchdog(tmp_path / "work", tmp_path / "watchdog.lock")

    assert result.returncode == 0, result.stderr
    assert "WATCHDOG_DECISION=start_lease_active" in result.stdout
    assert lease.is_dir()


def test_old_malformed_start_lease_expires_by_directory_mtime(tmp_path: Path) -> None:
    lease = tmp_path / "work" / "logs" / "runner_start.lock"
    lease.mkdir(parents=True)
    (lease / "created_at").write_text("broken\n", encoding="ascii")
    old = time.time() - 1900
    os.utime(lease, (old, old))

    result = run_watchdog(tmp_path / "work", tmp_path / "watchdog.lock")

    assert result.returncode == 0, result.stderr
    assert "WATCHDOG_DECISION=runner_never_started_clean" in result.stdout
    assert not lease.exists()


def test_stale_watchdog_lock_is_reclaimed(tmp_path: Path) -> None:
    lock = tmp_path / "watchdog.lock"
    lock.mkdir()
    (lock / "owner").write_text("999999 1\n", encoding="ascii")

    result = run_watchdog(tmp_path / "work", lock)

    assert result.returncode == 0, result.stderr
    assert "WATCHDOG_DECISION=runner_never_started_clean" in result.stdout
    assert not lock.exists()


def test_alive_runner_with_current_progress_is_preserved(tmp_path: Path) -> None:
    if not Path("/proc/self/stat").exists():
        pytest.skip("watchdog runner PID binding is Linux-specific")
    logs = tmp_path / "work" / "logs"
    logs.mkdir(parents=True)
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        stat_fields = Path(f"/proc/{sleeper.pid}/stat").read_text().split()
        (logs / "remote_runner.pid").write_text(
            f"{sleeper.pid} {stat_fields[21]} running\n", encoding="ascii"
        )
        write_bound_file(
            logs / "runner_progress.env",
            f"PHASE=training\nPROGRESS_TIMESTAMP={int(time.time())}\n",
        )

        result = run_watchdog(tmp_path / "work", tmp_path / "watchdog.lock")

        assert result.returncode == 0, result.stderr
        assert "WATCHDOG_DECISION=runner_progress_fresh" in result.stdout
    finally:
        sleeper.terminate()
        sleeper.wait(timeout=5)


def test_alive_runner_with_stale_progress_hits_billing_deadline(tmp_path: Path) -> None:
    if not Path("/proc/self/stat").exists():
        pytest.skip("watchdog runner PID binding is Linux-specific")
    logs = tmp_path / "work" / "logs"
    logs.mkdir(parents=True)
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        stat_fields = Path(f"/proc/{sleeper.pid}/stat").read_text().split()
        (logs / "remote_runner.pid").write_text(
            f"{sleeper.pid} {stat_fields[21]} running\n", encoding="ascii"
        )
        write_bound_file(
            logs / "runner_progress.env",
            f"PHASE=training\nPROGRESS_TIMESTAMP={int(time.time()) - 10}\n",
        )

        result = run_watchdog(tmp_path / "work", tmp_path / "watchdog.lock")

        assert result.returncode == 0, result.stderr
        assert "WATCHDOG_DECISION=runner_progress_deadline" in result.stdout
        assert sleeper.poll() is None, "watchdog must never signal the runner"
    finally:
        sleeper.terminate()
        sleeper.wait(timeout=5)


def test_emergency_retry_has_total_stop_deadline(tmp_path: Path) -> None:
    if not Path("/proc/self/stat").exists():
        pytest.skip("watchdog runner PID binding is Linux-specific")
    logs = tmp_path / "work" / "logs"
    logs.mkdir(parents=True)
    marker = Path("/tmp/dexposure-123-test-run-emergency.started")
    marker.unlink(missing_ok=True)
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        stat_fields = Path(f"/proc/{sleeper.pid}/stat").read_text().split()
        (logs / "remote_runner.pid").write_text(
            f"{sleeper.pid} {stat_fields[21]} running\n", encoding="ascii"
        )
        write_bound_file(
            logs / "runner_progress.env",
            f"PHASE=emergency_retry\nPROGRESS_TIMESTAMP={int(time.time())}\n",
        )
        marker.write_text(f"{int(time.time()) - 10}\n", encoding="ascii")

        result = run_watchdog(
            tmp_path / "work",
            tmp_path / "watchdog.lock",
            extra_environment={"VAST_WATCHDOG_EMERGENCY_PHASE_MAX_SECONDS": "1"},
        )

        assert result.returncode == 0, result.stderr
        assert "WATCHDOG_DECISION=emergency_retry_stalled" in result.stdout
        assert sleeper.poll() is None, "watchdog must stop the instance, not signal runner"
    finally:
        marker.unlink(missing_ok=True)
        sleeper.terminate()
        sleeper.wait(timeout=5)


def test_emergency_deadline_overrides_current_controller_heartbeat(tmp_path: Path) -> None:
    logs = tmp_path / "work" / "logs"
    logs.mkdir(parents=True)
    marker = Path("/tmp/dexposure-123-test-run-emergency.started")
    marker.unlink(missing_ok=True)
    try:
        write_bound_file(
            logs / "controller_heartbeat.env",
            f"CONTROLLER_TIMESTAMP={int(time.time())}\n",
        )
        write_bound_file(
            logs / "runner_progress.env",
            f"PHASE=emergency_retry\nPROGRESS_TIMESTAMP={int(time.time())}\n",
        )
        marker.write_text(f"{int(time.time()) - 10}\n", encoding="ascii")

        result = run_watchdog(
            tmp_path / "work",
            tmp_path / "watchdog.lock",
            extra_environment={"VAST_WATCHDOG_EMERGENCY_PHASE_MAX_SECONDS": "1"},
        )

        assert result.returncode == 0, result.stderr
        assert "WATCHDOG_DECISION=emergency_retry_stalled" in result.stdout
        assert "WATCHDOG_DECISION=controller_alive" not in result.stdout
    finally:
        marker.unlink(missing_ok=True)


def test_runner_progress_deadline_overrides_current_controller(tmp_path: Path) -> None:
    logs = tmp_path / "work" / "logs"
    logs.mkdir(parents=True)
    write_bound_file(
        logs / "controller_heartbeat.env",
        f"CONTROLLER_TIMESTAMP={int(time.time())}\n",
    )
    write_bound_file(
        logs / "runner_progress.env",
        f"PHASE=uploading_logs\nPROGRESS_TIMESTAMP={int(time.time()) - 10}\n",
    )

    result = run_watchdog(tmp_path / "work", tmp_path / "watchdog.lock")

    assert result.returncode == 0, result.stderr
    assert "WATCHDOG_DECISION=runner_progress_deadline" in result.stdout
    assert "WATCHDOG_DECISION=controller_alive" not in result.stdout


def test_local_recovery_authorization_stops_after_controller_death(tmp_path: Path) -> None:
    logs = tmp_path / "work" / "logs"
    logs.mkdir(parents=True)
    write_bound_file(
        logs / "watchdog_destroy_authorized.env",
        "LOCAL_RECOVERY_VERIFIED=1\n"
        "HF_ROUNDTRIP_VERIFIED=0\n"
        "EVIDENCE_LEVEL=failure_inventory\n"
        "MODEL_SHA256=none\n"
        "SNAPSHOT_SHA256=none\n",
    )

    result = run_watchdog(tmp_path / "work", tmp_path / "watchdog.lock")

    assert result.returncode == 0, result.stderr
    assert "WATCHDOG_DECISION=local_recovery_stop_ready" in result.stdout


def test_remote_authorization_in_grace_is_not_republished(tmp_path: Path) -> None:
    logs = tmp_path / "work" / "logs"
    logs.mkdir(parents=True)
    write_bound_file(logs / "run_identity.env", "")
    authorization = logs / "watchdog_destroy_authorized.env"
    write_bound_file(
        authorization,
        "LOCAL_RECOVERY_VERIFIED=0\n"
        "HF_ROUNDTRIP_VERIFIED=1\n"
        "EVIDENCE_LEVEL=remote_snapshot\n"
        "MODEL_SHA256=none\n"
        f"SNAPSHOT_SHA256={'b' * 64}\n",
    )
    authorization_mtime = authorization.stat().st_mtime_ns

    result = run_watchdog(
        tmp_path / "work",
        tmp_path / "watchdog.lock",
        remote_grace_seconds=60,
    )

    assert result.returncode == 0, result.stderr
    assert "WATCHDOG_DECISION=authorization_grace" in result.stdout
    assert "WATCHDOG_DECISION=emergency_snapshot_required" not in result.stdout
    assert authorization.stat().st_mtime_ns == authorization_mtime


def test_remote_only_authorization_stops_but_never_allows_destroy(tmp_path: Path) -> None:
    logs = tmp_path / "work" / "logs"
    logs.mkdir(parents=True)
    write_bound_file(
        logs / "controller_heartbeat.env",
        f"CONTROLLER_TIMESTAMP={int(time.time())}\n",
    )
    write_bound_file(
        logs / "watchdog_destroy_authorized.env",
        "LOCAL_RECOVERY_VERIFIED=0\n"
        "HF_ROUNDTRIP_VERIFIED=1\n"
        "EVIDENCE_LEVEL=remote_snapshot\n"
        "MODEL_SHA256=none\n"
        f"SNAPSHOT_SHA256={'b' * 64}\n",
    )

    result = run_watchdog(
        tmp_path / "work", tmp_path / "watchdog.lock", remote_grace_seconds=0
    )

    assert result.returncode == 0, result.stderr
    assert "WATCHDOG_DECISION=remote_snapshot_stop_ready" in result.stdout
    assert "WATCHDOG_DECISION=teardown_allowed" not in result.stdout


def test_remote_watchdog_is_stop_only_and_never_reads_account_state() -> None:
    source = WATCHDOG.read_text(encoding="utf-8")
    local_recovery = source[
        source.index("local_recovery_stop_ready() {") : source.index(
            "remote_snapshot_stop_ready() {"
        )
    ]
    assert "LOCAL_RECOVERY_VERIFIED" in local_recovery
    assert "vastai stop instance \"$INSTANCE_ID\"" in source
    assert "vastai destroy instance" not in source
    assert "show instances-v1" not in source
    assert "show user" not in source


def test_term_exits_and_releases_watchdog_lock(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    logs = workdir / "logs"
    lock = tmp_path / "watchdog.lock"
    fake_bin = tmp_path / "bin"
    fake_calls = tmp_path / "unexpected-vast-calls.log"
    fake_bin.mkdir()
    (fake_bin / "vastai").write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >>\"$FAKE_VAST_LOG\"\nexit 1\n",
        encoding="ascii",
    )
    os.chmod(fake_bin / "vastai", 0o755)
    logs.mkdir(parents=True)
    write_bound_file(
        logs / "controller_heartbeat.env",
        f"CONTROLLER_TIMESTAMP={int(time.time())}\n",
    )
    environment = watchdog_environment(workdir, lock, decision_only=False)
    environment.update(
        {
            "HOME": str(tmp_path / "fake-home"),
            "PATH": f"{fake_bin}:/usr/bin:/bin:/usr/sbin:/sbin",
            "FAKE_VAST_LOG": str(fake_calls),
        }
    )
    process = subprocess.Popen(
        watchdog_command(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        deadline = time.time() + 5
        while time.time() < deadline and not (lock / "owner").is_file():
            time.sleep(0.02)
        assert (lock / "owner").is_file()
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 130, (stdout, stderr)
        assert not lock.exists()
        assert not fake_calls.exists(), "current controller must block every Vast mutation"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_unreliable_start_uses_stop_not_destroy(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "vast-calls.log"
    (fake_bin / "vastai").write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >>\"$FAKE_VAST_LOG\"\n"
        "printf 'stopping instance %s.\\n' \"$3\"\n",
        encoding="ascii",
    )
    (fake_bin / "timeout").write_text(
        "#!/usr/bin/env bash\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  case \"$1\" in --signal=*|--kill-after=*) shift ;; *) shift; break ;; esac\n"
        "done\n"
        "exec \"$@\"\n",
        encoding="ascii",
    )
    os.chmod(fake_bin / "vastai", 0o755)
    os.chmod(fake_bin / "timeout", 0o755)

    result = run_watchdog(
        tmp_path / "work",
        tmp_path / "watchdog.lock",
        decision_only=False,
        extra_environment={
            "HOME": str(tmp_path / "fake-home"),
            "PATH": f"{fake_bin}:/usr/bin:/bin:/usr/sbin:/sbin",
            "FAKE_VAST_LOG": str(calls),
        },
    )

    assert result.returncode == 0, result.stderr
    assert calls.read_text(encoding="ascii").splitlines() == ["stop instance 123"]
    assert "disk is retained" in result.stdout


def test_watchdog_has_no_direct_runner_termination() -> None:
    source = WATCHDOG.read_text(encoding="utf-8")
    assert "recover_stalled_runner" not in source
    assert 'kill -TERM -- "-$pgid"' not in source
    assert 'kill -KILL -- "-$pgid"' not in source
