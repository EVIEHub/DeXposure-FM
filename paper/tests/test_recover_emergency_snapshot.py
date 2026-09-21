from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RECOVER = ROOT / "cloud" / "recover_emergency_snapshot.py"


def add_file(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def write_archive(path: Path, members: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, content in members:
            add_file(archive, name, content)


def write_checksum(archive: Path, checksum: Path, digest: str | None = None) -> str:
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum.write_text(f"{digest or actual}  {archive.name}\n", encoding="ascii")
    return actual


def run_recovery(archive: Path, checksum: Path, destination: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(RECOVER),
            "--archive",
            str(archive),
            "--checksum",
            str(checksum),
            "--destination",
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_valid_archive_is_verified_and_atomically_recovered(tmp_path: Path) -> None:
    archive = tmp_path / "artifact_snapshot.tar"
    checksum = tmp_path / "artifact_snapshot.sha256"
    destination = tmp_path / "recovered"
    destination.mkdir()
    write_archive(
        archive,
        [
            ("logs/train.log", b"epoch=20\n"),
            ("checkpoints/main2025_v2_h1/model.pt", b"model weights"),
        ],
    )
    snapshot_sha256 = write_checksum(archive, checksum)

    result = run_recovery(archive, checksum, destination)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "verified"
    assert payload["snapshot_sha256"] == snapshot_sha256
    assert payload["destination"] == str(destination)
    assert payload["files"] == [
        {
            "path": "checkpoints/main2025_v2_h1/model.pt",
            "sha256": hashlib.sha256(b"model weights").hexdigest(),
            "size": len(b"model weights"),
        },
        {
            "path": "logs/train.log",
            "sha256": hashlib.sha256(b"epoch=20\n").hexdigest(),
            "size": len(b"epoch=20\n"),
        },
    ]
    assert (destination / "logs" / "train.log").read_bytes() == b"epoch=20\n"
    assert (
        destination / "checkpoints" / "main2025_v2_h1" / "model.pt"
    ).read_bytes() == b"model weights"
    assert not list(tmp_path.glob(".recovered.recover-*"))


def test_traversal_member_is_rejected_without_destination(tmp_path: Path) -> None:
    archive = tmp_path / "artifact_snapshot.tar"
    checksum = tmp_path / "artifact_snapshot.sha256"
    destination = tmp_path / "recovered"
    write_archive(archive, [("logs/../../escaped", b"no")])
    write_checksum(archive, checksum)

    result = run_recovery(archive, checksum, destination)

    assert result.returncode != 0
    assert json.loads(result.stderr)["status"] == "error"
    assert not destination.exists()
    assert not (tmp_path / "escaped").exists()
    assert not list(tmp_path.glob(".recovered.recover-*"))


def test_symlink_member_is_rejected_without_destination(tmp_path: Path) -> None:
    archive = tmp_path / "artifact_snapshot.tar"
    checksum = tmp_path / "artifact_snapshot.sha256"
    destination = tmp_path / "recovered"
    with tarfile.open(archive, "w") as output:
        member = tarfile.TarInfo("logs/latest")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../escaped"
        output.addfile(member)
    write_checksum(archive, checksum)

    result = run_recovery(archive, checksum, destination)

    assert result.returncode != 0
    assert "not a regular file or directory" in json.loads(result.stderr)["error"]
    assert not destination.exists()


def test_bad_checksum_is_rejected_without_destination(tmp_path: Path) -> None:
    archive = tmp_path / "artifact_snapshot.tar"
    checksum = tmp_path / "artifact_snapshot.sha256"
    destination = tmp_path / "recovered"
    write_archive(archive, [("logs/train.log", b"complete")])
    write_checksum(archive, checksum, "0" * 64)

    result = run_recovery(archive, checksum, destination)

    assert result.returncode != 0
    assert "SHA-256 mismatch" in json.loads(result.stderr)["error"]
    assert not destination.exists()
    assert not list(tmp_path.glob(".recovered.recover-*"))


@pytest.mark.parametrize(
    "member_type",
    [tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.FIFOTYPE],
)
def test_other_special_members_are_rejected(
    tmp_path: Path, member_type: bytes
) -> None:
    archive = tmp_path / "artifact_snapshot.tar"
    checksum = tmp_path / "artifact_snapshot.sha256"
    destination = tmp_path / "recovered"
    with tarfile.open(archive, "w") as output:
        member = tarfile.TarInfo("logs/special")
        member.type = member_type
        if member_type == tarfile.LNKTYPE:
            member.linkname = "logs/train.log"
        output.addfile(member)
    write_checksum(archive, checksum)

    result = run_recovery(archive, checksum, destination)

    assert result.returncode != 0
    assert not destination.exists()


def test_duplicate_member_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "artifact_snapshot.tar"
    checksum = tmp_path / "artifact_snapshot.sha256"
    destination = tmp_path / "recovered"
    with tarfile.open(archive, "w") as output:
        add_file(output, "logs/train.log", b"first")
        add_file(output, "logs/train.log", b"second")
    write_checksum(archive, checksum)

    result = run_recovery(archive, checksum, destination)

    assert result.returncode != 0
    assert "duplicate archive member" in json.loads(result.stderr)["error"]
    assert not destination.exists()


def test_wrong_top_level_directory_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "artifact_snapshot.tar"
    checksum = tmp_path / "artifact_snapshot.sha256"
    destination = tmp_path / "recovered"
    write_archive(archive, [("outputs/model.pt", b"wrong root")])
    write_checksum(archive, checksum)

    result = run_recovery(archive, checksum, destination)

    assert result.returncode != 0
    assert "outside logs/ and checkpoints/" in json.loads(result.stderr)["error"]
    assert not destination.exists()
