#!/usr/bin/env python3
"""Verify and atomically recover a Vast emergency artifact snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO


ALLOWED_ROOTS = frozenset({"logs", "checkpoints"})
CHECKSUM_PATTERN = re.compile(rb"([0-9a-f]{64})  ([A-Za-z0-9._-]+)\n")
COPY_CHUNK_SIZE = 1024 * 1024


class RecoveryError(Exception):
    """An emergency snapshot failed a recovery contract check."""


def read_checksum(path: Path, archive_name: str) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RecoveryError(f"cannot read checksum file: {exc}") from exc
    match = CHECKSUM_PATTERN.fullmatch(raw)
    if match is None:
        raise RecoveryError(
            "checksum must be one lowercase SHA-256 line in '<hash>  <file>\\n' format"
        )
    checksum_name = match.group(2).decode("ascii")
    if checksum_name != archive_name:
        raise RecoveryError(
            f"checksum names {checksum_name!r}, expected {archive_name!r}"
        )
    return match.group(1).decode("ascii")


def open_regular_file(path: Path) -> BinaryIO:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RecoveryError(f"cannot open archive as a regular file: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RecoveryError("archive is not a regular file")
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


def copy_and_hash(source: BinaryIO, destination: Path) -> str:
    digest = hashlib.sha256()
    try:
        with destination.open("xb") as output:
            while chunk := source.read(COPY_CHUNK_SIZE):
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    except OSError as exc:
        raise RecoveryError(f"cannot stage archive: {exc}") from exc
    return digest.hexdigest()


def canonical_member_path(member: tarfile.TarInfo) -> PurePosixPath:
    name = member.name
    if not name or "\x00" in name or name.startswith("/"):
        raise RecoveryError(f"unsafe archive member path: {name!r}")
    if member.isdir() and name.endswith("/"):
        name = name[:-1]
    raw_parts = name.split("/")
    if not raw_parts or any(part in {"", ".", ".."} for part in raw_parts):
        raise RecoveryError(f"non-canonical archive member path: {member.name!r}")
    path = PurePosixPath(*raw_parts)
    if path.is_absolute() or ".." in path.parts:
        raise RecoveryError(f"unsafe archive member path: {member.name!r}")
    if path.parts[0] not in ALLOWED_ROOTS:
        raise RecoveryError(
            f"archive member is outside logs/ and checkpoints/: {member.name!r}"
        )
    if len(path.parts) == 1 and not member.isdir():
        raise RecoveryError(f"archive root must be a directory: {member.name!r}")
    return path


def validate_members(archive: tarfile.TarFile) -> list[tuple[tarfile.TarInfo, PurePosixPath]]:
    validated: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
    types: dict[PurePosixPath, str] = {}
    try:
        members = archive.getmembers()
    except (OSError, tarfile.TarError) as exc:
        raise RecoveryError(f"cannot enumerate archive: {exc}") from exc

    for member in members:
        path = canonical_member_path(member)
        if member.type in (tarfile.REGTYPE, tarfile.AREGTYPE):
            kind = "file"
        elif member.type == tarfile.DIRTYPE:
            kind = "directory"
        else:
            raise RecoveryError(
                f"archive member is not a regular file or directory: {member.name!r}"
            )
        if path in types:
            raise RecoveryError(f"duplicate archive member: {member.name!r}")
        types[path] = kind
        validated.append((member, path))

    if not any(kind == "file" for kind in types.values()):
        raise RecoveryError("archive contains no regular files")

    for path in types:
        for parent in path.parents:
            if str(parent) == ".":
                break
            if types.get(parent) == "file":
                raise RecoveryError(
                    f"archive file is also a parent directory: {str(parent)!r}"
                )
    return validated


def extract_members(
    archive: tarfile.TarFile,
    members: list[tuple[tarfile.TarInfo, PurePosixPath]],
    staging: Path,
) -> list[dict[str, object]]:
    directories = sorted(
        (path for member, path in members if member.type == tarfile.DIRTYPE),
        key=lambda path: (len(path.parts), str(path)),
    )
    try:
        for path in directories:
            (staging / Path(*path.parts)).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RecoveryError(f"cannot create staged directory: {exc}") from exc

    files: list[dict[str, object]] = []
    for member, path in members:
        if member.type == tarfile.DIRTYPE:
            continue
        target = staging / Path(*path.parts)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RecoveryError(f"cannot read archive member: {member.name!r}")
            digest = hashlib.sha256()
            size = 0
            with source, target.open("xb") as output:
                while chunk := source.read(COPY_CHUNK_SIZE):
                    digest.update(chunk)
                    size += len(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        except RecoveryError:
            raise
        except (OSError, tarfile.TarError) as exc:
            raise RecoveryError(f"cannot extract {member.name!r}: {exc}") from exc
        if size != member.size:
            raise RecoveryError(
                f"archive member size changed for {member.name!r}: {size} != {member.size}"
            )
        files.append(
            {"path": path.as_posix(), "sha256": digest.hexdigest(), "size": size}
        )
    return sorted(files, key=lambda item: str(item["path"]))


def checked_destination(raw_destination: Path) -> Path:
    if raw_destination.name in {"", ".", ".."}:
        raise RecoveryError("destination must name a directory below an existing parent")
    try:
        parent = raw_destination.expanduser().parent.resolve(strict=True)
    except OSError as exc:
        raise RecoveryError(f"destination parent is unavailable: {exc}") from exc
    if not parent.is_dir():
        raise RecoveryError("destination parent is not a directory")
    destination = parent / raw_destination.name
    try:
        metadata = destination.lstat()
    except FileNotFoundError:
        return destination
    except OSError as exc:
        raise RecoveryError(f"cannot inspect destination: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise RecoveryError("destination exists and is not a real directory")
    try:
        with os.scandir(destination) as entries:
            if next(entries, None) is not None:
                raise RecoveryError("destination directory is not empty")
    except OSError as exc:
        raise RecoveryError(f"cannot inspect destination contents: {exc}") from exc
    return destination


def recover(archive_path: Path, checksum_path: Path, raw_destination: Path) -> dict[str, object]:
    expected_sha256 = read_checksum(checksum_path, archive_path.name)
    destination = checked_destination(raw_destination)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.recover-", dir=destination.parent)
    )
    staged_archive = staging / ".verified-snapshot.tar"
    renamed = False
    try:
        with open_regular_file(archive_path) as source:
            actual_sha256 = copy_and_hash(source, staged_archive)
        if actual_sha256 != expected_sha256:
            raise RecoveryError(
                f"archive SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
            )
        try:
            with tarfile.open(staged_archive, mode="r:*") as archive:
                members = validate_members(archive)
                files = extract_members(archive, members, staging)
        except RecoveryError:
            raise
        except (OSError, tarfile.TarError) as exc:
            raise RecoveryError(f"cannot read archive: {exc}") from exc
        try:
            staged_archive.unlink()
            os.replace(staging, destination)
            renamed = True
        except OSError as exc:
            raise RecoveryError(f"cannot atomically publish destination: {exc}") from exc
    finally:
        if not renamed:
            shutil.rmtree(staging, ignore_errors=True)

    return {
        "status": "verified",
        "snapshot_sha256": actual_sha256,
        "destination": str(destination),
        "files": files,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checksum", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = recover(args.archive, args.checksum, args.destination)
    except RecoveryError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
