#!/usr/bin/env bash
# Fetch the immutable DeXposure-FM v2 inputs inside the detached Vast runner.
set -euo pipefail

ROOT="$PWD"
PY="${V2_INPUT_PYTHON:-$ROOT/.venv/bin/python}"
HF_REPO="losdwind/graph-dexposure-ckpt"
DEFAULT_REVISION="cd8fcf4264054b58109e8a50c35cd69f6b2ff72a"
REVISION="${HF_V2_INPUT_REVISION:-$DEFAULT_REVISION}"
MAX_ATTEMPTS="${HF_V2_INPUT_MAX_ATTEMPTS:-5}"

test -x "$PY" || { echo "FATAL: Python environment is unavailable: $PY" >&2; exit 1; }
test -n "${HF_TOKEN:-}" || { echo "FATAL: HF_TOKEN is required for the private input repository" >&2; exit 1; }
case "$REVISION" in
  *[!0-9a-f]*|'') echo "FATAL: HF_V2_INPUT_REVISION must be a lowercase commit SHA" >&2; exit 1 ;;
esac
[ "${#REVISION}" -eq 40 ] || {
  echo "FATAL: HF_V2_INPUT_REVISION must contain exactly 40 hexadecimal characters" >&2
  exit 1
}
case "$MAX_ATTEMPTS" in
  *[!0-9]*|'') echo "FATAL: HF_V2_INPUT_MAX_ATTEMPTS must be a positive integer" >&2; exit 1 ;;
esac
[ "$MAX_ATTEMPTS" -ge 1 ] && [ "$MAX_ATTEMPTS" -le 10 ] || {
  echo "FATAL: HF_V2_INPUT_MAX_ATTEMPTS must be between 1 and 10" >&2
  exit 1
}

export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_HUB_DISABLE_PROGRESS_BARS=1

HF_REPO="$HF_REPO" REVISION="$REVISION" MAX_ATTEMPTS="$MAX_ATTEMPTS" ROOT="$ROOT" \
  "$PY" - <<'PY'
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import sys
import time

from huggingface_hub import hf_hub_download


ROOT = Path(os.environ["ROOT"]).resolve()
REPO = os.environ["HF_REPO"]
REVISION = os.environ["REVISION"]
MAX_ATTEMPTS = int(os.environ["MAX_ATTEMPTS"])
TOKEN = os.environ["HF_TOKEN"]
CACHE_DIR = ROOT / ".cache" / "huggingface" / "v2_inputs"

# remote path, exact worker path, byte size, SHA-256
INPUTS = (
    (
        "v2_inputs/main2025/historical-network_week_2020-03-30.json",
        "data/historical-network_week_2020-03-30.json",
        1_126_067_323,
        "aa330bbb8fbf99719fc85d49625d7df7bd68f2b042f5806ded080cec99bad3f8",
    ),
    (
        "v2_inputs/main2025/historical-network_week_2025-07-01.json",
        "data/historical-network_week_2025-07-01.json",
        78_805_269,
        "d77920a7212847dd3cbfbbaabc8622c25e6523d2e7b2006d79d71b85851e1d0b",
    ),
    (
        "v2_inputs/main2025/meta_df.csv",
        "data/meta_df.csv",
        131_025,
        "a8306889fc4473972e843d8e847c9db68776cb86014746f1029fee574e254305",
    ),
    (
        "v2_inputs/main2025/graphpfn-v1.ckpt",
        "checkpoints/graphpfn-v1.ckpt",
        14_307_362,
        "5543a25b07b4be523490b7ef14adbb6c7a9ea763d93a4ef7a22584c0cee9a76d",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid(path: Path, expected_size: int, expected_sha: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_size
        and sha256_file(path) == expected_sha
    )


def sanitized_error(exc: Exception) -> str:
    message = str(exc)
    if TOKEN:
        message = message.replace(TOKEN, "<redacted>")
    return f"{type(exc).__name__}: {message}"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def download(remote_path: str, target: Path, expected_size: int, expected_sha: str) -> None:
    if valid(target, expected_size, expected_sha):
        print(f"INPUT OK existing path={target.relative_to(ROOT)} bytes={expected_size}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.partial")
    temporary.unlink(missing_ok=True)
    last_error = "download did not start"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(
            f"INPUT FETCH attempt={attempt}/{MAX_ATTEMPTS} "
            f"revision={REVISION} remote={remote_path}"
        )
        try:
            cached = Path(
                hf_hub_download(
                    repo_id=REPO,
                    filename=remote_path,
                    repo_type="model",
                    revision=REVISION,
                    token=TOKEN,
                    cache_dir=CACHE_DIR,
                    resume_download=True,
                )
            )
            shutil.copyfile(cached, temporary)
            if temporary.stat().st_size != expected_size:
                raise RuntimeError(
                    f"size mismatch for {remote_path}: expected {expected_size}, "
                    f"got {temporary.stat().st_size}"
                )
            observed_sha = sha256_file(temporary)
            if observed_sha != expected_sha:
                raise RuntimeError(
                    f"SHA-256 mismatch for {remote_path}: expected {expected_sha}, "
                    f"got {observed_sha}"
                )
            os.replace(temporary, target)
            print(
                f"INPUT VERIFIED path={target.relative_to(ROOT)} bytes={expected_size} "
                f"sha256={expected_sha}"
            )
            return
        except Exception as exc:  # retry is deliberately limited and observable
            temporary.unlink(missing_ok=True)
            last_error = sanitized_error(exc)
            print(
                f"INPUT RETRY remote={remote_path} attempt={attempt} error={last_error}",
                file=sys.stderr,
            )
            if attempt < MAX_ATTEMPTS:
                time.sleep(min(5 * (2 ** (attempt - 1)), 60))
    raise SystemExit(
        f"FATAL: input fetch failed after {MAX_ATTEMPTS} attempts: "
        f"remote={remote_path} error={last_error}"
    )


for remote_path, host_path, expected_size, expected_sha in INPUTS:
    download(remote_path, ROOT / host_path, expected_size, expected_sha)

manifest_lines = [
    f"{expected_sha}  {host_path}"
    for _, host_path, _, expected_sha in INPUTS
]
atomic_write(
    ROOT / "logs" / "input_manifest.sha256",
    "\n".join(manifest_lines) + "\n",
)
atomic_write(
    ROOT / "logs" / "input_transport_status.env",
    "\n".join(
        (
            "STATUS=verified",
            f"REPO={REPO}",
            f"REVISION={REVISION}",
            f"FILES={len(INPUTS)}",
        )
    )
    + "\n",
)
print(f"V2 INPUTS VERIFIED repo={REPO} revision={REVISION} files={len(INPUTS)}")
PY
