#!/usr/bin/env python3
"""Fail-closed, zero-training-cost checks for cloud experiment launches.

The preflight deliberately uses only the Python standard library.  It hashes
the files that are synced to the worker, validates the immutable input
manifest, binds TRAIN_CMD to a named experiment, and verifies output contracts
after a run.  It never imports torch, starts a service, contacts an API, or
invokes an LLM.

Examples::

    python cloud/preflight.py --phase local --train-cmd 'bash cloud/pre2022_run.sh'
    PREFLIGHT_SOURCE_SHA256=... python cloud/preflight.py --phase host \
        --train-cmd 'bash cloud/pre2022_run.sh'
    python cloud/preflight.py --phase post --train-cmd 'bash cloud/pre2022_run.sh'

An unclean checkout is accepted only when an explicit attestation file contains
the current source digest and its own SHA-256 is supplied through the
``PREFLIGHT_SOURCE_ATTESTATION*`` variables. The worker still checks that the
synced source digest equals the digest sent by ``cloud/train.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "cloud" / "preflight_manifest.json"
SOURCE_DIGEST_ENV = "PREFLIGHT_SOURCE_SHA256"
ATTESTATION_PATH_ENV = "PREFLIGHT_SOURCE_ATTESTATION"
ATTESTATION_SHA_ENV = "PREFLIGHT_SOURCE_ATTESTATION_SHA256"


class PreflightError(RuntimeError):
    """A required launch invariant is absent or does not match."""


def _fail(message: str) -> None:
    raise PreflightError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read manifest {path}: {exc}")
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        _fail(f"unsupported preflight manifest schema: {path}")
    return payload


def _expand_source_paths(
    root: Path, patterns: Iterable[str], *, optional: bool = False
) -> list[Path]:
    """Expand source globs into a stable, duplicate-free file list."""

    paths: dict[str, Path] = {}
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern or Path(pattern).is_absolute():
            _fail(f"source manifest contains an invalid path pattern: {pattern!r}")
        matches = list(root.glob(pattern))
        if not matches:
            if optional:
                continue
            _fail(f"source manifest path did not match any file: {pattern}")
        for match in matches:
            if not match.is_file():
                continue
            relative = match.relative_to(root).as_posix()
            paths[relative] = match
    if not paths and not optional:
        _fail("source manifest expanded to no files")
    return [paths[name] for name in sorted(paths)]


def source_digest(root: Path, manifest: dict[str, Any]) -> str:
    """Hash source path names and bytes, independent of mtimes and git state."""

    source = manifest.get("source")
    if not isinstance(source, dict):
        _fail("manifest.source is missing")
    algorithm = source.get("hash_algorithm")
    if algorithm != "sha256-path-content-v1":
        _fail(f"unsupported source hash algorithm: {algorithm!r}")
    files = _expand_source_paths(root, source.get("paths", []))
    files.extend(
        _expand_source_paths(root, source.get("optional_paths", []), optional=True)
    )
    files = sorted(set(files), key=lambda path: path.relative_to(root).as_posix())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(str(len(content)).encode("ascii"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def _git_status(root: Path) -> list[str] | None:
    git_dir = root / ".git"
    if not git_dir.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        _fail(f"cannot inspect git source state: {exc}")
    return [line for line in result.stdout.splitlines() if line.strip()]


def _validate_source_state(root: Path, manifest: dict[str, Any], phase: str) -> str:
    digest = source_digest(root, manifest)
    source = manifest["source"]
    if phase == "local":
        status = _git_status(root)
        if status:
            attestation_name = os.environ.get(
                source.get("attestation_path_env", ATTESTATION_PATH_ENV), ""
            )
            attestation_sha = os.environ.get(
                source.get("attestation_sha256_env", ATTESTATION_SHA_ENV), ""
            )
            if not attestation_name or not attestation_sha:
                _fail(
                    "working tree is dirty; provide a source-digest attestation and "
                    f"matching {ATTESTATION_PATH_ENV}/{ATTESTATION_SHA_ENV} before launch"
                )
            attestation = Path(attestation_name).expanduser()
            if not attestation.is_file():
                _fail(f"source attestation is missing: {attestation}")
            actual_attestation_sha = _sha256_file(attestation)
            if actual_attestation_sha != attestation_sha:
                _fail(
                    f"source attestation SHA-256 mismatch: expected {attestation_sha}, "
                    f"got {actual_attestation_sha}"
                )
            try:
                attested_source_digest = attestation.read_text(
                    encoding="utf-8"
                ).strip()
            except (OSError, UnicodeDecodeError) as exc:
                _fail(f"cannot read source attestation: {attestation}: {exc}")
            if attested_source_digest != digest:
                _fail(
                    "source attestation does not bind the current source digest: "
                    f"expected {digest}, got {attested_source_digest!r}"
                )
        # A clean source and an explicitly attested source digest are both valid.
        return digest

    if phase in {"host", "post"}:
        expected = os.environ.get(source.get("host_digest_env", SOURCE_DIGEST_ENV), "")
        if not expected:
            _fail(f"{SOURCE_DIGEST_ENV} is required on the worker")
        if digest != expected:
            _fail(f"synced source SHA-256 mismatch: expected {expected}, got {digest}")
    return digest


def _input_entries(manifest: dict[str, Any], section: str) -> list[dict[str, Any]]:
    value = manifest.get("inputs", {}).get(section, [])
    if not isinstance(value, list):
        _fail(f"manifest.inputs.{section} must be a list")
    entries: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, dict):
            _fail(f"manifest.inputs.{section} contains a non-object")
        if not all(isinstance(entry.get(key), str) and entry.get(key) for key in ("name", "path", "sha256")):
            _fail(f"manifest input entry is incomplete: {entry!r}")
        expected = entry["sha256"]
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected.lower()):
            _fail(f"manifest input has invalid SHA-256: {entry!r}")
        size_bytes = entry.get("size_bytes")
        if size_bytes is not None and (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes <= 0
        ):
            _fail(f"manifest input has invalid size_bytes: {entry!r}")
        entries.append(entry)
    return entries


def _check_inputs(
    root: Path, manifest: dict[str, Any], phase: str, binding: dict[str, Any]
) -> dict[str, str]:
    required_sections = binding.get("required_inputs", ["local"])
    if not isinstance(required_sections, list) or not required_sections:
        _fail("binding required_inputs must be a non-empty list")
    available_sections = manifest.get("inputs", {})
    if not isinstance(available_sections, dict) or any(
        section not in available_sections for section in required_sections
    ):
        _fail(f"binding contains an unknown input group: {required_sections!r}")
    entries: list[dict[str, Any]] = []
    for section in required_sections:
        entries.extend(_input_entries(manifest, section))
    actual: dict[str, str] = {}
    for entry in entries:
        relative = entry["path"] if phase == "local" else entry.get("host_path", entry["path"])
        path = root / relative
        if not path.is_file():
            _fail(f"required {entry['name']} is missing: {path}")
        expected_size = entry.get("size_bytes")
        if expected_size is not None and path.stat().st_size != expected_size:
            _fail(
                f"{entry['name']} size mismatch at {path}: "
                f"expected {expected_size}, got {path.stat().st_size}"
            )
        observed = _sha256_file(path)
        actual[entry["name"]] = observed
        if observed != entry["sha256"]:
            _fail(
                f"{entry['name']} SHA-256 mismatch at {path}: "
                f"expected {entry['sha256']}, got {observed}"
            )
    return actual


def _normal_command(command: str) -> str:
    if not isinstance(command, str) or not command.strip():
        _fail("TRAIN_CMD must be non-empty")
    try:
        return " ".join(shlex.split(command))
    except ValueError as exc:
        _fail(f"TRAIN_CMD is not shell-parseable: {exc}")
    raise AssertionError("unreachable")


def _binding_for_command(manifest: dict[str, Any], command: str) -> tuple[str, dict[str, Any]]:
    normalized = _normal_command(command)
    bindings = manifest.get("bindings")
    if not isinstance(bindings, dict):
        _fail("manifest.bindings is missing")
    for name, binding in bindings.items():
        if isinstance(binding, dict) and _normal_command(str(binding.get("command", ""))) == normalized:
            return str(name), binding
    if "run_full_experiment.py" in normalized:
        _fail(
            "direct run_full_experiment.py is not an allowlisted TRAIN_CMD; "
            "use the bound cloud script so h=4 and upload checks cannot be bypassed"
        )
    _fail(f"TRAIN_CMD is not bound to preflight manifest: {normalized!r}")
    raise AssertionError("unreachable")


def _validated_pi_min(name: str, binding: dict[str, Any]) -> float | None:
    if not binding.get("requires_pi_min"):
        return None
    raw = os.environ.get("PREFLIGHT_PI_MIN")
    if raw is None or not raw.strip():
        _fail(f"binding {name} requires PREFLIGHT_PI_MIN before launch")
    try:
        value = float(raw)
    except ValueError as exc:
        _fail(f"PREFLIGHT_PI_MIN must be a finite float in [0,1]: {exc}")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        _fail("PREFLIGHT_PI_MIN must be a finite float in [0,1]")
    return value


def _validate_remaining_v2_contract(
    manifest: dict[str, Any], binding: dict[str, Any], script_text: str
) -> None:
    """Bind the one-worker h=4,8,12 run to immutable inputs and outputs."""

    name = ("main2025_v2_repeat20" if binding.get("command", "").endswith(" --all")
            else "main2025_v2_remaining_checkpoints")
    all_horizons = name == "main2025_v2_repeat20"
    if binding.get("horizons") != ([1, 4, 8, 12] if all_horizons else [4, 8, 12]):
        _fail("v2 remaining checkpoint binding must declare [4,8,12]")
    required_script_fragments = (
        "TRAIN_HORIZONS=4,8,12",
        'TRAIN_ROOT="checkpoints/main2025_v2_hremaining_train"',
        'RELEASE_ROOT="checkpoints/main2025_v2_hremaining"',
        "--mode dexposure-fm",
        "--holdout-start 2025-01-01",
        "--val-weeks 24 --epochs 20 --seed 42",
        '--horizons "$TRAIN_HORIZONS"',
        "for h in $HORIZON_LIST; do",
    )
    missing_fragments = [
        fragment
        for fragment in required_script_fragments
        if fragment not in script_text
    ]
    if missing_fragments:
        _fail(
            "v2 remaining checkpoint script is missing fixed training settings: "
            f"{missing_fragments}"
        )
    if script_text.count("cloud/train_src/run_full_experiment.py") != 1:
        _fail("v2 remaining checkpoint script must invoke run_full_experiment.py once")

    expected_fetch = {
        "network_data": (
            "v2_inputs/main2025/historical-network_week_2020-03-30.json",
            "data/historical-network_week_2020-03-30.json",
            1_126_067_323,
            "aa330bbb8fbf99719fc85d49625d7df7bd68f2b042f5806ded080cec99bad3f8",
        ),
        "spillover_data": (
            "v2_inputs/main2025/historical-network_week_2025-07-01.json",
            "data/historical-network_week_2025-07-01.json",
            78_805_269,
            "d77920a7212847dd3cbfbbaabc8622c25e6523d2e7b2006d79d71b85851e1d0b",
        ),
        "metadata": (
            "v2_inputs/main2025/meta_df.csv",
            "data/meta_df.csv",
            131_025,
            "a8306889fc4473972e843d8e847c9db68776cb86014746f1029fee574e254305",
        ),
        "base_checkpoint": (
            "v2_inputs/main2025/graphpfn-v1.ckpt",
            "checkpoints/graphpfn-v1.ckpt",
            14_307_362,
            "5543a25b07b4be523490b7ef14adbb6c7a9ea763d93a4ef7a22584c0cee9a76d",
        ),
    }
    fetch = binding.get("input_fetch")
    if not isinstance(fetch, dict):
        _fail(f"binding {name} has no private Hugging Face input fetch contract")
    expected_revision = "cd8fcf4264054b58109e8a50c35cd69f6b2ff72a"
    if fetch.get("repo") != "losdwind/graph-dexposure-ckpt":
        _fail(f"binding {name} input repository is not fixed")
    if fetch.get("revision") != expected_revision:
        _fail(f"binding {name} input revision is not the staged immutable commit")
    if fetch.get("command") != "bash cloud/fetch_v2_inputs_from_hf.sh":
        _fail(f"binding {name} input fetch command is not fixed")
    if fetch.get("script") != "cloud/fetch_v2_inputs_from_hf.sh":
        _fail(f"binding {name} input fetch script is not fixed")
    fetch_script = ROOT / str(fetch.get("script", ""))
    if not fetch_script.is_file():
        _fail(f"binding {name} input fetch script is missing: {fetch_script}")
    fetch_text = fetch_script.read_text(encoding="utf-8")
    for required in (
        expected_revision,
        "resume_download=True",
        "os.replace(temporary, target)",
        "HF_V2_INPUT_MAX_ATTEMPTS",
        'ROOT / "logs" / "input_transport_status.env"',
        'ROOT / "logs" / "input_manifest.sha256"',
    ):
        if required not in fetch_text:
            _fail(f"binding {name} input fetch script lacks {required!r}")

    files = fetch.get("files")
    if not isinstance(files, list) or len(files) != len(expected_fetch):
        _fail(f"binding {name} must fetch exactly four immutable inputs")
    expected_order = list(expected_fetch)
    observed_order = [
        item.get("name") if isinstance(item, dict) else None for item in files
    ]
    if observed_order != expected_order:
        _fail(f"binding {name} immutable input fetch order is not fixed")
    observed_fetch: dict[str, tuple[Any, Any, Any, Any]] = {}
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            _fail(f"binding {name} has an invalid input fetch item: {item!r}")
        item_name = item["name"]
        if item_name in observed_fetch:
            _fail(f"binding {name} repeats input fetch item {item_name}")
        observed_fetch[item_name] = (
            item.get("remote_path"),
            item.get("host_path"),
            item.get("size_bytes"),
            item.get("sha256"),
        )
    if observed_fetch != expected_fetch:
        _fail(f"binding {name} immutable input fetch files do not match v2 inputs")
    normalized_fetch_text = fetch_text.replace("_", "")
    for remote_path, host_path, size_bytes, sha256 in expected_fetch.values():
        for required in (remote_path, host_path, sha256):
            if required not in fetch_text:
                _fail(
                    f"binding {name} input fetch script lacks immutable value "
                    f"{required!r}"
                )
        if str(size_bytes) not in normalized_fetch_text:
            _fail(
                f"binding {name} input fetch script lacks byte size {size_bytes}"
            )

    local_inputs = {
        entry["name"]: (
            entry.get("host_path"),
            entry.get("size_bytes"),
            entry.get("sha256"),
        )
        for entry in _input_entries(manifest, "local")
    }
    for item_name, (_, host_path, size_bytes, sha256) in expected_fetch.items():
        if local_inputs.get(item_name) != (host_path, size_bytes, sha256):
            _fail(
                f"binding {name} fetch contract disagrees with manifest input "
                f"{item_name}"
            )

    release_root = "checkpoints/main2025_v2_hremaining"
    train_root = "checkpoints/main2025_v2_hremaining_train/finetuned"
    expected_outputs = {
        f"{release_root}/dexposure-fm-h4.pt",
        f"{release_root}/dexposure-fm-h8.pt",
        f"{release_root}/dexposure-fm-h12.pt",
        f"{release_root}/feature_schema.json",
        f"{release_root}/task1_metrics.json",
        f"{release_root}/run_config_h4.json",
        f"{release_root}/run_config_h8.json",
        f"{release_root}/run_config_h12.json",
        f"{release_root}/SHA256SUMS",
        f"{train_root}/best_model_h4.pt",
        f"{train_root}/best_model_h8.pt",
        f"{train_root}/best_model_h12.pt",
        f"{train_root}/feature_schema.json",
        f"{train_root}/experiment_results.json",
        f"{train_root}/data_quality.json",
        f"{train_root}/metrics.json",
        f"{train_root}/all_results.json",
    }
    if all_horizons:
        expected_outputs.update({
            f"{release_root}/dexposure-fm-h1.pt",
            f"{release_root}/run_config_h1.json",
            f"{train_root}/best_model_h1.pt",
        })
        if "--all) TRAIN_HORIZONS=1,4,8,12" not in script_text:
            _fail("four-horizon repeat flag is missing")
    output_items = manifest.get("outputs", {}).get(name)
    if not isinstance(output_items, list):
        _fail(f"binding {name} has no output contract")
    observed_outputs = {
        item.get("path") for item in output_items if isinstance(item, dict)
    }
    if observed_outputs != expected_outputs:
        _fail(f"binding {name} output paths do not match the h=4,8,12 contract")

    json_contract = binding.get("json_contract")
    if not isinstance(json_contract, dict):
        _fail(f"binding {name} has no JSON contract")
    required_json = set(json_contract.get("required_files", []))
    expected_json = {
        f"{train_root}/feature_schema.json",
        f"{train_root}/experiment_results.json",
        f"{train_root}/data_quality.json",
        f"{train_root}/metrics.json",
        f"{train_root}/all_results.json",
        f"{release_root}/feature_schema.json",
        f"{release_root}/task1_metrics.json",
        f"{release_root}/run_config_h4.json",
        f"{release_root}/run_config_h8.json",
        f"{release_root}/run_config_h12.json",
    }
    if all_horizons:
        expected_json.add(f"{release_root}/run_config_h1.json")
    if required_json != expected_json or json_contract.get("minimum_count") != len(expected_json):
        _fail(f"binding {name} JSON contract does not cover all required results")


def _verify_remaining_input_transport(root: Path, binding: dict[str, Any]) -> None:
    """Verify the remote-only input receipt before training or postflight."""

    fetch = binding.get("input_fetch")
    if not isinstance(fetch, dict):
        _fail("v2 remaining input fetch contract is missing")
    files = fetch.get("files")
    if not isinstance(files, list) or len(files) != 4:
        _fail("v2 remaining input fetch file contract is incomplete")

    status_path = root / "logs" / "input_transport_status.env"
    expected_status = "\n".join(
        (
            "STATUS=verified",
            f"REPO={fetch.get('repo')}",
            f"REVISION={fetch.get('revision')}",
            "FILES=4",
            "",
        )
    )
    if not status_path.is_file():
        _fail(f"v2 remaining input transport status is missing: {status_path}")
    if status_path.read_text(encoding="utf-8") != expected_status:
        _fail(f"v2 remaining input transport status is invalid: {status_path}")

    manifest_path = root / "logs" / "input_manifest.sha256"
    expected_manifest = "".join(
        f"{item.get('sha256')}  {item.get('host_path')}\n" for item in files
    )
    if not manifest_path.is_file():
        _fail(f"v2 remaining input manifest is missing: {manifest_path}")
    if manifest_path.read_text(encoding="utf-8") != expected_manifest:
        _fail(f"v2 remaining input manifest is invalid: {manifest_path}")


def _validate_binding(
    manifest: dict[str, Any], command: str
) -> tuple[str, dict[str, Any]]:
    name, binding = _binding_for_command(manifest, command)
    if binding.get("enabled", True) is not True:
        reason = binding.get("disabled_reason", "no reason recorded")
        _fail(f"binding {name} is disabled: {reason}")
    _validated_pi_min(name, binding)
    required_env = binding.get("requires_env")
    if required_env and os.environ.get(str(required_env)) != "1":
        _fail(f"binding {name} requires explicit {required_env}=1")
    if binding.get("requires_openrouter"):
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            env_file = ROOT / ".env"
            if env_file.is_file():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("OPENROUTER_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip("\"'")
                        break
        if not key:
            _fail(f"binding {name} requires OPENROUTER_API_KEY before launch")
    if binding.get("allow_independent_m7") is not False:
        _fail(f"binding {name} must explicitly disallow independent M7 sampling")

    # A command that mentions M7 must be an export-only command.  This check
    # is intentionally textual and happens before a service or LLM can start.
    normalized = _normal_command(command)
    if "m7_fm_llm_gated" in normalized and "--export-prompts-only" not in normalized:
        _fail("independent M7 sampling is forbidden by the cloud manifest")
    script = ROOT / str(binding.get("script", ""))
    if not script.is_file():
        _fail(f"bound command script is missing: {script}")
    script_text = script.read_text(encoding="utf-8")
    if "m7_fm_llm_gated" in script_text and "--export-prompts-only" not in script_text:
        _fail(f"bound script contains an independent M7 path: {script}")

    expected = manifest.get("primary", {}).get("horizons")
    horizons = binding.get("horizons")
    if not isinstance(horizons, list) or not horizons:
        _fail(f"binding {name} has no forecast horizon contract")
    if name == manifest.get("primary", {}).get("binding"):
        if horizons != expected or horizons != [4]:
            _fail("primary cloud binding must be h=4 only")
        if "TRAIN_HORIZONS=4" not in script_text and "--horizons 4" not in script_text:
            _fail("primary script does not declare its h=4-only training configuration")
    if name == "pre2022_rq1" and horizons != [1, 4, 8, 12]:
        _fail("RQ1 all-horizon binding must declare [1,4,8,12]")
    if name in {"main2025_v2_all_horizons", "main2025_v2_checkpoints"}:
        if horizons != [1, 4, 8, 12]:
            _fail("v2 reconstruction binding must declare [1,4,8,12]")
        if "TRAIN_HORIZONS=1,4,8,12" not in script_text:
            _fail("v2 reconstruction script does not declare all four horizons")
    if name in {"main2025_v2_remaining_checkpoints", "main2025_v2_repeat20"}:
        _validate_remaining_v2_contract(manifest, binding, script_text)
    if name == "main2025_fm_tuning":
        if horizons != [1, 4, 8, 12] or binding.get("epochs") != 20:
            _fail("FM tuning requires all four horizons and a 20-epoch finalist cap")
        reference = manifest["bindings"]["main2025_v2_repeat20"]
        if binding.get("input_fetch") != reference.get("input_fetch"):
            _fail("FM tuning must reuse the immutable 2025 input contract")
        if "cloud/train_src/tune_fm.py --run" not in script_text:
            _fail("FM tuning must use the reviewed validation-only runner")
        sys.path.insert(0, str(ROOT))
        from cloud.train_src.tune_fm import validate_plan
        validate_plan(json.loads((ROOT / "cloud/fm_tune_plan.json").read_text()))
    if name == "main2025_fm_sequential":
        if horizons != [1, 4, 8, 12] or binding.get("epochs") != 40:
            _fail("FM sequential training requires all four horizons and a 40-epoch per-stage cap")
        reference = manifest["bindings"]["main2025_v2_repeat20"]
        if binding.get("input_fetch") != reference.get("input_fetch"):
            _fail("FM sequential training must reuse the immutable 2025 input contract")
        if "cloud/train_src/sequential_fm.py --run" not in script_text:
            _fail("FM sequential training must use the reviewed validation-only runner")
        sys.path.insert(0, str(ROOT))
        from cloud.train_src.sequential_fm import validate_plan
        validate_plan(json.loads((ROOT / "cloud/fm_sequential_plan.json").read_text()))
    if name == "main2025_fm_eval_paper":
        if (horizons != [1, 4, 8, 12] or binding.get("epochs") != 0
                or binding.get("holdout_start") != "2025-01-01"
                or binding.get("val_weeks") != 24 or binding.get("seed") != 42):
            _fail("paper eval requires four horizons, epochs=0, and the fixed 2025 split")
        reference = manifest["bindings"]["main2025_v2_repeat20"]
        if binding.get("input_fetch") != reference.get("input_fetch"):
            _fail("paper eval must reuse the immutable 2025 input contract")
        if "cloud/train_src/eval_paper.py --run" not in script_text:
            _fail("paper eval must use the reviewed release-eval runner")
        if "tune_fm.py --run" in script_text:
            _fail("paper eval must not launch the hyperparameter search")
        sys.path.insert(0, str(ROOT))
        from cloud.train_src.eval_paper import validate_plan as validate_eval_plan
        validate_eval_plan(json.loads((ROOT / "cloud/fm_eval_paper_plan.json").read_text()))
    if name == "main2025_v2_h12_extended":
        if (horizons != [12] or binding.get("epochs") != 40
                or binding.get("early_stop_patience") != 5
                or binding.get("holdout_start") != "2025-01-01"
                or binding.get("val_weeks") != 24 or binding.get("seed") != 42):
            _fail("h12 extended training contract must be 40 epochs, patience 5, fixed v2 split and seed")
        for required in ('EPOCHS=40', 'PATIENCE=5', '--epochs "$EPOCHS"', '--patience "$PATIENCE"'):
            if required not in script_text:
                _fail(f"extended training script lacks {required}")
    if name == "main2025_v2_h12":
        if horizons != [12]:
            _fail("v2 h=12 reconstruction binding must declare [12]")
        if "TRAIN_HORIZONS=12" not in script_text:
            _fail("v2 h=12 reconstruction script does not declare h=12")
    if binding.get("smoke_weeks") == 2 and horizons != [4]:
        _fail("two-week smoke binding must remain h=4 only")
    return name, binding


def _validate_upload_contract(manifest: dict[str, Any]) -> None:
    upload = manifest.get("artifact_upload")
    if not isinstance(upload, dict) or upload.get("required") is not True:
        _fail("artifact upload is not marked required in the manifest")
    paths = upload.get("paths")
    if not isinstance(paths, list) or not paths or any(not isinstance(path, str) for path in paths):
        _fail("artifact upload manifest must name checkpoints, logs, and output paths")
    if not upload.get("destination") or not upload.get("manifest_name"):
        _fail("artifact upload destination/manifest name is missing")


def _output_specs(manifest: dict[str, Any], binding_name: str) -> list[dict[str, Any]]:
    outputs = manifest.get("outputs", {}).get(binding_name)
    if not isinstance(outputs, list) or not outputs:
        _fail(f"no output manifest exists for binding {binding_name}")
    result: list[dict[str, Any]] = []
    for item in outputs:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            _fail(f"invalid output manifest item for {binding_name}: {item!r}")
        kind = item.get("kind", "file")
        if kind not in {"file", "directory"}:
            _fail(f"invalid output kind for {binding_name}: {kind!r}")
        result.append({"path": item["path"], "kind": kind})
    return result


def _json_contract(manifest: dict[str, Any], binding_name: str) -> tuple[list[str], int]:
    binding = manifest.get("bindings", {}).get(binding_name)
    contract = binding.get("json_contract") if isinstance(binding, dict) else None
    if not isinstance(contract, dict):
        _fail(f"binding {binding_name} has no JSON artifact contract")
    required = contract.get("required_files")
    minimum = contract.get("minimum_count")
    if (
        not isinstance(required, list)
        or any(not isinstance(path, str) or not path for path in required)
    ):
        _fail(f"binding {binding_name} JSON contract has invalid required_files")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < len(required):
        _fail(f"binding {binding_name} JSON contract has invalid minimum_count")
    return list(dict.fromkeys(required)), minimum


def _check_output_parents(root: Path, specs: list[dict[str, Any]]) -> None:
    for spec in specs:
        path = root / spec["path"]
        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)
        if not os.access(parent, os.W_OK):
            _fail(f"output parent is not writable: {parent}")
        if path.exists() and os.environ.get("PREFLIGHT_ALLOW_OUTPUT_REUSE") != "1":
            _fail(
                f"output target already exists: {path}; set PREFLIGHT_ALLOW_OUTPUT_REUSE=1 "
                "only for an explicitly reviewed rerun"
            )


def _check_outputs(root: Path, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in specs:
        path = root / spec["path"]
        if spec["kind"] == "file":
            if not path.is_file():
                _fail(f"required output is missing: {path}")
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "kind": "file",
                    "size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
            continue
        if not path.is_dir():
            _fail(f"required output directory is missing: {path}")
        files = sorted(item for item in path.rglob("*") if item.is_file())
        if not files:
            _fail(f"required output directory is empty: {path}")
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "kind": "directory",
                "file_count": len(files),
                "files": [
                    {
                        "path": item.relative_to(root).as_posix(),
                        "size": item.stat().st_size,
                        "sha256": _sha256_file(item),
                    }
                    for item in files
                ],
            }
        )
    return records


def _verify_checkpoint_invariants(
    root: Path, manifest: dict[str, Any], binding_name: str, input_shas: dict[str, str]
) -> None:
    """Check cheap checkpoint/schema invariants without loading model weights."""

    specs = _output_specs(manifest, binding_name)
    checkpoint_files: list[Path] = []
    for spec in specs:
        path = root / spec["path"]
        if spec["path"].endswith(".pt"):
            checkpoint_files.append(path)
        elif spec["kind"] == "directory" and path.is_dir():
            checkpoint_files.extend(item for item in path.rglob("*.pt") if item.is_file())
    if not checkpoint_files:
        _fail(f"binding {binding_name} has no checkpoint output in its manifest")
    for path in checkpoint_files:
        if not path.is_file() or path.stat().st_size == 0:
            _fail(f"checkpoint invariant failed for {path}")

    schema_candidates: list[Path] = []
    for spec in specs:
        path = root / spec["path"]
        if path.name == "feature_schema.json" and path.is_file():
            schema_candidates.append(path)
        elif path.is_dir():
            schema_candidates.extend(item for item in path.rglob("feature_schema.json") if item.is_file())
    if not schema_candidates:
        _fail(f"binding {binding_name} has no feature_schema.json output")
    expected_meta = input_shas.get("metadata")
    for schema_path in schema_candidates:
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _fail(f"feature schema is not valid JSON: {schema_path}: {exc}")
        if not isinstance(schema, dict) or not isinstance(schema.get("categories"), list):
            _fail(f"feature schema invariant failed: categories missing in {schema_path}")
        if expected_meta and schema.get("metadata_sha256") not in {None, expected_meta}:
            _fail(
                f"feature schema metadata SHA-256 mismatch in {schema_path}: "
                f"expected {expected_meta}, got {schema.get('metadata_sha256')}"
            )


def _import_verifiers(root: Path, manifest: dict[str, Any]) -> tuple[Any, Any, Any]:
    verifiers = manifest.get("verifiers", {})
    if not isinstance(verifiers, dict):
        _fail("manifest.verifiers is missing")
    sys.path.insert(0, str(root / "paper"))
    try:
        from scripts.replay_coverage_intervention_frontier import (
            apply_action_gate,
            evidence_hash,
            validate_decision_output,
        )
    except Exception as exc:
        _fail(f"callable evidence/candidate/gate verifiers unavailable: {exc}")
    for key in ("evidence", "candidate", "gate"):
        if not isinstance(verifiers.get(key), str) or ":" not in verifiers[key]:
            _fail(f"manifest verifier {key} is not a callable reference")
    return evidence_hash, validate_decision_output, apply_action_gate


def _verify_crisis_horizon_rows(
    payload: Any,
    expected_horizons: list[int],
) -> None:
    """Require one horizon-aligned B5 row for every crisis/method/horizon cell."""
    if not isinstance(payload, list) or not payload:
        _fail("b5_crisis.json must be a non-empty list")
    crises = {"terra_luna", "ftx", "svb"}
    methods = {"m5_fm_rules", "m1_persistence_rules"}
    expected = {
        (crisis, method, horizon)
        for crisis in crises
        for method in methods
        for horizon in expected_horizons
    }
    observed: set[tuple[str, str, int]] = set()
    for row in payload:
        if not isinstance(row, dict):
            _fail("b5_crisis.json contains a non-object row")
        prediction_horizon = row.get("prediction_horizon")
        label_horizon = row.get("label_horizon")
        if prediction_horizon != label_horizon:
            _fail(
                "b5 crisis prediction/label horizon mismatch: "
                f"prediction={prediction_horizon}, label={label_horizon}"
            )
        crisis = row.get("crisis")
        method = row.get("method")
        if (
            crisis not in crises
            or method not in methods
            or prediction_horizon not in expected_horizons
        ):
            _fail(
                "unexpected b5 crisis horizon row: "
                f"crisis={crisis}, method={method}, horizon={prediction_horizon}"
            )
        key = (crisis, method, prediction_horizon)
        if key in observed:
            _fail(f"duplicate b5 crisis horizon row: {key}")
        observed.add(key)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        _fail(
            "b5 crisis horizon grid is incomplete or unexpected: "
            f"missing={missing}, extra={extra}"
        )


def _verify_json_artifacts(root: Path, binding_name: str, manifest: dict[str, Any]) -> dict[str, int]:
    """Verify cheap evidence/candidate/gate invariants for produced JSON."""

    required_files, minimum_count = _json_contract(manifest, binding_name)
    required_paths = {root / relative for relative in required_files}
    search_root = root / "logs"
    json_paths = set(required_paths)
    if search_root.exists():
        json_paths.update(search_root.rglob("*.json"))

    payloads: dict[Path, Any] = {}
    for path in sorted(json_paths):
        if path in required_paths and not path.is_file():
            _fail(f"required JSON artifact is missing: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if path in required_paths:
                _fail(f"required JSON artifact is unreadable: {path}: {exc}")
            continue
        if path in required_paths and (
            not isinstance(payload, (list, dict)) or not payload
        ):
            _fail(f"required JSON artifact is empty or malformed: {path}")
        payloads[path] = payload

    if len(payloads) < minimum_count:
        _fail(
            f"binding {binding_name} produced {len(payloads)} JSON artifacts; "
            f"at least {minimum_count} are required"
        )

    evidence_hash, validate_candidate, apply_gate = _import_verifiers(root, manifest)
    summary = {
        "evidence": 0,
        "candidates": 0,
        "gates": 0,
        "prompt_files": 0,
        "json_files": len(payloads),
    }

    for path in sorted(payloads):
        name = path.name
        payload = payloads[path]

        if name == "b5_crisis.json" and binding_name in {
            "pre2022_crisis_h4",
            "pre2022_rq1",
        }:
            binding = manifest["bindings"][binding_name]
            _verify_crisis_horizon_rows(payload, list(binding["horizons"]))
            continue

        if name.startswith("prompts_m6_fm_llm"):
            if not isinstance(payload, list) or not payload:
                _fail(f"M6 prompt artifact is not a non-empty list: {path}")
            try:
                from experiments.b5_decision_quality import CandidateSet
            except Exception as exc:
                _fail(f"CandidateSet verifier is unavailable: {exc}")
            for row in payload:
                if not isinstance(row, dict) or row.get("method") != "m6_fm_llm":
                    _fail(f"invalid M6 prompt row: {path}")
                if "llm_outputs" in row or "raw_response" in row:
                    _fail(f"M6 prompt artifact contains a sampled output: {path}")
                prompts = (row.get("system_prompt"), row.get("user_prompt"))
                if not all(isinstance(item, str) and item for item in prompts):
                    _fail(f"M6 prompt row has incomplete evidence: {path}")
                bundle = row.get("evidence_bundle")
                if not isinstance(bundle, dict):
                    _fail(f"M6 prompt row has no evidence bundle: {path}")
                candidate_rows = bundle.get("candidate_set")
                if not isinstance(candidate_rows, list) or not candidate_rows:
                    _fail(f"M6 prompt row has no candidate set: {path}")
                try:
                    candidate_set = CandidateSet.from_rows(candidate_rows)
                except (AttributeError, TypeError, ValueError, KeyError) as exc:
                    _fail(f"M6 candidate set is malformed: {path}: {exc}")
                if bundle.get("candidate_set_sha256") != candidate_set.candidate_set_sha256:
                    _fail(f"M6 candidate-set hash mismatch: {path}")
                if bundle.get("candidate_membership_sha256") != candidate_set.candidate_membership_sha256:
                    _fail(f"M6 candidate-membership hash mismatch: {path}")
                expected = bundle.get("evidence_sha256")
                observed = evidence_hash(prompts[0], prompts[1], candidate_set)
                if not expected or expected != observed:
                    _fail(f"M6 evidence hash mismatch: {path}")
                summary["evidence"] += 1
            summary["prompt_files"] += 1
            continue

        if name.startswith("prompts_m7_fm_llm_gated"):
            if not isinstance(payload, list) or not payload:
                _fail(f"M7 prompt artifact is not a non-empty list: {path}")
            for row in payload:
                if not isinstance(row, dict) or row.get("method") != "m7_fm_llm_gated":
                    _fail(f"invalid M7 prompt row: {path}")
                if "llm_outputs" in row or "raw_response" in row:
                    _fail(f"M7 artifact contains an independently sampled output: {path}")
                if not row.get("system_prompt") or not row.get("user_prompt"):
                    _fail(f"M7 prompt row has incomplete evidence: {path}")
                summary["evidence"] += 1
            summary["prompt_files"] += 1
            continue

        if name == "raw_m6_fm_llm.json":
            if not isinstance(payload, list) or not payload:
                _fail(f"M6 raw artifact is not a list: {path}")
            try:
                from experiments.b5_decision_quality import CandidateSet
            except Exception as exc:
                _fail(f"CandidateSet verifier is unavailable: {exc}")
            for row in payload:
                if not isinstance(row, dict) or row.get("method") != "m6_fm_llm":
                    _fail(f"unexpected method in M6 artifact: {path}")
                prompts = (row.get("system_prompt"), row.get("user_prompt"))
                if not all(isinstance(item, str) and item for item in prompts):
                    _fail(f"M6 evidence bundle is incomplete: {path}")
                bundle = row.get("evidence_bundle")
                if not isinstance(bundle, dict):
                    _fail(f"M6 raw row has no evidence bundle: {path}")
                candidate_rows = bundle.get("candidate_set")
                if not isinstance(candidate_rows, list) or not candidate_rows:
                    _fail(f"M6 raw row has no candidate set: {path}")
                try:
                    candidate_set = CandidateSet.from_rows(candidate_rows)
                except (AttributeError, TypeError, ValueError, KeyError) as exc:
                    _fail(f"M6 candidate set is malformed: {path}: {exc}")
                if bundle.get("candidate_set_sha256") != candidate_set.candidate_set_sha256:
                    _fail(f"M6 candidate-set hash mismatch: {path}")
                if bundle.get("candidate_membership_sha256") != candidate_set.candidate_membership_sha256:
                    _fail(f"M6 candidate-membership hash mismatch: {path}")
                observed_evidence = evidence_hash(prompts[0], prompts[1], candidate_set)
                if bundle.get("evidence_sha256") != observed_evidence:
                    _fail(f"M6 evidence hash mismatch: {path}")
                outputs = row.get("raw_llm_outputs", row.get("llm_outputs"))
                if not isinstance(outputs, list) or not outputs:
                    _fail(f"M6 raw row has no sampled outputs: {path}")
                for candidate in outputs:
                    validate_candidate(candidate, candidate_set=candidate_set)
                    summary["candidates"] += 1
                    gated, _ = apply_gate(candidate)
                    validate_candidate(gated, candidate_set=candidate_set)
                    summary["gates"] += 1
                summary["evidence"] += 1
            summary["prompt_files"] += 1
            continue

        if name == "frozen_evidence.json":
            cases = payload.get("cases") if isinstance(payload, dict) else None
            if not isinstance(cases, list) or not cases:
                _fail(f"frozen evidence has no cases: {path}")
            try:
                from experiments.b5_decision_quality import CandidateSet
            except Exception as exc:
                _fail(f"CandidateSet verifier is unavailable: {exc}")
            for case in cases:
                evidence = case.get("evidence") if isinstance(case, dict) else None
                if not isinstance(evidence, dict):
                    _fail(f"frozen evidence case is malformed: {path}")
                candidate_rows = evidence.get("candidate_set")
                if not isinstance(candidate_rows, list) or not candidate_rows:
                    _fail(f"frozen evidence has no candidate set: {path}")
                try:
                    candidate_set = CandidateSet.from_rows(candidate_rows)
                except (AttributeError, TypeError, ValueError, KeyError) as exc:
                    _fail(f"frozen CandidateSet is malformed: {path}: {exc}")
                if evidence.get("candidate_set_sha256") != candidate_set.candidate_set_sha256:
                    _fail(f"frozen candidate-set hash mismatch: {path}")
                if evidence.get("candidate_membership_sha256") != candidate_set.candidate_membership_sha256:
                    _fail(f"frozen candidate-membership hash mismatch: {path}")
                expected = evidence.get("evidence_sha256")
                observed = evidence_hash(
                    evidence.get("system_prompt", ""),
                    evidence.get("user_prompt", ""),
                    candidate_set,
                )
                if not expected or expected != observed:
                    _fail(f"evidence hash mismatch: {path}")
                summary["evidence"] += 1
            continue

        if name == "evidence_alignment.json":
            if not isinstance(payload, dict) or payload.get("horizon") != 4:
                _fail(f"evidence alignment is not h=4: {path}")
            rows = payload.get("rows", [])
            if not rows or payload.get("evidence_matches") != payload.get("n_weeks"):
                _fail(f"evidence alignment is incomplete: {path}")
            if not all(isinstance(row, dict) and row.get("evidence_match") is True for row in rows):
                _fail(f"evidence alignment contains a mismatch: {path}")
            summary["evidence"] += len(rows)

    return summary


def _write_output_manifest(
    root: Path,
    manifest: dict[str, Any],
    binding_name: str,
    source_sha: str,
    input_shas: dict[str, str],
    records: list[dict[str, Any]],
    verifier_summary: dict[str, int],
    pi_min: float | None,
) -> Path:
    upload_name = str(manifest["artifact_upload"]["manifest_name"])
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    output = logs / upload_name
    payload = {
        "schema_version": 1,
        "binding": binding_name,
        "source_sha256": source_sha,
        "input_sha256": input_shas,
        "pi_min": pi_min,
        "outputs": records,
        "verifiers": verifier_summary,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def run_preflight(
    *,
    root: Path,
    manifest_path: Path,
    phase: str,
    train_cmd: str,
) -> dict[str, Any]:
    manifest = _read_manifest(manifest_path)
    _validate_upload_contract(manifest)
    binding_name, binding = _validate_binding(manifest, train_cmd)
    pi_min = _validated_pi_min(binding_name, binding)
    source_sha = _validate_source_state(root, manifest, phase)
    input_shas = _check_inputs(root, manifest, phase, binding)
    if phase in {"host", "post"} and binding_name in {
        "main2025_v2_remaining_checkpoints", "main2025_v2_repeat20",
        "main2025_fm_tuning", "main2025_fm_eval_paper", "main2025_fm_sequential",
    }:
        _verify_remaining_input_transport(root, binding)
    specs = _output_specs(manifest, binding_name)

    if phase == "host":
        _check_output_parents(root, specs)
    verifier_summary = {
        "evidence": 0,
        "candidates": 0,
        "gates": 0,
        "prompt_files": 0,
        "json_files": 0,
    }
    records: list[dict[str, Any]] = []
    if phase == "post":
        records = _check_outputs(root, specs)
        if binding_name == "main2025_fm_tuning":
            from cloud.train_src.tune_fm import verify_bundle
            verify_bundle(root, source_sha)
        if binding_name == "main2025_fm_sequential":
            from cloud.train_src.sequential_fm import verify_bundle
            verify_bundle(root, source_sha)
        if binding_name == "main2025_fm_eval_paper":
            from cloud.train_src.eval_paper import verify_bundle as verify_eval_bundle
            verify_eval_bundle(root, source_sha)
        _verify_checkpoint_invariants(root, manifest, binding_name, input_shas)
        verifier_summary = _verify_json_artifacts(root, binding_name, manifest)
        _write_output_manifest(
            root,
            manifest,
            binding_name,
            source_sha,
            input_shas,
            records,
            verifier_summary,
            pi_min,
        )

    return {
        "phase": phase,
        "binding": binding_name,
        "horizons": binding.get("horizons"),
        "source_sha256": source_sha,
        "pi_min": pi_min,
        "inputs": input_shas,
        "outputs_checked": len(records),
        "verifiers": verifier_summary,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--phase", choices=("local", "host", "post"), default="local")
    parser.add_argument("--train-cmd", default="")
    parser.add_argument("--print-source-digest", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        manifest = _read_manifest(args.manifest.resolve())
        if args.print_source_digest:
            print(source_digest(ROOT, manifest))
            return 0
        if not args.train_cmd:
            _fail("--train-cmd is required unless --print-source-digest is used")
        report = run_preflight(
            root=ROOT,
            manifest_path=args.manifest.resolve(),
            phase=args.phase,
            train_cmd=args.train_cmd,
        )
        if args.as_json:
            print(json.dumps(report, sort_keys=True))
        else:
            print(
                f"PREFLIGHT OK phase={report['phase']} binding={report['binding']} "
                f"horizons={report['horizons']} source_sha256={report['source_sha256']}"
            )
        return 0
    except PreflightError as exc:
        print(f"PREFLIGHT BLOCKED: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError) as exc:
        print(f"PREFLIGHT BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
