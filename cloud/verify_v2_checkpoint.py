#!/usr/bin/env python3
"""Fail closed unless a downloaded v2 checkpoint matches its run contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import torch


EXPECTED_SPLIT_COUNTS = {"train": 226, "val": 24, "test": 33}
EXPECTED_SNAPSHOT_COUNT = 283
EXPECTED_HOLDOUT_START = "2025-01-01"
EXPECTED_FIRST_DATE = "2020-03-23"
EXPECTED_LAST_DATE = "2025-08-18"
SPLIT_DIGEST_ALGORITHM = "sha256-canonical-json-v1"


def fail(message: str) -> None:
    raise SystemExit(f"CHECKPOINT VERIFICATION FAILED: {message}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read JSON {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON root is not an object: {path}")
    return value


def expected_input(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    for item in manifest.get("inputs", {}).get("local", []):
        if not isinstance(item, dict):
            fail("manifest local input entry is malformed")
        if item.get("name") == name:
            return item
    fail(f"manifest input is missing: {name}")
    raise AssertionError("unreachable")


def expected_input_hash(manifest: dict[str, Any], name: str) -> str:
    value = expected_input(manifest, name).get("sha256")
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value.lower()
    ):
        fail(f"manifest input SHA-256 is invalid: {name}")
    return value


def require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        fail(f"{field} expected {expected!r}, got {actual!r}")


def canonical_split_sha256(groups: dict[str, list[str]]) -> str:
    payload = json.dumps(
        groups, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_exact_split(
    manifest: dict[str, Any], split: Any
) -> dict[str, list[str]]:
    if not isinstance(split, dict) or set(split) != set(EXPECTED_SPLIT_COUNTS):
        fail("provenance.split must contain exactly train, val, and test")

    groups: dict[str, list[str]] = {}
    for name, expected_count in EXPECTED_SPLIT_COUNTS.items():
        values = split.get(name)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            fail(f"provenance.split.{name} is malformed")
        require_equal(len(values), expected_count, f"{name} snapshot count")
        if values != sorted(values):
            fail(f"provenance.split.{name} is not chronologically sorted")
        try:
            for item in values:
                date.fromisoformat(item)
        except ValueError as exc:
            fail(f"provenance.split.{name} contains an invalid date: {exc}")
        groups[name] = values

    all_dates = groups["train"] + groups["val"] + groups["test"]
    require_equal(len(all_dates), EXPECTED_SNAPSHOT_COUNT, "total snapshot count")
    require_equal(len(set(all_dates)), EXPECTED_SNAPSHOT_COUNT, "unique snapshot count")
    if all_dates != sorted(all_dates):
        fail("provenance split groups are not in strict chronological order")

    network_input = expected_input(manifest, "network_data")
    contract = network_input.get("temporal_split")
    if not isinstance(contract, dict):
        fail("manifest network_data temporal_split contract is missing")
    require_equal(
        contract.get("algorithm"), SPLIT_DIGEST_ALGORITHM, "split digest algorithm"
    )
    require_equal(contract.get("counts"), EXPECTED_SPLIT_COUNTS, "manifest split counts")
    require_equal(
        contract.get("snapshot_count"), EXPECTED_SNAPSHOT_COUNT, "manifest snapshot count"
    )
    require_equal(
        contract.get("holdout_start"), EXPECTED_HOLDOUT_START, "manifest holdout start"
    )
    require_equal(contract.get("first_date"), EXPECTED_FIRST_DATE, "manifest first date")
    require_equal(contract.get("last_date"), EXPECTED_LAST_DATE, "manifest last date")
    require_equal(all_dates[0], EXPECTED_FIRST_DATE, "split first date")
    require_equal(all_dates[-1], EXPECTED_LAST_DATE, "split last date")

    expected_digest = contract.get("dates_sha256")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64 or any(
        char not in "0123456789abcdef" for char in expected_digest.lower()
    ):
        fail("manifest split dates SHA-256 is invalid")
    require_equal(
        canonical_split_sha256(groups), expected_digest, "provenance split dates SHA-256"
    )

    holdout = date.fromisoformat(EXPECTED_HOLDOUT_START)
    if any(date.fromisoformat(item) >= holdout for item in groups["train"] + groups["val"]):
        fail("train or validation split reaches the 2025 holdout")
    if any(date.fromisoformat(item) < holdout for item in groups["test"]):
        fail("test split starts before the 2025 holdout")
    return groups


def require_finite_metrics(metrics: dict[str, Any], horizon: int) -> dict[str, Any]:
    result = metrics.get(f"h{horizon}")
    if not isinstance(result, dict):
        fail(f"Task I metrics do not contain h{horizon}")
    fields = {
        "exist": ("auprc", "auroc"),
        "weight": ("mae", "rmse", "weighted_mae"),
        "node": ("mae", "rmse"),
    }
    for group, names in fields.items():
        values = result.get(group)
        if not isinstance(values, dict):
            fail(f"Task I metrics group is missing: {group}")
        for name in names:
            value = values.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                fail(f"Task I metric is not finite: {group}.{name}={value!r}")
    return result


def verify(args: argparse.Namespace) -> None:
    manifest = load_json(args.manifest)
    metrics = load_json(args.metrics)
    config = load_json(args.config)
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:
        fail(f"cannot load checkpoint {args.checkpoint}: {exc}")
    if not isinstance(checkpoint, dict):
        fail("checkpoint root is not a mapping")

    require_equal(checkpoint.get("artifact_status"), "complete", "artifact_status")
    require_equal(checkpoint.get("horizon"), args.horizon, "checkpoint.horizon")
    if not isinstance(checkpoint.get("model"), dict) or not checkpoint["model"]:
        fail("checkpoint model state is missing")

    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, dict):
        fail("checkpoint provenance is missing")
    require_equal(provenance.get("horizon"), args.horizon, "provenance.horizon")
    require_equal(provenance.get("seed"), 42, "provenance.seed")
    expected_epochs = getattr(args, "epochs", 20)
    require_equal(provenance.get("epochs"), expected_epochs, "provenance.epochs")
    completed_epochs = provenance.get("completed_epochs")
    if not isinstance(completed_epochs, int) or not 1 <= completed_epochs <= expected_epochs:
        fail(f"provenance.completed_epochs is invalid: {completed_epochs!r}")
    require_equal(
        provenance.get("data_sha256"),
        expected_input_hash(manifest, "network_data"),
        "provenance.data_sha256",
    )
    require_equal(
        provenance.get("metadata_sha256"),
        expected_input_hash(manifest, "metadata"),
        "provenance.metadata_sha256",
    )
    require_equal(
        provenance.get("base_checkpoint_sha256"),
        expected_input_hash(manifest, "base_checkpoint"),
        "provenance.base_checkpoint_sha256",
    )
    source = provenance.get("source")
    if not isinstance(source, dict):
        fail("provenance.source is missing")
    require_equal(
        source.get("preflight_source_sha256"),
        args.source_sha256,
        "provenance.source.preflight_source_sha256",
    )

    require_exact_split(manifest, provenance.get("split"))

    require_equal(config.get("artifact_status"), "complete", "config.artifact_status")
    require_equal(config.get("forecast_horizon_weeks"), args.horizon, "config.horizon")
    require_equal(
        config.get("holdout_start"), EXPECTED_HOLDOUT_START, "config.holdout_start"
    )
    require_equal(
        config.get("validation_weeks"), EXPECTED_SPLIT_COUNTS["val"], "config.validation_weeks"
    )
    require_equal(config.get("epochs"), expected_epochs, "config.epochs")
    require_equal(config.get("seed"), 42, "config.seed")
    require_equal(
        config.get("network_snapshots"), EXPECTED_SNAPSHOT_COUNT, "config.network_snapshots"
    )
    require_equal(config.get("source_sha256"), args.source_sha256, "config.source_sha256")

    result = require_finite_metrics(metrics, args.horizon)
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "horizon": args.horizon,
                "completed_epochs": completed_epochs,
                "test_auprc": result["exist"]["auprc"],
                "test_auroc": result["exist"]["auroc"],
                "status": "verified",
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=(1, 4, 8, 12), required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--epochs", type=int, choices=(20, 40), default=20)
    args = parser.parse_args()
    if len(args.source_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in args.source_sha256.lower()
    ):
        fail("source SHA-256 is not 64 hexadecimal characters")
    verify(args)


if __name__ == "__main__":
    main()
