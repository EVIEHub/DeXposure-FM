from __future__ import annotations

import argparse
import importlib.util
from datetime import date, timedelta
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
VERIFY_PATH = ROOT / "cloud" / "verify_v2_checkpoint.py"


@pytest.fixture()
def verifier():
    spec = importlib.util.spec_from_file_location("verify_v2_checkpoint_contract", VERIFY_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def real_split() -> dict[str, list[str]]:
    dates = [str(date(2020, 3, 23) + timedelta(days=7 * index)) for index in range(283)]
    return {"train": dates[:226], "val": dates[226:250], "test": dates[250:]}


def payloads(verifier, split: dict[str, list[str]]):
    source = "a" * 64
    input_hashes = {
        "network_data": "1" * 64,
        "metadata": "2" * 64,
        "base_checkpoint": "3" * 64,
    }
    manifest = {
        "inputs": {
            "local": [
                {
                    "name": name,
                    "sha256": digest,
                    **(
                        {
                            "temporal_split": {
                                "algorithm": verifier.SPLIT_DIGEST_ALGORITHM,
                                "snapshot_count": 283,
                                "holdout_start": "2025-01-01",
                                "first_date": "2020-03-23",
                                "last_date": "2025-08-18",
                                "counts": {"train": 226, "val": 24, "test": 33},
                                "dates_sha256": verifier.canonical_split_sha256(real_split()),
                            }
                        }
                        if name == "network_data"
                        else {}
                    ),
                }
                for name, digest in input_hashes.items()
            ]
        }
    }
    checkpoint = {
        "artifact_status": "complete",
        "horizon": 1,
        "model": {"weight": 1},
        "provenance": {
            "horizon": 1,
            "seed": 42,
            "epochs": 20,
            "completed_epochs": 1,
            "data_sha256": input_hashes["network_data"],
            "metadata_sha256": input_hashes["metadata"],
            "base_checkpoint_sha256": input_hashes["base_checkpoint"],
            "source": {"preflight_source_sha256": source},
            "split": split,
        },
    }
    metrics = {
        "h1": {
            "exist": {"auprc": 0.5, "auroc": 0.5},
            "weight": {"mae": 1.0, "rmse": 1.0, "weighted_mae": 1.0},
            "node": {"mae": 1.0, "rmse": 1.0},
        }
    }
    config = {
        "artifact_status": "complete",
        "forecast_horizon_weeks": 1,
        "holdout_start": "2025-01-01",
        "validation_weeks": 24,
        "epochs": 20,
        "seed": 42,
        "network_snapshots": 283,
        "source_sha256": source,
    }
    args = argparse.Namespace(
        manifest=Path("manifest"),
        metrics=Path("metrics"),
        config=Path("config"),
        checkpoint=Path("checkpoint"),
        horizon=1,
        source_sha256=source,
    )
    return args, manifest, metrics, config, checkpoint


def run_verify(monkeypatch, verifier, split):
    args, manifest, metrics, config, checkpoint = payloads(verifier, split)
    documents = {"manifest": manifest, "metrics": metrics, "config": config}
    monkeypatch.setattr(verifier, "load_json", lambda path: documents[path.name])
    monkeypatch.setattr(verifier.torch, "load", lambda *args, **kwargs: checkpoint)
    verifier.verify(args)


def test_exact_v2_split_is_accepted(monkeypatch, verifier, capsys):
    run_verify(monkeypatch, verifier, real_split())
    assert '"status": "verified"' in capsys.readouterr().out


def test_extended_epochs_require_explicit_verifier_contract(monkeypatch, verifier):
    args, manifest, metrics, config, checkpoint = payloads(verifier, real_split())
    config["epochs"] = 40
    checkpoint["provenance"]["epochs"] = 40
    checkpoint["provenance"]["completed_epochs"] = 32
    documents = {"manifest": manifest, "metrics": metrics, "config": config}
    monkeypatch.setattr(verifier, "load_json", lambda path: documents[path.name])
    monkeypatch.setattr(verifier.torch, "load", lambda *args, **kwargs: checkpoint)
    with pytest.raises(SystemExit, match="provenance.epochs"):
        verifier.verify(args)
    args.epochs = 40
    verifier.verify(args)


def test_wrong_split_counts_are_rejected(monkeypatch, verifier):
    split = real_split()
    split["train"] = split["train"][:-1]
    split["test"] = split["test"] + ["2025-08-25"]
    with pytest.raises(SystemExit, match="train snapshot count expected 226, got 225"):
        run_verify(monkeypatch, verifier, split)


def test_forged_interior_date_with_correct_counts_is_rejected(monkeypatch, verifier):
    split = real_split()
    split["test"][10] = "2025-03-18"
    with pytest.raises(SystemExit, match="provenance split dates SHA-256"):
        run_verify(monkeypatch, verifier, split)


def test_manifest_records_real_main_data_split_digest(verifier):
    import json

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    contract = next(
        item for item in manifest["inputs"]["local"] if item["name"] == "network_data"
    )["temporal_split"]
    assert contract["counts"] == {"train": 226, "val": 24, "test": 33}
    assert contract["dates_sha256"] == verifier.canonical_split_sha256(real_split())
