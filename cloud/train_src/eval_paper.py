#!/usr/bin/env python3
"""Evaluate pinned EVIEHub DeXposure-FM v1.0 weights on the 2025 holdout."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cloud" / "train_src"))
from tune_fm import (  # noqa: E402
    METRICS,
    PLAN_PATH as TUNE_PLAN_PATH,
    RELEASE_REL,
    TRAIN_REL,
    archive_sources,
    pairs_fingerprint,
    prepare_output_dirs,
    read_json,
    save_predictions,
    sha256,
    values,
    write_json,
)

PLAN_PATH = ROOT / "cloud/fm_eval_paper_plan.json"
PAPER_REVISION = "e0c9667dea5ef07df49ad8cfad413ba19f4ad00e"
DIRECTIONS = (1, 1, -1, -1, -1, -1)


def legacy_self_node_head(embed_dim: int, hidden_dim: int):
    import torch.nn as nn

    class LegacySelfNodeHead(nn.Module):
        """Paper h4/h8-h12 head: self embedding only, two linear layers."""

        def __init__(self, embed_dim: int, hidden_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, h, edge_src=None, edge_dst=None, edge_weight=None):
            return self.net(h).squeeze(-1)

    return LegacySelfNodeHead(embed_dim, hidden_dim)


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def detect_layout(state: dict) -> str:
    weight = state["node_head.net.0.weight"]
    has2 = "node_head.net.2.weight" in state
    has3 = "node_head.net.3.weight" in state
    has5 = "node_head.net.5.weight" in state
    width = int(weight.shape[1])
    if width == 576 and has3 and has5 and not has2:
        return "current"
    if width == 192 and has2 and not has3 and not has5:
        return "legacy_self"
    raise ValueError(
        f"Unsupported node head: shape={tuple(weight.shape)} has2={has2} has3={has3} has5={has5}"
    )


def model_state(payload):
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("Release checkpoint is not a {'model': state_dict} payload")
    state = payload["model"]
    if not isinstance(state, dict):
        raise ValueError("Release checkpoint model payload is not a state dict")
    return state


def compare(observed, printed):
    differences = [(v - p) * d for v, p, d in zip(observed, printed, DIRECTIONS)]
    return {
        "signed_improvements": differences,
        "all_six_above_printed": all(item > 0 for item in differences),
        "all_six_beyond_rounding": all(item > 0.0005 for item in differences),
    }


def validate_plan(plan):
    if plan.get("experiment_kind") != "paper_release_2025_holdout_eval":
        raise ValueError("Unreviewed eval experiment kind")
    if plan.get("paper_repo") != "EVIEHub/DeXposure-FM":
        raise ValueError("Paper repository is not EVIEHub/DeXposure-FM")
    if plan.get("paper_revision") != PAPER_REVISION:
        raise ValueError("Paper revision is not the pinned v1.0 release")
    for key, expected in (("horizons", [1, 4, 8, 12]), ("seed", 42),
                          ("holdout_start", "2025-01-01"), ("validation_weeks", 24),
                          ("neg_ratio", 5), ("max_train_seconds", 14400),
                          ("max_offer_dph", 0.81), ("max_gpu_memory_mib", 49152)):
        if plan.get(key) != expected:
            raise ValueError(f"Unreviewed eval contract: {key}")
    files = plan["paper_files"]
    if set(files) != {"1", "4", "8", "12"}:
        raise ValueError("Paper file contract must cover h1, h4, h8, and h12")
    if files["8"] != files["12"]:
        raise ValueError("h8 and h12 must share the single h8-h12 release file")
    if files["8"]["name"] != "dexposure-fm-h8-h12.pt":
        raise ValueError("h8/h12 source file must be dexposure-fm-h8-h12.pt")
    for spec in files.values():
        digest = spec["sha256"]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("Paper file SHA-256 is malformed")
    if sha256(ROOT / "cloud/train_src/run_full_experiment.py") != plan["core_sha256"]:
        raise ValueError("Training/evaluation core changed; review before eval")
    printed = read_json(TUNE_PLAN_PATH)["paper_printed"]
    if plan["paper_printed"] != printed:
        raise ValueError("Eval paper_printed does not match the tuning plan")
    search = plan["search_20260913"]
    if search.get("run_tag") != "20260913_fm_search_validation_only":
        raise ValueError("Search comparison run tag is not the completed 20260913 search")
    for horizon in plan["horizons"]:
        values({"exist": {"auprc": search["metrics"][str(horizon)][0],
                          "auroc": search["metrics"][str(horizon)][1]},
                "weight": {"mae": search["metrics"][str(horizon)][2],
                           "rmse": search["metrics"][str(horizon)][3]},
                "node": {"mae": search["metrics"][str(horizon)][4],
                         "rmse": search["metrics"][str(horizon)][5]}})


def fetch_paper_weights(plan, directory):
    from huggingface_hub import hf_hub_download
    directory.mkdir(parents=True, exist_ok=True)
    unique = {}
    for spec in plan["paper_files"].values():
        unique[spec["name"]] = spec["sha256"]
    for name, digest in unique.items():
        cached = hf_hub_download(
            plan["paper_repo"], name, revision=plan["paper_revision"],
            token=os.environ["HF_TOKEN"],
        )
        target = directory / name
        shutil.copyfile(cached, target)
        if sha256(target) != digest:
            raise ValueError(f"Paper weight SHA mismatch {name}")
        log(f"PAPER_WEIGHT {name} sha256={digest}")
    return unique


def load_release_model(core, config, state):
    import torch
    encoder = core.load_graphpfn_encoder(config.checkpoint_path, torch.device(config.device))
    embed_dim = encoder.tfm.embed_dim
    model = core.GraphPFNLinkPredictor(encoder, embed_dim, config.hidden_dim).to(config.device)
    layout = detect_layout(state)
    if layout == "legacy_self":
        model.node_head = legacy_self_node_head(embed_dim, config.hidden_dim).to(config.device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, layout


def write_eval_manifest(run_root):
    manifest = run_root / TRAIN_REL / "eval_manifest.sha256"
    files = sorted(
        path for base in (run_root / TRAIN_REL, run_root / RELEASE_REL)
        for path in base.rglob("*")
        if path.is_file() and path != manifest and not path.name.endswith(".tmp")
    )
    manifest.write_text("".join(
        f"{sha256(path)}  {path.relative_to(run_root).as_posix()}\n" for path in files
    ))


def verify_bundle(run_root, expected_source=None):
    run_root = Path(run_root).resolve()
    train, release = run_root / TRAIN_REL, run_root / RELEASE_REL
    plan = read_json(train / "plan.json")
    validate_plan(plan)
    result = read_json(train / "result.json")
    if result.get("status") != "complete" or result.get("experiment_kind") != plan["experiment_kind"]:
        raise ValueError("Eval result is incomplete")
    observed = set()
    for line in (train / "eval_manifest.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = (run_root / relative).resolve()
        if (not (path.is_relative_to(run_root / TRAIN_REL) or path.is_relative_to(run_root / RELEASE_REL))
                or relative in observed or not path.is_file() or sha256(path) != digest):
            raise ValueError(f"Invalid eval artifact {relative}")
        observed.add(relative)
    expected = {p.relative_to(run_root).as_posix() for base in (train, release)
                for p in base.rglob("*") if p.is_file() and p.name != "eval_manifest.sha256"
                and ".cache" not in p.parts}
    if expected != observed:
        raise ValueError(f"Eval manifest inventory mismatch: {expected ^ observed}")
    if expected_source is not None:
        for horizon in plan["horizons"]:
            cfg = read_json(release / f"run_config_h{horizon}.json")
            if cfg.get("source_sha256") != expected_source:
                raise ValueError("Wrong source digest")
    if sha256(release / "task1_metrics.json") != sha256(train / "finetuned" / "metrics.json"):
        raise ValueError("Release and finetuned metrics diverged")
    if sha256(release / "dexposure-fm-h8.pt") != sha256(release / "dexposure-fm-h12.pt"):
        raise ValueError("h8 and h12 copies of the shared release file diverged")
    if sha256(release / "dexposure-fm-h8.pt") != plan["paper_files"]["8"]["sha256"]:
        raise ValueError("Shared h8-h12 release copy SHA mismatch")
    for horizon, spec in plan["paper_files"].items():
        path = release / f"dexposure-fm-h{horizon}.pt"
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"Release copy SHA mismatch h={horizon}")
        if sha256(path) != sha256(train / "finetuned" / f"best_model_h{horizon}.pt"):
            raise ValueError(f"best_model copy SHA mismatch h={horizon}")
        config = read_json(release / f"run_config_h{horizon}.json")
        if config.get("paper_revision") != PAPER_REVISION:
            raise ValueError(f"run_config revision mismatch h={horizon}")
        if config.get("source_file") != spec["name"]:
            raise ValueError(f"run_config source_file mismatch h={horizon}")
        values(result["test"][f"h{horizon}"])
    if result["test"]["h8"] is result["test"]["h12"]:
        raise ValueError("h8 and h12 metrics must be separately computed")
    return {"status": "verified", "files": sum(1 for _ in train.rglob('*') if _.is_file()) +
            sum(1 for _ in release.rglob('*') if _.is_file()),
            "paper_revision": PAPER_REVISION}


def run():
    import torch
    train, release = prepare_output_dirs(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "cloud"))
    import run_full_experiment as core
    from verify_v2_checkpoint import require_exact_split
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    if not core.DGL_CUDA_AVAILABLE:
        raise RuntimeError("CUDA DGL unavailable; refusing paid CPU fallback")
    memory = torch.cuda.get_device_properties(0).total_memory
    if memory > plan["max_gpu_memory_mib"] * 1024 * 1024:
        raise RuntimeError("GPU exceeds 48 GiB ceiling")
    write_json(train / "plan.json", plan)
    archive_sources(train / "source.tar.gz")
    finetuned = train / "finetuned"
    config = core.ExperimentConfig(device="cuda", seed=plan["seed"])
    meta, categories, category_map = core.load_metadata(config.meta_path)
    schema = core.build_feature_schema(config.meta_path, categories)
    data = core.load_network_data(config.data_path)
    dates = sorted(data)
    split = core.get_single_split(dates, plan["holdout_start"], plan["validation_weeks"])
    require_exact_split(read_json(ROOT / "cloud/preflight_manifest.json"), split)
    snapshots = core.enrich_snapshots_with_historical_features([
        core.build_snapshot(d, data[d], meta, category_map, categories) for d in dates])
    quality = core.compute_data_quality(snapshots, data)
    write_json(finetuned / "data_quality.json", core.json_safe(quality))
    del data
    gc.collect()
    by_date = {s["date"]: s for s in snapshots}
    groups = {key: [by_date[d] for d in part] for key, part in split.items()}
    for directory in (release, finetuned):
        write_json(directory / "feature_schema.json", schema)
    write_json(train / "split.json", split)
    weights_dir = train / "paper_weights"
    fetch_paper_weights(plan, weights_dir)
    for horizon, spec in plan["paper_files"].items():
        source = weights_dir / spec["name"]
        for target in (release / f"dexposure-fm-h{horizon}.pt",
                       finetuned / f"best_model_h{horizon}.pt"):
            shutil.copyfile(source, target)
    test_results, comparison, fingerprints = {"model": "DeXposure-FM EVIEHub v1.0 e0c9667"}, {}, {}
    for horizon in plan["horizons"]:
        spec = plan["paper_files"][str(horizon)]
        payload = torch.load(weights_dir / spec["name"], map_location="cpu", weights_only=False)
        state = model_state(payload)
        model, layout = load_release_model(core, config, state)
        log(f"LOADED h={horizon} file={spec['name']} layout={layout}")
        pairs = core.build_week_pairs(groups["test"], config.neg_ratio, config.seed, horizon)
        fingerprints[str(horizon)] = {"test": pairs_fingerprint(pairs)}
        write_json(train / "pair_hashes.json", fingerprints)
        predictions = core.predict_graphpfn(model, pairs, config)
        metrics = core.evaluate_predictions(predictions)
        save_predictions(predictions, train / "predictions" / f"h{horizon}")
        observed = values(metrics)
        test_results[f"h{horizon}"] = metrics
        comparison[str(horizon)] = {
            "source_file": spec["name"],
            "layout": layout,
            "release_on_2025_test": observed,
            "paper_printed": plan["paper_printed"][str(horizon)],
            "search_20260913": plan["search_20260913"]["metrics"][str(horizon)],
            "vs_printed": compare(observed, plan["paper_printed"][str(horizon)]),
            "vs_search": compare(observed, plan["search_20260913"]["metrics"][str(horizon)]),
        }
        write_json(release / f"run_config_h{horizon}.json", {
            "artifact_status": "complete",
            "experiment_kind": plan["experiment_kind"],
            "forecast_horizon_weeks": horizon,
            "holdout_start": plan["holdout_start"],
            "validation_weeks": 24,
            "seed": 42,
            "network_snapshots": 283,
            "source_sha256": os.environ["PREFLIGHT_SOURCE_SHA256"],
            "paper_repo": plan["paper_repo"],
            "paper_revision": plan["paper_revision"],
            "source_file": spec["name"],
            "layout": layout,
            "shared_h8_h12_weights": spec["name"] == "dexposure-fm-h8-h12.pt",
        })
        log(f"PAPER_TEST h={horizon} metrics={observed} vs_printed={comparison[str(horizon)]['vs_printed']}")
        del model, payload, state, predictions, pairs
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "status": "complete",
        "experiment_kind": plan["experiment_kind"],
        "paper_repo": plan["paper_repo"],
        "paper_revision": plan["paper_revision"],
        "test": test_results,
        "comparison": comparison,
        "paper_all_24_better": all(row["vs_printed"]["all_six_beyond_rounding"]
                                   for row in comparison.values()),
        "limitations": [
            "single seed; no significance claim",
            "h8 and h12 evaluate the same released file dexposure-fm-h8-h12.pt",
            "printed paper values rounded to three decimals",
            "release checkpoints have no embedded split provenance",
        ],
    }
    write_json(train / "result.json", result)
    write_json(train / "comparison.json", comparison)
    write_json(release / "task1_metrics.json", test_results)
    write_json(finetuned / "metrics.json", test_results)
    write_json(finetuned / "all_results.json", result)
    write_json(finetuned / "experiment_results.json", {**result, "config": plan})
    (release / "SHA256SUMS").write_text("".join(
        f"{sha256(path)}  {path.name}\n" for path in sorted(release.iterdir()) if path.is_file()
    ))
    write_eval_manifest(ROOT)
    log(f"PAPER_EVAL_COMPLETE {verify_bundle(ROOT, os.environ['PREFLIGHT_SOURCE_SHA256'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--source-sha256")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_bundle(args.verify, args.source_sha256)))
    elif args.run:
        try:
            run()
        except Exception:
            failure = ROOT / TRAIN_REL / "failure.txt"
            failure.parent.mkdir(parents=True, exist_ok=True)
            failure.write_text(traceback.format_exc())
            raise
    else:
        parser.error("choose --run or --verify")


if __name__ == "__main__":
    main()
