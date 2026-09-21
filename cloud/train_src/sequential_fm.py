#!/usr/bin/env python3
"""Fixed-parameter sequential horizons, with matched model/Adam/RNG warm starts."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

# Shared data, atomic saves, training loop and hash archive; no duplicated model code.
try:
    from . import tune_fm as common
except ImportError:
    import tune_fm as common
ROOT = common.ROOT
PLAN_PATH = ROOT / "cloud/fm_sequential_plan.json"
TRAIN_REL, RELEASE_REL = common.TRAIN_REL, common.RELEASE_REL
read_json, write_json, sha256 = common.read_json, common.write_json, common.sha256
prepare_output_dirs, OutputDirectoryError = common.prepare_output_dirs, common.OutputDirectoryError
archive_sources, write_tree_manifest = common.archive_sources, common.write_tree_manifest
build_model, pairs_fingerprint = common.build_model, common.pairs_fingerprint
save_predictions, values, log = common.save_predictions, common.values, common.log


def validate_plan(plan):
    for key, expected in (("experiment_kind", "sequential_horizon_training"),
                          ("horizons", [1, 4, 8, 12]), ("seed", 42), ("epochs", 40),
                          ("early_stop_patience", 5), ("neg_ratio", 5),
                          ("holdout_start", "2025-01-01"), ("validation_weeks", 24),
                          ("rank_mode", "validation_auprc")):
        if plan.get(key) != expected:
            raise ValueError(f"Unreviewed sequential contract: {key}")
    if "stages" in plan:
        raise ValueError("Per-horizon parameter overrides are not allowed")
    expected = {"name": "baseline", "lr": 0.0005, "encoder_lr": 0.00005,
                "exist_loss_weight": 2.0, "weight_loss_weight": 0.5, "node_loss_weight": 20.0}
    if plan.get("parameters") != expected:
        raise ValueError("Unreviewed shared parameter selection")
    if sha256(ROOT / "cloud/train_src/run_full_experiment.py") != plan["core_sha256"]:
        raise ValueError("Training/evaluation core changed")


def verify_bundle(run_root, expected_source=None):
    root = Path(run_root).resolve()
    train, release = root / TRAIN_REL, root / RELEASE_REL
    plan = read_json(train / "plan.json")
    validate_plan(plan)
    selection = read_json(train / "selection_frozen.json")
    result = read_json(train / "result.json")
    if result.get("status") != "complete" or result.get("experiment_kind") != plan["experiment_kind"]:
        raise ValueError("Sequential campaign incomplete")
    if (selection["selection_data"] != "pre2025_validation_only"
            or selection["plan_sha256"] != sha256(train / "plan.json")
            or result["selection_sha256"] != sha256(train / "selection_frozen.json")
            or set(selection["selected"]) != {"1", "4", "8", "12"}):
        raise ValueError("Selection freeze mismatch")
    observed = set()
    for line in (train / "tuning_manifest.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if (not (path.is_relative_to(train) or path.is_relative_to(release))
                or relative in observed or not path.is_file() or sha256(path) != digest):
            raise ValueError(f"Invalid sequential artifact {relative}")
        observed.add(relative)
    expected = {p.relative_to(root).as_posix() for base in (train, release) for p in base.rglob("*")
                if p.is_file() and p.name != "tuning_manifest.sha256" and ".cache" not in p.parts}
    if expected != observed:
        raise ValueError("Sequential artifact inventory mismatch")
    parent_sha, inherited_epochs = None, 0
    for horizon in plan["horizons"]:
        chosen = selection["selected"][str(horizon)]
        candidate = plan["parameters"]
        directory = train / "trials" / f"h{horizon}" / candidate["name"]
        summary, history = read_json(directory / "summary.json"), read_json(directory / "history.json")
        if (summary["status"] != "complete" or summary["candidate"] != candidate["name"]
                or not 1 <= len(history) <= plan["epochs"]
                or summary["completed_epochs"] != len(history)
                or [row["epoch"] for row in history] != list(range(1, len(history) + 1))):
            raise ValueError(f"Incomplete stage h={horizon}")
        best = max(history, key=lambda row: common.rank_metrics(row["validation"], None, plan, horizon))
        if (summary["selected_epoch"] != best["epoch"]
                or (len(history) < plan["epochs"] and len(history) - best["epoch"] != plan["early_stop_patience"])
                or summary["parent_sha256"] != parent_sha
                or summary["cumulative_selected_epochs"] != inherited_epochs + best["epoch"]):
            raise ValueError(f"Invalid stage selection or parent h={horizon}")
        model_sha = sha256(directory / f"best_model_h{horizon}.pt")
        if (chosen != {**summary, "model_sha256": model_sha}
                or model_sha != sha256(release / f"dexposure-fm-h{horizon}.pt")
                or model_sha != sha256(train / "finetuned" / f"best_model_h{horizon}.pt")):
            raise ValueError("Selected weights changed")
        cfg = read_json(release / f"run_config_h{horizon}.json")
        if (cfg["experiment_kind"] != plan["experiment_kind"] or cfg["candidate"] != candidate
                or cfg["parent_sha256"] != parent_sha
                or (expected_source is not None and cfg["source_sha256"] != expected_source)):
            raise ValueError("Run config provenance mismatch")
        for row in history:
            values(row["validation"])
        values(result["test"][f"h{horizon}"])
        if len(list((train / "predictions" / f"h{horizon}").glob("week_*.npz"))) != 33 - horizon:
            raise ValueError("Missing prediction weeks")
        parent_sha, inherited_epochs = model_sha, summary["cumulative_selected_epochs"]
    return {"status": "verified", "files": len(observed), "horizons": plan["horizons"]}


def run():
    existing = ROOT / TRAIN_REL / "plan.json"
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    if existing.exists() and read_json(existing) != plan:
        raise OutputDirectoryError("Existing run has a different plan")
    train, release = prepare_output_dirs(ROOT, allow_existing=existing.exists())
    if (train / "result.json").exists():
        verify_bundle(ROOT, os.environ["PREFLIGHT_SOURCE_SHA256"])
        log("SEQUENTIAL_ALREADY_COMPLETE")
        return
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "cloud"))
    import numpy as np
    import torch
    import run_full_experiment as core
    from verify_v2_checkpoint import require_exact_split
    if not core.DGL_CUDA_AVAILABLE:
        raise RuntimeError("CUDA DGL unavailable; refusing paid CPU fallback")
    memory = torch.cuda.get_device_properties(0).total_memory
    if memory > plan["max_gpu_memory_mib"] * 1024 * 1024:
        raise RuntimeError("GPU exceeds planned memory ceiling")
    write_json(train / "plan.json", plan)
    archive_sources(train / "source.tar.gz")
    finetuned = train / "finetuned"
    config = core.ExperimentConfig(device="cuda", epochs=plan["epochs"], seed=plan["seed"], early_stop_patience=plan["early_stop_patience"])
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
    fingerprints, selected = {}, {}
    parent = None
    for horizon in plan["horizons"]:
        candidate = plan["parameters"]
        directory = train / "trials" / f"h{horizon}" / candidate["name"]
        source = directory / f"best_model_h{horizon}.pt"
        summary_path = directory / "summary.json"
        parent_sha = sha256(parent) if parent is not None else None
        if summary_path.exists():
            completed = read_json(summary_path)
            if completed.get("status") != "complete" or completed.get("parent_sha256") != parent_sha:
                raise ValueError("Existing stage summary does not match the completed parent")
            fingerprints = read_json(train / "pair_hashes.json")
            log(f"HORIZON_SKIP_COMPLETE h={horizon}")
        else:
            log(f"BUILD_PAIRS h={horizon} parent_sha256={parent_sha}")
            train_pairs = core.build_week_pairs(groups["train"], config.neg_ratio, config.seed, horizon)
            val_pairs = core.build_week_pairs(groups["val"], config.neg_ratio, config.seed, horizon)
            fingerprints[str(horizon)] = {"train": pairs_fingerprint(train_pairs), "val": pairs_fingerprint(val_pairs)}
            write_json(train / "pair_hashes.json", fingerprints)
            completed = common.train_trial(core, config, candidate, horizon, train_pairs, val_pairs,
                None, split, schema, directory, plan["epochs"], plan,
                resume=(directory / f"resume_h{horizon}.partial.pt").exists(), parent=parent)
            del train_pairs, val_pairs
            gc.collect()
        for target in (release / f"dexposure-fm-h{horizon}.pt", finetuned / f"best_model_h{horizon}.pt"):
            shutil.copyfile(source, target)
        selected[str(horizon)] = {**completed, "model_sha256": sha256(source)}
        write_json(release / f"run_config_h{horizon}.json", {
            "artifact_status": "complete", "experiment_kind": plan["experiment_kind"],
            "forecast_horizon_weeks": horizon, "holdout_start": plan["holdout_start"],
            "validation_weeks": 24, "epochs": plan["epochs"], "early_stop_patience": 5,
            "seed": 42, "network_snapshots": len(dates),
            "source_sha256": os.environ["PREFLIGHT_SOURCE_SHA256"], "candidate": candidate,
            "selected_epoch": completed["selected_epoch"], "completed_epochs": completed["completed_epochs"],
            "parent_sha256": parent_sha, "cumulative_selected_epochs": completed["cumulative_selected_epochs"],
            "warm_start": "matched_best_model_optimizer_scaler_rng" if parent else "base_graphpfn_pretrained_encoder",
        })
        parent = source
    freeze = {"selected": selected, "frozen_at": datetime.now(timezone.utc).isoformat(),
              "selection_data": "pre2025_validation_only", "plan_sha256": sha256(train / "plan.json")}
    write_json(train / "selection_frozen.json", freeze)
    selection_sha = sha256(train / "selection_frozen.json")
    log(f"ALL_FOUR_SELECTIONS_FROZEN sha256={selection_sha}; final holdout evaluation begins")
    test_results, comparison = {"model": "DeXposure-FM sequential h1-h4-h8-h12"}, {}
    for horizon in plan["horizons"]:
        model = build_model(core, config)
        state = torch.load(release / f"dexposure-fm-h{horizon}.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        pairs = core.build_week_pairs(groups["test"], config.neg_ratio, config.seed, horizon)
        fingerprints[str(horizon)]["test"] = pairs_fingerprint(pairs)
        write_json(train / "pair_hashes.json", fingerprints)
        predictions = core.predict_graphpfn(model, pairs, config)
        metrics = core.evaluate_predictions(predictions)
        save_predictions(predictions, train / "predictions" / f"h{horizon}")
        test_results[f"h{horizon}"] = metrics
        directions = (1, 1, -1, -1, -1, -1)
        differences = [(v - p) * d for v, p, d in zip(values(metrics), plan["paper_printed"][str(horizon)], directions)]
        comparison[str(horizon)] = {"signed_improvements": differences,
                                    "all_six_above_printed": all(d > 0 for d in differences),
                                    "all_six_beyond_rounding": all(d > 0.0005 for d in differences)}
        log(f"FINAL_TEST h={horizon} metrics={values(metrics)} comparison={comparison[str(horizon)]}")
        del model, state, predictions, pairs
        gc.collect()
        torch.cuda.empty_cache()
    if sha256(train / "selection_frozen.json") != selection_sha:
        raise ValueError("Selection changed during test evaluation")
    result = {"status": "complete", "experiment_kind": plan["experiment_kind"], "selection_sha256": selection_sha,
              "test": test_results, "paper_comparison": comparison,
              "paper_all_24_better": all(v["all_six_beyond_rounding"] for v in comparison.values()),
              "limitations": ["single seed; no significance claim", "2025 benchmark previously inspected",
                              "Task II not evaluated", "printed paper values rounded to three decimals",
                              "new sequential experiment; not an exact replay of the historical paper"]}
    write_json(train / "result.json", result)
    write_json(release / "task1_metrics.json", test_results)
    write_json(finetuned / "metrics.json", test_results)
    write_json(finetuned / "all_results.json", result)
    write_json(finetuned / "experiment_results.json", {**result, "config": plan, "selected": selected})
    (release / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in sorted(release.iterdir()) if p.is_file()))
    write_tree_manifest(ROOT)
    log(f"SEQUENTIAL_COMPLETE {verify_bundle(ROOT, os.environ['PREFLIGHT_SOURCE_SHA256'])}")


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
        except OutputDirectoryError:
            # The outer runner captures stderr; leave the rejected directory untouched.
            raise
        except BaseException:
            failure = ROOT / TRAIN_REL / "failure.txt"
            failure.parent.mkdir(parents=True, exist_ok=True)
            failure.write_text(traceback.format_exc())
            raise
    else:
        parser.error("choose --run or --verify")


if __name__ == "__main__":
    main()
