#!/usr/bin/env python3
"""Bounded, validation-only FM tuning. Reuses the unchanged training/evaluator core."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "cloud/fm_tune_plan.json"
TRAIN_REL = Path("checkpoints/main2025_v2_hremaining_train")
RELEASE_REL = Path("checkpoints/main2025_v2_hremaining")
METRICS = (("exist", "auprc"), ("exist", "auroc"), ("weight", "mae"),
           ("weight", "rmse"), ("node", "mae"), ("node", "rmse"))


class OutputDirectoryError(RuntimeError):
    """Output ownership is unproven; do not write even a failure log there."""


def has_trial_artifacts(root):
    train = root / TRAIN_REL
    return any(train.glob("trials/**/*.json")) or any(train.glob("trials/**/*.pt"))


def load_screen_summaries(train, horizon, plan):
    summaries = []
    for candidate in plan["candidates"]:
        path = train / "trials" / f"h{horizon}" / candidate["name"] / "summary.json"
        if not path.is_file():
            return None
        summary = read_json(path)
        if (summary.get("candidate") != candidate["name"]
                or summary.get("horizon") != horizon
                or int(summary.get("completed_epochs") or 0) < plan["screen_epochs"]):
            return None
        summaries.append(summary)
    return summaries


def prepare_output_dirs(root, allow_existing=False):
    """Allow preflight's directory-only scaffolding, never unexpected artifacts."""
    train, release = root / TRAIN_REL, root / RELEASE_REL
    try:
        parent = train.parent
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise OutputDirectoryError(f"Unsafe output parent: {parent}")
        pending = [p for p in (train, release) if p.exists() or p.is_symlink()]
        while pending:
            path = pending.pop()
            if path.is_symlink():
                raise OutputDirectoryError(f"Existing artifact or unsafe output path: {path}")
            if path.is_dir():
                pending.extend(path.iterdir())
            elif not allow_existing:
                raise OutputDirectoryError(f"Existing artifact or unsafe output path: {path}")
        for path in (train, release, train / "finetuned"):
            path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OutputDirectoryError(f"Cannot safely prepare output directories: {error}") from error
    return train, release


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def values(metrics):
    result = [float(metrics[g][m]) for g, m in METRICS]
    if not all(math.isfinite(v) for v in result):
        raise ValueError("Non-finite required metric")
    return result


def validation_rank(metrics, reference):
    """No test or paper values accepted. Protect the worst of all six metrics."""
    current, baseline = values(metrics), values(reference)
    for i in (0, 1):
        current[i], baseline[i] = 1 - current[i], 1 - baseline[i]
    gains = [(b - v) / max(b, 1e-8) for v, b in zip(current, baseline)]
    return min(gains), sum(gains) / len(gains)


def paper_surplus_rank(metrics, printed):
    """Validation only. Prefer beating the printed table, then the weakest gap, then AUPRC."""
    directions = (1, 1, -1, -1, -1, -1)
    diffs = [(v - p) * d for v, p, d in zip(values(metrics), printed, directions)]
    return (sum(d > 0.0005 for d in diffs), min(diffs), diffs[0])


def rank_metrics(metrics, reference, plan, horizon):
    mode = plan.get("rank_mode", "validation_vs_baseline")
    if mode == "validation_auprc":
        values(metrics)
        return metrics["exist"]["auprc"], metrics["exist"]["auroc"]
    if mode == "paper_surplus":
        return paper_surplus_rank(metrics, plan["paper_printed"][str(horizon)])
    if mode == "validation_vs_baseline":
        return validation_rank(metrics, reference)
    raise ValueError(f"Unreviewed rank_mode: {mode}")


def require_agreement(observed, expected, plan):
    differences = {}
    for (group, metric), actual, target in zip(METRICS, values(observed), values(expected)):
        differences[f"{group}.{metric}"] = actual - target
        atol = (plan.get("regression_agreement_atol", plan["agreement_atol"])
                if group in {"weight", "node"} else plan["agreement_atol"])
        if not math.isclose(actual, target, abs_tol=atol,
                            rel_tol=plan["agreement_rtol"]):
            raise ValueError(f"Baseline evaluation mismatch {group}.{metric}: {actual} vs {target} "
                             f"(atol={atol}, rtol={plan['agreement_rtol']})")
    return differences


def validate_plan(plan):
    for key, expected in (("horizons", [1, 4, 8, 12]), ("seed", 42),
                          ("epochs", 20), ("screen_epochs", 6), ("neg_ratio", 5),
                          ("holdout_start", "2025-01-01"), ("validation_weeks", 24),
                          ("baseline_policy", "validation_reference_only"),
                          ("agreement_atol", 1e-5), ("agreement_rtol", 1e-5),
                          ("regression_agreement_atol", 1e-4)):
        if plan.get(key) != expected:
            raise ValueError(f"Unreviewed tuning contract: {key}")
    if plan.get("rank_mode") not in {"validation_vs_baseline", "paper_surplus"}:
        raise ValueError("Unreviewed tuning contract: rank_mode")
    candidates = plan["candidates"]
    if len(candidates) != 5 or len({c["name"] for c in candidates}) != 5:
        raise ValueError("Expected five unique candidates")
    for candidate in candidates:
        if not candidate["name"].replace("_", "").isalnum():
            raise ValueError("Unsafe candidate name")
        for key in ("lr", "encoder_lr", "exist_loss_weight", "weight_loss_weight", "node_loss_weight"):
            if not math.isfinite(candidate[key]) or candidate[key] <= 0:
                raise ValueError(f"Invalid candidate {key}")
    if sha256(ROOT / "cloud/train_src/run_full_experiment.py") != plan["core_sha256"]:
        raise ValueError("Training/evaluation core changed; review before tuning")


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def pairs_fingerprint(pairs):
    digest = hashlib.sha256()
    for pair in pairs:
        digest.update(json.dumps([pair.time_t, pair.time_t1, pair.node_ids], separators=(",", ":")).encode())
        for key in ("pair_src", "pair_dst", "y_exist", "y_weight", "weight_mask", "y_node", "node_mask"):
            array = getattr(pair, key)
            digest.update(key.encode())
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def save_predictions(predictions, directory):
    """Per-example binary arrays avoid the existing all-rows-in-RAM CSV writer."""
    import numpy as np
    directory.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(predictions):
        arrays = {key: np.asarray(value) for key, value in row.items()}
        if any(value.dtype.kind == "O" for value in arrays.values()):
            raise ValueError("Prediction archives must not contain pickled object arrays")
        np.savez_compressed(directory / f"week_{index:03d}.npz", **arrays)


def atomic_torch_save(value, path):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def build_model(core, config):
    import torch
    core.set_seed(config.seed)
    encoder = core.load_graphpfn_encoder(config.checkpoint_path, torch.device(config.device))
    core.set_seed(config.seed)
    return core.GraphPFNLinkPredictor(encoder, encoder.tfm.embed_dim, config.hidden_dim).to(config.device)


def rng_state():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def train_trial(core, config, candidate, horizon, train_pairs, val_pairs, reference,
                split, schema, directory, stop_epoch, plan, resume=False, parent=None):
    """Only train/validation pairs enter this function; test inference is separate."""
    import torch
    config = replace(config, forecast_horizons=[horizon],
                     **{k: v for k, v in candidate.items() if k not in {"name", "encoder_lr"}})
    model = build_model(core, config)
    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": candidate["encoder_lr"]},
        {"params": [p for p in model.parameters() if id(p) not in encoder_ids], "lr": config.lr},
    ], weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
    last_path = directory / f"resume_h{horizon}.partial.pt"
    best_path = directory / f"best_model_h{horizon}.pt"
    history_path = directory / "history.json"
    history, best_rank, best_epoch, first_epoch = [], (-float("inf"), -float("inf")), 0, 1
    sequential = plan.get("experiment_kind") == "sequential_horizon_training"
    parent_sha = sha256(parent) if parent is not None else None
    inherited_epochs = 0
    if parent is not None:
        saved = torch.load(parent, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        inherited_epochs = saved["cumulative_selected_epochs"]
        # Keep Adam moments, but apply the next stage's predeclared learning rates.
        for group, lr in zip(optimizer.param_groups, (candidate["encoder_lr"], config.lr)):
            group["lr"] = lr
        del saved
    if resume:
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if sequential and saved.get("parent_sha256") != parent_sha:
            raise ValueError("Resume checkpoint belongs to a different stage parent")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        best_rank, best_epoch = tuple(saved["best_rank"]), saved["best_epoch"]
        first_epoch = saved["epoch"] + 1
        history = (saved["history"] if sequential else read_json(history_path)[:saved["epoch"]])
        if sequential:
            # The epoch checkpoint commits model, best model, and history together.
            # Recover even when interruption happened between the separate file writes.
            atomic_torch_save(saved["best_snapshot"], best_path)
            write_json(history_path, history)
        del saved
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", {**asdict(config), **candidate})
    provenance = core.build_checkpoint_provenance(
        data_path=config.data_path, metadata_path=config.meta_path, split=split,
        seed=config.seed, epochs=config.epochs, completed_epochs=first_epoch,
        horizon=horizon, feature_schema=schema, best_validation_metric=0.0,
        best_validation_metric_name=("auprc" if sequential else "six_metric_validation_maximin"),
        checkpoint_path=config.checkpoint_path,
    )
    log(f"TRIAL_START h={horizon} candidate={candidate['name']} epochs={first_epoch}-{stop_epoch}")
    for epoch in range(first_epoch, stop_epoch + 1):
        if sequential and best_epoch and epoch - 1 - best_epoch >= plan["early_stop_patience"]:
            break
        log(f"EPOCH_START h={horizon} candidate={candidate['name']} epoch={epoch}/{config.epochs}")
        started = time.monotonic()
        losses, _ = core.train_graphpfn_epoch(model, train_pairs, optimizer, config, True, scaler=scaler)
        if not all(math.isfinite(float(v)) for v in losses.values()):
            raise RuntimeError("Non-finite training loss")
        metrics = core.evaluate_predictions(core.predict_graphpfn(model, val_pairs, config))
        rank = rank_metrics(metrics, reference, plan, horizon)
        record = {"epoch": epoch, "losses": losses, "validation": metrics,
                  "rank": list(rank), "seconds": time.monotonic() - started,
                  "learning_rates": [g["lr"] for g in optimizer.param_groups],
                  "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated()}
        history.append(record)
        write_json(history_path, history)
        if rank > best_rank:
            best_rank, best_epoch = rank, epoch
            provenance.update(completed_epochs=epoch, best_validation_metric=rank[0])
            continuation = ({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                             "rng": rng_state(), "parent_sha256": parent_sha,
                             "cumulative_selected_epochs": inherited_epochs + epoch}
                            if sequential else {})
            atomic_torch_save({"artifact_status": "in_progress", "model": model.state_dict(),
                               "horizon": horizon, "feature_schema": schema,
                               "provenance": copy.deepcopy(provenance),
                               "training_config": {**asdict(config), **candidate},
                               "selected_epoch": epoch, "validation_metrics": metrics,
                               **continuation}, best_path)
        recovery = ({"history": history,
                     "best_snapshot": torch.load(best_path, map_location="cpu", weights_only=False)}
                    if sequential else {})
        atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scaler": scaler.state_dict(), "rng": rng_state(), "epoch": epoch,
                           "best_rank": best_rank, "best_epoch": best_epoch,
                           "parent_sha256": parent_sha, **recovery}, last_path)
        del recovery
        log(f"TRIAL h={horizon} candidate={candidate['name']} epoch={epoch}/{config.epochs} "
            f"validation={values(metrics)} rank={rank} losses={losses} seconds={record['seconds']:.1f}")
    completed_epochs = history[-1]["epoch"]
    summary = {"candidate": candidate["name"], "horizon": horizon, "completed_epochs": completed_epochs,
               "selected_epoch": best_epoch, "rank": list(best_rank), "status": "screened"}
    if sequential:
        summary.update(parent_sha256=parent_sha, cumulative_selected_epochs=inherited_epochs + best_epoch)
    if stop_epoch == config.epochs:
        saved = torch.load(best_path, map_location="cpu", weights_only=False)
        saved["artifact_status"] = "complete"
        saved["provenance"]["completed_epochs"] = completed_epochs
        atomic_torch_save(saved, best_path)
        summary["status"] = "complete"
        del saved
    write_json(directory / "summary.json", summary)
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def fetch_baselines(plan, target):
    from huggingface_hub import hf_hub_download
    target.mkdir(parents=True, exist_ok=True)
    for name in [f"dexposure-fm-h{h}.pt" for h in plan["horizons"]] + ["task1_metrics.json"]:
        cached = hf_hub_download(plan["baseline_repo"], plan["baseline_prefix"] + "/" + name,
                                 revision=plan["baseline_revision"], token=os.environ["HF_TOKEN"])
        shutil.copyfile(cached, target / name)
    for horizon in plan["horizons"]:
        if sha256(target / f"dexposure-fm-h{horizon}.pt") != plan["baseline_sha256"][str(horizon)]:
            raise ValueError(f"Baseline checkpoint SHA mismatch h={horizon}")


def archive_sources(path):
    # Only code paths explicitly listed by the preflight contract; no .env or credentials.
    manifest = read_json(ROOT / "cloud/preflight_manifest.json")
    paths = set()
    for pattern in manifest["source"]["paths"]:
        paths.update(p for p in ROOT.glob(pattern) if p.is_file())
    paths.update(p for p in (ROOT / "lib").rglob("*.py") if p.is_file())
    with tarfile.open(path, "w:gz") as archive:
        for file in sorted(paths):
            archive.add(file, arcname=file.relative_to(ROOT), recursive=False)


def write_tree_manifest(run_root):
    manifest = run_root / TRAIN_REL / "tuning_manifest.sha256"
    files = sorted(p for base in (run_root / TRAIN_REL, run_root / RELEASE_REL)
                   for p in base.rglob("*") if p.is_file() and p != manifest and not p.name.endswith(".tmp"))
    manifest.write_text("".join(f"{sha256(p)}  {p.relative_to(run_root).as_posix()}\n" for p in files))


def verify_baseline_evidence(train, plan):
    """Keep old replay gates strict; new searches require validation references only."""
    gate = read_json(train / "baseline_gate.json")
    horizons = {str(h) for h in plan["horizons"]}
    if plan.get("baseline_policy") == "validation_reference_only":
        if gate != {"status": "not_run", "policy": "validation_reference_only",
                    "reason": "historical_test_agreement_not_required_for_new_search"}:
            raise ValueError("New search must not claim historical replay passed")
        references = read_json(train / "baseline_validation.json")
        if set(references) != horizons:
            raise ValueError("Incomplete validation references")
        for horizon in horizons:
            values(references[horizon])
            if sha256(train / "baseline" / f"dexposure-fm-h{horizon}.pt") != plan["baseline_sha256"][horizon]:
                raise ValueError(f"Baseline checkpoint SHA mismatch h={horizon}")
    elif "baseline_policy" not in plan:
        if gate.get("status") != "passed" or set(gate.get("horizons", {})) != horizons:
            raise ValueError("Baseline agreement gate did not pass")
        for row in gate["horizons"].values():
            require_agreement(row["metrics"], row["expected"], plan)
    else:
        raise ValueError("Unknown baseline policy")


def verify_bundle(run_root, expected_source=None):
    """Hash every trial/prediction/source file, not only the four published candidates."""
    run_root = Path(run_root).resolve()
    train = run_root / TRAIN_REL
    plan = read_json(train / "plan.json")
    selection = read_json(train / "selection_frozen.json")
    result = read_json(train / "result.json")
    if result.get("status") != "complete" or set(selection["selected"]) != {str(h) for h in plan["horizons"]}:
        raise ValueError("Incomplete tuning campaign")
    verify_baseline_evidence(train, plan)
    if sha256(train / "selection_frozen.json") != result["selection_sha256"]:
        raise ValueError("Selection changed after freeze")
    observed = set()
    for line in (train / "tuning_manifest.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = (run_root / relative).resolve()
        if (not (path.is_relative_to(run_root / TRAIN_REL) or path.is_relative_to(run_root / RELEASE_REL))
                or relative in observed or not path.is_file() or sha256(path) != digest):
            raise ValueError(f"Invalid tuning artifact {relative}")
        observed.add(relative)
    expected = {p.relative_to(run_root).as_posix() for base in (train, run_root / RELEASE_REL)
                for p in base.rglob("*") if p.is_file() and p.name != "tuning_manifest.sha256"
                and ".cache" not in p.parts}
    if expected != observed:
        raise ValueError(f"Tuning manifest inventory mismatch: {expected ^ observed}")
    for horizon in plan["horizons"]:
        selected = selection["selected"][str(horizon)]
        candidate = selected["candidate"]
        if candidate not in {c["name"] for c in plan["candidates"]}:
            raise ValueError("Unplanned candidate selected")
        for planned in plan["candidates"]:
            trial_dir = train / "trials" / f"h{horizon}" / planned["name"]
            trial_summary = read_json(trial_dir / "summary.json")
            history = read_json(trial_dir / "history.json")
            completed_epochs = plan["epochs"] if planned["name"] == candidate else plan["screen_epochs"]
            if (trial_summary["completed_epochs"] != completed_epochs or
                    [entry["epoch"] for entry in history] != list(range(1, completed_epochs + 1))):
                raise ValueError(f"Incomplete planned trial h={horizon} {planned['name']}")
            for entry in history:
                values(entry["validation"])
        directory = train / "trials" / f"h{horizon}" / candidate
        summary = read_json(directory / "summary.json")
        if summary["completed_epochs"] != plan["epochs"] or summary["status"] != "complete":
            raise ValueError(f"Selected trial incomplete h={horizon}")
        model_sha = sha256(directory / f"best_model_h{horizon}.pt")
        if model_sha != selected["model_sha256"] or model_sha != sha256(run_root / RELEASE_REL / f"dexposure-fm-h{horizon}.pt"):
            raise ValueError(f"Selected checkpoint changed h={horizon}")
        cfg = read_json(run_root / RELEASE_REL / f"run_config_h{horizon}.json")
        if expected_source is not None and cfg["source_sha256"] != expected_source:
            raise ValueError("Wrong source digest")
        if cfg["experiment_kind"] != "new_hyperparameter_search_not_original_v2":
            raise ValueError("Wrong experiment identity")
        values(result["test"][f"h{horizon}"])
        expected_files = 33 - horizon
        prediction_directories = (("predictions",) if plan.get("baseline_policy") == "validation_reference_only"
                                  else ("baseline_predictions", "predictions"))
        for subdirectory in prediction_directories:
            if len(list((train / subdirectory / f"h{horizon}").glob("week_*.npz"))) != expected_files:
                raise ValueError(f"Prediction weeks missing h={horizon} {subdirectory}")
    return {"status": "verified", "files": len(observed), "paper_all_24_better": result["paper_all_24_better"]}


def run():
    train, release = prepare_output_dirs(ROOT, allow_existing=has_trial_artifacts(ROOT))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "cloud"))
    import numpy as np
    import torch
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
    config = core.ExperimentConfig(device="cuda", epochs=plan["epochs"], seed=plan["seed"])
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
    fetch_baselines(plan, train / "baseline")
    write_json(train / "baseline_gate.json", {
        "status": "not_run", "policy": "validation_reference_only",
        "reason": "historical_test_agreement_not_required_for_new_search"})
    references, fingerprints, selected = {}, {}, {}
    # New model development is independent of exact replay of old test scores.
    # Establish each horizon's validation reference immediately before its search.
    for horizon in plan["horizons"]:
        log(f"VALIDATION_REFERENCE h={horizon}; historical test replay is not a training gate")
        val_pairs = core.build_week_pairs(groups["val"], config.neg_ratio, config.seed, horizon)
        fingerprints[str(horizon)] = {"val": pairs_fingerprint(val_pairs)}
        model = build_model(core, config)
        state = torch.load(train / "baseline" / f"dexposure-fm-h{horizon}.pt", map_location="cpu", weights_only=False)
        require_exact_split(read_json(ROOT / "cloud/preflight_manifest.json"), state["provenance"]["split"])
        model.load_state_dict(state["model"], strict=True)
        references[str(horizon)] = core.evaluate_predictions(core.predict_graphpfn(model, val_pairs, config))
        values(references[str(horizon)])
        write_json(train / "baseline_validation.json", references)
        write_json(train / "pair_hashes.json", fingerprints)
        del model, state
        gc.collect()
        torch.cuda.empty_cache()
        disk_summaries = load_screen_summaries(train, horizon, plan)
        if disk_summaries is not None:
            winner = max(disk_summaries, key=lambda item: tuple(item["rank"]))
            candidate = next(c for c in plan["candidates"] if c["name"] == winner["candidate"])
            directory = train / "trials" / f"h{horizon}" / candidate["name"]
            disk_summary = read_json(directory / "summary.json")
            source = directory / f"best_model_h{horizon}.pt"
            if (disk_summary.get("status") == "complete"
                    and disk_summary.get("completed_epochs") == plan["epochs"]
                    and source.is_file()):
                for target in (release / f"dexposure-fm-h{horizon}.pt", finetuned / f"best_model_h{horizon}.pt"):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.is_file() or sha256(target) != sha256(source):
                        shutil.copyfile(source, target)
                selected[str(horizon)] = {**disk_summary, "model_sha256": sha256(source)}
                log(f"HORIZON_SKIP_COMPLETE h={horizon} candidate={candidate['name']}")
                continue
        log(f"BUILD_TRAIN_PAIRS h={horizon}")
        train_pairs = core.build_week_pairs(groups["train"], config.neg_ratio, config.seed, horizon)
        if pairs_fingerprint(val_pairs) != fingerprints[str(horizon)]["val"]:
            raise ValueError("Validation pairs changed")
        if disk_summaries is None:
            summaries = []
            for candidate in plan["candidates"]:
                directory = train / "trials" / f"h{horizon}" / candidate["name"]
                summaries.append(train_trial(core, config, candidate, horizon, train_pairs, val_pairs,
                                             references[str(horizon)], split, schema, directory,
                                             plan["screen_epochs"], plan))
            winner = max(summaries, key=lambda item: tuple(item["rank"]))
        candidate = next(c for c in plan["candidates"] if c["name"] == winner["candidate"])
        directory = train / "trials" / f"h{horizon}" / candidate["name"]
        log(f"SCREEN_SELECTED h={horizon} candidate={candidate['name']} rank={winner['rank']}")
        completed = train_trial(core, config, candidate, horizon, train_pairs, val_pairs,
                                references[str(horizon)], split, schema, directory,
                                plan["epochs"], plan, resume=True)
        if pairs_fingerprint(val_pairs) != fingerprints[str(horizon)]["val"]:
            raise ValueError("Validation pairs changed during training")
        source = directory / f"best_model_h{horizon}.pt"
        for target in (release / f"dexposure-fm-h{horizon}.pt", finetuned / f"best_model_h{horizon}.pt"):
            shutil.copyfile(source, target)
        selected[str(horizon)] = {**completed, "model_sha256": sha256(source)}
        write_json(release / f"run_config_h{horizon}.json", {
            "artifact_status": "complete", "experiment_kind": plan["experiment_kind"],
            "forecast_horizon_weeks": horizon, "holdout_start": plan["holdout_start"],
            "validation_weeks": 24, "epochs": 20, "seed": 42, "network_snapshots": 283,
            "source_sha256": os.environ["PREFLIGHT_SOURCE_SHA256"], "candidate": candidate,
            "selected_epoch": completed["selected_epoch"], "completed_epochs": 20,
        })
        del train_pairs, val_pairs
        gc.collect()
    freeze = {"selected": selected, "frozen_at": datetime.now(timezone.utc).isoformat(),
              "selection_data": "pre2025_validation_only", "plan_sha256": sha256(PLAN_PATH)}
    write_json(train / "selection_frozen.json", freeze)
    selection_sha = sha256(train / "selection_frozen.json")
    log(f"ALL_FOUR_SELECTIONS_FROZEN sha256={selection_sha}; final holdout evaluation begins")
    test_results, comparison = {"model": "DeXposure-FM tuned candidate"}, {}
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
                              "historical test-score agreement not required for this new-model search"]}
    write_json(train / "result.json", result)
    write_json(release / "task1_metrics.json", test_results)
    write_json(finetuned / "metrics.json", test_results)
    write_json(finetuned / "all_results.json", result)
    write_json(finetuned / "experiment_results.json", {**result, "config": plan, "selected": selected})
    (release / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in sorted(release.iterdir()) if p.is_file()))
    write_tree_manifest(ROOT)
    log(f"TUNING_COMPLETE {verify_bundle(ROOT, os.environ['PREFLIGHT_SOURCE_SHA256'])}")


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
