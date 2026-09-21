#!/usr/bin/env python3
"""
Macroprudential Tools CLI (Observed + Predictive) for DeXposure-FM.

This script exposes the "forecast-then-measure" workflow as a usable tool:
  1) observed: compute SIS / spillovers / contagion on an observed snapshot G_t
  2) predict : forecast a future snapshot \\hat{G}_{t+h} with DeXposure-FM,
               then compute the same tools on the predicted graph

Examples:
  # Observed snapshot tools
  uv run python run_macroprudential_tools.py observed --date 2025-06-30 \\
    --data-path data/historical-network_week_2025-07-01.json

  # Forecast-then-measure (trains or loads a cached model)
  uv run python run_macroprudential_tools.py predict --date 2025-06-30 --horizon 4 \\
    --data-path data/historical-network_week_2025-07-01.json --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from dexposure_fm.macroprudential_tools import (  # noqa: E402
    array_snap_to_dict_snap,
    compute_network_risk_metrics,
    compute_sector_spillover_index,
    compute_systemic_importance_score,
    simulate_contagion,
    spillover_hhi,
)

torch = None  # Lazy import for observed-only workflows

GRAPHPFN_AVAILABLE = None
ExperimentConfig = None
GraphPFNLinkPredictor = None
WeekPair = None
build_snapshot = None
build_week_pairs = None
find_nearest_date = None
load_graphpfn_encoder = None
load_metadata = None
load_network_data = None
predict_graphpfn = None
set_seed = None
train_graphpfn_epoch = None
EmbeddingCache = None


def _import_graphpfn_deps() -> None:
    """Lazy import heavy deps so observed-mode can run on lighter environments."""
    global GRAPHPFN_AVAILABLE
    global ExperimentConfig
    global GraphPFNLinkPredictor
    global WeekPair
    global build_snapshot
    global build_week_pairs
    global find_nearest_date
    global load_graphpfn_encoder
    global load_metadata
    global load_network_data
    global predict_graphpfn
    global set_seed
    global train_graphpfn_epoch
    global EmbeddingCache
    global torch

    if GRAPHPFN_AVAILABLE is not None:
        return

    import torch as _torch

    torch = _torch

    from run_full_experiment import (
        GRAPHPFN_AVAILABLE as _GRAPHPFN_AVAILABLE,
        ExperimentConfig as _ExperimentConfig,
        WeekPair as _WeekPair,
        build_snapshot as _build_snapshot,
        build_week_pairs as _build_week_pairs,
        find_nearest_date as _find_nearest_date,
        load_metadata as _load_metadata,
        load_network_data as _load_network_data,
        set_seed as _set_seed,
    )

    GRAPHPFN_AVAILABLE = _GRAPHPFN_AVAILABLE
    ExperimentConfig = _ExperimentConfig
    WeekPair = _WeekPair
    build_snapshot = _build_snapshot
    build_week_pairs = _build_week_pairs
    find_nearest_date = _find_nearest_date
    load_metadata = _load_metadata
    load_network_data = _load_network_data
    set_seed = _set_seed

    if GRAPHPFN_AVAILABLE:
        from run_full_experiment import (
            EmbeddingCache as _EmbeddingCache,
            GraphPFNLinkPredictor as _GraphPFNLinkPredictor,
            load_graphpfn_encoder as _load_graphpfn_encoder,
            predict_graphpfn as _predict_graphpfn,
            train_graphpfn_epoch as _train_graphpfn_epoch,
        )

        GraphPFNLinkPredictor = _GraphPFNLinkPredictor
        EmbeddingCache = _EmbeddingCache
        load_graphpfn_encoder = _load_graphpfn_encoder
        predict_graphpfn = _predict_graphpfn
        train_graphpfn_epoch = _train_graphpfn_epoch
    else:
        GraphPFNLinkPredictor = None
        EmbeddingCache = None
        load_graphpfn_encoder = None
        predict_graphpfn = None
        train_graphpfn_epoch = None


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("macro-tools")


MODEL_CACHE_DIR_DEFAULT = Path("output/model_cache")
OUTPUT_DIR_DEFAULT = Path("output/macroprudential_tools")


DEFAULT_SHOCK_SCENARIOS = [
    {"name": "Top Protocol Shock", "top_n": 1, "shock_ratio": 0.5},
    {"name": "Top 5 Protocols Shock", "top_n": 5, "shock_ratio": 0.3},
    {"name": "Bridge Sector Shock", "sector": "Bridge", "shock_ratio": 1.0},
]


def _topk_scores(scores: Mapping[str, float], k: int) -> List[Dict[str, Any]]:
    k_eff = max(0, int(k))
    if k_eff <= 0 or not scores:
        return []
    items = sorted(scores.items(), key=lambda kv: float(kv[1]), reverse=True)[:k_eff]
    return [{"node": str(n), "score": float(s)} for n, s in items]


def _summarize_snapshot(
    name: str,
    snap_array: Dict[str, Any],
    *,
    meta_category: Optional[Mapping[str, str]],
    edge_weight_is_log: bool,
    top_k: int,
    include_full_sis: bool,
    include_spillover_matrix: bool,
) -> Dict[str, Any]:
    risk = compute_network_risk_metrics(
        snap_array, meta_category=meta_category, edge_weight_is_log=edge_weight_is_log
    )
    dict_snap = array_snap_to_dict_snap(snap_array, edge_weight_is_log=edge_weight_is_log)

    sis = compute_systemic_importance_score(dict_snap)
    sector_exposure = (
        compute_sector_spillover_index(dict_snap, meta_category) if meta_category else {}
    )
    spill_total, spill_idx = spillover_hhi(sector_exposure)

    out: Dict[str, Any] = {
        "name": name,
        "date": snap_array.get("date", ""),
        "risk_metrics": risk,
        "sis_top": _topk_scores(sis, top_k),
        "spillover": {
            "spillover_total": float(spill_total),
            "spillover_index": float(spill_idx),
        },
    }
    if include_full_sis:
        out["sis_scores"] = {str(k): float(v) for k, v in sis.items()}
    if include_spillover_matrix:
        out["spillover"]["sector_exposure"] = sector_exposure
    return out


def _resolve_date(requested: str, all_dates: Sequence[str]) -> str:
    if requested in all_dates:
        return requested
    if find_nearest_date is None:
        return requested
    nearest = find_nearest_date(requested, list(all_dates))
    if nearest is None:
        raise ValueError("No dates available in dataset.")
    logger.warning(f"Date {requested!r} not found; using nearest available {nearest!r}.")
    return nearest


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)

def _contagion_scenarios_for_pair(
    pair: Any,
    meta_category: Mapping[str, str],
    scenarios: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build shock sets using information available at time t."""
    node_ids = list(pair.node_ids)
    sizes_t = np.asarray(pair.sizes_t, dtype=float)
    node_mask = np.asarray(pair.node_mask, dtype=bool)

    valid_idx = np.where(node_mask)[0]
    out = []
    for s in scenarios:
        if "top_n" in s:
            n = int(s["top_n"])
            if valid_idx.size == 0 or n <= 0:
                shocked = []
            else:
                order = valid_idx[np.argsort(sizes_t[valid_idx])[::-1]]
                shocked = [node_ids[int(i)] for i in order[:n]]
        elif "sector" in s:
            sector = str(s["sector"])
            s_lower = sector.lower()
            shocked = [
                node_ids[int(i)]
                for i in valid_idx
                if s_lower in str(meta_category.get(node_ids[int(i)], "") or "").lower()
            ]
        else:
            shocked = []

        out.append({**s, "shocked_nodes": shocked, "shocked_nodes_count": len(shocked)})
    return out


def load_or_train_model(
    config: Any,
    train_pairs: List[Any],
    *,
    model_cache_dir: Path,
    force_retrain: bool = False,
    frozen: bool = False,
    cache_tag: Optional[str] = None,
    horizon: int = 1,
) -> Any:
    """Load cached model or train if missing (cache key includes horizon + optional tag)."""
    _import_graphpfn_deps()
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    model_name = "graphpfn_frozen" if frozen else "dexposure_fm"
    name_parts = [model_name, f"h{int(horizon)}"]
    if cache_tag:
        safe_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in cache_tag)
        name_parts.append(safe_tag)
    cache_stem = "__".join(name_parts)
    cache_path = model_cache_dir / f"{cache_stem}.pt"
    legacy_cache_path = model_cache_dir / f"{model_name}.pt"

    device = torch.device(config.device)
    encoder = load_graphpfn_encoder(config.checkpoint_path, device)
    embed_dim = encoder.tfm.embed_dim
    model = GraphPFNLinkPredictor(encoder, embed_dim, config.hidden_dim).to(device)

    if cache_path.exists() and not force_retrain:
        logger.info(f"Loading cached model from {cache_path}")
        state = torch.load(cache_path, map_location=device)
        model.load_state_dict(state["model"], strict=True)
        return model
    if horizon == 1 and not cache_tag and legacy_cache_path.exists() and not force_retrain:
        logger.info(f"Loading legacy cached model from {legacy_cache_path} (treating as h=1)")
        state = torch.load(legacy_cache_path, map_location=device)
        model.load_state_dict(state["model"], strict=True)
        return model

    finetune_encoder = not frozen
    for p in model.encoder.parameters():
        p.requires_grad = finetune_encoder

    weight_decay = float(getattr(config, "weight_decay", 0.0))
    if finetune_encoder:
        encoder_params = list(model.encoder.parameters())
        encoder_param_set = set(encoder_params)
        head_params = [p for p in model.parameters() if p not in encoder_param_set]
        optimizer = torch.optim.Adam(
            [
                {"params": encoder_params, "lr": float(config.lr) * 0.1},
                {"params": head_params, "lr": float(config.lr)},
            ],
            weight_decay=weight_decay,
        )
        logger.info(
            f"Training {'DeXposure-FM (fine-tuned)' if finetune_encoder else 'GraphPFN-Frozen'} "
            f"(h={horizon}): lr_head={float(config.lr):.1e}, lr_enc={float(config.lr)*0.1:.1e}, wd={weight_decay:.1e}"
        )
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=float(config.lr), weight_decay=weight_decay)
        logger.info(
            f"Training GraphPFN-Frozen (h={horizon}): lr={float(config.lr):.1e}, wd={weight_decay:.1e}"
        )

    scaler = None
    if bool(getattr(config, "use_amp", False)) and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        logger.info("AMP enabled")

    embedding_cache = None
    if (not finetune_encoder) and bool(getattr(config, "cache_frozen_embeddings", False)):
        embedding_cache = EmbeddingCache(model, device)
        logger.info("Embedding cache enabled (frozen encoder)")

    best_state: Optional[Dict[str, Any]] = None
    best_loss = float("inf")
    prev_embeddings = None

    for epoch in range(int(config.epochs)):
        losses, prev_embeddings = train_graphpfn_epoch(
            model,
            train_pairs,
            optimizer,
            config,
            finetune_encoder=finetune_encoder,
            prev_embeddings=prev_embeddings,
            scaler=scaler,
            embedding_cache=embedding_cache,
        )
        total_loss = (
            float(getattr(config, "exist_loss_weight", 1.0)) * float(losses.get("exist_loss", 0.0))
            + float(getattr(config, "weight_loss_weight", 1.0)) * float(losses.get("weight_loss", 0.0))
            + float(getattr(config, "node_loss_weight", 1.0)) * float(losses.get("node_loss", 0.0))
            + float(getattr(config, "stats_loss_weight", 0.0)) * float(losses.get("stats_loss", 0.0))
            + float(getattr(config, "impute_loss_weight", 0.0)) * float(losses.get("impute_loss", 0.0))
            + float(getattr(config, "scen_loss_weight", 0.0)) * float(losses.get("scen_loss", 0.0))
            + float(getattr(config, "smooth_loss_weight", 0.0)) * float(losses.get("smooth_loss", 0.0))
        )

        if total_loss < best_loss:
            best_loss = total_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0:
            logger.info(
                f"  Epoch {epoch+1}: exist={losses.get('exist_loss', 0.0):.4f}, "
                f"weight={losses.get('weight_loss', 0.0):.4f}, node={losses.get('node_loss', 0.0):.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({"model": model.state_dict()}, cache_path)
    logger.info(f"Model saved to {cache_path}")
    return model


def reconstruct_network_from_predictions(
    pred: Dict[str, Any],
    pair: Any,
    *,
    edge_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Reconstruct an array-format snapshot from model predictions (log-weight edges)."""
    node_mask = np.asarray(pair.node_mask, dtype=bool)

    exist_prob = 1.0 / (1.0 + np.exp(-np.asarray(pred["exist_logits"], dtype=float)))
    edge_mask = exist_prob > float(edge_threshold)

    pred_src = np.asarray(pair.pair_src, dtype=int)[edge_mask]
    pred_dst = np.asarray(pair.pair_dst, dtype=int)[edge_mask]
    pred_weights = np.asarray(pred["weight_pred"], dtype=float)[edge_mask]

    valid_edge_mask = node_mask[pred_src] & node_mask[pred_dst]
    pred_src = pred_src[valid_edge_mask]
    pred_dst = pred_dst[valid_edge_mask]
    pred_weights = pred_weights[valid_edge_mask]

    sizes_t = np.asarray(pair.sizes_t, dtype=float)
    log_size_t = np.log1p(np.maximum(sizes_t, 0.0))
    pred_log_tvl = log_size_t + np.asarray(pred["node_pred"], dtype=float)
    pred_sizes = np.expm1(np.maximum(pred_log_tvl, 0.0))

    valid_node_idx = np.where(node_mask)[0]
    valid_node_ids = [pair.node_ids[int(i)] for i in valid_node_idx]
    valid_categories = [pair.categories[int(i)] for i in valid_node_idx]
    valid_sizes = pred_sizes[node_mask]
    valid_features = pair.features_t[node_mask] if pair.features_t is not None else None

    old_to_new = {int(old): new for new, old in enumerate(valid_node_idx.tolist())}
    remapped_src = np.array([old_to_new[int(s)] for s in pred_src], dtype=np.int64)
    remapped_dst = np.array([old_to_new[int(d)] for d in pred_dst], dtype=np.int64)

    return {
        "date": pred.get("time_t1", "predicted"),
        "node_ids": valid_node_ids,
        "categories": valid_categories,
        "sizes": valid_sizes,
        "edge_src": remapped_src,
        "edge_dst": remapped_dst,
        "edge_weight": pred_weights,
        "features": valid_features,
    }


def reconstruct_actual_network(pred: Dict[str, Any], pair: Any) -> Dict[str, Any]:
    """Reconstruct the actual array-format snapshot at t+h from labels (log-weight edges)."""
    node_mask = np.asarray(pair.node_mask, dtype=bool)

    edge_mask = np.asarray(pair.y_exist, dtype=float) > 0.5
    actual_src = np.asarray(pair.pair_src, dtype=int)[edge_mask]
    actual_dst = np.asarray(pair.pair_dst, dtype=int)[edge_mask]
    actual_weights = np.asarray(pair.y_weight, dtype=float)[edge_mask]

    valid_edge_mask = node_mask[actual_src] & node_mask[actual_dst]
    actual_src = actual_src[valid_edge_mask]
    actual_dst = actual_dst[valid_edge_mask]
    actual_weights = actual_weights[valid_edge_mask]

    sizes_t = np.asarray(pair.sizes_t, dtype=float)
    log_size_t = np.log1p(np.maximum(sizes_t, 0.0))
    actual_log_tvl = log_size_t + np.asarray(pair.y_node, dtype=float)
    actual_sizes = np.expm1(np.maximum(actual_log_tvl, 0.0))

    valid_node_idx = np.where(node_mask)[0]
    valid_node_ids = [pair.node_ids[int(i)] for i in valid_node_idx]
    valid_categories = [pair.categories[int(i)] for i in valid_node_idx]
    valid_sizes = actual_sizes[node_mask]
    valid_features = pair.features_t[node_mask] if pair.features_t is not None else None

    old_to_new = {int(old): new for new, old in enumerate(valid_node_idx.tolist())}
    remapped_src = np.array([old_to_new[int(s)] for s in actual_src], dtype=np.int64)
    remapped_dst = np.array([old_to_new[int(d)] for d in actual_dst], dtype=np.int64)

    return {
        "date": pred.get("time_t1", "actual"),
        "node_ids": valid_node_ids,
        "categories": valid_categories,
        "sizes": valid_sizes,
        "edge_src": remapped_src,
        "edge_dst": remapped_dst,
        "edge_weight": actual_weights,
        "features": valid_features,
    }


def cmd_observed(args: argparse.Namespace) -> None:
    _import_graphpfn_deps()

    meta_category, category_list, category_to_idx = load_metadata(args.meta_path)
    network_data = load_network_data(args.data_path)
    all_dates = sorted(network_data.keys())
    if not all_dates:
        raise RuntimeError("No snapshots found in data file.")

    date = _resolve_date(args.date, all_dates)
    snap = build_snapshot(date, network_data[date], meta_category, category_to_idx, category_list)

    summary = _summarize_snapshot(
        "observed",
        snap,
        meta_category=meta_category,
        edge_weight_is_log=False,  # build_snapshot uses linear weights
        top_k=args.top_k,
        include_full_sis=bool(args.full_sis),
        include_spillover_matrix=bool(args.spillover_matrix),
    )

    out: Dict[str, Any] = {
        "mode": "observed",
        "data_path": str(args.data_path),
        "meta_path": str(args.meta_path),
        "snapshot": summary,
    }

    if args.contagion:
        dict_snap = array_snap_to_dict_snap(snap, edge_weight_is_log=False)
        contagion_rows = []
        for sc in DEFAULT_SHOCK_SCENARIOS:
            shocked_nodes: List[str]
            if "top_n" in sc:
                n = int(sc["top_n"])
                order = np.argsort(np.asarray(snap["sizes"], dtype=float))[::-1]
                shocked_nodes = [snap["node_ids"][int(i)] for i in order[:n]]
            elif "sector" in sc:
                sector = str(sc["sector"]).lower()
                shocked_nodes = [
                    nid for nid in snap["node_ids"]
                    if sector in str(meta_category.get(nid, "") or "").lower()
                ]
            else:
                shocked_nodes = []

            contagion_rows.append(
                {
                    "scenario": sc,
                    "shocked_nodes": shocked_nodes,
                    "result": simulate_contagion(
                        dict_snap, shocked_nodes, shock_fraction=float(sc["shock_ratio"])
                    ),
                }
            )

        out["contagion"] = contagion_rows

    out_dir = Path(args.output_dir)
    out_path = (
        Path(args.output)
        if args.output
        else out_dir / f"observed_{date}.json"
    )
    _write_json(out_path, out)
    logger.info(f"Wrote {out_path}")


def cmd_predict(args: argparse.Namespace) -> None:
    _import_graphpfn_deps()

    if not GRAPHPFN_AVAILABLE:
        raise RuntimeError("GraphPFN not available; cannot run predictive mode.")

    config = ExperimentConfig(
        epochs=int(args.epochs),
        seed=int(args.seed),
        device=str(args.device) if args.device else ExperimentConfig().device,
    )
    set_seed(config.seed)

    meta_category, category_list, category_to_idx = load_metadata(args.meta_path)
    network_data = load_network_data(args.data_path)
    all_dates = sorted(network_data.keys())
    if not all_dates:
        raise RuntimeError("No snapshots found in data file.")

    anchor_date = _resolve_date(args.date, all_dates)
    h = int(args.horizon)
    if h <= 0:
        raise ValueError("--horizon must be >= 1")

    anchor_idx = all_dates.index(anchor_date)
    target_idx = anchor_idx + h
    if target_idx >= len(all_dates):
        raise ValueError(
            f"Not enough future snapshots for date {anchor_date} with horizon h={h} "
            f"(need index {target_idx}, max index {len(all_dates)-1})."
        )
    target_date = all_dates[target_idx]

    # Build snapshots only as needed.
    train_cutoff = args.train_cutoff or anchor_date
    train_cutoff_resolved = _resolve_date(train_cutoff, all_dates)
    train_dates = [d for d in all_dates if d < train_cutoff_resolved]
    if len(train_dates) < max(2, h + 1):
        raise ValueError(
            f"Insufficient training history (< {train_cutoff_resolved}). "
            f"Have {len(train_dates)} weeks; need at least {max(2, h + 1)}."
        )

    logger.info(f"Predict: t={anchor_date}, t+h={target_date} (h={h})")
    logger.info(f"Training cutoff: < {train_cutoff_resolved} ({len(train_dates)} weeks)")

    # Training snapshots for horizon-h supervision.
    train_snaps = [
        build_snapshot(d, network_data[d], meta_category, category_to_idx, category_list)
        for d in train_dates
    ]
    train_pairs = build_week_pairs(train_snaps, config.neg_ratio, config.seed, horizon=h)
    if not train_pairs:
        raise RuntimeError("No training pairs built (check data coverage and horizon).")

    if args.max_train_pairs > 0 and len(train_pairs) > int(args.max_train_pairs):
        train_pairs = train_pairs[-int(args.max_train_pairs) :]
        logger.info(f"Subsampled train_pairs to {len(train_pairs)} (max_train_pairs={args.max_train_pairs})")

    model_cache_dir = Path(args.model_cache_dir)
    cache_tag = args.cache_tag
    if cache_tag is None:
        cache_tag = f"train_lt_{train_cutoff_resolved}"

    model = load_or_train_model(
        config,
        train_pairs,
        model_cache_dir=model_cache_dir,
        force_retrain=bool(args.force_retrain),
        frozen=bool(args.frozen),
        cache_tag=cache_tag,
        horizon=h,
    )

    # Evaluation slice (t..t+h inclusive)
    eval_dates = all_dates[anchor_idx : target_idx + 1]
    eval_snaps = [
        build_snapshot(d, network_data[d], meta_category, category_to_idx, category_list)
        for d in eval_dates
    ]
    eval_pairs = build_week_pairs(eval_snaps, config.neg_ratio, config.seed, horizon=h)
    if not eval_pairs:
        raise RuntimeError("Failed to build evaluation pair for the requested date/horizon.")

    pair = eval_pairs[0]
    preds = predict_graphpfn(model, [pair], config)
    pred = preds[0]

    pred_net = reconstruct_network_from_predictions(pred, pair, edge_threshold=float(args.edge_threshold))
    actual_net = reconstruct_actual_network(pred, pair)

    observed_net = {
        "date": pair.time_t,
        "node_ids": pair.node_ids,
        "categories": pair.categories,
        "sizes": pair.sizes_t,
        "edge_src": pair.edge_src_t,
        "edge_dst": pair.edge_dst_t,
        "edge_weight": pair.edge_weight_t,  # log1p
        "features": pair.features_t,
    }

    summaries = {
        "observed_t": _summarize_snapshot(
            "observed_t",
            observed_net,
            meta_category=meta_category,
            edge_weight_is_log=True,
            top_k=args.top_k,
            include_full_sis=bool(args.full_sis),
            include_spillover_matrix=bool(args.spillover_matrix),
        ),
        "predicted_t_h": _summarize_snapshot(
            "predicted_t_h",
            pred_net,
            meta_category=meta_category,
            edge_weight_is_log=True,
            top_k=args.top_k,
            include_full_sis=bool(args.full_sis),
            include_spillover_matrix=bool(args.spillover_matrix),
        ),
        "actual_t_h": _summarize_snapshot(
            "actual_t_h",
            actual_net,
            meta_category=meta_category,
            edge_weight_is_log=True,
            top_k=args.top_k,
            include_full_sis=bool(args.full_sis),
            include_spillover_matrix=bool(args.spillover_matrix),
        ),
    }

    out: Dict[str, Any] = {
        "mode": "predict",
        "data_path": str(args.data_path),
        "meta_path": str(args.meta_path),
        "anchor_date": anchor_date,
        "target_date": target_date,
        "horizon": h,
        "edge_threshold": float(args.edge_threshold),
        "train_cutoff": train_cutoff_resolved,
        "cache_tag": cache_tag,
        "model": {
            "frozen": bool(args.frozen),
            "force_retrain": bool(args.force_retrain),
            "epochs": int(args.epochs),
            "seed": int(args.seed),
            "device": str(config.device),
            "cache_dir": str(model_cache_dir),
        },
        "snapshots": summaries,
    }

    if args.contagion:
        scenarios = _contagion_scenarios_for_pair(pair, meta_category, DEFAULT_SHOCK_SCENARIOS)
        obs_dict = array_snap_to_dict_snap(observed_net, edge_weight_is_log=True)
        pred_dict = array_snap_to_dict_snap(pred_net, edge_weight_is_log=True)
        act_dict = array_snap_to_dict_snap(actual_net, edge_weight_is_log=True)

        rows = []
        for sc in scenarios:
            shocked_nodes = sc.get("shocked_nodes", [])
            if not shocked_nodes:
                continue
            shock_ratio = float(sc.get("shock_ratio", 1.0))
            rows.append(
                {
                    "scenario": {k: v for k, v in sc.items() if k not in ("shocked_nodes",)},
                    "shocked_nodes": shocked_nodes,
                    "observed_t": simulate_contagion(obs_dict, shocked_nodes, shock_fraction=shock_ratio),
                    "predicted_t_h": simulate_contagion(pred_dict, shocked_nodes, shock_fraction=shock_ratio),
                    "actual_t_h": simulate_contagion(act_dict, shocked_nodes, shock_fraction=shock_ratio),
                }
            )
        out["contagion"] = rows

    out_dir = Path(args.output_dir)
    out_path = (
        Path(args.output)
        if args.output
        else out_dir / f"predict_{anchor_date}_h{h}.json"
    )
    _write_json(out_path, out)
    logger.info(f"Wrote {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeXposure-FM macroprudential tools (observed + predictive)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-path", type=str, default=None, help="Network JSON path (default: ExperimentConfig.data_path)")
        p.add_argument("--meta-path", type=str, default=None, help="Metadata CSV path (default: ExperimentConfig.meta_path)")
        p.add_argument("--top-k", type=int, default=20, help="Top-K nodes to report for SIS ranking")
        p.add_argument("--full-sis", action="store_true", help="Include full SIS scores in output JSON")
        p.add_argument("--spillover-matrix", action="store_true", help="Include full spillover matrix in output JSON")
        p.add_argument("--contagion", action="store_true", help="Run default contagion scenarios")
        p.add_argument("--output", type=str, default="", help="Write JSON to this path (default: output_dir/<auto>.json)")
        p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR_DEFAULT), help="Output directory")

    p_obs = sub.add_parser("observed", help="Compute tools on observed snapshot G_t")
    add_common(p_obs)
    p_obs.add_argument("--date", type=str, required=True, help="Snapshot date (YYYY-MM-DD)")

    p_pred = sub.add_parser("predict", help="Forecast \\hat{G}_{t+h} then compute tools")
    add_common(p_pred)
    p_pred.add_argument("--date", type=str, required=True, help="Anchor date t (YYYY-MM-DD)")
    p_pred.add_argument("--horizon", type=int, required=True, help="Forecast horizon h (weeks)")
    p_pred.add_argument("--edge-threshold", type=float, default=0.5, help="Edge existence probability threshold")
    p_pred.add_argument("--epochs", type=int, default=20, help="Training epochs (if training)")
    p_pred.add_argument("--seed", type=int, default=42, help="Random seed")
    p_pred.add_argument("--device", type=str, default=None, help="Device (cpu/cuda); default from ExperimentConfig")
    p_pred.add_argument("--frozen", action="store_true", help="Use frozen GraphPFN encoder (faster)")
    p_pred.add_argument("--force-retrain", action="store_true", help="Force retraining even if cache exists")
    p_pred.add_argument("--model-cache-dir", type=str, default=str(MODEL_CACHE_DIR_DEFAULT), help="Model cache directory")
    p_pred.add_argument("--cache-tag", type=str, default=None, help="Optional cache tag (default: train_lt_<train_cutoff>)")
    p_pred.add_argument("--train-cutoff", type=str, default="", help="Train on dates < this cutoff (default: < anchor date)")
    p_pred.add_argument("--max-train-pairs", type=int, default=0, help="Limit number of training pairs (0=all)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "data_path", None) in (None, ""):
        args.data_path = "data/historical-network_week_2020-03-30.json"
    if getattr(args, "meta_path", None) in (None, ""):
        args.meta_path = "data/meta_df.csv"

    if args.cmd == "observed":
        cmd_observed(args)
        return
    if args.cmd == "predict":
        cmd_predict(args)
        return
    raise RuntimeError(f"Unknown command {args.cmd!r}")


if __name__ == "__main__":
    main()
