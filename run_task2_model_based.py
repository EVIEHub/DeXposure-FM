#!/usr/bin/env python3
"""
Task II: Model-based Forward-looking Financial Stability Analysis

This script implements the IMPROVED Task II experiments that demonstrate
the Foundation Model's predictive capabilities for risk assessment.

Core Experiments:
1. Forward-looking Risk Metric Prediction
   - Use DeXposure-FM to predict network at t+h
   - Compute risk metrics on predicted network
   - Compare with actual risk metrics at t+h

2. Predictive Contagion Simulation
   - Run contagion simulation on predicted networks
   - Compare: observed(t) vs predicted(t->t+h) vs actual(t+h)
   - Show forward-looking stress testing capability

3. Shock Event Early Warning
   - Predict network structure before Terra/Luna, FTX events
   - Check if predicted risk metrics rise before actual shocks
   - Demonstrate early warning capability

Key Insight: Previous experiments only used descriptive statistics on OBSERVED
networks. This version shows how the trained model enables FORWARD-LOOKING
risk assessment - the core value proposition of a Foundation Model.

Usage:
    # Run on GPU server
    ssh gpu-server
    cd /path/to/graph-dexposure
    python run_task2_model_based.py --experiment all
    python run_task2_model_based.py --experiment all --no-plot          # save JSON only
    python run_task2_model_based.py --experiment forward_risk
    python run_task2_model_based.py --experiment predictive_contagion
    python run_task2_model_based.py --experiment early_warning

    # Regenerate figures quickly from saved JSON (no model/data needed)
    python run_task2_model_based.py --plot-only --output-dir output/task2_model_based

    # Minimal robustness check (no model training/inference)
    python run_task2_model_based.py --experiment sis_sensitivity --output-dir output/task2_model_based
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

torch = None  # Lazy import for plot-only workflows

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add project paths
GRAPHPFN_ROOT = Path(__file__).parent
sys.path.insert(0, str(GRAPHPFN_ROOT))

GRAPHPFN_AVAILABLE = None
ExperimentConfig = None
GraphPFNLinkPredictor = None
WeekPair = None
build_snapshot = None
build_week_pairs = None
get_single_split = None
load_graphpfn_encoder = None
load_metadata = None
load_network_data = None
predict_graphpfn = None
set_seed = None
train_graphpfn_epoch = None
EmbeddingCache = None


def _import_graphpfn_deps() -> None:
    """Lazy import heavy deps so --plot-only can run on a lightweight environment."""
    global GRAPHPFN_AVAILABLE
    global ExperimentConfig
    global GraphPFNLinkPredictor
    global WeekPair
    global build_snapshot
    global build_week_pairs
    global get_single_split
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
        GraphPFNLinkPredictor as _GraphPFNLinkPredictor,
        EmbeddingCache as _EmbeddingCache,
        WeekPair as _WeekPair,
        build_snapshot as _build_snapshot,
        build_week_pairs as _build_week_pairs,
        get_single_split as _get_single_split,
        load_graphpfn_encoder as _load_graphpfn_encoder,
        load_metadata as _load_metadata,
        load_network_data as _load_network_data,
        predict_graphpfn as _predict_graphpfn,
        set_seed as _set_seed,
        train_graphpfn_epoch as _train_graphpfn_epoch,
    )

    GRAPHPFN_AVAILABLE = _GRAPHPFN_AVAILABLE
    ExperimentConfig = _ExperimentConfig
    GraphPFNLinkPredictor = _GraphPFNLinkPredictor
    EmbeddingCache = _EmbeddingCache
    WeekPair = _WeekPair
    build_snapshot = _build_snapshot
    build_week_pairs = _build_week_pairs
    get_single_split = _get_single_split
    load_graphpfn_encoder = _load_graphpfn_encoder
    load_metadata = _load_metadata
    load_network_data = _load_network_data
    predict_graphpfn = _predict_graphpfn
    set_seed = _set_seed
    train_graphpfn_epoch = _train_graphpfn_epoch


_PLOTTING_CONFIGURED = False


def _import_matplotlib() -> Any:
    """Lazy import matplotlib so compute-only runs don't require it."""
    global _PLOTTING_CONFIGURED
    try:
        import matplotlib

        if "matplotlib.pyplot" not in sys.modules:
            matplotlib.use("Agg")  # Non-interactive backend for servers

        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it (e.g. `pip install matplotlib`) "
            "or run this script with `--no-plot` and later regenerate figures with "
            "`--plot-only` on a machine that has matplotlib."
        ) from e

    if not _PLOTTING_CONFIGURED:
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 10,
                "axes.labelsize": 11,
                "axes.titlesize": 12,
                "legend.fontsize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "axes.grid": True,
                "grid.alpha": 0.3,
            }
        )
        _PLOTTING_CONFIGURED = True

    return plt


# ============== Snapshot Format Adapter ==============


def array_snap_to_dict_snap(
    snap: Dict[str, Any],
    edge_weight_is_log: bool = True,
) -> Dict[str, Any]:
    """
    Convert array-format snapshot to dict-format snapshot for contagion/SIS functions.

    Array format (from reconstruct_network_from_predictions):
        - node_ids: List[str]
        - sizes: np.ndarray (TVL values)
        - edge_src: np.ndarray (source indices)
        - edge_dst: np.ndarray (destination indices)
        - edge_weight: np.ndarray (log1p or linear weights)
        - categories: List[str]

    Dict format (expected by simulate_contagion, compute_systemic_importance_score):
        - nodes: Dict[str, {"tvlUsd": float, "category": str}]
        - edges: List[{"source": str, "target": str, "weight": float}]
    """
    # Check if already in dict format
    if "nodes" in snap and isinstance(snap.get("nodes"), dict):
        return snap

    node_ids = snap.get("node_ids", [])
    sizes = snap.get("sizes", [])
    categories = snap.get("categories", [])
    edge_src = snap.get("edge_src", [])
    edge_dst = snap.get("edge_dst", [])
    edge_weight = snap.get("edge_weight", [])

    # Build nodes dict
    nodes = {}
    for i, nid in enumerate(node_ids):
        tvl = float(sizes[i]) if i < len(sizes) else 0.0
        cat = categories[i] if i < len(categories) else "Unknown"
        nodes[nid] = {"tvlUsd": tvl, "category": cat}

    # Build edges list
    edges = []
    for i in range(len(edge_src)):
        src_idx = int(edge_src[i])
        dst_idx = int(edge_dst[i])
        if src_idx < len(node_ids) and dst_idx < len(node_ids):
            w = float(edge_weight[i]) if i < len(edge_weight) else 0.0
            # Convert from log scale if needed
            if edge_weight_is_log:
                w = float(np.expm1(w))
                w = max(0.0, w)
            edges.append({
                "source": node_ids[src_idx],
                "target": node_ids[dst_idx],
                "weight": w,
            })

    return {"nodes": nodes, "edges": edges, "date": snap.get("date", "")}


# ============== Risk Analysis Functions (Task II specific) ==============


def gini_coefficient(values: List[float]) -> float:
    """Compute Gini coefficient of a distribution."""
    values = np.array(values, dtype=float)
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values)))


def compute_systemic_importance_score(
    snap: Dict,
    alpha: float = 1 / 3,
    beta: float = 1 / 3,
    gamma: float = 1 / 3,
    tail_k: int = 5,
    pagerank_max_iter: int = 100,
) -> Dict[str, float]:
    """
    Compute systemic importance score (SIS) for each node.

    SIS_p = alpha * PageRank_p + beta * TailExposure_p + gamma * log(1 + TVL_p)

    - PageRank captures network interconnectedness (weighted by exposures).
    - TailExposure measures concentration of the node's largest outgoing exposures:
      sum(top-k outgoing exposure weights) / sum(all outgoing exposure weights).
    - log(1+TVL) captures protocol scale.
    """
    nodes = snap.get("nodes", {})
    edges = snap.get("edges", [])

    if not nodes:
        return {}

    components = compute_sis_components(snap, tail_k=tail_k, pagerank_max_iter=pagerank_max_iter)
    pagerank_norm = components["pagerank"]
    tail_exposure = components["tail_exposure"]
    log_tvl_norm = components["log_tvl_norm"]

    # PageRank, TailExposure, and log(1+TVL) are normalized to comparable scales.
    scores: Dict[str, float] = {}
    for n in nodes:
        scores[n] = (
            alpha * float(pagerank_norm.get(n, 0.0))
            + beta * float(tail_exposure.get(n, 0.0))
            + gamma * float(log_tvl_norm.get(n, 0.0))
        )

    return scores


def compute_sis_components(
    snap: Dict[str, Any],
    tail_k: int = 5,
    pagerank_max_iter: int = 100,
) -> Dict[str, Dict[str, float]]:
    """
    Compute normalized SIS components for a snapshot:
    - pagerank: weighted PageRank, normalized to [0,1] by dividing max value
    - tail_exposure: top-k outgoing exposure share in [0,1]
    - log_tvl_norm: log(1+TVL) normalized to [0,1] by dividing max value
    """
    nodes = snap.get("nodes", {})
    edges = snap.get("edges", [])
    if not nodes:
        return {"pagerank": {}, "tail_exposure": {}, "log_tvl_norm": {}}

    import networkx as nx

    # 1) Weighted PageRank on exposure graph
    G = nx.DiGraph()
    for node_id, data in nodes.items():
        G.add_node(node_id, tvl=float(data.get("tvlUsd", 0.0)))

    for edge in edges:
        src, dst = edge.get("source"), edge.get("target")
        w = float(edge.get("weight", 0.0))
        if src in nodes and dst in nodes and w > 0:
            G.add_edge(src, dst, weight=w)

    try:
        pagerank_raw = nx.pagerank(G, weight="weight", max_iter=pagerank_max_iter)
    except Exception:
        pagerank_raw = {n: 0.0 for n in nodes}

    pr_max = max(pagerank_raw.values()) if pagerank_raw else 0.0
    if pr_max > 0:
        pagerank = {n: float(pagerank_raw.get(n, 0.0)) / pr_max for n in nodes}
    else:
        pagerank = {n: 0.0 for n in nodes}

    # 2) TailExposure: concentration of outgoing exposures
    outgoing_weights: Dict[str, List[float]] = {n: [] for n in nodes}
    for edge in edges:
        src = edge.get("source")
        w = float(edge.get("weight", 0.0))
        if src in outgoing_weights and w > 0:
            outgoing_weights[src].append(w)

    tail_exposure: Dict[str, float] = {}
    for n, ws in outgoing_weights.items():
        if not ws:
            tail_exposure[n] = 0.0
            continue
        total = float(np.sum(ws))
        if total <= 0:
            tail_exposure[n] = 0.0
            continue
        top_k_sum = float(np.sum(sorted(ws, reverse=True)[:tail_k]))
        tail_exposure[n] = top_k_sum / total

    # 3) log(1+TVL) term (normalized to [0,1])
    tvl_values = {n: float(data.get("tvlUsd", 0.0)) for n, data in nodes.items()}
    log_tvl = {n: float(np.log1p(max(v, 0.0))) for n, v in tvl_values.items()}
    log_tvl_max = max(log_tvl.values()) if log_tvl else 0.0
    if log_tvl_max > 0:
        log_tvl_norm = {n: v / log_tvl_max for n, v in log_tvl.items()}
    else:
        log_tvl_norm = {n: 0.0 for n in nodes}

    return {"pagerank": pagerank, "tail_exposure": tail_exposure, "log_tvl_norm": log_tvl_norm}


def compute_sector_spillover_index(
    snap: Dict,
    sector_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute sector-level spillover index based on cross-sector edge weights.
    """
    nodes = snap.get("nodes", {})
    edges = snap.get("edges", [])

    # Default sector mapping based on protocol category
    if sector_map is None:
        sector_map = {}
        for node_id, data in nodes.items():
            category = data.get("category", "other")
            sector_map[node_id] = category

    # Aggregate cross-sector exposures
    sector_exposure = {}
    for edge in edges:
        src, dst = edge.get("source"), edge.get("target")
        weight = edge.get("weight", 0)

        src_sector = sector_map.get(src, "other")
        dst_sector = sector_map.get(dst, "other")

        if src_sector != dst_sector:
            if src_sector not in sector_exposure:
                sector_exposure[src_sector] = {}
            if dst_sector not in sector_exposure[src_sector]:
                sector_exposure[src_sector][dst_sector] = 0
            sector_exposure[src_sector][dst_sector] += weight

    return sector_exposure


def simulate_contagion(
    snap: Dict,
    shocked_nodes: List[str],
    shock_fraction: float = 1.0,
    distress_threshold: float = 0.1,
    max_rounds: int = 10,
) -> Dict[str, Any]:
    """
    Simulate DebtRank-style contagion from shocked nodes.

    Args:
        snap: Network snapshot with nodes and edges
        shocked_nodes: List of initially shocked node IDs
        shock_fraction: Fraction of TVL lost by shocked nodes (0-1)
        distress_threshold: Loss/TVL ratio that triggers distress
        max_rounds: Maximum propagation rounds

    Returns:
        Dictionary with contagion results
    """
    nodes = snap.get("nodes", {})
    edges = snap.get("edges", [])

    # Initialize TVL and losses
    tvl = {n: float(data.get("tvlUsd", 0.0) or 0.0) for n, data in nodes.items()}
    losses = {n: 0.0 for n in nodes}

    # Build exposure index for propagation.
    # We treat an edge `creditor -> debtor` as "creditor has exposure to debtor".
    # When debtor is distressed, losses propagate to its creditors (incoming neighbors).
    exposures_in = {}
    for edge in edges:
        src, dst = edge.get("source"), edge.get("target")
        weight = float(edge.get("weight", 0) or 0.0)
        if weight <= 0.0:
            continue
        if src is None or dst is None:
            continue
        if dst not in exposures_in:
            exposures_in[dst] = {}
        exposures_in[dst][src] = exposures_in[dst].get(src, 0.0) + weight

    # Initial shock
    distressed = set()
    active = set()
    for node in shocked_nodes:
        if node in tvl:
            tvl_n = float(tvl[node])
            if tvl_n <= 0:
                continue
            losses[node] = shock_fraction * tvl_n
            distressed.add(node)
            active.add(node)

    # Propagation rounds
    affected_history = [len(distressed)]
    for round_num in range(max_rounds):
        if not active:
            break

        new_active = set()
        for node in active:
            creditors = exposures_in.get(node, {})
            if not creditors:
                continue
            total_exposure = float(np.sum(list(creditors.values())))
            if total_exposure <= 0:
                continue

            node_loss = float(losses.get(node, 0.0))
            if node_loss <= 0:
                continue

            # Propagate debtor losses to its creditors proportionally.
            for creditor, exposure in creditors.items():
                exposure = float(exposure)
                if exposure <= 0:
                    continue
                if creditor not in tvl:
                    continue

                loss_share = exposure / total_exposure
                propagated_loss = node_loss * loss_share
                losses[creditor] += propagated_loss

                tvl_c = float(tvl.get(creditor, 0.0))
                if tvl_c > 0:
                    losses[creditor] = min(losses[creditor], tvl_c)

                    # Check if creditor becomes distressed (only once, when crossing threshold)
                    if creditor not in distressed and losses[creditor] > distress_threshold * tvl_c:
                        distressed.add(creditor)
                        new_active.add(creditor)

        if not new_active:
            break

        active = new_active
        affected_history.append(len(distressed))

    # Compute aggregate statistics
    total_tvl = sum(tvl.values())
    total_loss = sum(losses.values())

    return {
        "shocked_nodes": shocked_nodes,
        "shock_fraction": shock_fraction,
        "total_loss": total_loss,
        "total_loss_pct": 100.0 * total_loss / total_tvl if total_tvl > 0 else 0,
        "loss_fraction": total_loss / total_tvl if total_tvl > 0 else 0,
        "affected_count": len(distressed),
        "distressed_count": len(distressed),
        "distressed_nodes": list(distressed),
        "propagation_rounds": len(affected_history) - 1,
        "affected_history": affected_history,
    }

# ============== Configuration ==============

OUTPUT_DIR = Path("output/task2_model_based")
MODEL_CACHE_DIR = Path("output/model_cache")

SHOCK_EVENTS = {
    "terra_luna": {
        "name": "Terra/Luna Collapse",
        "event_date": "2022-05-09",
        "pre_window": ("2022-03-28", "2022-05-02"),  # 5 weeks before
        "event_window": ("2022-05-02", "2022-05-23"),
        "description": "UST algorithmic stablecoin depeg and death spiral",
    },
    "ftx": {
        "name": "FTX Collapse",
        "event_date": "2022-11-07",
        "pre_window": ("2022-09-26", "2022-10-31"),  # 5 weeks before
        "event_window": ("2022-10-31", "2022-11-21"),
        "description": "Exchange failure and FTT token collapse",
    },
}


# ============== Visualization Functions ==============

COLORS = {
    "predicted": "#2E86AB",  # Blue
    "actual": "#E94F37",     # Red
    "observed": "#7B8D8E",   # Gray
    "event": "#F39C12",      # Orange
}


def plot_early_warning_timeseries(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Plot Early Warning time series for Terra/Luna and FTX events.

    Creates a 2-row figure showing:
    - Risk concentration (HHI): Predicted vs Actual
    - Forward-looking stress loss: Predicted vs Actual
    """
    plt = _import_matplotlib()

    events_data = results.get("events", {})
    if not events_data:
        logger.warning("No early warning data to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Early Warning Analysis: Model Predictions Before Major Events", fontsize=14, fontweight="bold")

    preferred_order = [k for k in ["terra_luna", "ftx"] if k in events_data]
    remaining = [k for k in events_data.keys() if k not in preferred_order]
    event_ids = preferred_order + remaining

    for row_idx, event_id in enumerate(event_ids):
        if row_idx >= 2:
            break
        event_data = events_data[event_id]

        event_info = event_data.get("event_info", {})
        predictions = event_data.get("predictions", [])
        baseline = event_data.get("baseline_metrics", {})

        if not predictions:
            continue

        event_name = event_info.get("name", event_id)
        event_date = event_info.get("event_date", "")

        # Extract time series data
        dates = [p["target_date"] for p in predictions]
        pred_hhi = [p["predicted_metrics"].get("tvl_hhi", np.nan) for p in predictions]
        actual_hhi = [p["actual_metrics"].get("tvl_hhi", np.nan) for p in predictions]
        pred_contagion = [p.get("predicted_contagion_loss", np.nan) for p in predictions]
        actual_contagion = [p.get("actual_contagion_loss", np.nan) for p in predictions]

        baseline_hhi = baseline.get("tvl_hhi", np.nan)

        # Plot HHI (left column)
        ax1 = axes[row_idx, 0]
        x_pos = np.arange(len(dates))

        ax1.plot(x_pos, pred_hhi, "o-", color=COLORS["predicted"], label="Predicted", linewidth=2, markersize=6)
        ax1.plot(x_pos, actual_hhi, "s--", color=COLORS["actual"], label="Actual", linewidth=2, markersize=6)
        ax1.axhline(y=baseline_hhi, color=COLORS["observed"], linestyle=":", linewidth=1.5, label=f"Baseline (pre-event)")

        # Mark event window
        ax1.axvspan(-0.5, len(dates)-0.5, alpha=0.1, color=COLORS["event"], label="Event window")

        ax1.set_xlabel("Weeks into event period")
        ax1.set_ylabel("TVL HHI (Concentration)")
        title_suffix = f" (event {event_date})" if event_date else ""
        ax1.set_title(f"{event_name}{title_suffix}: Risk Concentration")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"W{i+1}" for i in range(len(dates))], rotation=45)
        ax1.legend(loc="best", framealpha=0.9)
        ax1.set_ylim(bottom=0)

        # Plot Contagion Loss (right column)
        ax2 = axes[row_idx, 1]

        pred_arr = np.array(pred_contagion, dtype=float)
        actual_arr = np.array(actual_contagion, dtype=float)

        bar_pred = np.nan_to_num(pred_arr, nan=0.0)
        bar_actual = np.nan_to_num(actual_arr, nan=0.0)

        width = 0.35
        ax2.bar(x_pos - width/2, bar_pred, width, color=COLORS["predicted"], label="Predicted", alpha=0.8)
        ax2.bar(x_pos + width/2, bar_actual, width, color=COLORS["actual"], label="Actual", alpha=0.8)

        ax2.set_xlabel("Weeks into event period")
        ax2.set_ylabel("Contagion Loss (%)")
        ax2.set_title(f"{event_name}: Stress Test Loss")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"W{i+1}" for i in range(len(dates))], rotation=45)
        ax2.legend(loc="best", framealpha=0.9)
        ax2.set_ylim(bottom=0)

        # Add annotation for prediction accuracy
        valid_mask = ~np.isnan(pred_arr) & ~np.isnan(actual_arr)
        if np.any(valid_mask):
            mae = float(np.mean(np.abs(pred_arr[valid_mask] - actual_arr[valid_mask])))
            ax2.text(
                0.95,
                0.95,
                f"MAE: {mae:.2f}%",
                transform=ax2.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    plt.tight_layout()

    fig_path = output_dir / "figures" / "fig_early_warning.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path)
    plt.savefig(fig_path.with_suffix(".png"))
    plt.close()

    logger.info(f"Saved Early Warning figure to {fig_path}")


def plot_contagion_comparison(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Plot Contagion simulation comparison: Observed(t) vs Predicted(t+h) vs Actual(t+h).

    Creates grouped bar chart showing system loss across different shock scenarios.

    Note: The naive baseline is the observed network at time t (\"Observed (t)\"),
    which corresponds to a persistence assumption for stress testing.
    """
    horizons_data = results.get("horizons", {})
    if not horizons_data:
        logger.warning("No contagion data to plot")
        return

    plt = _import_matplotlib()

    scenarios = [s.get("name") for s in results.get("scenarios", []) if isinstance(s, dict) and s.get("name")]
    if not scenarios:
        scenarios = ["Top Protocol Shock", "Top 5 Protocols Shock", "Bridge Sector Shock"]

    def _short_scenario_label(name: str) -> str:
        n = (name or "").lower()
        if "top" in n and ("protocol" in n or "protocols" in n):
            if "5" in n:
                return "Top-5"
            return "Top-1"
        if "bridge" in n:
            return "Bridge"
        return name

    def _parse_horizon_key(key: str) -> int:
        # Expecting "h=1", "h=4", ...
        try:
            return int(key.split("=")[1])
        except Exception:
            return 10**9

    horizon_items = sorted(horizons_data.items(), key=lambda kv: _parse_horizon_key(kv[0]))

    # Use a 2x2 layout (or 2 columns for fewer horizons) for readability in paper.
    n_panels = len(horizon_items)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 4.8 * nrows),
        sharey=True,
    )
    axes = np.array(axes).reshape(-1).tolist()

    fig.suptitle("Predictive Contagion: Model vs Naive Baseline", fontsize=14, fontweight="bold", y=0.98)

    for ax_idx, (horizon_key, horizon_data) in enumerate(horizon_items):
        ax = axes[ax_idx]
        samples = horizon_data.get("samples", [])

        if not samples:
            continue

        # Aggregate across samples for each scenario
        scenario_results = {s: {"observed": [], "predicted": [], "actual": []} for s in scenarios}

        for sample in samples:
            for scenario_name, scenario_data in sample.get("scenarios", {}).items():
                if scenario_name in scenario_results:
                    scenario_results[scenario_name]["observed"].append(
                        scenario_data.get("observed_t", {}).get("total_loss_pct", 0)
                    )
                    scenario_results[scenario_name]["predicted"].append(
                        scenario_data.get("predicted_t_h", {}).get("total_loss_pct", 0)
                    )
                    scenario_results[scenario_name]["actual"].append(
                        scenario_data.get("actual_t_h", {}).get("total_loss_pct", 0)
                    )

        # Plot grouped bars
        x = np.arange(len(scenarios))
        width = 0.25

        observed_means = [np.mean(scenario_results[s]["observed"]) if scenario_results[s]["observed"] else 0 for s in scenarios]
        predicted_means = [np.mean(scenario_results[s]["predicted"]) if scenario_results[s]["predicted"] else 0 for s in scenarios]
        actual_means = [np.mean(scenario_results[s]["actual"]) if scenario_results[s]["actual"] else 0 for s in scenarios]

        observed_stds = [np.std(scenario_results[s]["observed"]) if len(scenario_results[s]["observed"]) > 1 else 0 for s in scenarios]
        predicted_stds = [np.std(scenario_results[s]["predicted"]) if len(scenario_results[s]["predicted"]) > 1 else 0 for s in scenarios]
        actual_stds = [np.std(scenario_results[s]["actual"]) if len(scenario_results[s]["actual"]) > 1 else 0 for s in scenarios]

        ax.bar(
            x - width,
            observed_means,
            width,
            yerr=observed_stds,
            label="Baseline (Obs t)",
            color=COLORS["observed"],
            alpha=0.8,
            capsize=3,
        )
        ax.bar(
            x,
            predicted_means,
            width,
            yerr=predicted_stds,
            label="Model (Pred t+h)",
            color=COLORS["predicted"],
            alpha=0.8,
            capsize=3,
        )
        ax.bar(
            x + width,
            actual_means,
            width,
            yerr=actual_stds,
            label="Actual (t+h)",
            color=COLORS["actual"],
            alpha=0.8,
            capsize=3,
        )

        ax.set_xlabel("Shock Scenario")
        ax.set_ylabel("System Loss (%)" if (ax_idx % ncols) == 0 else "")
        ax.set_title(f"Horizon {horizon_key}")
        ax.set_xticks(x)
        ax.set_xticklabels([_short_scenario_label(s) for s in scenarios], rotation=0)
        # Add a small headroom so the summary box does not overlap bars.
        upper_vals = []
        upper_vals.extend([m + s for m, s in zip(observed_means, observed_stds)])
        upper_vals.extend([m + s for m, s in zip(predicted_means, predicted_stds)])
        upper_vals.extend([m + s for m, s in zip(actual_means, actual_stds)])
        y_max = max(upper_vals) if upper_vals else 0.0
        if y_max > 0:
            # Extra headroom so the in-axes legend and MAE box don't cover bars.
            ax.set_ylim(0, y_max * 1.40)
        else:
            ax.set_ylim(bottom=0)

        # Match Early Warning style: legend inside axes with a light frame.
        ax.legend(loc="best", framealpha=0.9, ncol=1)

        # Horizon-level summary annotation (uses MAE across samples, not MAE of means)
        pred_all = np.concatenate([np.abs(np.array(scenario_results[s]["predicted"]) - np.array(scenario_results[s]["actual"])) for s in scenarios if scenario_results[s]["predicted"] and scenario_results[s]["actual"]])
        base_all = np.concatenate([np.abs(np.array(scenario_results[s]["observed"]) - np.array(scenario_results[s]["actual"])) for s in scenarios if scenario_results[s]["observed"] and scenario_results[s]["actual"]])
        pred_overall = float(pred_all.mean()) if pred_all.size else 0.0
        base_overall = float(base_all.mean()) if base_all.size else 0.0
        improvement = base_overall - pred_overall
        ax.text(
            0.95,
            0.95,
            f"MAE(model): {pred_overall:.2f}%\nMAE(baseline): {base_overall:.2f}%\nΔ: {improvement:+.2f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Hide any unused panels (e.g., if horizons != 4).
    for extra_ax in axes[len(horizon_items):]:
        extra_ax.set_visible(False)

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    fig_path = output_dir / "figures" / "fig_contagion_comparison.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path)
    plt.savefig(fig_path.with_suffix(".png"))
    plt.close()

    logger.info(f"Saved Contagion Comparison figure to {fig_path}")


def plot_contagion_advantage(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Plot "advantage regimes" where the model beats the persistence baseline.

    Uses Exp2 post-processing fields (delta_mae_all vs delta_mae_worst) computed in
    `run_predictive_contagion()`. Positive values mean the model improves over baseline.
    """
    horizons_data = results.get("horizons", {})
    if not horizons_data:
        logger.warning("No contagion data to plot (advantage)")
        return

    plt = _import_matplotlib()

    scenarios = [s.get("name") for s in results.get("scenarios", []) if isinstance(s, dict) and s.get("name")]
    if not scenarios:
        scenarios = ["Top Protocol Shock", "Top 5 Protocols Shock", "Bridge Sector Shock"]

    has_adv = any(
        f"{scenario_name}_advantage" in horizon_data
        for horizon_data in horizons_data.values()
        for scenario_name in scenarios
    )
    if not has_adv:
        logger.warning("Contagion advantage stats not found in results; rerun predictive_contagion to generate them.")
        return

    def _short_scenario_label(name: str) -> str:
        n = (name or "").lower()
        if "top" in n and ("protocol" in n or "protocols" in n):
            if "5" in n:
                return "Top-5"
            return "Top-1"
        if "bridge" in n:
            return "Bridge"
        return name

    def _parse_horizon_key(key: str) -> int:
        try:
            return int(key.split("=")[1])
        except Exception:
            return 10**9

    horizon_items = sorted(horizons_data.items(), key=lambda kv: _parse_horizon_key(kv[0]))

    # Use a 2x2 layout (or 2 columns for fewer horizons) for readability in paper.
    n_panels = len(horizon_items)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 4.4 * nrows),
        sharey=True,
    )
    axes = np.array(axes).reshape(-1).tolist()

    worst_frac = None
    for _, hdata in horizon_items:
        ws = hdata.get("advantage_overall", {}).get("worst_frac")
        if ws is not None:
            worst_frac = float(ws)
            break
    if worst_frac is None:
        worst_frac = 0.2

    fig.suptitle(
        "Predictive Contagion: Where the Model Beats Persistence",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for ax_idx, (horizon_key, horizon_data) in enumerate(horizon_items):
        ax = axes[ax_idx]

        overall_impr = []
        worst_impr = []
        for s in scenarios:
            adv = horizon_data.get(f"{s}_advantage", {})
            overall_impr.append(float(adv.get("delta_mae_all", np.nan)))
            worst_impr.append(float(adv.get("delta_mae_worst", np.nan)))

        x = np.arange(len(scenarios))
        width = 0.35

        ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.35)
        ax.bar(
            x - width / 2,
            overall_impr,
            width,
            color=COLORS["observed"],
            alpha=0.85,
            label="Overall",
        )
        ax.bar(
            x + width / 2,
            worst_impr,
            width,
            color=COLORS["predicted"],
            alpha=0.85,
            label=f"Worst {int(round(100 * worst_frac))}%",
        )

        ax.set_title(f"Horizon {horizon_key}")
        ax.set_xlabel("Shock Scenario")
        ax.set_ylabel("ΔMAE (baseline − model) [% pts]" if (ax_idx % ncols) == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels([_short_scenario_label(s) for s in scenarios], rotation=0)

        # Symmetric y-limits for readability.
        vals = np.array(overall_impr + worst_impr, dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size:
            m = float(np.max(np.abs(vals)))
            if m > 0:
                ax.set_ylim(-1.25 * m, 1.25 * m)

        ax.legend(loc="best", framealpha=0.9)

        overall = horizon_data.get("advantage_overall", {})
        if overall:
            ax.text(
                0.95,
                0.95,
                f"Δ(all): {overall.get('delta_mae_all', float('nan')):+.2f}%\n"
                f"Δ(worst): {overall.get('delta_mae_worst', float('nan')):+.2f}%\n"
                f"Win@worst: {100*float(overall.get('win_rate_worst', float('nan'))):.0f}%",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    # Hide any unused panels (e.g., if horizons != 4).
    for extra_ax in axes[len(horizon_items):]:
        extra_ax.set_visible(False)

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    fig_path = output_dir / "figures" / "fig_contagion_advantage.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path)
    plt.savefig(fig_path.with_suffix(".png"))
    plt.close()

    logger.info(f"Saved Contagion Advantage figure to {fig_path}")


def plot_forward_risk_scatter(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Plot Forward Risk Prediction scatter plots: Predicted vs Actual for key metrics.

    Creates 2x2 panel showing correlation between predicted and actual values
    for HHI, Density, SIS, and Spillover metrics across horizons.
    """
    horizons_data = results.get("horizons", {})
    if not horizons_data:
        logger.warning("No forward risk data to plot")
        return

    plt = _import_matplotlib()

    def _parse_horizon_key(key: str) -> int:
        try:
            return int(key.split("=")[1])
        except Exception:
            return 10**9

    horizon_items = sorted(horizons_data.items(), key=lambda kv: _parse_horizon_key(kv[0]))

    metrics_to_plot = [
        ("tvl_hhi", "TVL Concentration (HHI)"),
        ("density", "Network Density"),
        ("mean_sis", "Mean Systemic Importance (SIS)"),
        ("spillover_index", "Cross-sector Spillover Concentration (HHI)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Forward-looking Risk Metric Prediction Accuracy", fontsize=14, fontweight="bold")

    horizon_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(horizon_items)))

    for ax_idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        ax = axes[ax_idx // 2, ax_idx % 2]

        all_pred = []
        all_actual = []

        for h_idx, (horizon_key, horizon_data) in enumerate(horizon_items):
            pred_metrics = horizon_data.get("predicted_metrics", [])
            actual_metrics = horizon_data.get("actual_metrics", [])

            pred_vals = [m.get(metric_key, np.nan) for m in pred_metrics]
            actual_vals = [m.get(metric_key, np.nan) for m in actual_metrics]

            # Filter valid pairs
            pred_arr = np.array(pred_vals, dtype=float)
            actual_arr = np.array(actual_vals, dtype=float)
            valid_mask = ~np.isnan(pred_arr) & ~np.isnan(actual_arr)
            pred_arr = pred_arr[valid_mask]
            actual_arr = actual_arr[valid_mask]

            if len(pred_arr) > 0:
                ax.scatter(actual_arr, pred_arr, c=[horizon_colors[h_idx]],
                          label=horizon_key, alpha=0.6, s=30, edgecolors="white", linewidth=0.5)
                all_pred.extend(pred_arr.tolist())
                all_actual.extend(actual_arr.tolist())

        if all_pred and all_actual:
            # Add 45-degree reference line
            all_vals = np.concatenate([all_pred, all_actual])
            min_val, max_val = np.min(all_vals), np.max(all_vals)
            margin = (max_val - min_val) * 0.1
            line_range = [min_val - margin, max_val + margin]
            ax.plot(line_range, line_range, "k--", alpha=0.5, linewidth=1, label="Perfect prediction")

            # Compute R²
            corr = np.corrcoef(all_pred, all_actual)[0, 1]
            r_squared = corr ** 2
            ax.text(0.05, 0.95, f"$R^2$ = {r_squared:.3f}", transform=ax.transAxes,
                   ha="left", va="top", fontsize=10,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.set_xlim(line_range)
            ax.set_ylim(line_range)

        ax.set_xlabel(f"Actual {metric_label}")
        ax.set_ylabel(f"Predicted {metric_label}")
        ax.set_title(metric_label)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    fig_path = output_dir / "figures" / "fig_forward_risk_scatter.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path)
    plt.savefig(fig_path.with_suffix(".png"))
    plt.close()

    logger.info(f"Saved Forward Risk Scatter figure to {fig_path}")


def plot_all_figures(
    forward_risk_results: Optional[Dict] = None,
    contagion_results: Optional[Dict] = None,
    early_warning_results: Optional[Dict] = None,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Generate all Task II figures."""
    logger.info("Generating Task II visualization figures...")

    if forward_risk_results:
        plot_forward_risk_scatter(forward_risk_results, output_dir)

    if contagion_results:
        plot_contagion_comparison(contagion_results, output_dir)
        plot_contagion_advantage(contagion_results, output_dir)

    if early_warning_results:
        plot_early_warning_timeseries(early_warning_results, output_dir)

    logger.info(f"All figures saved to {output_dir / 'figures'}")


# ============== Results I/O ==============


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_saved_task2_results(output_dir: Path) -> Dict[str, Any]:
    """
    Load previously saved results from `output_dir`.

    Prefers the consolidated `all_results.json`; fills any missing experiments from per-experiment json files.
    Returns a dict with optional keys: forward_risk, predictive_contagion, early_warning.
    """
    results: Dict[str, Any] = {}

    all_path = output_dir / "all_results.json"
    if all_path.exists():
        data = _load_json(all_path)
        if isinstance(data, dict):
            results.update(data)

    paths = {
        "forward_risk": output_dir / "exp1_forward_risk.json",
        "predictive_contagion": output_dir / "exp2_predictive_contagion.json",
        "early_warning": output_dir / "exp3_early_warning.json",
    }
    for key, path in paths.items():
        if key not in results and path.exists():
            results[key] = _load_json(path)
    return results


def parse_int_list(spec: str) -> List[int]:
    """
    Parse a comma-separated integer list like "1,4,8,12".

    Returns [] if spec is empty.
    """
    spec = (spec or "").strip()
    if not spec:
        return []
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    values: List[int] = []
    for p in parts:
        try:
            values.append(int(p))
        except ValueError as e:
            raise ValueError(f"Invalid integer in list: {p!r} (spec={spec!r})") from e
    return values


# ============== Model Loading ==============

def load_or_train_model(
    config: ExperimentConfig,
    train_pairs: List[WeekPair],
    force_retrain: bool = False,
    frozen: bool = False,
    cache_tag: Optional[str] = None,
    horizon: int = 1,
) -> GraphPFNLinkPredictor:
    """
    Load cached model or train if not exists.

    Args:
        config: Experiment configuration
        train_pairs: Training data pairs
        force_retrain: Force retraining even if cache exists
        frozen: If True, use frozen encoder (faster); if False, fine-tune (better)
        cache_tag: Optional tag for isolated caches (e.g., per-event models).
        horizon: Prediction horizon used to build `train_pairs`. We cache models per-horizon
            to match `run_full_experiment.py` (separate model per horizon).
    """
    _import_graphpfn_deps()
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Different cache paths for frozen vs finetuned
    model_name = "graphpfn_frozen" if frozen else "dexposure_fm"
    name_parts = [model_name, f"h{int(horizon)}"]
    if cache_tag:
        safe_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in cache_tag)
        name_parts.append(safe_tag)
    cache_stem = "__".join(name_parts)
    cache_path = MODEL_CACHE_DIR / f"{cache_stem}.pt"
    legacy_cache_path = MODEL_CACHE_DIR / f"{model_name}.pt"
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
        # Backward compatible: older runs cached only the h=1 model without the horizon suffix.
        logger.info(f"Loading legacy cached model from {legacy_cache_path} (treating as h=1)")
        state = torch.load(legacy_cache_path, map_location=device)
        model.load_state_dict(state["model"], strict=True)
        return model

    # Set encoder trainability
    finetune_encoder = not frozen
    for p in model.encoder.parameters():
        p.requires_grad = finetune_encoder

    mode_name = "GraphPFN-Frozen" if frozen else "DeXposure-FM (fine-tuned)"
    logger.info(f"Training {mode_name} (h={horizon})...")

    # Match `run_full_experiment.py` optimizer setup (layer-wise LR + weight decay).
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
            f"  Optimizer: layer-wise LR (encoder={float(config.lr)*0.1:.1e}, head={float(config.lr):.1e}), wd={weight_decay:.1e}"
        )
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            trainable_params, lr=float(config.lr), weight_decay=weight_decay
        )
        logger.info(f"  Optimizer: frozen head-only (lr={float(config.lr):.1e}, wd={weight_decay:.1e})")

    scaler = None
    if bool(getattr(config, "use_amp", False)) and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        logger.info("  AMP enabled (mixed precision training)")

    embedding_cache = None
    if (not finetune_encoder) and bool(getattr(config, "cache_frozen_embeddings", False)):
        embedding_cache = EmbeddingCache(model, device)
        logger.info("  Embedding cache enabled (frozen encoder)")

    best_state: Optional[Dict[str, Any]] = None
    best_loss = float("inf")
    prev_embeddings = None

    for epoch in range(config.epochs):
        losses, prev_embeddings = train_graphpfn_epoch(
            model, train_pairs, optimizer, config,
            finetune_encoder=finetune_encoder, prev_embeddings=prev_embeddings,
            scaler=scaler, embedding_cache=embedding_cache,
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
            logger.info(f"  Epoch {epoch+1}: exist={losses['exist_loss']:.4f}, "
                       f"weight={losses['weight_loss']:.4f}, node={losses['node_loss']:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    torch.save({"model": model.state_dict()}, cache_path)
    logger.info(f"Model saved to {cache_path}")

    return model


# ============== Network Reconstruction from Predictions ==============

def reconstruct_network_from_predictions(
    pred: Dict[str, Any],
    pair: WeekPair,
    edge_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Reconstruct a network snapshot from model predictions.

    Args:
        pred: Model prediction output
        pair: Original WeekPair with node info
        edge_threshold: Probability threshold for edge existence

    Returns:
        Reconstructed snapshot dictionary compatible with risk analysis functions
    """
    # Get node_mask: only include nodes that exist at both t and t+h
    node_mask = pair.node_mask

    # Edge predictions - filter edges to only include valid nodes
    exist_prob = 1 / (1 + np.exp(-pred["exist_logits"]))
    edge_mask = exist_prob > edge_threshold

    pred_src = pair.pair_src[edge_mask]
    pred_dst = pair.pair_dst[edge_mask]
    pred_weights = pred["weight_pred"][edge_mask]

    # Further filter edges: both endpoints must be valid nodes
    valid_edge_mask = node_mask[pred_src] & node_mask[pred_dst]
    pred_src = pred_src[valid_edge_mask]
    pred_dst = pred_dst[valid_edge_mask]
    pred_weights = pred_weights[valid_edge_mask]

    # Node predictions (TVL change) - only for valid nodes
    sizes_t = pair.sizes_t
    log_size_t = np.log1p(np.maximum(sizes_t, 0))
    pred_log_tvl = log_size_t + pred["node_pred"]
    pred_sizes = np.expm1(np.maximum(pred_log_tvl, 0))

    # Filter to valid nodes only
    valid_node_ids = [nid for i, nid in enumerate(pair.node_ids) if node_mask[i]]
    valid_categories = [cat for i, cat in enumerate(pair.categories) if node_mask[i]]
    valid_sizes = pred_sizes[node_mask]
    valid_features = pair.features_t[node_mask] if pair.features_t is not None else None

    # Remap edge indices to new node ordering
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(node_mask)[0])}
    remapped_src = np.array([old_to_new[s] for s in pred_src])
    remapped_dst = np.array([old_to_new[d] for d in pred_dst])

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


def reconstruct_actual_network(
    pred: Dict[str, Any],
    pair: WeekPair,
) -> Dict[str, Any]:
    """Reconstruct actual network at t+h from ground truth.

    Only includes nodes that exist at both t and t+h (node_mask=True).
    """
    # Get node_mask: only include nodes that exist at both t and t+h
    node_mask = pair.node_mask

    # Edge selection based on ground truth
    edge_mask = pair.y_exist > 0.5
    actual_src = pair.pair_src[edge_mask]
    actual_dst = pair.pair_dst[edge_mask]
    actual_weights = pair.y_weight[edge_mask]

    # Further filter edges: both endpoints must be valid nodes
    valid_edge_mask = node_mask[actual_src] & node_mask[actual_dst]
    actual_src = actual_src[valid_edge_mask]
    actual_dst = actual_dst[valid_edge_mask]
    actual_weights = actual_weights[valid_edge_mask]

    # Node sizes from ground truth
    sizes_t = pair.sizes_t
    log_size_t = np.log1p(np.maximum(sizes_t, 0))
    actual_log_tvl = log_size_t + pair.y_node
    actual_sizes = np.expm1(np.maximum(actual_log_tvl, 0))

    # Filter to valid nodes only
    valid_node_ids = [nid for i, nid in enumerate(pair.node_ids) if node_mask[i]]
    valid_categories = [cat for i, cat in enumerate(pair.categories) if node_mask[i]]
    valid_sizes = actual_sizes[node_mask]
    valid_features = pair.features_t[node_mask] if pair.features_t is not None else None

    # Remap edge indices to new node ordering
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(node_mask)[0])}
    remapped_src = np.array([old_to_new[s] for s in actual_src])
    remapped_dst = np.array([old_to_new[d] for d in actual_dst])

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


# ============== Risk Metrics Computation ==============

def compute_network_risk_metrics(
    snap: Dict[str, Any],
    meta_category: Dict,
    edge_weight_is_log: bool = True,
) -> Dict[str, float]:
    """
    Compute comprehensive risk metrics on a network snapshot.

    Args:
        snap: Network snapshot in array format (node_ids, sizes, edge_src, edge_dst, edge_weight)
        meta_category: Protocol category mapping
        edge_weight_is_log: If True, edge_weight is log1p-scaled; if False, linear

    Returns metrics that are relevant for financial stability assessment.
    """
    node_ids = snap["node_ids"]
    sizes = np.array(snap["sizes"], dtype=np.float64)
    edge_src = snap["edge_src"]
    edge_dst = snap["edge_dst"]
    edge_weight = snap["edge_weight"]

    n_nodes = len(node_ids)
    n_edges = len(edge_src)

    metrics = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "total_tvl": float(np.sum(sizes)),
    }

    if n_nodes == 0:
        return metrics

    # 1. TVL concentration (HHI)
    total_tvl = np.sum(sizes)
    if total_tvl > 0:
        tvl_shares = sizes / total_tvl
        metrics["tvl_hhi"] = float(np.sum(tvl_shares ** 2))
        sorted_shares = np.sort(tvl_shares)[::-1]
        metrics["top10_tvl_share"] = float(np.sum(sorted_shares[:10]))
        metrics["tvl_gini"] = gini_coefficient(sizes)
    else:
        metrics["tvl_hhi"] = 0.0
        metrics["top10_tvl_share"] = 0.0
        metrics["tvl_gini"] = 0.0

    # 2. Edge weight concentration
    if len(edge_weight) > 0:
        # Convert to linear scale if needed
        if edge_weight_is_log:
            weights_linear = np.expm1(np.array(edge_weight))
        else:
            weights_linear = np.array(edge_weight)
        weights_linear = np.maximum(weights_linear, 0)
        total_weight = np.sum(weights_linear)
        if total_weight > 0:
            weight_shares = weights_linear / total_weight
            metrics["edge_hhi"] = float(np.sum(weight_shares ** 2))
            sorted_weights = np.sort(weight_shares)[::-1]
            metrics["top10_edge_share"] = float(np.sum(sorted_weights[:10]))
        else:
            metrics["edge_hhi"] = 0.0
            metrics["top10_edge_share"] = 0.0
    else:
        metrics["edge_hhi"] = 0.0
        metrics["top10_edge_share"] = 0.0

    # 3. Network density
    max_edges = n_nodes * (n_nodes - 1)
    metrics["density"] = n_edges / max_edges if max_edges > 0 else 0.0

    # 4. Degree distribution
    in_degree = np.zeros(n_nodes)
    out_degree = np.zeros(n_nodes)
    for src, dst in zip(edge_src, edge_dst):
        if src < n_nodes:
            out_degree[src] += 1
        if dst < n_nodes:
            in_degree[dst] += 1

    metrics["mean_in_degree"] = float(np.mean(in_degree))
    metrics["max_in_degree"] = float(np.max(in_degree)) if n_nodes > 0 else 0
    metrics["degree_gini"] = gini_coefficient(in_degree + out_degree)

    # 5. SIS scores (if available)
    # Need to convert array format to dict format for compute_systemic_importance_score
    try:
        dict_snap = array_snap_to_dict_snap(snap, edge_weight_is_log=edge_weight_is_log)
        sis = compute_systemic_importance_score(dict_snap)
        if sis:
            sis_values = list(sis.values())
            metrics["mean_sis"] = float(np.mean(sis_values))
            metrics["max_sis"] = float(np.max(sis_values))
            metrics["sis_gini"] = gini_coefficient(sis_values)
    except Exception as e:
        logger.warning(f"Could not compute SIS: {e}")

    # 6. Sector spillover (if meta_category available)
    # Note: compute_sector_spillover_index returns a dict of sector->sector->weight
    # We compute:
    # - spillover_total: total cross-sector exposure (scale-dependent)
    # - spillover_index: HHI concentration of off-diagonal sector exposures (scale-invariant)
    try:
        dict_snap = array_snap_to_dict_snap(snap, edge_weight_is_log=edge_weight_is_log)
        sector_exposure = compute_sector_spillover_index(dict_snap, meta_category)
        off_diag_weights = [
            float(w)
            for targets in sector_exposure.values()
            for w in targets.values()
            if float(w) > 0
        ]
        total_spillover = float(np.sum(off_diag_weights)) if off_diag_weights else 0.0
        metrics["spillover_total"] = total_spillover

        if total_spillover > 0:
            shares = np.array(off_diag_weights, dtype=float) / total_spillover
            metrics["spillover_index"] = float(np.sum(shares**2))
        else:
            metrics["spillover_index"] = 0.0
    except Exception as e:
        logger.warning(f"Could not compute spillover: {e}")

    return metrics


# ============== Robustness: SIS Weight Sensitivity ==============


def _rank_desc(values: np.ndarray) -> np.ndarray:
    """Rank values in descending order (0 = highest rank)."""
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(len(values))
    return ranks.astype(np.float64)


def _spearman_rho_from_scores(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = sorted(set(a.keys()) & set(b.keys()))
    if len(keys) < 2:
        return float("nan")
    a_vals = np.array([a[k] for k in keys], dtype=np.float64)
    b_vals = np.array([b[k] for k in keys], dtype=np.float64)
    ra = _rank_desc(a_vals)
    rb = _rank_desc(b_vals)
    if len(keys) < 2:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _topk_overlap_from_scores(a: Dict[str, float], b: Dict[str, float], k: int = 20) -> float:
    keys = list(set(a.keys()) & set(b.keys()))
    if not keys:
        return 0.0
    k_eff = min(k, len(keys))
    a_top = sorted(keys, key=lambda x: a.get(x, float("-inf")), reverse=True)[:k_eff]
    b_top = sorted(keys, key=lambda x: b.get(x, float("-inf")), reverse=True)[:k_eff]
    return float(len(set(a_top) & set(b_top)) / k_eff) if k_eff > 0 else 0.0


def run_sis_weight_sensitivity(
    observed_snapshots: List[Dict[str, Any]],
    output_dir: Path,
    sample_weeks: int = 10,
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    Minimal robustness check for SIS: show rankings are stable under alternative weights.

    This is a post-processing analysis on observed networks only (no model training/inference).
    """
    rng = np.random.default_rng(42)
    if not observed_snapshots:
        return {"error": "no_snapshots"}

    if sample_weeks <= 0:
        sampled = observed_snapshots
    else:
        n = min(sample_weeks, len(observed_snapshots))
        idx = rng.choice(len(observed_snapshots), size=n, replace=False)
        sampled = [observed_snapshots[i] for i in sorted(idx.tolist())]

    weight_sets = [
        {"name": "equal", "alpha": 1 / 3, "beta": 1 / 3, "gamma": 1 / 3},
        {"name": "pagerank-heavy", "alpha": 0.50, "beta": 0.25, "gamma": 0.25},
        {"name": "tail-heavy", "alpha": 0.25, "beta": 0.50, "gamma": 0.25},
        {"name": "tvl-heavy", "alpha": 0.25, "beta": 0.25, "gamma": 0.50},
    ]

    def _combine(components: Dict[str, Dict[str, float]], w: Dict[str, float]) -> Dict[str, float]:
        pr = components["pagerank"]
        tail = components["tail_exposure"]
        log_tvl = components["log_tvl_norm"]
        nodes = pr.keys()
        return {
            n: w["alpha"] * pr.get(n, 0.0) + w["beta"] * tail.get(n, 0.0) + w["gamma"] * log_tvl.get(n, 0.0)
            for n in nodes
        }

    baseline_w = weight_sets[0]
    alt_ws = weight_sets[1:]

    per_week = []
    by_alt = {w["name"]: {"spearman_rho": [], "topk_overlap": []} for w in alt_ws}

    for snap in sampled:
        # build_snapshot() produces array-format snapshots with LINEAR edge weights
        dict_snap = array_snap_to_dict_snap(snap, edge_weight_is_log=False)
        components = compute_sis_components(dict_snap)
        baseline_scores = _combine(components, baseline_w)

        week_row = {
            "date": snap.get("date", ""),
            "n_nodes": len(dict_snap.get("nodes", {})),
            "n_edges": len(dict_snap.get("edges", [])),
            "comparisons": {},
        }

        for w in alt_ws:
            alt_scores = _combine(components, w)
            rho = _spearman_rho_from_scores(baseline_scores, alt_scores)
            overlap = _topk_overlap_from_scores(baseline_scores, alt_scores, k=top_k)
            week_row["comparisons"][w["name"]] = {"spearman_rho": rho, "topk_overlap": overlap}
            by_alt[w["name"]]["spearman_rho"].append(rho)
            by_alt[w["name"]]["topk_overlap"].append(overlap)

        per_week.append(week_row)

    summary = {}
    for w in alt_ws:
        name = w["name"]
        rhos = np.array(by_alt[name]["spearman_rho"], dtype=np.float64)
        overlaps = np.array(by_alt[name]["topk_overlap"], dtype=np.float64)
        summary[name] = {
            "weights": {"alpha": w["alpha"], "beta": w["beta"], "gamma": w["gamma"]},
            "spearman_rho_mean": float(np.nanmean(rhos)) if rhos.size else float("nan"),
            "spearman_rho_std": float(np.nanstd(rhos)) if rhos.size else float("nan"),
            "topk_overlap_mean": float(np.nanmean(overlaps)) if overlaps.size else float("nan"),
            "topk_overlap_std": float(np.nanstd(overlaps)) if overlaps.size else float("nan"),
        }

    results = {
        "baseline": baseline_w,
        "alternatives": alt_ws,
        "sample_weeks": len(sampled),
        "top_k": top_k,
        "per_week": per_week,
        "summary": summary,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "sis_weight_sensitivity.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)

    # Write a small LaTeX table for easy paste into appendix
    out_tex = output_dir / "sis_weight_sensitivity_table.tex"
    lines = []
    lines.append(r"\\begin{table}[t]")
    lines.append(r"\\centering")
    lines.append(r"\\small")
    lines.append(r"\\begin{tabular}{lcc}")
    lines.append(r"\\hline")
    lines.append(r"Weights $(\\alpha,\\beta,\\gamma)$ & Spearman $\\rho$ & Top-%d overlap\\\\" % top_k)
    lines.append(r"\\hline")
    for w in alt_ws:
        name = w["name"]
        s = summary[name]
        w_str = f"({w['alpha']:.2f},{w['beta']:.2f},{w['gamma']:.2f})"
        rho_str = f"{s['spearman_rho_mean']:.3f} $\\pm$ {s['spearman_rho_std']:.3f}"
        ov_str = f"{s['topk_overlap_mean']:.3f} $\\pm$ {s['topk_overlap_std']:.3f}"
        lines.append(f"{w['name']} {w_str} & {rho_str} & {ov_str}\\\\")
    lines.append(r"\\hline")
    lines.append(r"\\end{tabular}")
    lines.append(
        r"\\caption{SIS weight sensitivity (robustness check). We report stability against the equal-weight baseline "
        r"$(\\alpha,\\beta,\\gamma)=(1/3,1/3,1/3)$ over %d sampled weeks.}" % len(sampled)
    )
    lines.append(r"\\label{tab:sis-sensitivity}")
    lines.append(r"\\end{table}")
    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"SIS sensitivity saved to {out_json}")
    logger.info(f"LaTeX table saved to {out_tex}")

    return results


# ============== Experiment 1: Forward-looking Risk Metric Prediction ==============

def run_forward_risk_prediction(
    config: ExperimentConfig,
    models_by_horizon: Dict[int, GraphPFNLinkPredictor],
    test_snapshots: List[Dict],
    meta_category: Dict,
    horizons: List[int] = [1, 4, 8, 12],
    max_pairs_per_horizon: int = 0,
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, Any]:
    """
    Experiment 1: Forward-looking Risk Metric Prediction

    Demonstrates that the model can predict future network risk metrics.

    For each horizon h:
    1. Use model to predict network at t+h
    2. Compute risk metrics on predicted network
    3. Compare with actual risk metrics at t+h
    4. Report prediction accuracy for each risk metric
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1: Forward-looking Risk Metric Prediction")
    logger.info("=" * 70)
    logger.info("Goal: Show model can predict future network risk metrics")

    results = {"horizons": {}, "summary": {}}

    for horizon in horizons:
        logger.info(f"\n--- Horizon h={horizon} weeks ---")
        model = models_by_horizon.get(horizon)
        if model is None:
            raise ValueError(
                f"Missing trained model for horizon h={horizon}. "
                f"Available: {sorted(models_by_horizon.keys())}"
            )

        # Build test pairs for this horizon
        test_pairs = build_week_pairs(test_snapshots, config.neg_ratio, config.seed, horizon=horizon)
        if not test_pairs:
            logger.warning(f"No test pairs for horizon {horizon}")
            continue

        if max_pairs_per_horizon > 0 and len(test_pairs) > max_pairs_per_horizon:
            rng = np.random.default_rng(config.seed + horizon)
            idx = rng.choice(len(test_pairs), size=max_pairs_per_horizon, replace=False)
            idx = sorted(idx.tolist())
            test_pairs = [test_pairs[i] for i in idx]
            logger.info(
                f"  Subsampled to {len(test_pairs)} pairs (max_pairs_per_horizon={max_pairs_per_horizon})"
            )

        # Get predictions
        preds = predict_graphpfn(model, test_pairs, config)

        horizon_results = {
            "n_samples": len(test_pairs),
            "predicted_metrics": [],
            "actual_metrics": [],
            "metric_errors": {},
        }

        for pair, pred in zip(test_pairs, preds):
            # Reconstruct predicted and actual networks
            pred_net = reconstruct_network_from_predictions(pred, pair)
            actual_net = reconstruct_actual_network(pred, pair)

            # Compute risk metrics
            pred_metrics = compute_network_risk_metrics(pred_net, meta_category)
            actual_metrics = compute_network_risk_metrics(actual_net, meta_category)

            horizon_results["predicted_metrics"].append(pred_metrics)
            horizon_results["actual_metrics"].append(actual_metrics)

        # Compute errors for each metric
        key_metrics = [
            "tvl_hhi", "top10_tvl_share", "tvl_gini",
            "edge_hhi", "density", "degree_gini",
            "mean_sis", "spillover_index", "total_tvl"
        ]

        for key in key_metrics:
            pred_vals = [m.get(key, np.nan) for m in horizon_results["predicted_metrics"]]
            actual_vals = [m.get(key, np.nan) for m in horizon_results["actual_metrics"]]

            # Filter out NaN
            valid_mask = ~(np.isnan(pred_vals) | np.isnan(actual_vals))
            pred_vals = np.array(pred_vals)[valid_mask]
            actual_vals = np.array(actual_vals)[valid_mask]

            if len(pred_vals) > 0:
                mae = float(np.mean(np.abs(pred_vals - actual_vals)))
                rmse = float(np.sqrt(np.mean((pred_vals - actual_vals) ** 2)))
                mean_actual = float(np.mean(actual_vals))
                mean_pred = float(np.mean(pred_vals))
                corr = float(np.corrcoef(pred_vals, actual_vals)[0, 1]) if len(pred_vals) > 1 else np.nan

                horizon_results["metric_errors"][key] = {
                    "mae": mae,
                    "rmse": rmse,
                    "mean_actual": mean_actual,
                    "mean_predicted": mean_pred,
                    "relative_error": mae / (abs(mean_actual) + 1e-10),
                    "correlation": corr,
                }

        results["horizons"][f"h={horizon}"] = horizon_results

        # Print summary for this horizon
        logger.info(f"\n  Risk Metric Prediction Accuracy (h={horizon}):")
        logger.info(f"  {'Metric':<20} {'MAE':<12} {'Corr':<12} {'Rel.Err':<12}")
        logger.info(f"  {'-'*56}")
        for key in ["tvl_hhi", "edge_hhi", "density", "mean_sis"]:
            if key in horizon_results["metric_errors"]:
                err = horizon_results["metric_errors"][key]
                logger.info(f"  {key:<20} {err['mae']:<12.4f} {err['correlation']:<12.3f} "
                           f"{err['relative_error']:<12.2%}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "exp1_forward_risk.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    logger.info(f"\nResults saved to {output_dir / 'exp1_forward_risk.json'}")

    return results


# ============== Experiment 2: Predictive Contagion Simulation ==============

def run_predictive_contagion(
    config: ExperimentConfig,
    models_by_horizon: Dict[int, GraphPFNLinkPredictor],
    test_snapshots: List[Dict],
    meta_category: Dict,
    horizons: List[int] = [1, 4, 8],
    max_samples_per_horizon: int = 10,
    output_dir: Path = OUTPUT_DIR,
    save_json: bool = True,
) -> Dict[str, Any]:
    """
    Experiment 2: Predictive Contagion Simulation

    Demonstrates forward-looking stress testing capability.

    For each test sample:
    1. Run contagion on observed network at t
    2. Run contagion on predicted network (t -> t+h)
    3. Run contagion on actual network at t+h
    4. Compare: Does predicted contagion match actual future contagion?
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 2: Predictive Contagion Simulation")
    logger.info("=" * 70)
    logger.info("Goal: Show model enables forward-looking stress testing")

    WORST_FRAC = 0.2  # evaluate "advantage regime" on top-20% baseline-error samples
    SIGN_TOL = 1e-6

    def _safe_jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        inter = len(a.intersection(b))
        union = len(a.union(b))
        return float(inter / union) if union > 0 else 0.0

    def _topn_set(net: Dict[str, Any], n: int) -> set:
        node_ids = net.get("node_ids", [])
        sizes = np.array(net.get("sizes", []), dtype=float)
        if n <= 0 or len(node_ids) == 0 or sizes.size == 0:
            return set()
        n_eff = min(int(n), len(node_ids))
        order = np.argsort(sizes)[::-1][:n_eff]
        return {node_ids[int(i)] for i in order if int(i) < len(node_ids)}

    def _sector_set(net: Dict[str, Any], sector: str) -> set:
        node_ids = net.get("node_ids", [])
        s = (sector or "").lower()
        out = set()
        for nid in node_ids:
            if s and s in (meta_category.get(nid, "") or "").lower():
                out.add(nid)
        return out

    def _sign_with_tol(x: np.ndarray, tol: float = SIGN_TOL) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.where(x > tol, 1.0, np.where(x < -tol, -1.0, 0.0))

    def _topk_idx(values: np.ndarray, frac: float = WORST_FRAC) -> np.ndarray:
        """Return indices of the largest `frac` values (at least 1)."""
        v = np.asarray(values, dtype=float)
        n = int(v.size)
        if n <= 0:
            return np.array([], dtype=int)
        k = max(1, int(np.ceil(float(frac) * n)))
        return np.argsort(v)[::-1][:k]

    # Define shock scenarios
    scenarios = [
        {"name": "Top Protocol Shock", "top_n": 1, "shock_ratio": 0.5},
        {"name": "Top 5 Protocols Shock", "top_n": 5, "shock_ratio": 0.3},
        {"name": "Bridge Sector Shock", "sector": "Bridge", "shock_ratio": 1.0},
    ]

    results = {"horizons": {}, "scenarios": scenarios}

    for horizon in horizons:
        logger.info(f"\n--- Horizon h={horizon} weeks ---")
        model = models_by_horizon.get(horizon)
        if model is None:
            raise ValueError(
                f"Missing trained model for horizon h={horizon}. "
                f"Available: {sorted(models_by_horizon.keys())}"
            )

        test_pairs = build_week_pairs(test_snapshots, config.neg_ratio, config.seed, horizon=horizon)
        if not test_pairs or len(test_pairs) < 3:
            logger.warning(f"Insufficient test pairs for horizon {horizon}")
            continue

        # Sample a few pairs for detailed analysis
        sample_n = min(max_samples_per_horizon, len(test_pairs)) if max_samples_per_horizon > 0 else len(test_pairs)
        if sample_n >= len(test_pairs):
            sample_pairs = test_pairs
        else:
            rng = np.random.default_rng(config.seed + horizon)
            idx = rng.choice(len(test_pairs), size=sample_n, replace=False)
            idx = sorted(idx.tolist())
            sample_pairs = [test_pairs[i] for i in idx]
        preds = predict_graphpfn(model, sample_pairs, config)

        horizon_results = {"samples": []}

        for pair, pred in zip(sample_pairs, preds):
            sample_result = {
                "time_t": pair.time_t,
                "time_t1": pair.time_t1,
                "scenarios": {},
            }

            # Reconstruct networks
            pred_net = reconstruct_network_from_predictions(pred, pair)
            actual_net = reconstruct_actual_network(pred, pair)

            # Get observed network at t (from pair)
            observed_net = {
                "date": pair.time_t,
                "node_ids": pair.node_ids,
                "categories": pair.categories,
                "sizes": pair.sizes_t,
                "edge_src": pair.edge_src_t,
                "edge_dst": pair.edge_dst_t,
                "edge_weight": pair.edge_weight_t,
            }

            for scenario in scenarios:
                # Determine shocked nodes
                if "top_n" in scenario:
                    # Shock top N protocols by TVL
                    tvl_order = np.argsort(pred_net["sizes"])[::-1]
                    shocked_indices = tvl_order[:scenario["top_n"]]
                    shocked_nodes = [pred_net["node_ids"][i] for i in shocked_indices if i < len(pred_net["node_ids"])]
                elif "sector" in scenario:
                    # Shock all protocols in a sector
                    shocked_nodes = [
                        nid for nid in pred_net["node_ids"]
                        if scenario["sector"].lower() in meta_category.get(nid, "").lower()
                    ]
                else:
                    shocked_nodes = []

                if not shocked_nodes:
                    continue

                shock_ratio = scenario["shock_ratio"]

                # Run contagion on all three networks
                # Convert array format to dict format for simulate_contagion
                try:
                    # --- Diagnostics: how stable is the shocked-node set? ---
                    # For Top-N, compare top-N sets under observed/predicted/actual.
                    # For Sector, compare sector membership sets under observed/predicted/actual.
                    diag = {}
                    if "top_n" in scenario:
                        n = int(scenario["top_n"])
                        obs_set = _topn_set(observed_net, n)
                        pred_set = _topn_set(pred_net, n)
                        act_set = _topn_set(actual_net, n)
                        used_set = set(shocked_nodes)
                        diag = {
                            "shock_set_type": "top_n",
                            "top_n": n,
                            "jaccard_obs_vs_actual": _safe_jaccard(obs_set, act_set),
                            "jaccard_pred_vs_actual": _safe_jaccard(pred_set, act_set),
                            "jaccard_used_vs_actual": _safe_jaccard(used_set, act_set),
                            "match_obs_eq_actual": bool(obs_set == act_set),
                            "match_pred_eq_actual": bool(pred_set == act_set),
                            "match_used_eq_actual": bool(used_set == act_set),
                            "obs_size": len(obs_set),
                            "pred_size": len(pred_set),
                            "actual_size": len(act_set),
                            "used_size": len(used_set),
                        }
                    elif "sector" in scenario:
                        sector = str(scenario["sector"])
                        obs_set = _sector_set(observed_net, sector)
                        pred_set = _sector_set(pred_net, sector)
                        act_set = _sector_set(actual_net, sector)
                        used_set = set(shocked_nodes)
                        diag = {
                            "shock_set_type": "sector",
                            "sector": sector,
                            "jaccard_obs_vs_actual": _safe_jaccard(obs_set, act_set),
                            "jaccard_pred_vs_actual": _safe_jaccard(pred_set, act_set),
                            "jaccard_used_vs_actual": _safe_jaccard(used_set, act_set),
                            "match_obs_eq_actual": bool(obs_set == act_set),
                            "match_pred_eq_actual": bool(pred_set == act_set),
                            "match_used_eq_actual": bool(used_set == act_set),
                            "obs_size": len(obs_set),
                            "pred_size": len(pred_set),
                            "actual_size": len(act_set),
                            "used_size": len(used_set),
                        }

                    observed_dict = array_snap_to_dict_snap(observed_net, edge_weight_is_log=True)
                    predicted_dict = array_snap_to_dict_snap(pred_net, edge_weight_is_log=True)
                    actual_dict = array_snap_to_dict_snap(actual_net, edge_weight_is_log=True)

                    contagion_observed = simulate_contagion(observed_dict, shocked_nodes, shock_ratio)
                    contagion_predicted = simulate_contagion(predicted_dict, shocked_nodes, shock_ratio)
                    contagion_actual = simulate_contagion(actual_dict, shocked_nodes, shock_ratio)

                    sample_result["scenarios"][scenario["name"]] = {
                        "shocked_nodes_count": len(shocked_nodes),
                        "observed_t": {
                            "total_loss_pct": contagion_observed["total_loss_pct"],
                            "affected_count": contagion_observed["affected_count"],
                            "depth": contagion_observed["propagation_rounds"],
                        },
                        "predicted_t_h": {
                            "total_loss_pct": contagion_predicted["total_loss_pct"],
                            "affected_count": contagion_predicted["affected_count"],
                            "depth": contagion_predicted["propagation_rounds"],
                        },
                        "actual_t_h": {
                            "total_loss_pct": contagion_actual["total_loss_pct"],
                            "affected_count": contagion_actual["affected_count"],
                            "depth": contagion_actual["propagation_rounds"],
                        },
                        "shock_set_diagnostics": diag,
                    }
                except Exception as e:
                    logger.warning(f"Contagion simulation failed: {e}")

            horizon_results["samples"].append(sample_result)

        # Compute aggregate accuracy
        for scenario in scenarios:
            predicted_losses = []
            actual_losses = []
            observed_losses = []

            for sample in horizon_results["samples"]:
                if scenario["name"] in sample["scenarios"]:
                    s = sample["scenarios"][scenario["name"]]
                    predicted_losses.append(s["predicted_t_h"]["total_loss_pct"])
                    actual_losses.append(s["actual_t_h"]["total_loss_pct"])
                    observed_losses.append(s["observed_t"]["total_loss_pct"])

            if predicted_losses:
                pred_arr = np.array(predicted_losses, dtype=float)
                act_arr = np.array(actual_losses, dtype=float)
                obs_arr = np.array(observed_losses, dtype=float)

                model_abs_err = np.abs(pred_arr - act_arr)
                base_abs_err = np.abs(obs_arr - act_arr)

                mae = float(np.mean(model_abs_err))
                corr = float(np.corrcoef(pred_arr, act_arr)[0, 1]) if len(pred_arr) > 1 else float("nan")
                baseline_mae = float(np.mean(base_abs_err))

                # "Advantage regime" evaluation: where baseline fails most.
                worst_idx = _topk_idx(base_abs_err, frac=WORST_FRAC)
                worst_k = int(worst_idx.size)
                mae_model_worst = float(np.mean(model_abs_err[worst_idx])) if worst_k else float("nan")
                mae_base_worst = float(np.mean(base_abs_err[worst_idx])) if worst_k else float("nan")
                win_rate_all = float(np.mean(model_abs_err < base_abs_err)) if pred_arr.size else float("nan")
                win_rate_worst = float(np.mean(model_abs_err[worst_idx] < base_abs_err[worst_idx])) if worst_k else float("nan")

                # Delta(contagion loss) analysis: focuses on change rather than level.
                delta_actual = act_arr - obs_arr
                delta_model = pred_arr - obs_arr
                corr_delta = float(np.corrcoef(delta_model, delta_actual)[0, 1]) if len(delta_model) > 1 else float("nan")
                sign_acc_delta = float(np.mean(_sign_with_tol(delta_model) == _sign_with_tol(delta_actual))) if delta_model.size else float("nan")
                sign_acc_delta_worst = (
                    float(np.mean(_sign_with_tol(delta_model[worst_idx]) == _sign_with_tol(delta_actual[worst_idx])))
                    if worst_k
                    else float("nan")
                )

                horizon_results[f"{scenario['name']}_accuracy"] = {
                    "mae_loss_pct": float(mae),
                    "correlation": float(corr),
                    "n_samples": len(predicted_losses),
                    "baseline_mae_loss_pct": float(baseline_mae),
                }

                horizon_results[f"{scenario['name']}_advantage"] = {
                    "n_samples": int(len(pred_arr)),
                    "worst_frac": float(WORST_FRAC),
                    "worst_k": int(worst_k),
                    "mae_model_all": float(mae),
                    "mae_baseline_all": float(baseline_mae),
                    "delta_mae_all": float(baseline_mae - mae),
                    "win_rate_all": float(win_rate_all),
                    "mae_model_worst": float(mae_model_worst),
                    "mae_baseline_worst": float(mae_base_worst),
                    "delta_mae_worst": float(mae_base_worst - mae_model_worst) if np.isfinite(mae_base_worst) and np.isfinite(mae_model_worst) else float("nan"),
                    "win_rate_worst": float(win_rate_worst),
                    "delta_corr_all": float(corr_delta),
                    "delta_sign_acc_all": float(sign_acc_delta),
                    "delta_sign_acc_worst": float(sign_acc_delta_worst),
                    "avg_abs_delta_actual_all": float(np.mean(np.abs(delta_actual))),
                    "avg_abs_delta_actual_worst": float(np.mean(np.abs(delta_actual[worst_idx]))) if worst_k else float("nan"),
                }

                logger.info(f"  {scenario['name']}: MAE={mae:.2f}%, Corr={corr:.3f}")

        # Horizon-wide advantage summary (concatenate scenarios; avoids "MAE of means")
        all_pred, all_act, all_obs = [], [], []
        for scenario in scenarios:
            for sample in horizon_results["samples"]:
                if scenario["name"] not in sample.get("scenarios", {}):
                    continue
                s = sample["scenarios"][scenario["name"]]
                all_pred.append(float(s["predicted_t_h"]["total_loss_pct"]))
                all_act.append(float(s["actual_t_h"]["total_loss_pct"]))
                all_obs.append(float(s["observed_t"]["total_loss_pct"]))

        if all_pred:
            pred_arr = np.array(all_pred, dtype=float)
            act_arr = np.array(all_act, dtype=float)
            obs_arr = np.array(all_obs, dtype=float)
            model_abs_err = np.abs(pred_arr - act_arr)
            base_abs_err = np.abs(obs_arr - act_arr)
            worst_idx = _topk_idx(base_abs_err, frac=WORST_FRAC)
            worst_k = int(worst_idx.size)
            delta_actual = act_arr - obs_arr
            delta_model = pred_arr - obs_arr
            horizon_results["advantage_overall"] = {
                "n_samples": int(pred_arr.size),
                "worst_frac": float(WORST_FRAC),
                "worst_k": int(worst_k),
                "mae_model_all": float(np.mean(model_abs_err)),
                "mae_baseline_all": float(np.mean(base_abs_err)),
                "delta_mae_all": float(np.mean(base_abs_err) - np.mean(model_abs_err)),
                "win_rate_all": float(np.mean(model_abs_err < base_abs_err)),
                "mae_model_worst": float(np.mean(model_abs_err[worst_idx])) if worst_k else float("nan"),
                "mae_baseline_worst": float(np.mean(base_abs_err[worst_idx])) if worst_k else float("nan"),
                "delta_mae_worst": (
                    float(np.mean(base_abs_err[worst_idx]) - np.mean(model_abs_err[worst_idx])) if worst_k else float("nan")
                ),
                "win_rate_worst": float(np.mean(model_abs_err[worst_idx] < base_abs_err[worst_idx])) if worst_k else float("nan"),
                "delta_corr_all": float(np.corrcoef(delta_model, delta_actual)[0, 1]) if pred_arr.size > 1 else float("nan"),
                "delta_sign_acc_all": float(np.mean(_sign_with_tol(delta_model) == _sign_with_tol(delta_actual))),
                "delta_sign_acc_worst": (
                    float(np.mean(_sign_with_tol(delta_model[worst_idx]) == _sign_with_tol(delta_actual[worst_idx])))
                    if worst_k
                    else float("nan")
                ),
                "avg_abs_delta_actual_all": float(np.mean(np.abs(delta_actual))),
                "avg_abs_delta_actual_worst": float(np.mean(np.abs(delta_actual[worst_idx]))) if worst_k else float("nan"),
            }

        # Aggregate shock-set diagnostics per scenario
        for scenario in scenarios:
            j_obs, j_pred, j_used = [], [], []
            eq_obs, eq_pred, eq_used = [], [], []
            sizes = []

            for sample in horizon_results["samples"]:
                sdata = sample.get("scenarios", {}).get(scenario["name"], {})
                diag = sdata.get("shock_set_diagnostics", {})
                if not diag:
                    continue
                j_obs.append(diag.get("jaccard_obs_vs_actual", np.nan))
                j_pred.append(diag.get("jaccard_pred_vs_actual", np.nan))
                j_used.append(diag.get("jaccard_used_vs_actual", np.nan))
                eq_obs.append(1.0 if diag.get("match_obs_eq_actual") else 0.0)
                eq_pred.append(1.0 if diag.get("match_pred_eq_actual") else 0.0)
                eq_used.append(1.0 if diag.get("match_used_eq_actual") else 0.0)
                sizes.append(
                    {
                        "obs": diag.get("obs_size", 0),
                        "pred": diag.get("pred_size", 0),
                        "actual": diag.get("actual_size", 0),
                        "used": diag.get("used_size", 0),
                    }
                )

            if j_obs:
                j_obs_arr = np.array(j_obs, dtype=float)
                j_pred_arr = np.array(j_pred, dtype=float)
                j_used_arr = np.array(j_used, dtype=float)
                eq_obs_arr = np.array(eq_obs, dtype=float)
                eq_pred_arr = np.array(eq_pred, dtype=float)
                eq_used_arr = np.array(eq_used, dtype=float)

                horizon_results[f"{scenario['name']}_shockset"] = {
                    "n_samples": int(len(j_obs_arr)),
                    "jaccard_obs_vs_actual_mean": float(np.nanmean(j_obs_arr)),
                    "jaccard_pred_vs_actual_mean": float(np.nanmean(j_pred_arr)),
                    "jaccard_used_vs_actual_mean": float(np.nanmean(j_used_arr)),
                    "exact_match_obs_eq_actual_rate": float(np.nanmean(eq_obs_arr)),
                    "exact_match_pred_eq_actual_rate": float(np.nanmean(eq_pred_arr)),
                    "exact_match_used_eq_actual_rate": float(np.nanmean(eq_used_arr)),
                    "avg_set_size": {
                        "obs": float(np.mean([x["obs"] for x in sizes])) if sizes else 0.0,
                        "pred": float(np.mean([x["pred"] for x in sizes])) if sizes else 0.0,
                        "actual": float(np.mean([x["actual"] for x in sizes])) if sizes else 0.0,
                        "used": float(np.mean([x["used"] for x in sizes])) if sizes else 0.0,
                    },
                }

        results["horizons"][f"h={horizon}"] = horizon_results

    if save_json:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "exp2_predictive_contagion.json", "w") as f:
            json.dump(results, f, indent=2, default=float)
        logger.info(f"\nResults saved to {output_dir / 'exp2_predictive_contagion.json'}")

    return results


# ============== Experiment 3: Shock Early Warning ==============

def run_early_warning_analysis(
    config: ExperimentConfig,
    all_snapshots: List[Dict],
    date_to_snap: Dict[str, Dict],
    meta_category: Dict,
    frozen: bool = False,
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, Any]:
    """
    Experiment 3: Shock Early Warning Analysis

    Demonstrates the model's ability to predict risk before major events.

    IMPORTANT: To avoid data leakage, we train a SEPARATE model for each event
    using ONLY data from before the event start date. This ensures the model
    has never seen any information about the shock event during training.

    For each shock event (Terra/Luna, FTX):
    1. Train model on data strictly before event start (no leakage!)
    2. Use model to predict risk metrics for event period
    3. Compare predicted vs actual risk trajectories
    4. Assess early warning capability
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 3: Shock Early Warning Analysis")
    logger.info("=" * 70)
    logger.info("Goal: Show model can predict elevated risk before major shocks")
    logger.info("NOTE: Training separate model for each event to avoid data leakage")

    all_dates = sorted([s["date"] for s in all_snapshots])
    results = {"events": {}}

    for event_id, event_info in SHOCK_EVENTS.items():
        logger.info(f"\n--- Analyzing {event_info['name']} ---")
        logger.info(f"    Event date: {event_info['event_date']}")

        pre_start, pre_end = event_info["pre_window"]
        event_start, event_end = event_info["event_window"]

        # Get pre-event and event-period snapshots
        event_dates = [d for d in all_dates if event_start <= d <= event_end]
        if not event_dates:
            logger.warning(f"Insufficient data for {event_info['name']}")
            continue

        # Use the last available snapshot strictly before the first event week as anchor.
        first_event_idx = all_dates.index(event_dates[0])
        if first_event_idx <= 0:
            logger.warning(f"Insufficient pre-event data for {event_info['name']}")
            continue

        anchor_date = all_dates[first_event_idx - 1]
        pre_dates = [d for d in all_dates if pre_start <= d <= anchor_date]

        if len(pre_dates) < 2:
            logger.warning(f"Insufficient data for {event_info['name']}")
            continue

        logger.info(f"    Pre-event weeks: {len(pre_dates)}")
        logger.info(f"    Event weeks: {len(event_dates)}")

        # =================================================================
        # CRITICAL FIX: Train model only on data BEFORE event start
        # This avoids data leakage - model never sees the shock event
        # =================================================================
        train_cutoff = event_start  # All data before event window
        train_dates = [d for d in all_dates if d < train_cutoff]
        train_snapshots = [date_to_snap[d] for d in train_dates if d in date_to_snap]

        logger.info(f"    Training on {len(train_snapshots)} weeks (< {train_cutoff})")

        if len(train_snapshots) < 10:
            logger.warning(f"Insufficient training data for {event_info['name']}")
            continue

        # Train a fresh model for this event
        train_pairs = build_week_pairs(train_snapshots, config.neg_ratio, config.seed, horizon=1)
        logger.info(f"    Training pairs: {len(train_pairs)}")

        event_model = load_or_train_model(
            config,
            train_pairs,
            force_retrain=True,
            frozen=frozen,
            cache_tag=f"early_warning_{event_id}_cutoff_{train_cutoff}",
            horizon=1,
        )
        logger.info(f"    Model trained for {event_info['name']} (no event data leakage)")

        pre_snapshots = [date_to_snap[d] for d in pre_dates if d in date_to_snap]

        event_results = {
            "event_info": event_info,
            "pre_dates": pre_dates,
            "event_dates": event_dates,
            "train_cutoff": train_cutoff,
            "train_weeks": len(train_snapshots),
            "predictions": [],
            "actual_event_metrics": [],
        }

        # Anchor is the last observed week before the event window.
        last_pre_snap = date_to_snap.get(anchor_date, pre_snapshots[-1])

        # For early warning, use 1-step-ahead predictions:
        # predict each event-week network from the immediately preceding observed week.
        for week_idx, event_date in enumerate(event_dates):
            if event_date not in date_to_snap:
                continue

            event_snap = date_to_snap[event_date]

            prev_idx = all_dates.index(event_date) - 1
            if prev_idx < 0:
                continue
            prev_date = all_dates[prev_idx]
            if prev_date not in date_to_snap:
                continue
            prev_snap = date_to_snap[prev_date]

            # Build prediction pair with h=1 (one-step ahead, truly consecutive in time)
            test_pairs = build_week_pairs(
                [prev_snap, event_snap],
                config.neg_ratio,
                config.seed,
                horizon=1  # Always use h=1 for early warning
            )

            if not test_pairs:
                logger.warning(f"No pairs for {event_date}")
                continue

            # Use event-specific model (trained without leakage)
            preds = predict_graphpfn(event_model, test_pairs[:1], config)
            pred = preds[0]
            pair = test_pairs[0]

            # Reconstruct networks
            pred_net = reconstruct_network_from_predictions(pred, pair)
            actual_net = reconstruct_actual_network(pred, pair)

            # Compute risk metrics
            pred_metrics = compute_network_risk_metrics(pred_net, meta_category)
            actual_metrics = compute_network_risk_metrics(actual_net, meta_category)

            # Run contagion simulation (convert to dict format first)
            try:
                # Shock top 5 protocols by TVL at time t (within the evaluation universe).
                # We restrict to nodes that exist at both t and t+1 (pair.node_mask) so the
                # shock is applied consistently on predicted/actual networks.
                valid_idx = np.where(pair.node_mask)[0]
                if valid_idx.size > 0:
                    order = valid_idx[np.argsort(pair.sizes_t[valid_idx])[::-1]]
                    shocked_nodes = [pair.node_ids[int(i)] for i in order[:5]]
                else:
                    shocked_nodes = []

                pred_dict = array_snap_to_dict_snap(pred_net, edge_weight_is_log=True)
                actual_dict = array_snap_to_dict_snap(actual_net, edge_weight_is_log=True)

                contagion_pred = simulate_contagion(pred_dict, shocked_nodes, 0.5)
                contagion_actual = simulate_contagion(actual_dict, shocked_nodes, 0.5)
            except Exception as e:
                logger.warning(f"Contagion failed: {e}")
                contagion_pred = {"total_loss_pct": np.nan}
                contagion_actual = {"total_loss_pct": np.nan}

            week_result = {
                "weeks_ahead": week_idx + 1,
                "target_date": event_date,
                "predicted_metrics": pred_metrics,
                "actual_metrics": actual_metrics,
                "predicted_contagion_loss": contagion_pred["total_loss_pct"],
                "actual_contagion_loss": contagion_actual["total_loss_pct"],
            }

            event_results["predictions"].append(week_result)

            # Print progress
            logger.info(f"    week {week_idx+1} ({event_date}): "
                       f"Pred HHI={pred_metrics.get('tvl_hhi', 0):.4f}, "
                       f"Actual HHI={actual_metrics.get('tvl_hhi', 0):.4f}, "
                       f"Pred Contagion={contagion_pred['total_loss_pct']:.2f}%")

        # Also compute metrics for observed pre-event period (baseline)
        # Note: last_pre_snap has LINEAR edge weights (from build_snapshot)
        baseline_metrics = compute_network_risk_metrics(
            last_pre_snap, meta_category, edge_weight_is_log=False
        )
        event_results["baseline_metrics"] = baseline_metrics

        # Compute summary statistics
        if event_results["predictions"]:
            pred_hhi = [p["predicted_metrics"].get("tvl_hhi", np.nan) for p in event_results["predictions"]]
            actual_hhi = [p["actual_metrics"].get("tvl_hhi", np.nan) for p in event_results["predictions"]]

            event_results["summary"] = {
                "baseline_hhi": baseline_metrics.get("tvl_hhi", np.nan),
                "mean_predicted_hhi": float(np.nanmean(pred_hhi)),
                "mean_actual_hhi": float(np.nanmean(actual_hhi)),
                "hhi_prediction_mae": float(np.nanmean(np.abs(np.array(pred_hhi) - np.array(actual_hhi)))),
                "hhi_change_predicted": float(np.nanmean(pred_hhi) - baseline_metrics.get("tvl_hhi", 0)),
                "hhi_change_actual": float(np.nanmean(actual_hhi) - baseline_metrics.get("tvl_hhi", 0)),
            }

            logger.info(f"\n    Summary for {event_info['name']}:")
            logger.info(f"    Baseline HHI: {event_results['summary']['baseline_hhi']:.4f}")
            logger.info(f"    Predicted HHI during event: {event_results['summary']['mean_predicted_hhi']:.4f}")
            logger.info(f"    Actual HHI during event: {event_results['summary']['mean_actual_hhi']:.4f}")
            logger.info(f"    HHI Change (Predicted): {event_results['summary']['hhi_change_predicted']:+.4f}")
            logger.info(f"    HHI Change (Actual): {event_results['summary']['hhi_change_actual']:+.4f}")

        results["events"][event_id] = event_results

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "exp3_early_warning.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    logger.info(f"\nResults saved to {output_dir / 'exp3_early_warning.json'}")

    return results


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description="Task II: Model-based Financial Stability Analysis")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "forward_risk", "predictive_contagion", "early_warning", "sis_sensitivity"],
        help="Which experiment to run",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs if model needs training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: inferred from ExperimentConfig; unused with --plot-only)",
    )
    parser.add_argument("--force-retrain", action="store_true", help="Force model retraining")
    parser.add_argument(
        "--frozen",
        action="store_true",
        help="Use frozen encoder (GraphPFN-Frozen) instead of fine-tuning (faster, for quick validation)",
    )
    parser.add_argument("--output-dir", type=str, default="output/task2_model_based")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke-test mode: fewer epochs/pairs and smaller horizons for faster iteration",
    )
    parser.add_argument(
        "--forward-horizons",
        type=str,
        default="1,4,8,12",
        help="Comma-separated horizons for forward_risk (default: 1,4,8,12)",
    )
    parser.add_argument(
        "--contagion-horizons",
        type=str,
        default="1,4,8,12",
        help="Comma-separated horizons for predictive_contagion (default: 1,4,8,12)",
    )
    parser.add_argument(
        "--shared-model-h1",
        action="store_true",
        help=(
            "Train a single model on horizon=1 and reuse it for all horizons. "
            "This reduces compute cost but is less strict than training a separate model per horizon."
        ),
    )
    parser.add_argument(
        "--max-train-pairs",
        type=int,
        default=0,
        help="Limit number of training week-pairs (0 = all). Useful for quick runs.",
    )
    parser.add_argument(
        "--max-forward-pairs",
        type=int,
        default=0,
        help="Max number of test pairs per horizon for forward_risk (0 = all).",
    )
    parser.add_argument(
        "--max-contagion-samples",
        type=int,
        default=10,
        help="Max number of test pairs per horizon for predictive_contagion (default: 10).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only regenerate figures from saved JSON results in --output-dir (no model/data needed)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating figures (useful on servers without matplotlib)",
    )
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="If exp*.json exists in --output-dir, load it instead of recomputing that experiment",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add file handler for logging
    file_handler = logging.FileHandler(output_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    logger.addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("Task II: Model-based Forward-looking Financial Stability Analysis")
    logger.info("=" * 70)
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Output: {output_dir}")

    if args.plot_only:
        logger.info("Mode: plot-only (loading saved results, no model/data)")
        saved = load_saved_task2_results(output_dir)
        if not saved:
            logger.error(
                f"No saved results found in {output_dir}. Expected `all_results.json` "
                "or `exp1_forward_risk.json` / `exp2_predictive_contagion.json` / "
                "`exp3_early_warning.json`."
            )
            return

        try:
            plot_all_figures(
                forward_risk_results=saved.get("forward_risk") if args.experiment in ["all", "forward_risk"] else None,
                contagion_results=saved.get("predictive_contagion") if args.experiment in ["all", "predictive_contagion"] else None,
                early_warning_results=saved.get("early_warning") if args.experiment in ["all", "early_warning"] else None,
                output_dir=output_dir,
            )
        except RuntimeError as e:
            logger.error(str(e))
            return
        return

    DEFAULT_EPOCHS = 20
    DEFAULT_FORWARD_HORIZONS = "1,4,8,12"
    DEFAULT_CONTAGION_HORIZONS = "1,4,8,12"
    DEFAULT_MAX_FORWARD_PAIRS = 0
    DEFAULT_MAX_CONTAGION_SAMPLES = 10
    DEFAULT_MAX_TRAIN_PAIRS = 0

    if args.quick:
        if args.epochs == DEFAULT_EPOCHS:
            args.epochs = 3
        if args.forward_horizons == DEFAULT_FORWARD_HORIZONS:
            args.forward_horizons = "1"
        if args.contagion_horizons == DEFAULT_CONTAGION_HORIZONS:
            args.contagion_horizons = "1"
        if args.max_forward_pairs == DEFAULT_MAX_FORWARD_PAIRS:
            args.max_forward_pairs = 5
        if args.max_contagion_samples == DEFAULT_MAX_CONTAGION_SAMPLES:
            args.max_contagion_samples = 5
        if args.max_train_pairs == DEFAULT_MAX_TRAIN_PAIRS:
            args.max_train_pairs = 20
        logger.info(
            f"Quick mode: epochs={args.epochs}, forward_horizons={args.forward_horizons}, "
            f"contagion_horizons={args.contagion_horizons}, max_train_pairs={args.max_train_pairs}, "
            f"max_forward_pairs={args.max_forward_pairs}, max_contagion_samples={args.max_contagion_samples}"
        )

    _import_graphpfn_deps()
    default_device = ExperimentConfig().device
    if args.device is None:
        args.device = default_device
    logger.info(f"Device: {args.device}")
    logger.info(f"Model: {'GraphPFN-Frozen' if args.frozen else 'DeXposure-FM (fine-tuned)'}")

    if args.device.startswith("cuda") and default_device == "cpu":
        logger.warning("CUDA requested but DGL CUDA is not available; falling back to CPU.")
        args.device = "cpu"
        logger.info(f"Device: {args.device}")

    if not GRAPHPFN_AVAILABLE:
        logger.error("GraphPFN not available. Please install required dependencies.")
        return

    # Setup config
    config = ExperimentConfig(
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
    )
    set_seed(config.seed)

    # Load data
    logger.info("\nLoading data...")
    meta_category, category_list, category_to_idx = load_metadata(config.meta_path)
    network_data = load_network_data(config.data_path)
    all_dates = sorted(network_data.keys())

    logger.info(f"Loaded {len(all_dates)} weekly snapshots")

    # Build snapshots
    snapshots = [
        build_snapshot(date, network_data[date], meta_category, category_to_idx, category_list)
        for date in all_dates
    ]
    date_to_snap = {s["date"]: s for s in snapshots}

    # Get train/test split
    date_splits = get_single_split(all_dates)
    train_snapshots = [date_to_snap[d] for d in date_splits["train"] if d in date_to_snap]
    test_snapshots = [date_to_snap[d] for d in date_splits["test"] if d in date_to_snap]

    logger.info(f"Train snapshots: {len(train_snapshots)}")
    logger.info(f"Test snapshots: {len(test_snapshots)}")

    if args.experiment == "sis_sensitivity":
        logger.info("\nRunning SIS weight sensitivity (robustness check)...")
        _ = run_sis_weight_sensitivity(test_snapshots, output_dir=output_dir, sample_weeks=10, top_k=20)
        return

    forward_horizons = parse_int_list(args.forward_horizons)
    contagion_horizons = parse_int_list(args.contagion_horizons)

    # Build training pairs and load/train model(s).
    #
    # IMPORTANT: `run_full_experiment.py` trains a separate model per horizon because the supervision
    # targets change with h. We follow the same approach here to keep training/model behavior consistent.
    required_horizons: List[int] = []
    if args.experiment in ["all", "forward_risk"]:
        required_horizons.extend(forward_horizons)
    if args.experiment in ["all", "predictive_contagion"]:
        required_horizons.extend(contagion_horizons)
    required_horizons = sorted({int(h) for h in required_horizons if int(h) > 0})

    models_by_horizon: Dict[int, GraphPFNLinkPredictor] = {}
    if required_horizons:
        logger.info("\nPreparing model(s)...")
        if args.shared_model_h1:
            logger.info(
                f"Using shared h=1 model for horizons: {required_horizons} "
                "(compute-saving setting; fixed-parameter pseudo-OOS evaluation)."
            )
            set_seed(config.seed)
            train_pairs = build_week_pairs(
                train_snapshots, config.neg_ratio, config.seed, horizon=1
            )
            if args.max_train_pairs > 0 and len(train_pairs) > args.max_train_pairs:
                train_pairs = train_pairs[-args.max_train_pairs :]
                logger.info(
                    f"Subsampled train_pairs(h=1) to {len(train_pairs)} (max_train_pairs={args.max_train_pairs})"
                )
            shared_model = load_or_train_model(
                config,
                train_pairs,
                force_retrain=args.force_retrain,
                frozen=args.frozen,
                horizon=1,
            )
            for horizon in required_horizons:
                models_by_horizon[horizon] = shared_model
        else:
            logger.info(f"Training horizons: {required_horizons}")
            for horizon in required_horizons:
                set_seed(config.seed)
                train_pairs = build_week_pairs(
                    train_snapshots, config.neg_ratio, config.seed, horizon=horizon
                )
                if args.max_train_pairs > 0 and len(train_pairs) > args.max_train_pairs:
                    train_pairs = train_pairs[-args.max_train_pairs :]
                    logger.info(
                        f"Subsampled train_pairs(h={horizon}) to {len(train_pairs)} (max_train_pairs={args.max_train_pairs})"
                    )
                models_by_horizon[horizon] = load_or_train_model(
                    config,
                    train_pairs,
                    force_retrain=args.force_retrain,
                    frozen=args.frozen,
                    horizon=horizon,
                )

    # Run experiments
    results = {}

    if args.experiment in ["all", "forward_risk"]:
        exp_path = output_dir / "exp1_forward_risk.json"
        if args.reuse_results and exp_path.exists():
            logger.info(f"\nLoading cached forward_risk results from {exp_path}")
            results["forward_risk"] = _load_json(exp_path)
        else:
            results["forward_risk"] = run_forward_risk_prediction(
                config,
                models_by_horizon,
                test_snapshots,
                meta_category,
                horizons=forward_horizons,
                max_pairs_per_horizon=args.max_forward_pairs,
                output_dir=output_dir,
            )

    if args.experiment in ["all", "predictive_contagion"]:
        exp_path = output_dir / "exp2_predictive_contagion.json"
        requested_horizons = sorted({int(h) for h in contagion_horizons if int(h) > 0})

        if args.reuse_results and exp_path.exists():
            existing = _load_json(exp_path)
            if not isinstance(existing, dict):
                existing = {}

            existing_horizons: set[int] = set()
            existing_horizons_dict = existing.get("horizons", {})
            if isinstance(existing_horizons_dict, dict):
                for key in existing_horizons_dict.keys():
                    try:
                        existing_horizons.add(int(str(key).split("=", 1)[1]))
                    except Exception:
                        continue

            missing_horizons = sorted(set(requested_horizons) - existing_horizons)
            if missing_horizons:
                logger.info(
                    f"\nReusing predictive_contagion results for horizons {sorted(existing_horizons)}; "
                    f"computing missing horizons {missing_horizons}"
                )
                new_part = run_predictive_contagion(
                    config,
                    models_by_horizon,
                    test_snapshots,
                    meta_category,
                    horizons=missing_horizons,
                    max_samples_per_horizon=args.max_contagion_samples,
                    output_dir=output_dir,
                    save_json=False,
                )

                merged = existing
                if "scenarios" not in merged and "scenarios" in new_part:
                    merged["scenarios"] = new_part["scenarios"]

                merged_horizons = merged.get("horizons", {})
                if not isinstance(merged_horizons, dict):
                    merged_horizons = {}
                merged_horizons.update(new_part.get("horizons", {}))
                merged["horizons"] = merged_horizons

                with open(exp_path, "w") as f:
                    json.dump(merged, f, indent=2, default=float)
                logger.info(f"Updated {exp_path} (merged horizons)")
                results["predictive_contagion"] = merged
            else:
                logger.info(f"\nLoading cached predictive_contagion results from {exp_path}")
                results["predictive_contagion"] = existing
        else:
            results["predictive_contagion"] = run_predictive_contagion(
                config,
                models_by_horizon,
                test_snapshots,
                meta_category,
                horizons=requested_horizons,
                max_samples_per_horizon=args.max_contagion_samples,
                output_dir=output_dir,
            )

    if args.experiment in ["all", "early_warning"]:
        exp_path = output_dir / "exp3_early_warning.json"
        if args.reuse_results and exp_path.exists():
            logger.info(f"\nLoading cached early_warning results from {exp_path}")
            results["early_warning"] = _load_json(exp_path)
        else:
            # Note: run_early_warning_analysis now trains its own model per-event
            # to avoid data leakage (model never sees shock event data during training)
            results["early_warning"] = run_early_warning_analysis(
                config, snapshots, date_to_snap, meta_category,
                frozen=args.frozen, output_dir=output_dir
            )

    # Save combined results
    all_path = output_dir / "all_results.json"
    merged_results: Dict[str, Any] = {}
    if all_path.exists():
        try:
            prev = _load_json(all_path)
            if isinstance(prev, dict):
                merged_results.update(prev)
        except Exception:
            pass
    merged_results.update(results)
    with open(all_path, "w") as f:
        json.dump(merged_results, f, indent=2, default=float)

    # Generate visualization figures
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING VISUALIZATION FIGURES")
    logger.info("=" * 70)

    if args.no_plot:
        logger.info("Skipping figure generation (--no-plot).")
    else:
        try:
            plot_all_figures(
                forward_risk_results=results.get("forward_risk"),
                contagion_results=results.get("predictive_contagion"),
                early_warning_results=results.get("early_warning"),
                output_dir=output_dir,
            )
        except RuntimeError as e:
            logger.error(str(e))
            logger.error("Tip: run with `--no-plot` and later regenerate with `--plot-only` on a machine with matplotlib.")
            return

    logger.info("\n" + "=" * 70)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Figures saved to: {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
