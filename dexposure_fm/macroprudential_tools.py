#!/usr/bin/env python3
"""
Macroprudential tools for DeXposure-FM.

These utilities turn an exposure graph snapshot into:
- Systemic importance scores (SIS)
- Sector-to-sector spillover exposures and concentration indices
- Simple DebtRank-style contagion stress tests

The key idea for "model-enabled tools" is *forecast-then-measure*:
use DeXposure-FM to forecast a future graph \\hat{G}_{t+h}, then apply the
same tools to the predicted snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def array_snap_to_dict_snap(
    snap: dict[str, Any],
    *,
    edge_weight_is_log: bool = True,
) -> dict[str, Any]:
    """
    Convert an array-format snapshot into a dict-format snapshot.

    Array format:
      - node_ids: List[str]
      - sizes: np.ndarray (TVL values)
      - edge_src: np.ndarray (source indices)
      - edge_dst: np.ndarray (destination indices)
      - edge_weight: np.ndarray (log1p or linear weights)
      - categories: List[str] (optional)

    Dict format:
      - nodes: Dict[str, {"tvlUsd": float, "category": str}]
      - edges: List[{"source": str, "target": str, "weight": float}]
    """
    if "nodes" in snap and isinstance(snap.get("nodes"), dict):
        return snap

    node_ids: list[str] = list(snap.get("node_ids", []) or [])
    sizes = np.asarray(snap.get("sizes", []), dtype=float)
    categories: list[str] = list(snap.get("categories", []) or [])
    edge_src = np.asarray(snap.get("edge_src", []), dtype=int)
    edge_dst = np.asarray(snap.get("edge_dst", []), dtype=int)
    edge_weight = np.asarray(snap.get("edge_weight", []), dtype=float)

    nodes: dict[str, dict[str, Any]] = {}
    for i, nid in enumerate(node_ids):
        tvl = float(sizes[i]) if i < sizes.size else 0.0
        cat = categories[i] if i < len(categories) else "Unknown"
        nodes[str(nid)] = {"tvlUsd": tvl, "category": cat}

    edges: list[dict[str, Any]] = []
    m = int(min(edge_src.size, edge_dst.size))
    for i in range(m):
        src_idx = int(edge_src[i])
        dst_idx = int(edge_dst[i])
        if src_idx < 0 or dst_idx < 0:
            continue
        if src_idx >= len(node_ids) or dst_idx >= len(node_ids):
            continue
        w = float(edge_weight[i]) if i < edge_weight.size else 0.0
        if edge_weight_is_log:
            w = float(np.expm1(w))
        if w <= 0.0:
            continue
        edges.append(
            {"source": node_ids[src_idx], "target": node_ids[dst_idx], "weight": w}
        )

    return {"nodes": nodes, "edges": edges, "date": snap.get("date", "")}


def gini_coefficient(values: Sequence[float]) -> float:
    """Gini coefficient in [0,1] (0 = equal, 1 = maximally unequal)."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    arr = np.sort(arr)
    n = int(arr.size)
    index = np.arange(1, n + 1, dtype=float)
    denom = float(n * np.sum(arr))
    if denom <= 0:
        return 0.0
    return float((2.0 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / denom)


def compute_sis_components(
    snap: Mapping[str, Any],
    *,
    tail_k: int = 5,
    pagerank_max_iter: int = 100,
) -> dict[str, dict[str, float]]:
    """
    Compute normalized SIS components for a dict-format snapshot:
      - pagerank: weighted PageRank normalized to [0,1] by dividing by max
      - tail_exposure: top-k outgoing exposure share in [0,1]
      - log_tvl_norm: log(1+TVL) normalized to [0,1] by dividing by max
    """
    nodes = snap.get("nodes", {}) or {}
    edges = snap.get("edges", []) or []
    if not isinstance(nodes, dict) or not nodes:
        return {"pagerank": {}, "tail_exposure": {}, "log_tvl_norm": {}}

    import networkx as nx

    g = nx.DiGraph()
    for node_id, data in nodes.items():
        tvl = float((data or {}).get("tvlUsd", 0.0) or 0.0)
        g.add_node(node_id, tvl=tvl)

    for e in edges:
        src = (e or {}).get("source")
        dst = (e or {}).get("target")
        w = float((e or {}).get("weight", 0.0) or 0.0)
        if src in nodes and dst in nodes and w > 0:
            g.add_edge(src, dst, weight=w)

    try:
        pagerank_raw = nx.pagerank(g, weight="weight", max_iter=pagerank_max_iter)
    except Exception:
        pagerank_raw = {n: 0.0 for n in nodes}

    pr_max = float(max(pagerank_raw.values())) if pagerank_raw else 0.0
    if pr_max > 0:
        pagerank = {n: float(pagerank_raw.get(n, 0.0)) / pr_max for n in nodes}
    else:
        pagerank = {n: 0.0 for n in nodes}

    outgoing_weights: dict[str, list[float]] = {n: [] for n in nodes}
    for e in edges:
        src = (e or {}).get("source")
        w = float((e or {}).get("weight", 0.0) or 0.0)
        if src in outgoing_weights and w > 0:
            outgoing_weights[str(src)].append(w)

    tail_exposure: dict[str, float] = {}
    k = int(max(1, tail_k))
    for n, ws in outgoing_weights.items():
        if not ws:
            tail_exposure[n] = 0.0
            continue
        total = float(np.sum(ws))
        if total <= 0:
            tail_exposure[n] = 0.0
            continue
        top_k_sum = float(np.sum(sorted(ws, reverse=True)[:k]))
        tail_exposure[n] = top_k_sum / total

    tvl_values = {
        n: float((data or {}).get("tvlUsd", 0.0) or 0.0) for n, data in nodes.items()
    }
    log_tvl = {n: float(np.log1p(max(v, 0.0))) for n, v in tvl_values.items()}
    log_tvl_max = float(max(log_tvl.values())) if log_tvl else 0.0
    if log_tvl_max > 0:
        log_tvl_norm = {n: float(v) / log_tvl_max for n, v in log_tvl.items()}
    else:
        log_tvl_norm = {n: 0.0 for n in nodes}

    return {
        "pagerank": pagerank,
        "tail_exposure": tail_exposure,
        "log_tvl_norm": log_tvl_norm,
    }


def compute_systemic_importance_score(
    snap: Mapping[str, Any],
    *,
    alpha: float = 1 / 3,
    beta: float = 1 / 3,
    gamma: float = 1 / 3,
    tail_k: int = 5,
    pagerank_max_iter: int = 100,
) -> dict[str, float]:
    """
    Systemic Importance Score (SIS) per node:

        SIS_i = alpha * PageRank_i + beta * TailExposure_i + gamma * log(1 + TVL_i)

    Each component is normalized to a comparable [0,1] scale in
    `compute_sis_components`.
    """
    nodes = snap.get("nodes", {}) or {}
    if not isinstance(nodes, dict) or not nodes:
        return {}

    components = compute_sis_components(
        snap, tail_k=tail_k, pagerank_max_iter=pagerank_max_iter
    )
    pagerank = components["pagerank"]
    tail = components["tail_exposure"]
    log_tvl = components["log_tvl_norm"]

    scores: dict[str, float] = {}
    for n in nodes:
        scores[n] = (
            float(alpha) * float(pagerank.get(n, 0.0))
            + float(beta) * float(tail.get(n, 0.0))
            + float(gamma) * float(log_tvl.get(n, 0.0))
        )
    return scores


def compute_sector_spillover_index(
    snap: Mapping[str, Any],
    sector_map: Mapping[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Sector-to-sector spillover exposures based on cross-sector edge weights.

    Returns:
        sector_exposure[src_sector][dst_sector] = total cross-sector exposure weight
    """
    nodes = snap.get("nodes", {}) or {}
    edges = snap.get("edges", []) or []
    if not isinstance(nodes, dict) or not nodes:
        return {}

    if sector_map is None:
        sector_map = {
            node_id: str((data or {}).get("category", "other") or "other")
            for node_id, data in nodes.items()
        }

    sector_exposure: dict[str, dict[str, float]] = {}
    for e in edges:
        src = (e or {}).get("source")
        dst = (e or {}).get("target")
        w = float((e or {}).get("weight", 0.0) or 0.0)
        if not src or not dst or w <= 0:
            continue
        src_sector = str(sector_map.get(str(src), "other") or "other")
        dst_sector = str(sector_map.get(str(dst), "other") or "other")
        if src_sector == dst_sector:
            continue
        sector_exposure.setdefault(src_sector, {})
        sector_exposure[src_sector][dst_sector] = (
            sector_exposure[src_sector].get(dst_sector, 0.0) + w
        )

    return sector_exposure


def spillover_hhi(
    sector_exposure: Mapping[str, Mapping[str, float]],
) -> tuple[float, float]:
    """
    Compute spillover_total and spillover_index (HHI) from a sector_exposure dict.

    - spillover_total is scale-dependent (sum of off-diagonal weights).
    - spillover_index is scale-invariant (HHI over off-diagonal flows).
    """
    off_diag = [
        float(w)
        for targets in sector_exposure.values()
        for w in (targets or {}).values()
        if float(w) > 0
    ]
    total = float(np.sum(off_diag)) if off_diag else 0.0
    if total <= 0:
        return 0.0, 0.0
    shares = np.asarray(off_diag, dtype=float) / total
    return total, float(np.sum(shares**2))


def simulate_contagion(
    snap: Mapping[str, Any],
    shocked_nodes: Sequence[str],
    *,
    shock_fraction: float = 1.0,
    distress_threshold: float = 0.1,
    max_rounds: int = 10,
) -> dict[str, Any]:
    """
    DebtRank-style contagion:
    interpret edge (creditor -> debtor) as creditor exposure to debtor.
    """
    nodes = snap.get("nodes", {}) or {}
    edges = snap.get("edges", []) or []

    tvl = {
        n: float((data or {}).get("tvlUsd", 0.0) or 0.0) for n, data in nodes.items()
    }
    losses = {n: 0.0 for n in nodes}

    exposures_in: dict[str, dict[str, float]] = {}
    for e in edges:
        src = (e or {}).get("source")
        dst = (e or {}).get("target")
        w = float((e or {}).get("weight", 0.0) or 0.0)
        if not src or not dst or w <= 0:
            continue
        dst = str(dst)
        src = str(src)
        exposures_in.setdefault(dst, {})
        exposures_in[dst][src] = exposures_in[dst].get(src, 0.0) + w

    distressed: set[str] = set()
    active: set[str] = set()
    for node in shocked_nodes:
        node = str(node)
        if node not in tvl:
            continue
        tvl_n = float(tvl[node])
        if tvl_n <= 0:
            continue
        losses[node] = float(shock_fraction) * tvl_n
        distressed.add(node)
        active.add(node)

    affected_history = [len(distressed)]
    for _ in range(int(max_rounds)):
        if not active:
            break

        new_active: set[str] = set()
        for debtor in active:
            creditors = exposures_in.get(debtor, {})
            if not creditors:
                continue
            total_exposure = float(np.sum(list(creditors.values())))
            if total_exposure <= 0:
                continue
            debtor_loss = float(losses.get(debtor, 0.0))
            if debtor_loss <= 0:
                continue

            for creditor, exposure in creditors.items():
                exposure = float(exposure)
                if exposure <= 0:
                    continue
                if creditor not in tvl:
                    continue
                loss_share = exposure / total_exposure
                propagated = debtor_loss * loss_share
                losses[creditor] += propagated

                tvl_c = float(tvl.get(creditor, 0.0))
                if tvl_c > 0:
                    losses[creditor] = min(losses[creditor], tvl_c)
                    if (
                        creditor not in distressed
                        and losses[creditor] > float(distress_threshold) * tvl_c
                    ):
                        distressed.add(creditor)
                        new_active.add(creditor)

        if not new_active:
            break
        active = new_active
        affected_history.append(len(distressed))

    total_tvl = float(np.sum(list(tvl.values()))) if tvl else 0.0
    total_loss = float(np.sum(list(losses.values()))) if losses else 0.0
    return {
        "shocked_nodes": list(map(str, shocked_nodes)),
        "shock_fraction": float(shock_fraction),
        "total_loss": float(total_loss),
        "total_loss_pct": 100.0 * total_loss / total_tvl if total_tvl > 0 else 0.0,
        "loss_fraction": total_loss / total_tvl if total_tvl > 0 else 0.0,
        "affected_count": len(distressed),
        "distressed_count": len(distressed),
        "distressed_nodes": sorted(distressed),
        "propagation_rounds": int(len(affected_history) - 1),
        "affected_history": affected_history,
    }


def compute_network_risk_metrics(
    snap: Mapping[str, Any],
    *,
    meta_category: Mapping[str, str] | None = None,
    edge_weight_is_log: bool = True,
) -> dict[str, float]:
    """
    Compute a compact set of risk metrics on an array-format snapshot.

    Returns:
        - TVL concentration: HHI, Gini, top10 share
        - Edge weight concentration: HHI, top10 share
        - Density and degree concentration
        - Mean SIS (if computable)
        - Spillover concentration (if meta_category provided)
    """
    node_ids: list[str] = list(snap.get("node_ids", []) or [])
    sizes = np.asarray(snap.get("sizes", []), dtype=np.float64)
    edge_src = np.asarray(snap.get("edge_src", []), dtype=int)
    edge_dst = np.asarray(snap.get("edge_dst", []), dtype=int)
    edge_weight = np.asarray(snap.get("edge_weight", []), dtype=np.float64)

    n_nodes = len(node_ids)
    n_edges = edge_src.size

    metrics: dict[str, float] = {
        "n_nodes": float(n_nodes),
        "n_edges": float(n_edges),
        "total_tvl": float(np.sum(sizes)) if sizes.size else 0.0,
    }
    if n_nodes <= 0:
        return metrics

    total_tvl = float(np.sum(sizes)) if sizes.size else 0.0
    if total_tvl > 0:
        shares = sizes / total_tvl
        metrics["tvl_hhi"] = float(np.sum(shares**2))
        metrics["top10_tvl_share"] = float(np.sum(np.sort(shares)[::-1][:10]))
        metrics["tvl_gini"] = gini_coefficient(sizes.tolist())
    else:
        metrics["tvl_hhi"] = 0.0
        metrics["top10_tvl_share"] = 0.0
        metrics["tvl_gini"] = 0.0

    if edge_weight.size > 0:
        weights_linear = np.expm1(edge_weight) if edge_weight_is_log else edge_weight
        weights_linear = np.maximum(weights_linear, 0)
        total_w = float(np.sum(weights_linear))
        if total_w > 0:
            w_shares = weights_linear / total_w
            metrics["edge_hhi"] = float(np.sum(w_shares**2))
            metrics["top10_edge_share"] = float(np.sum(np.sort(w_shares)[::-1][:10]))
        else:
            metrics["edge_hhi"] = 0.0
            metrics["top10_edge_share"] = 0.0
    else:
        metrics["edge_hhi"] = 0.0
        metrics["top10_edge_share"] = 0.0

    max_edges = n_nodes * (n_nodes - 1)
    metrics["density"] = float(n_edges / max_edges) if max_edges > 0 else 0.0

    in_degree = np.zeros(n_nodes, dtype=float)
    out_degree = np.zeros(n_nodes, dtype=float)
    m = int(min(edge_src.size, edge_dst.size))
    for i in range(m):
        s = int(edge_src[i])
        d = int(edge_dst[i])
        if 0 <= s < n_nodes:
            out_degree[s] += 1
        if 0 <= d < n_nodes:
            in_degree[d] += 1

    metrics["mean_in_degree"] = float(np.mean(in_degree)) if n_nodes else 0.0
    metrics["max_in_degree"] = float(np.max(in_degree)) if n_nodes else 0.0
    metrics["degree_gini"] = gini_coefficient((in_degree + out_degree).tolist())

    # SIS on dict-format snapshot (always needs linear weights)
    try:
        dict_snap = array_snap_to_dict_snap(
            dict(snap), edge_weight_is_log=edge_weight_is_log
        )
        sis = compute_systemic_importance_score(dict_snap)
        if sis:
            vals = list(sis.values())
            metrics["mean_sis"] = float(np.mean(vals))
            metrics["max_sis"] = float(np.max(vals))
            metrics["sis_gini"] = gini_coefficient(vals)
    except Exception:
        pass

    if meta_category is not None:
        try:
            dict_snap = array_snap_to_dict_snap(
                dict(snap), edge_weight_is_log=edge_weight_is_log
            )
            sector_exposure = compute_sector_spillover_index(dict_snap, meta_category)
            total, idx = spillover_hhi(sector_exposure)
            metrics["spillover_total"] = float(total)
            metrics["spillover_index"] = float(idx)
        except Exception:
            metrics.setdefault("spillover_total", 0.0)
            metrics.setdefault("spillover_index", 0.0)

    return metrics
