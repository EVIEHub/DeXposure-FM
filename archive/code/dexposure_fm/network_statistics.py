#!/usr/bin/env python3
"""
Network Statistics Module for DeXposure-FM experiments.
Computes network-level metrics required by the paper (Section 5.2).

Metrics:
- Gini coefficient (degree centralization)
- HHI (Herfindahl-Hirschman Index for concentration)
- Network density
- Degree entropy
- Top 10% concentration
- Assortativity
- Sector connectivity (if sector labels available)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import networkx as nx


def gini_coefficient(values: np.ndarray) -> float:
    """
    Calculate the Gini coefficient of a list of values.
    Measures inequality in degree distribution.

    Returns:
        Gini coefficient in [0, 1], where 0 = perfect equality, 1 = perfect inequality
    """
    if len(values) == 0:
        return 0.0
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    numerator = 2 * np.sum(np.arange(1, n + 1) * sorted_values) - (n + 1) * cumsum[-1]
    denominator = n * cumsum[-1] + 1e-12
    return float(numerator / denominator)


def herfindahl_hirschman_index(values: np.ndarray) -> float:
    """
    Calculate HHI (Herfindahl-Hirschman Index) for concentration.
    Used to measure market/exposure concentration.

    Returns:
        HHI in [0, 1], where higher = more concentrated
    """
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0
    shares = values / (np.sum(values) + 1e-12)
    return float(np.sum(shares**2))


def network_density(num_nodes: int, num_edges: int, directed: bool = True) -> float:
    """
    Calculate network density.

    Returns:
        Density in [0, 1]
    """
    if num_nodes <= 1:
        return 0.0
    max_edges = num_nodes * (num_nodes - 1)
    if not directed:
        max_edges //= 2
    return num_edges / (max_edges + 1e-12)


def degree_entropy(degrees: np.ndarray) -> float:
    """
    Calculate entropy of degree distribution.
    H = -sum(p(k) * log(p(k)))

    Higher entropy = more uniform degree distribution
    """
    if len(degrees) == 0:
        return 0.0

    # Count degree frequencies
    unique, counts = np.unique(degrees, return_counts=True)
    probabilities = counts / len(degrees)

    # Calculate entropy (using natural log)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))
    return float(entropy)


def top_k_concentration(values: np.ndarray, k_percent: float = 0.1) -> float:
    """
    Calculate concentration in top K% of values.
    T = sum(top_k values) / sum(all values)

    Args:
        values: Array of values (e.g., degrees, exposures)
        k_percent: Fraction of top values to consider (default 10%)

    Returns:
        Concentration ratio in [0, 1]
    """
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0

    sorted_values = np.sort(values)[::-1]  # Descending
    k = max(1, int(len(sorted_values) * k_percent))
    top_k_sum = np.sum(sorted_values[:k])
    total_sum = np.sum(sorted_values)

    return float(top_k_sum / (total_sum + 1e-12))


def degree_assortativity(edge_index: np.ndarray, num_nodes: int) -> float:
    """
    Calculate degree assortativity coefficient.
    Measures tendency of nodes to connect to similar-degree nodes.

    Returns:
        Assortativity in [-1, 1], where:
        - Positive: high-degree nodes connect to high-degree nodes
        - Negative: high-degree nodes connect to low-degree nodes
    """
    if edge_index.shape[1] == 0:
        return 0.0

    # Build NetworkX graph
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    edges = [
        (int(edge_index[0, i]), int(edge_index[1, i]))
        for i in range(edge_index.shape[1])
    ]
    G.add_edges_from(edges)

    try:
        return float(nx.degree_assortativity_coefficient(G))
    except Exception:
        return 0.0


def sector_connectivity_matrix(
    edge_index: np.ndarray,
    edge_weights: np.ndarray,
    node_sectors: list[str],
    sector_list: list[str],
) -> np.ndarray:
    """
    Compute sector-to-sector exposure matrix.

    Args:
        edge_index: [2, E] edge indices
        edge_weights: [E] edge weights
        node_sectors: Sector label for each node
        sector_list: List of unique sectors

    Returns:
        [S, S] matrix where M[i,j] = total exposure from sector i to sector j
    """
    num_sectors = len(sector_list)
    sector_to_idx = {s: i for i, s in enumerate(sector_list)}

    matrix = np.zeros((num_sectors, num_sectors), dtype=np.float64)

    for e in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, e]), int(edge_index[1, e])
        weight = float(edge_weights[e]) if e < len(edge_weights) else 1.0

        src_sector = node_sectors[src] if src < len(node_sectors) else "Unknown"
        dst_sector = node_sectors[dst] if dst < len(node_sectors) else "Unknown"

        if src_sector in sector_to_idx and dst_sector in sector_to_idx:
            i, j = sector_to_idx[src_sector], sector_to_idx[dst_sector]
            matrix[i, j] += weight

    return matrix


def compute_all_network_statistics(
    edge_index: np.ndarray,
    num_nodes: int,
    edge_weights: np.ndarray | None = None,
    node_sizes: np.ndarray | None = None,
    node_sectors: list[str] | None = None,
    sector_list: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute all network-level statistics for a single snapshot.

    Args:
        edge_index: [2, E] edge indices
        num_nodes: Number of nodes
        edge_weights: [E] optional edge weights
        node_sizes: [N] optional node sizes (TVL)
        node_sectors: [N] optional sector labels
        sector_list: List of unique sectors

    Returns:
        Dictionary of statistics
    """
    num_edges = edge_index.shape[1] if edge_index.ndim == 2 else 0

    # Compute degrees
    out_degrees = np.zeros(num_nodes, dtype=np.float64)
    in_degrees = np.zeros(num_nodes, dtype=np.float64)

    for e in range(num_edges):
        src, dst = int(edge_index[0, e]), int(edge_index[1, e])
        out_degrees[src] += 1
        in_degrees[dst] += 1

    total_degrees = out_degrees + in_degrees

    # Basic statistics
    stats = {
        "num_nodes": float(num_nodes),
        "num_edges": float(num_edges),
        "density": network_density(num_nodes, num_edges, directed=True),
        "degree_gini": gini_coefficient(total_degrees),
        "degree_entropy": degree_entropy(total_degrees.astype(int)),
        "top_10_concentration": top_k_concentration(total_degrees, 0.1),
        "assortativity": degree_assortativity(edge_index, num_nodes),
    }

    # Weight-based statistics (if available)
    if edge_weights is not None and len(edge_weights) > 0:
        ew = np.array(edge_weights, dtype=np.float64)
        ew = ew[np.isfinite(ew)]
        ew = np.maximum(ew, 0.0)
        if ew.size > 0:
            stats["edge_weight_gini"] = gini_coefficient(ew)
            stats["edge_weight_hhi"] = herfindahl_hirschman_index(ew)
            stats["total_exposure"] = float(np.sum(ew))
            stats["mean_edge_weight"] = float(np.mean(ew))

    # Node size statistics (if available)
    if node_sizes is not None and len(node_sizes) > 0:
        ns = np.array(node_sizes, dtype=np.float64)
        ns = ns[np.isfinite(ns)]
        ns = np.maximum(ns, 0.0)
        if ns.size > 0:
            stats["tvl_gini"] = gini_coefficient(ns)
            stats["tvl_hhi"] = herfindahl_hirschman_index(ns)
            stats["total_tvl"] = float(np.sum(ns))
            stats["tvl_top_10_concentration"] = top_k_concentration(ns, 0.1)

    # Sector connectivity (if available)
    if (
        node_sectors is not None
        and sector_list is not None
        and edge_weights is not None
    ):
        sector_matrix = sector_connectivity_matrix(
            edge_index, edge_weights, node_sectors, sector_list
        )
        stats["sector_connectivity_density"] = float(
            np.count_nonzero(sector_matrix) / (sector_matrix.size + 1e-12)
        )
        stats["cross_sector_exposure_ratio"] = float(
            (np.sum(sector_matrix) - np.trace(sector_matrix))
            / (np.sum(sector_matrix) + 1e-12)
        )

    return stats


def compute_statistics_delta(
    stats_t: dict[str, float], stats_t1: dict[str, float]
) -> dict[str, float]:
    """
    Compute change in statistics between two time periods.

    Returns:
        Dictionary of deltas (t+1 - t)
    """
    delta = {}
    for key in stats_t:
        if key in stats_t1:
            delta[f"delta_{key}"] = stats_t1[key] - stats_t[key]
    return delta


# ============== Batch computation for temporal analysis ==============


def compute_rolling_statistics(
    snapshots: list[dict[str, Any]],
    sector_list: list[str] | None = None,
) -> list[dict[str, float | str]]:
    """
    Compute network statistics for a sequence of snapshots.

    Args:
        snapshots: List of snapshot dictionaries with keys:
            - edge_index: [2, E] array
            - num_nodes: int
            - edge_weights: [E] array (optional)
            - node_sizes: [N] array (optional)
            - node_sectors: [N] list (optional)
        sector_list: List of unique sectors

    Returns:
        List of statistics dictionaries
    """
    all_stats = []

    for snap in snapshots:
        stats = compute_all_network_statistics(
            edge_index=snap["edge_index"],
            num_nodes=snap["num_nodes"],
            edge_weights=snap.get("edge_weights"),
            node_sizes=snap.get("node_sizes"),
            node_sectors=snap.get("node_sectors"),
            sector_list=sector_list,
        )
        stats["date"] = snap.get("date", "unknown")
        all_stats.append(stats)

    return all_stats


if __name__ == "__main__":
    # Simple test
    # Create dummy graph
    num_nodes = 100
    edge_index = np.array(
        [
            [0, 1, 2, 3, 4, 0, 1, 2],
            [1, 2, 3, 4, 0, 2, 3, 4],
        ]
    )
    edge_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5])
    node_sizes = np.random.exponential(100, num_nodes)

    stats = compute_all_network_statistics(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
        node_sizes=node_sizes,
    )

    print("Network Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
