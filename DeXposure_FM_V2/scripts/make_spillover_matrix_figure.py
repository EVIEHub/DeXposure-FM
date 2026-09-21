#!/usr/bin/env python3
"""
Generate a sector-to-sector spillover (cross-exposure) matrix heatmap figure for the paper.

This script is intentionally lightweight and does NOT import `run_full_experiment.py` (which may
pull GPU/DGL dependencies). It reads `data/...json` directly and aggregates link `size` by
sector labels from `data/meta_df.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple


def _load_meta_category(meta_path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with meta_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("id") or "").strip()
            cat = (row.get("category") or "").strip()
            if pid:
                out[pid] = cat or "other"
    return out


def _load_payload_data(data_path: Path) -> Mapping[str, Dict]:
    with data_path.open("r") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], dict):
        return payload["data"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unexpected JSON format in {data_path}")


def _resolve_date(requested: str, dates: List[str]) -> str:
    if requested in dates:
        return requested
    # nearest by absolute days (dates are YYYY-MM-DD)
    def _to_ymd(s: str) -> Tuple[int, int, int]:
        y, m, d = s.split("-")
        return int(y), int(m), int(d)

    ry, rm, rd = _to_ymd(requested)

    def _score(s: str) -> int:
        y, m, d = _to_ymd(s)
        # crude but stable ordering for weekly data; avoids datetime dependency
        return abs((y - ry) * 372 + (m - rm) * 31 + (d - rd))

    return min(dates, key=_score)


def _compute_sector_exposure(
    snap: Mapping[str, object],
    meta_category: Mapping[str, str],
    *,
    min_edge_weight: float,
) -> Dict[str, Dict[str, float]]:
    links = snap.get("links") or []
    if not isinstance(links, list):
        return {}

    out: Dict[str, Dict[str, float]] = {}
    for e in links:
        if not isinstance(e, dict):
            continue
        src = str(e.get("source") or "")
        dst = str(e.get("target") or "")
        w = float(e.get("size") or 0.0)
        if not src or not dst or not math.isfinite(w) or w <= min_edge_weight:
            continue
        src_sector = str(meta_category.get(src, "other") or "other")
        dst_sector = str(meta_category.get(dst, "other") or "other")
        if src_sector == dst_sector:
            continue
        out.setdefault(src_sector, {})
        out[src_sector][dst_sector] = out[src_sector].get(dst_sector, 0.0) + w
    return out


def _iter_pairs(sector_exposure: Mapping[str, Mapping[str, float]]) -> Iterable[Tuple[str, str, float]]:
    for s, row in sector_exposure.items():
        if not isinstance(row, Mapping):
            continue
        for t, w in row.items():
            try:
                ww = float(w)
            except Exception:
                continue
            if math.isfinite(ww) and ww > 0:
                yield str(s), str(t), ww


def _select_sectors(
    sector_exposure: Mapping[str, Mapping[str, float]],
    *,
    max_sectors: int,
) -> List[str]:
    totals: Dict[str, float] = {}
    for s, t, w in _iter_pairs(sector_exposure):
        totals[s] = totals.get(s, 0.0) + w
        totals[t] = totals.get(t, 0.0) + w
    if not totals:
        return ["other"]
    ranked = sorted(totals.items(), key=lambda kv: float(kv[1]), reverse=True)
    keep = [k for k, _ in ranked[: max(1, int(max_sectors))]]
    # put "other" last if present
    keep = [k for k in keep if k != "other"] + (["other"] if "other" in totals else [])
    return keep


def _build_matrix(
    sector_exposure: Mapping[str, Mapping[str, float]],
    sectors: List[str],
    *,
    other_label: str = "other",
) -> List[List[float]]:
    idx = {s: i for i, s in enumerate(sectors)}
    n = len(sectors)
    m = [[0.0 for _ in range(n)] for _ in range(n)]

    for s, t, w in _iter_pairs(sector_exposure):
        ss = s if s in idx else other_label
        tt = t if t in idx else other_label
        if ss not in idx:
            continue
        if tt not in idx:
            continue
        i = idx[ss]
        j = idx[tt]
        if i == j:
            continue
        m[i][j] += float(w)
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, default="2025-06-30", help="Snapshot date (YYYY-MM-DD)")
    p.add_argument("--data-path", type=Path, default=Path("data/historical-network_week_2025-07-01.json"))
    p.add_argument("--meta-path", type=Path, default=Path("data/meta_df.csv"))
    p.add_argument("--out", type=Path, default=Path("DeXposure_FM_V2/figures/fig_spillover_matrix_example.pdf"))
    p.add_argument("--max-sectors", type=int, default=15, help="Max number of sectors to show (others aggregated to 'other')")
    p.add_argument("--min-edge-weight", type=float, default=0.0, help="Filter edges with weight <= this threshold")
    args = p.parse_args()

    meta_category = _load_meta_category(args.meta_path)
    data = _load_payload_data(args.data_path)
    dates = sorted(data.keys())
    if not dates:
        raise RuntimeError("No snapshots found in data file.")

    resolved_date = _resolve_date(args.date, dates)
    snap = data[resolved_date]
    sector_exposure = _compute_sector_exposure(snap, meta_category, min_edge_weight=float(args.min_edge_weight))
    sectors = _select_sectors(sector_exposure, max_sectors=int(args.max_sectors))
    if "other" not in sectors:
        sectors.append("other")
    matrix = _build_matrix(sector_exposure, sectors, other_label="other")

    # Plot
    import numpy as np  # noqa: E402

    import matplotlib.pyplot as plt  # noqa: E402

    arr = np.array(matrix, dtype=float)
    z = np.log10(1.0 + np.maximum(arr, 0.0))

    fig_w = max(7.5, 0.45 * len(sectors))
    fig_h = max(6.5, 0.45 * len(sectors))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(z, cmap="magma", aspect="auto")

    ax.set_title(f"Sector-to-sector cross-exposure matrix $S$ (t = {resolved_date})")
    ax.set_xlabel("Destination sector")
    ax.set_ylabel("Source sector")

    ax.set_xticks(range(len(sectors)))
    ax.set_yticks(range(len(sectors)))
    ax.set_xticklabels(sectors, rotation=45, ha="right")
    ax.set_yticklabels(sectors)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_{10}(1 + S_{ij})$")

    # Make room on the right for the annotation box + colorbar.
    fig.subplots_adjust(right=0.76)

    # annotate top links (by raw exposure)
    pairs = []
    for i, s in enumerate(sectors):
        for j, t in enumerate(sectors):
            if i == j:
                continue
            w = float(arr[i, j])
            if w > 0:
                pairs.append((w, s, t))
    pairs.sort(reverse=True)
    top_lines = [f"{s}→{t}: {w:,.0f}" for w, s, t in pairs[:10]]
    if top_lines:
        ax.text(
            1.02,
            0.5,
            "Top cross-sector links\\n" + "\\n".join(top_lines),
            transform=ax.transAxes,
            va="center",
            fontsize=9,
        )

    # Note: avoid tight_layout() here because the right-side annotation intentionally
    # extends beyond the Axes.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
