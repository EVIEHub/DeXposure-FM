#!/usr/bin/env python3
"""Build the v2 sector-to-sector spillover matrix figure."""

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
        for row in csv.DictReader(f):
            protocol_id = (row.get("id") or "").strip()
            category = (row.get("category") or "").strip()
            if protocol_id:
                out[protocol_id] = category or "other"
    return out


def _load_payload_data(data_path: Path) -> Mapping[str, Dict]:
    with data_path.open("r") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return payload["data"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unexpected JSON format in {data_path}")


def _resolve_date(requested: str, dates: List[str]) -> str:
    if requested in dates:
        return requested

    def to_ymd(value: str) -> Tuple[int, int, int]:
        year, month, day = value.split("-")
        return int(year), int(month), int(day)

    requested_year, requested_month, requested_day = to_ymd(requested)

    def score(value: str) -> int:
        year, month, day = to_ymd(value)
        return abs(
            (year - requested_year) * 372
            + (month - requested_month) * 31
            + (day - requested_day)
        )

    return min(dates, key=score)


def _compute_sector_exposure(
    snapshot: Mapping[str, object],
    metadata: Mapping[str, str],
    *,
    min_edge_weight: float,
) -> Dict[str, Dict[str, float]]:
    links = snapshot.get("links") or []
    if not isinstance(links, list):
        return {}

    output: Dict[str, Dict[str, float]] = {}
    for edge in links:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        weight = float(edge.get("size") or 0.0)
        if not source or not target or not math.isfinite(weight) or weight <= min_edge_weight:
            continue
        source_sector = str(metadata.get(source, "other") or "other")
        target_sector = str(metadata.get(target, "other") or "other")
        if source_sector == target_sector:
            continue
        output.setdefault(source_sector, {})
        output[source_sector][target_sector] = (
            output[source_sector].get(target_sector, 0.0) + weight
        )
    return output


def _iter_pairs(
    sector_exposure: Mapping[str, Mapping[str, float]],
) -> Iterable[Tuple[str, str, float]]:
    for source, row in sector_exposure.items():
        if not isinstance(row, Mapping):
            continue
        for target, weight in row.items():
            try:
                finite_weight = float(weight)
            except Exception:
                continue
            if math.isfinite(finite_weight) and finite_weight > 0:
                yield str(source), str(target), finite_weight


def _select_sectors(
    sector_exposure: Mapping[str, Mapping[str, float]], *, max_sectors: int
) -> List[str]:
    totals: Dict[str, float] = {}
    for source, target, weight in _iter_pairs(sector_exposure):
        totals[source] = totals.get(source, 0.0) + weight
        totals[target] = totals.get(target, 0.0) + weight
    if not totals:
        return ["other"]
    ranked = sorted(totals.items(), key=lambda item: float(item[1]), reverse=True)
    keep = [key for key, _ in ranked[: max(1, int(max_sectors))]]
    return [key for key in keep if key != "other"] + (
        ["other"] if "other" in totals else []
    )


def _build_matrix(
    sector_exposure: Mapping[str, Mapping[str, float]], sectors: List[str]
) -> List[List[float]]:
    index = {sector: position for position, sector in enumerate(sectors)}
    matrix = [[0.0 for _ in sectors] for _ in sectors]
    for source, target, weight in _iter_pairs(sector_exposure):
        source = source if source in index else "other"
        target = target if target in index else "other"
        if source not in index or target not in index or source == target:
            continue
        matrix[index[source]][index[target]] += weight
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2025-06-30")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/historical-network_week_2025-07-01.json"),
    )
    parser.add_argument("--meta-path", type=Path, default=Path("data/meta_df.csv"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/v2_task2/figures/fig_spillover_matrix_example.pdf"),
    )
    parser.add_argument("--max-sectors", type=int, default=15)
    parser.add_argument("--min-edge-weight", type=float, default=0.0)
    args = parser.parse_args()

    metadata = _load_meta_category(args.meta_path)
    data = _load_payload_data(args.data_path)
    dates = sorted(data.keys())
    if not dates:
        raise RuntimeError("No snapshots found in data file")

    resolved_date = _resolve_date(args.date, dates)
    exposure = _compute_sector_exposure(
        data[resolved_date], metadata, min_edge_weight=args.min_edge_weight
    )
    sectors = _select_sectors(exposure, max_sectors=args.max_sectors)
    if "other" not in sectors:
        sectors.append("other")

    import matplotlib.pyplot as plt
    import numpy as np

    raw = np.array(_build_matrix(exposure, sectors), dtype=float)
    display = np.log10(1.0 + np.maximum(raw, 0.0))
    figure, axis = plt.subplots(
        figsize=(max(7.5, 0.45 * len(sectors)), max(6.5, 0.45 * len(sectors)))
    )
    image = axis.imshow(display, cmap="magma", aspect="auto")
    axis.set_title(f"Sector-to-sector cross-exposure matrix $S$ (t = {resolved_date})")
    axis.set_xlabel("Destination sector")
    axis.set_ylabel("Source sector")
    axis.set_xticks(range(len(sectors)))
    axis.set_yticks(range(len(sectors)))
    axis.set_xticklabels(sectors, rotation=45, ha="right")
    axis.set_yticklabels(sectors)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(r"$\log_{10}(1 + S_{ij})$")
    figure.subplots_adjust(right=0.76)

    pairs = sorted(
        (
            (float(raw[i, j]), source, target)
            for i, source in enumerate(sectors)
            for j, target in enumerate(sectors)
            if i != j and float(raw[i, j]) > 0
        ),
        reverse=True,
    )
    if pairs:
        lines = [f"{source}→{target}: {weight:,.0f}" for weight, source, target in pairs[:10]]
        axis.text(
            1.02,
            0.5,
            "Top cross-sector links\n" + "\n".join(lines),
            transform=axis.transAxes,
            va="center",
            fontsize=9,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out)
    figure.savefig(args.out.with_suffix(".png"), dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
