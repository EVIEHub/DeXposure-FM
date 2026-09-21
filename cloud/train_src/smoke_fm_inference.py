#!/usr/bin/env python3
"""Cloud-only DGL smoke test for the released DeXposure-FM checkpoints."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper"))
sys.path.insert(0, str(ROOT))

import dgl  # noqa: E402
import torch  # noqa: E402

from dexposure_agent.data_loader import SnapshotLoader  # noqa: E402
from dexposure_agent.fm_predictor import FMPredictor  # noqa: E402


CHECKPOINT_DIR = ROOT / "checkpoints" / "dexposure-fm-release"
DATA_DIR = ROOT / "data"
OUTPUT_PATH = ROOT / "logs" / "fm_inference_smoke.json"
SMOKE_DATE = "2021-01-04"
HORIZONS = (1, 4, 8)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def graph_signature(graph) -> tuple[tuple, tuple]:
    nodes = tuple(
        sorted((node_id, round(features.log_size, 8)) for node_id, features in graph.nodes.items())
    )
    edges = tuple(
        sorted((edge.source, edge.target, round(edge.weight, 8)) for edge in graph.edges)
    )
    return nodes, edges


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger = logging.getLogger("fm_inference_smoke")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable on the cloud inference worker")

    predictor = FMPredictor(checkpoint_dir=str(CHECKPOINT_DIR), device="cuda")
    if predictor.available_horizons != list(HORIZONS):
        raise RuntimeError(
            f"Unexpected release horizons: {predictor.available_horizons}; "
            f"expected {list(HORIZONS)}"
        )

    loader = SnapshotLoader(data_dir=DATA_DIR)
    observed = loader.load_single(SMOKE_DATE)
    if len(observed.nodes) < 2 or not observed.edges:
        raise RuntimeError("Smoke snapshot must contain at least two nodes and one edge")

    observed_signature = graph_signature(observed)
    results = []
    for horizon in HORIZONS:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        predicted = predictor.predict(observed, horizon=horizon)
        elapsed = time.perf_counter() - started
        if predicted.date != observed.date:
            raise RuntimeError(f"h={horizon} changed the snapshot date")
        if set(predicted.nodes) != set(observed.nodes):
            raise RuntimeError(f"h={horizon} changed the observed node universe")
        if any(not math.isfinite(node.log_size) for node in predicted.nodes.values()):
            raise RuntimeError(f"h={horizon} produced a non-finite node value")
        if any(
            not math.isfinite(edge.weight) or edge.weight < 0.0
            for edge in predicted.edges
        ):
            raise RuntimeError(f"h={horizon} produced an invalid edge weight")
        changed = graph_signature(predicted) != observed_signature
        if not changed:
            raise RuntimeError(f"h={horizon} returned an unchanged persistence graph")
        row = {
            "horizon": horizon,
            "elapsed_seconds": round(elapsed, 3),
            "output_nodes": len(predicted.nodes),
            "output_edges": len(predicted.edges),
            "prediction_changed": changed,
            "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated(),
        }
        results.append(row)
        logger.info("h=%d smoke result %s", horizon, row)

    checkpoint_hashes = {
        path.name: sha256_file(path)
        for path in sorted(CHECKPOINT_DIR.glob("*.pt"))
        if path.name in {
            "dexposure-fm-h1.pt",
            "dexposure-fm-h4.pt",
            "dexposure-fm-h8-h12.pt",
        }
    }
    output = {
        "status": "ok",
        "smoke_date": SMOKE_DATE,
        "torch_version": torch.__version__,
        "dgl_version": dgl.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "feature_schema_sha256": sha256_file(CHECKPOINT_DIR / "feature_schema.json"),
        "checkpoint_sha256": checkpoint_hashes,
        "input_nodes": len(observed.nodes),
        "input_edges": len(observed.edges),
        "results": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n")
    logger.info("wrote %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
