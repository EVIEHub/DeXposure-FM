#!/usr/bin/env python3
"""
DeXposure-FM Full Experiment Suite

This script implements the complete evaluation framework for Task I: Multi-step Forecasting.

Task I: Multi-step Forecasting
- Horizon h ∈ {1, 4, 8, 12} weeks ahead prediction
- Metrics: AUPRC, AUROC for link existence; MAE, RMSE for edge weight

Rolling Walk-forward Evaluation (Expanding Window)
- Expanding train window with fixed val/test windows
- No look-ahead bias in evaluation

Model Comparison
- GraphPFN-Frozen (encoder): Linear probe on pretrained embeddings
- DeXposure-FM: End-to-end fine-tuning
- ROLAND baseline: Temporal GNN (GCN + GRU)

Network Statistics
- Gini coefficient (degree centralization)
- HHI (Herfindahl-Hirschman Index)
- Network density, entropy, assortativity
- TVL concentration metrics

Usage:
    python run_full_experiment.py --mode all
    python run_full_experiment.py --mode frozen           # GraphPFN-Frozen
    python run_full_experiment.py --mode dexposure-fm      # DeXposure-FM (alias: deposure-fm)
    python run_full_experiment.py --mode roland           # ROLAND Baseline
    python run_full_experiment.py --mode stats            # Network Statistics
    python run_full_experiment.py --mode compare          # Compare all models

Note: Task II (Model-based Financial Stability Analysis) is implemented in run_task2_model_based.py

Output Directory Structure:
    output/
    └── YYYY-MM-DD_HHMMSS/                # Timestamped experiment folder
        ├── frozen/                        # GraphPFN-Frozen outputs
        ├── finetuned/                     # DeXposure-FM outputs
        └── roland/                        # ROLAND baseline outputs
"""

import argparse
import os as _os
import atexit
import copy
import json
import logging
import math

_os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
_os.environ.setdefault("DGL_DISABLE_GRAPHBOLT", "1")
import os
import random
import signal
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import dgl
except ModuleNotFoundError:
    dgl = None
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

# Add project paths
GRAPHPFN_ROOT = Path(__file__).parent
sys.path.insert(0, str(GRAPHPFN_ROOT))

# Early logger handle + log_info so import-time warnings don't crash before logger init.
_logger = None


def log_info(msg: str):
    """Log info message (falls back to print if logger not initialized)."""
    if _logger:
        _logger.info(msg)
    else:
        print(msg)
    sys.stdout.flush()


# Import GraphPFN components
try:
    from lib.graphpfn.model import GraphPFN, GraphPFNLayerWrapper
    from lib.limix.model.layer import MultiheadAttention as MHA

    GRAPHPFN_AVAILABLE = True
except ImportError:
    GRAPHPFN_AVAILABLE = False
    log_info("Warning: GraphPFN not available, will skip GraphPFN experiments")

# Import network statistics
from dexposure_fm.network_statistics import (
    compute_rolling_statistics,
)

# ============== DGL CUDA Availability Check ==============


def check_dgl_cuda_available() -> bool:
    """Check if DGL CUDA support is available."""
    if dgl is None:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        # Try to create a small graph and move to CUDA
        test_graph = dgl.graph(([0, 1], [1, 0]))
        test_graph = test_graph.to("cuda")
        return True
    except Exception as e:
        log_info(f"DGL CUDA not available: {e}")
        return False


DGL_CUDA_AVAILABLE = check_dgl_cuda_available()
if torch.cuda.is_available() and not DGL_CUDA_AVAILABLE:
    log_info(
        "Warning: PyTorch CUDA is available but DGL CUDA is not. Using CPU for graph operations."
    )

# ============== Configuration ==============


@dataclass
class ExperimentConfig:
    # Data paths
    data_path: str = "data/historical-network_week_2020-03-30.json"
    meta_path: str = "data/meta_df.csv"
    output_dir: str = "output/full_experiment"
    checkpoint_path: str = "checkpoints/graphpfn-v1.ckpt"

    # Experiment settings
    seed: int = 42
    neg_ratio: int = 5
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    test_ratio: float = 0.20

    # Model settings
    hidden_dim: int = 256
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    edge_batch_size: int = 20000

    # Multi-step forecasting
    forecast_horizons: List[int] = field(default_factory=lambda: [1, 4, 8, 12])

    # Gradient clipping
    gradient_clip_norm: float = 1.0

    # Loss weights (7 components as per advisor requirements)
    # Core prediction losses - balanced for ~56%:21%:23% contribution ratio
    exist_loss_weight: float = 2.0   # L_edge: BCE for edge existence
    weight_loss_weight: float = 0.5  # L_link: SmoothL1 for edge weight (residual learning)
    node_loss_weight: float = 20.0   # L_node: SmoothL1 for node TVL change (increased)
    # Auxiliary losses
    stats_loss_weight: float = 0.0  # L_stats: MSE for graph statistics constraint
    impute_loss_weight: float = 0.0  # L_impute: SmoothL1 for missing value imputation
    scen_loss_weight: float = 0.0  # L_scen: CE/Contrastive for scenario classification
    smooth_loss_weight: float = 0.0  # L_smooth: Temporal smoothness regularization

    # Imputation masking ratio (used for L_impute)
    impute_mask_ratio: float = 0.15

    # Early stopping
    early_stop_patience: int = 3
    early_stop_metric: str = "auprc"  # "auprc" or "auroc"
    val_eval_every: int = 1

    # Performance optimizations
    use_amp: bool = True  # Mixed precision training (2-3x speedup on modern GPUs)
    cache_frozen_embeddings: bool = True  # Cache encoder outputs when frozen (3-5x speedup)

    # Random seeds for multiple runs
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 2024])

    # Device - use CUDA only if both PyTorch and DGL support it
    device: str = (
        "cuda" if (torch.cuda.is_available() and DGL_CUDA_AVAILABLE) else "cpu"
    )


EPS = 1e-12


# ============== Logging and Result Management ==============


class ExperimentLogger:
    """
    Unified logger that writes to both console and file with real-time flushing.
    """

    def __init__(self, output_dir: Path, name: str = "experiment"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create unique log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"{name}_{timestamp}.log"

        # Setup logging
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()  # Clear existing handlers

        # File handler - writes everything
        file_handler = logging.FileHandler(self.log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)

        # Console handler - INFO and above
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.info(f"Log file: {self.log_file}")

    def debug(self, msg: str):
        self.logger.debug(msg)
        self._flush()

    def info(self, msg: str):
        self.logger.info(msg)
        self._flush()

    def warning(self, msg: str):
        self.logger.warning(msg)
        self._flush()

    def error(self, msg: str):
        self.logger.error(msg)
        self._flush()

    def critical(self, msg: str):
        self.logger.critical(msg)
        self._flush()

    def _flush(self):
        """Force flush all handlers."""
        for handler in self.logger.handlers:
            handler.flush()
        sys.stdout.flush()
        sys.stderr.flush()


class ResultManager:
    """
    Manages experiment results with auto-save on crash/interrupt.
    Saves intermediate results after each task/horizon completion.
    """

    def __init__(self, output_dir: Path, logger: Optional[ExperimentLogger] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

        self.results: Dict[str, Any] = {
            "experiment_start": datetime.now().isoformat(),
            "status": "running",
        }
        self.results_file = self.output_dir / "experiment_results.json"
        self.backup_file = self.output_dir / "experiment_results_backup.json"

        # Register signal handlers for graceful shutdown
        self._register_handlers()

        # Initial save
        self._save()

        self._log(f"ResultManager initialized, saving to: {self.results_file}")

    def _log(self, msg: str, level: str = "info"):
        if self.logger:
            getattr(self.logger, level)(msg)
        else:
            print(msg)

    def _register_handlers(self):
        """Register handlers to save on exit/interrupt."""
        atexit.register(self._on_exit)
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    def _on_exit(self):
        """Called on normal exit."""
        if self.results.get("status") == "running":
            self.results["status"] = "interrupted_or_crashed"
            self.results["end_time"] = datetime.now().isoformat()
        self._save()
        self._log("Results saved on exit.")

    def _on_signal(self, signum, frame):
        """Called on SIGINT/SIGTERM."""
        self._log(f"Received signal {signum}, saving results...")
        self.results["status"] = f"interrupted_signal_{signum}"
        self.results["end_time"] = datetime.now().isoformat()
        self._save()
        self._log("Results saved. Exiting.")
        sys.exit(1)

    def _save(self):
        """Save results to JSON with backup."""
        try:
            # Backup existing file
            if self.results_file.exists():
                import shutil

                shutil.copy(self.results_file, self.backup_file)

            # Save with atomic write pattern
            temp_file = self.output_dir / "experiment_results_temp.json"
            with open(temp_file, "w") as f:
                json.dump(self.results, f, indent=2, default=str)
            temp_file.rename(self.results_file)

        except Exception as e:
            self._log(f"ERROR saving results: {e}", "error")

    def update(self, key: str, value: Any, save: bool = True):
        """Update a result key and optionally save."""
        self.results[key] = value
        if save:
            self._save()
            self._log(f"Saved result: {key}")

    def add_task_result(self, task_name: str, result: Dict[str, Any]):
        """Add result for a completed task and save immediately."""
        self.results[task_name] = result
        self.results[f"{task_name}_completed_at"] = datetime.now().isoformat()
        self._save()
        self._log(f"Task {task_name} results saved.")

    def mark_complete(self):
        """Mark experiment as complete."""
        self.results["status"] = "complete"
        self.results["end_time"] = datetime.now().isoformat()
        self._save()
        self._log("Experiment marked as complete.")

    def get_results(self) -> Dict[str, Any]:
        """Get all results."""
        return self.results.copy()


# Global instances (initialized in main)
_result_manager: Optional[ResultManager] = None
_run_lock_path: Optional[Path] = None


def init_run_context(output_dir: Path) -> None:
    """Initialize logger and result manager for a specific output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "experiment_results.json"
    if results_file.exists():
        backup_name = f"experiment_results_{datetime.now().strftime('%H%M%S')}.json.bak"
        results_file.rename(output_dir / backup_name)
        print(f"Backed up existing results to: {backup_name}")

    global _logger, _result_manager
    _logger = ExperimentLogger(output_dir, name="dexposure_experiment")
    _result_manager = ResultManager(output_dir, logger=_logger)


def get_logger() -> Optional[ExperimentLogger]:
    """Get global logger instance."""
    return _logger


def get_result_manager() -> Optional[ResultManager]:
    """Get global result manager instance."""
    return _result_manager


def log_debug(msg: str):
    """Log debug message."""
    if _logger:
        _logger.debug(msg)


def log_error(msg: str):
    """Log error message."""
    if _logger:
        _logger.error(msg)
    else:
        print(f"ERROR: {msg}", file=sys.stderr)
    sys.stderr.flush()


def acquire_run_lock(lock_path: Path):
    """
    Prevent multiple concurrent runs of this script.
    If a live PID is found in the lock file, exit early.
    """
    if lock_path.exists():
        try:
            payload = json.loads(lock_path.read_text())
            pid = int(payload.get("pid", -1))
        except Exception:
            pid = -1

        if pid > 0:
            try:
                os.kill(pid, 0)
                log_error(
                    f"Another run_full_experiment.py is already running (pid={pid}). "
                    "Exiting to avoid OOM."
                )
                sys.exit(2)
            except OSError:
                # Stale lock; proceed to overwrite
                pass

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_payload = {
        "pid": os.getpid(),
        "started_at": datetime.now().isoformat(),
    }
    lock_path.write_text(json.dumps(lock_payload))


def release_run_lock():
    if _run_lock_path and _run_lock_path.exists():
        try:
            _run_lock_path.unlink()
        except Exception:
            pass


def save_result(key: str, value: Any):
    """Save a result to the result manager."""
    if _result_manager:
        _result_manager.update(key, value)


def save_task_result(task_name: str, result: Dict[str, Any]):
    """Save a task result."""
    if _result_manager:
        _result_manager.add_task_result(task_name, result)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============== Data Loading ==============


def load_network_data(path: str) -> Dict:
    """Load network JSON data."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    size_mb = path.stat().st_size / (1024 * 1024)
    log_info(f"Loading {path} ({size_mb:.1f} MB)")

    with path.open("rb") as f:
        try:
            import ijson

            data = {k: v for k, v in ijson.kvitems(f, "data")}
            return data
        except Exception:
            f.seek(0)
            payload = json.load(f)
            return payload["data"]


def load_metadata(path: str) -> Tuple[Dict, List[str], Dict]:
    """Load protocol metadata."""
    meta_df = pd.read_csv(path)
    meta_df["id"] = meta_df["id"].astype(str)

    category_list = sorted(meta_df["category"].dropna().unique().tolist())
    if "Unknown" not in category_list:
        category_list.append("Unknown")

    category_to_idx = {c: i for i, c in enumerate(category_list)}
    meta_category = meta_df.set_index("id")["category"].to_dict()

    return meta_category, category_list, category_to_idx


def node_features(
    node: Dict, meta_category: Dict, category_to_idx: Dict, category_list: List
) -> Tuple[np.ndarray, float, str]:
    """Extract node features."""
    node_id = str(node.get("id"))
    size = float(node.get("size", 0.0))
    comp = node.get("composition", {}) or {}

    log_size = math.log1p(max(size, 0.0))
    num_tokens = float(len(comp))

    if size > 0 and comp:
        values = np.array(list(comp.values()), dtype=np.float64)
        values = np.maximum(values, 0.0)
        total = values.sum() + EPS
        shares = values / total
        max_share = float(shares.max())
        entropy = float(-(shares * np.log(shares + EPS)).sum())
    else:
        max_share = 0.0
        entropy = 0.0

    category = meta_category.get(node_id, "Unknown")
    idx = category_to_idx.get(category, category_to_idx["Unknown"])
    cat_vec = np.zeros(len(category_list), dtype=np.float32)
    cat_vec[idx] = 1.0

    feats = np.array([log_size, num_tokens, max_share, entropy], dtype=np.float32)
    feats = np.concatenate([feats, cat_vec], axis=0)

    return feats, size, category


def build_snapshot(
    date: str,
    snapshot: Dict,
    meta_category: Dict,
    category_to_idx: Dict,
    category_list: List,
) -> Dict:
    """Build a single snapshot."""
    nodes = snapshot.get("nodes", [])
    links = snapshot.get("links", [])

    node_ids, features, sizes, categories = [], [], [], []

    for node in nodes:
        node_id = node.get("id")
        if node_id is None:
            continue
        feats, size, category = node_features(
            node, meta_category, category_to_idx, category_list
        )
        node_ids.append(str(node_id))
        features.append(feats)
        sizes.append(size)
        categories.append(category)

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    src, dst, weights = [], [], []
    for link in links:
        source = link.get("source")
        target = link.get("target")
        if source is None or target is None:
            continue
        source, target = str(source), str(target)
        if source not in id_to_idx or target not in id_to_idx:
            continue
        src.append(id_to_idx[source])
        dst.append(id_to_idx[target])
        weights.append(float(link.get("size", 0.0)))

    return {
        "date": date,
        "node_ids": node_ids,
        "features": np.array(features, dtype=np.float32)
        if features
        else np.zeros((0, 4 + len(category_list)), dtype=np.float32),
        "sizes": np.array(sizes, dtype=np.float32),
        "categories": categories,
        "edge_src": np.array(src, dtype=np.int64),
        "edge_dst": np.array(dst, dtype=np.int64),
        "edge_weight": np.array(weights, dtype=np.float32),
    }


def enrich_snapshots_with_historical_features(snapshots: List[Dict]) -> List[Dict]:
    """
    Add historical and structural features to snapshots.

    New features added (3 total):
    - prev_tvl_change: log(size_t / size_{t-1}), the TVL change from previous week
    - in_degree: number of incoming edges (normalized by max)
    - out_degree: number of outgoing edges (normalized by max)

    These features help the model capture temporal dynamics and structural importance.
    """
    if not snapshots:
        return snapshots

    # Build node_id -> size mapping for each snapshot
    size_maps = []
    for snap in snapshots:
        size_map = {nid: size for nid, size in zip(snap["node_ids"], snap["sizes"])}
        size_maps.append(size_map)

    enriched = []
    for t, snap in enumerate(snapshots):
        n_nodes = len(snap["node_ids"])

        # 1. Previous TVL change (log-ratio)
        prev_tvl_change = np.zeros(n_nodes, dtype=np.float32)
        if t > 0:
            prev_size_map = size_maps[t - 1]
            for i, nid in enumerate(snap["node_ids"]):
                curr_size = snap["sizes"][i]
                prev_size = prev_size_map.get(nid, 0.0)
                if prev_size > 0 and curr_size > 0:
                    # log(curr/prev) = log(curr) - log(prev)
                    prev_tvl_change[i] = math.log1p(curr_size) - math.log1p(prev_size)
                # else: 0 (new node or no previous data)

        # 2. In-degree and out-degree (normalized)
        in_degree = np.zeros(n_nodes, dtype=np.float32)
        out_degree = np.zeros(n_nodes, dtype=np.float32)

        if len(snap["edge_src"]) > 0:
            for src_idx in snap["edge_src"]:
                out_degree[src_idx] += 1
            for dst_idx in snap["edge_dst"]:
                in_degree[dst_idx] += 1

            # Normalize by max degree (avoid division by zero)
            max_in = in_degree.max() if in_degree.max() > 0 else 1.0
            max_out = out_degree.max() if out_degree.max() > 0 else 1.0
            in_degree = in_degree / max_in
            out_degree = out_degree / max_out

        # Concatenate new features to existing features
        new_features = np.stack([prev_tvl_change, in_degree, out_degree], axis=1)
        enriched_features = np.concatenate([snap["features"], new_features], axis=1)

        # Create enriched snapshot (copy to avoid mutating original)
        enriched_snap = snap.copy()
        enriched_snap["features"] = enriched_features
        enriched.append(enriched_snap)

    return enriched


# ============== Strict Temporal Split ==============


def expanding_window_split(
    all_dates: List[str],
    holdout_start: str = "2025-01-01",
    min_train_weeks: int = 104,  # 最少2年训练数据
    val_weeks: int = 24,  # 验证集: 24周 (约6个月) - 确保h=12也有足够验证集
    test_weeks: int = 8,  # 每fold测试: 8周 (约2个月)
    step_weeks: int = 8,  # 每次向前滚动8周
) -> Dict[str, Any]:
    """
    Expanding Window Walk-Forward Validation (金融惯例).

    特点:
    1. 训练窗口随时间扩展 (Expanding Window)
    2. 验证集固定大小，紧跟训练集
    3. 测试集固定大小，紧跟验证集
    4. Hold-out 2025 数据用于最终评估

    Returns:
        {
            "folds": [
                {"train": [...], "val": [...], "test": [...]},
                ...
            ],
            "holdout": {
                "train": [...],  # 所有 < holdout_start 的数据
                "val": [...],    # 最后 val_weeks
                "test": [...]    # >= holdout_start 的数据 (NEVER seen)
            }
        }

    Timeline:
    ────────────────────────────────────────────────────────────────
    Fold 1: [==== Train (104w) ====][Val 24w][Test 8w]
    Fold 2: [====== Train (112w) ======][Val 24w][Test 8w]
    Fold 3: [======== Train (120w) ========][Val 24w][Test 8w]
    ...
    ────────────────────────────────────────────────────────────────
    Holdout: [============ All pre-2025 Train ============][Val 24w][ 2025 Test ]

    Note: val_weeks=24 确保即使预测horizon h=12时也有充足验证样本
          (build_week_pairs需要 len(val_snaps) > horizon 才能构建样本)
    """
    sorted_dates = sorted(all_dates)

    # 分离 holdout 数据
    pre_holdout = [d for d in sorted_dates if d < holdout_start]
    holdout_dates = [d for d in sorted_dates if d >= holdout_start]

    # 生成 rolling folds
    folds = []
    fold_idx = 0

    while True:
        # 训练集终点 (expanding)
        train_end_idx = min_train_weeks + fold_idx * step_weeks

        # 验证集范围
        val_start_idx = train_end_idx
        val_end_idx = val_start_idx + val_weeks

        # 测试集范围
        test_start_idx = val_end_idx
        test_end_idx = test_start_idx + test_weeks

        # 检查是否超出 pre_holdout 范围
        if test_end_idx > len(pre_holdout):
            break

        fold = {
            "fold_id": fold_idx + 1,
            "train": pre_holdout[:train_end_idx],
            "val": pre_holdout[val_start_idx:val_end_idx],
            "test": pre_holdout[test_start_idx:test_end_idx],
        }
        folds.append(fold)
        fold_idx += 1

    # Holdout split: 用所有 pre-2025 训练，2025 测试
    if len(pre_holdout) > val_weeks:
        holdout_train = pre_holdout[:-val_weeks]
        holdout_val = pre_holdout[-val_weeks:]
    else:
        holdout_train = pre_holdout
        holdout_val = []

    return {
        "folds": folds,
        "n_folds": len(folds),
        "holdout": {
            "train": holdout_train,
            "val": holdout_val,
            "test": holdout_dates,
        },
        "config": {
            "min_train_weeks": min_train_weeks,
            "val_weeks": val_weeks,
            "test_weeks": test_weeks,
            "step_weeks": step_weeks,
            "holdout_start": holdout_start,
        },
    }


def get_single_split(
    all_dates: List[str],
    holdout_start: str = "2025-01-01",
    val_weeks: int = 24,
) -> Dict[str, List[str]]:
    """
    简化版: 仅返回最终 holdout 划分 (用于快速实验).

    Train: 所有 < holdout_start - val_weeks
    Val:   holdout_start 前 val_weeks
    Test:  所有 >= holdout_start (2025 hold-out)
    """
    sorted_dates = sorted(all_dates)
    pre_holdout = [d for d in sorted_dates if d < holdout_start]
    holdout_dates = [d for d in sorted_dates if d >= holdout_start]

    if len(pre_holdout) > val_weeks:
        train_dates = pre_holdout[:-val_weeks]
        val_dates = pre_holdout[-val_weeks:]
    else:
        train_dates = pre_holdout
        val_dates = []

    return {
        "train": train_dates,
        "val": val_dates,
        "test": holdout_dates,
    }


# ============== Data Quality Statistics ==============


def compute_data_quality(
    snapshots: List[Dict], raw_network_data: Dict
) -> Dict[str, Any]:
    """Compute data quality statistics for each snapshot."""
    weekly_stats = []

    for i, snap in enumerate(snapshots):
        date = snap["date"]
        raw_snap = raw_network_data.get(date, {})
        raw_nodes = raw_snap.get("nodes", [])
        raw_links = raw_snap.get("links", [])

        # Count dropped edges
        n_target_null = sum(1 for link in raw_links if link.get("target") is None)
        n_endpoint_missing = len(raw_links) - n_target_null - len(snap["edge_src"])

        # Compute overlap with next snapshot
        overlap_ratio = 0.0
        if i < len(snapshots) - 1:
            next_ids = set(snapshots[i + 1]["node_ids"])
            current_ids = set(snap["node_ids"])
            if len(current_ids) > 0:
                overlap_ratio = len(current_ids & next_ids) / len(current_ids)

        # Count unknown categories
        n_unknown = sum(1 for cat in snap["categories"] if cat == "Unknown")

        weekly_stats.append(
            {
                "date": date,
                "N_nodes": len(snap["node_ids"]),
                "N_edges": len(snap["edge_src"]),
                "pct_target_null_dropped": n_target_null / max(len(raw_links), 1),
                "pct_endpoint_missing_dropped": n_endpoint_missing
                / max(len(raw_links), 1),
                "overlap_ratio_next_week": overlap_ratio,
                "pct_category_unknown": n_unknown / max(len(snap["node_ids"]), 1),
            }
        )

    # Summary statistics
    summary = {
        "mean_nodes_per_week": float(np.mean([s["N_nodes"] for s in weekly_stats])),
        "mean_edges_per_week": float(np.mean([s["N_edges"] for s in weekly_stats])),
        "std_nodes_per_week": float(np.std([s["N_nodes"] for s in weekly_stats])),
        "std_edges_per_week": float(np.std([s["N_edges"] for s in weekly_stats])),
        "mean_pct_target_null_dropped": float(
            np.mean([s["pct_target_null_dropped"] for s in weekly_stats])
        ),
        "mean_pct_endpoint_missing_dropped": float(
            np.mean([s["pct_endpoint_missing_dropped"] for s in weekly_stats])
        ),
        "mean_overlap_ratio": float(
            np.mean([s["overlap_ratio_next_week"] for s in weekly_stats[:-1]])
        )
        if len(weekly_stats) > 1
        else 0.0,
        "mean_pct_category_unknown": float(
            np.mean([s["pct_category_unknown"] for s in weekly_stats])
        ),
    }

    summary["pct_target_null_dropped"] = summary["mean_pct_target_null_dropped"]
    summary["pct_endpoint_missing_dropped"] = summary[
        "mean_pct_endpoint_missing_dropped"
    ]

    return {"summary": summary, "weekly": weekly_stats}


# ============== Predictions Output ==============


def save_predictions_csv(
    preds: List[Dict],
    output_dir: Path,
    model_name: str,
    horizon: int,
    fold_id: Optional[int] = None,
    append: bool = False,
) -> Tuple[Path, Path]:
    """Save predictions to CSV files."""
    edges_rows = []
    nodes_rows = []

    for item in preds:
        time_t = item["time_t"]
        time_t1 = item["time_t1"]
        node_ids = item["node_ids"]
        categories = item["categories"]
        sizes_t = item["sizes_t"]
        exist_prob = 1 / (1 + np.exp(-item["exist_logits"]))

        # Edge predictions
        for i in range(len(item["y_exist"])):
            u_idx = int(item["pair_src"][i])
            v_idx = int(item["pair_dst"][i])
            edges_rows.append(
                {
                    "time_t": time_t,
                    "time_t1": time_t1,
                    "horizon": horizon,
                    "fold_id": fold_id,
                    "u_id": node_ids[u_idx],
                    "v_id": node_ids[v_idx],
                    "y_exist_true": int(item["y_exist"][i]),
                    "y_exist_pred": float(exist_prob[i]),
                    "y_w_true": float(item["y_weight"][i]),
                    "y_w_pred": float(item["weight_pred"][i]),
                    "is_positive": bool(item["weight_mask"][i] > 0.5),
                }
            )

        # Node predictions
        node_mask = item["node_mask"]
        for i in range(len(node_mask)):
            if node_mask[i]:
                nodes_rows.append(
                    {
                        "time_t": time_t,
                        "time_t1": time_t1,
                        "horizon": horizon,
                        "fold_id": fold_id,
                        "node_id": node_ids[i],
                        "y_node_true": float(item["y_node"][i]),
                        "y_node_pred": float(item["node_pred"][i]),
                        "size_t": float(sizes_t[i]),
                        "category": categories[i],
                    }
                )

    # Save to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    edges_path = output_dir / "predictions_edges_test.csv"
    nodes_path = output_dir / "predictions_nodes_test.csv"

    if edges_rows:
        edges_df = pd.DataFrame(edges_rows)
        edges_df.to_csv(
            edges_path,
            index=False,
            mode="a" if append else "w",
            header=not (append and edges_path.exists()),
        )
    if nodes_rows:
        nodes_df = pd.DataFrame(nodes_rows)
        nodes_df.to_csv(
            nodes_path,
            index=False,
            mode="a" if append else "w",
            header=not (append and nodes_path.exists()),
        )

    return edges_path, nodes_path


def save_metrics_json(result: Dict[str, Any], output_dir: Path) -> Path:
    """Save metrics.json in the experiment plan format."""
    payload = {"model": result.get("model", "unknown")}
    results = result.get("results", {})
    for key in sorted(results.keys()):
        payload[key] = results[key]

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(payload, f, indent=2)

    return metrics_path


# ============== Week Pair Construction ==============


@dataclass
class WeekPair:
    time_t: str
    time_t1: str
    prev_time_t: Optional[str]
    node_ids: List[str]
    categories: List[str]
    sizes_t: np.ndarray
    features_t: np.ndarray
    edge_src_t: np.ndarray
    edge_dst_t: np.ndarray
    edge_weight_t: np.ndarray  # Edge weights at time t (for persistence baseline)
    pair_src: np.ndarray
    pair_dst: np.ndarray
    y_exist: np.ndarray
    y_weight: np.ndarray
    weight_mask: np.ndarray
    y_node: np.ndarray
    node_mask: np.ndarray
    pos_edge_count: int
    neg_edge_count: int
    scenario_label: Optional[int]
    # For residual learning: baseline weight at time t for each pair
    baseline_weight: np.ndarray = None  # Shape: (num_pairs,), log-scaled weight at t


def sample_negatives(
    num_nodes: int, pos_set: set, num_neg: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample negative edges."""
    neg_src, neg_dst = [], []
    seen = set(pos_set)
    attempts = 0
    max_attempts = num_neg * 10

    while len(neg_src) < num_neg and attempts < max_attempts:
        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        if u != v and (u, v) not in seen:
            seen.add((u, v))
            neg_src.append(u)
            neg_dst.append(v)
        attempts += 1

    return np.array(neg_src, dtype=np.int64), np.array(neg_dst, dtype=np.int64)


# Minimal shock event definitions for scenario loss during training
@dataclass
class ShockEvent:
    """Minimal shock event for scenario labeling."""
    name: str
    event_date: str


SHOCK_EVENTS = [
    ShockEvent("Terra/Luna", "2022-05-09"),
    ShockEvent("FTX", "2022-11-07"),
]


def find_nearest_date(target: str, dates: List[str]) -> Optional[str]:
    """Find the nearest date in the list to the target date."""
    if not dates:
        return None
    sorted_dates = sorted(dates)
    for d in sorted_dates:
        if d >= target:
            return d
    return sorted_dates[-1]


def build_scenario_label_map(
    dates: List[str], pre_weeks: int = 4, post_weeks: int = 4
) -> Dict[str, int]:
    """
    Build scenario labels for each date.

    Labels:
        0 = normal
        1 = pre-shock
        2 = shock
        3 = post-shock
    """
    label_map = {d: 0 for d in dates}
    if not dates:
        return label_map

    for event in SHOCK_EVENTS:
        event_date = find_nearest_date(event.event_date, dates)
        if event_date is None:
            continue
        event_idx = dates.index(event_date)

        # Shock week
        label_map[dates[event_idx]] = 2

        # Pre-shock window
        for i in range(max(0, event_idx - pre_weeks), event_idx):
            if label_map[dates[i]] < 2:
                label_map[dates[i]] = 1

        # Post-shock window
        for i in range(event_idx + 1, min(len(dates), event_idx + post_weeks + 1)):
            if label_map[dates[i]] < 2:
                label_map[dates[i]] = 3

    return label_map


def build_week_pairs(
    snapshots: List[Dict], neg_ratio: int, seed: int, horizon: int = 1
) -> List[WeekPair]:
    """
    Build week pairs for training/evaluation.

    Args:
        snapshots: List of snapshot dictionaries
        neg_ratio: Negative sampling ratio
        seed: Random seed
        horizon: Prediction horizon (default 1 = next week)
    """
    rng = np.random.default_rng(seed)
    pairs = []
    dates = [snap["date"] for snap in snapshots]
    scenario_labels = build_scenario_label_map(dates)

    for t in range(len(snapshots) - horizon):
        snap_t = snapshots[t]
        snap_t1 = snapshots[t + horizon]
        prev_time_t = snapshots[t - 1]["date"] if t > 0 else None

        id_to_idx = {nid: i for i, nid in enumerate(snap_t["node_ids"])}
        size_t1_map = {
            nid: size for nid, size in zip(snap_t1["node_ids"], snap_t1["sizes"])
        }

        # Build positive edges (edges that exist at t+h)
        pos_src, pos_dst, pos_w, pos_set = [], [], [], set()

        for src_idx, dst_idx, w in zip(
            snap_t1["edge_src"], snap_t1["edge_dst"], snap_t1["edge_weight"]
        ):
            src_id = snap_t1["node_ids"][int(src_idx)]
            dst_id = snap_t1["node_ids"][int(dst_idx)]
            if src_id not in id_to_idx or dst_id not in id_to_idx:
                continue
            u, v = id_to_idx[src_id], id_to_idx[dst_id]
            pos_src.append(u)
            pos_dst.append(v)
            pos_w.append(math.log1p(max(w, 0.0)))
            pos_set.add((u, v))

        pos_src = np.array(pos_src, dtype=np.int64)
        pos_dst = np.array(pos_dst, dtype=np.int64)
        pos_w = np.array(pos_w, dtype=np.float32)

        num_pos = len(pos_src)
        if num_pos == 0:
            continue

        # Sample negatives
        num_neg = num_pos * neg_ratio
        neg_src, neg_dst = sample_negatives(
            len(snap_t["node_ids"]), pos_set, num_neg, rng
        )

        # Combine and shuffle
        pair_src = np.concatenate([pos_src, neg_src])
        pair_dst = np.concatenate([pos_dst, neg_dst])
        y_exist = np.concatenate(
            [
                np.ones(num_pos, dtype=np.float32),
                np.zeros(len(neg_src), dtype=np.float32),
            ]
        )
        y_weight = np.concatenate([pos_w, np.zeros(len(neg_src), dtype=np.float32)])
        weight_mask = np.concatenate(
            [
                np.ones(num_pos, dtype=np.float32),
                np.zeros(len(neg_src), dtype=np.float32),
            ]
        )

        order = rng.permutation(len(pair_src))
        pair_src, pair_dst = pair_src[order], pair_dst[order]
        y_exist, y_weight, weight_mask = (
            y_exist[order],
            y_weight[order],
            weight_mask[order],
        )

        # Node-level labels
        y_node = np.zeros(len(snap_t["node_ids"]), dtype=np.float32)
        node_mask = np.zeros(len(snap_t["node_ids"]), dtype=bool)
        for i, nid in enumerate(snap_t["node_ids"]):
            if nid in size_t1_map:
                y_node[i] = math.log1p(max(size_t1_map[nid], 0.0)) - math.log1p(
                    max(snap_t["sizes"][i], 0.0)
                )
                node_mask[i] = True

        # Get edge weights at time t (log-scaled for persistence baseline)
        edge_weight_t = np.array(
            [math.log1p(max(w, 0.0)) for w in snap_t["edge_weight"]],
            dtype=np.float32
        )

        # Build lookup dict for edge weights at time t: (src_idx, dst_idx) -> log_weight
        edge_t_lookup = {}
        for src_idx, dst_idx, w in zip(
            snap_t["edge_src"], snap_t["edge_dst"], snap_t["edge_weight"]
        ):
            edge_t_lookup[(int(src_idx), int(dst_idx))] = math.log1p(max(w, 0.0))

        # Compute baseline_weight for each pair (weight at time t, 0 if edge didn't exist)
        baseline_weight = np.zeros(len(pair_src), dtype=np.float32)
        for i, (s, d) in enumerate(zip(pair_src, pair_dst)):
            baseline_weight[i] = edge_t_lookup.get((int(s), int(d)), 0.0)

        pairs.append(
            WeekPair(
                time_t=snap_t["date"],
                time_t1=snap_t1["date"],
                prev_time_t=prev_time_t,
                node_ids=snap_t["node_ids"],
                categories=snap_t["categories"],
                sizes_t=snap_t["sizes"],
                features_t=snap_t["features"],
                edge_src_t=snap_t["edge_src"],
                edge_dst_t=snap_t["edge_dst"],
                edge_weight_t=edge_weight_t,
                pair_src=pair_src,
                pair_dst=pair_dst,
                y_exist=y_exist,
                y_weight=y_weight,
                weight_mask=weight_mask,
                y_node=y_node,
                node_mask=node_mask,
                pos_edge_count=num_pos,
                neg_edge_count=len(neg_src),
                scenario_label=scenario_labels.get(snap_t["date"], 0),
                baseline_weight=baseline_weight,
            )
        )

    return pairs


# ============== Model Definitions ==============


class LinkScorer(nn.Module):
    """Link prediction head with exist and weight outputs."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        in_dim = 4 * embed_dim
        self.exist_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.weight_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor):
        h_u, h_v = h[src], h[dst]
        z = torch.cat([h_u, h_v, h_u * h_v, (h_u - h_v).abs()], dim=-1)
        return self.exist_head(z).squeeze(-1), self.weight_head(z).squeeze(-1)


class NodeHead(nn.Module):
    """Node-level prediction head with neighbor aggregation.

    Aggregates inflow/outflow neighbor embeddings weighted by edge weights
    to better predict TVL changes.
    """

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        # Input: [self_embed, inflow_embed, outflow_embed] = 3 * embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_src: torch.Tensor = None,
        edge_dst: torch.Tensor = None,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            h: Node embeddings [N, embed_dim]
            edge_src: Source node indices [E]
            edge_dst: Destination node indices [E]
            edge_weight: Edge weights (log-scaled) [E]
        """
        num_nodes = h.size(0)
        device = h.device

        if edge_src is None or edge_dst is None:
            # Fallback: no neighbor info, use zeros
            inflow = torch.zeros_like(h)
            outflow = torch.zeros_like(h)
        else:
            # Normalize edge weights for aggregation
            if edge_weight is not None:
                w = torch.softmax(edge_weight, dim=0)
            else:
                w = torch.ones(edge_src.size(0), device=device) / edge_src.size(0)

            # Inflow: edges coming INTO node i (edge_dst == i)
            # Weighted sum of source node embeddings
            inflow = torch.zeros_like(h)
            inflow.index_add_(0, edge_dst, h[edge_src] * w.unsqueeze(-1))

            # Outflow: edges going OUT of node i (edge_src == i)
            # Weighted sum of destination node embeddings
            outflow = torch.zeros_like(h)
            outflow.index_add_(0, edge_src, h[edge_dst] * w.unsqueeze(-1))

        # Concatenate self + inflow + outflow
        node_features = torch.cat([h, inflow, outflow], dim=-1)
        return self.net(node_features).squeeze(-1)


# ============== GraphPFN Encoder ==============

if GRAPHPFN_AVAILABLE:

    def graphpfn_encode(
        model: GraphPFN,
        graph: dgl.DGLGraph,
        features: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode graph using GraphPFN."""
        num_nodes = graph.num_nodes()

        train_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        train_mask[-1] = False  # At least 1 test node required

        n_train = int(train_mask.sum().item())
        y_train = torch.zeros(n_train, device=device)

        features = features.to(device)
        graph = graph.to(device)

        tfm_features = torch.cat([features[train_mask], features[~train_mask]], dim=0)

        for module in model.modules():
            if isinstance(module, GraphPFNLayerWrapper):
                module.train_mask = train_mask
                module.graph = graph
            if isinstance(module, MHA):
                module.batched = False

        # PyTorch 2.2 compatibility: use torch.backends.cuda.sdp_kernel context manager
        # instead of torch.nn.attention.sdpa_kernel (PyTorch 2.4+)
        if hasattr(torch.nn, "attention") and hasattr(
            torch.nn.attention, "sdpa_kernel"
        ):
            # PyTorch 2.4+ API
            if device.type == "cpu":
                sdpa_backends = [torch.nn.attention.SDPBackend.MATH]
            else:
                sdpa_backends = [
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                ]
            with torch.nn.attention.sdpa_kernel(sdpa_backends):
                out = model.tfm.forward(
                    x=tfm_features.unsqueeze(0),
                    y=y_train.unsqueeze(0),
                    eval_pos=n_train,
                    task_type="reg",
                    checkpointing=True,
                )
        else:
            # PyTorch 2.2/2.3 compatibility - use torch.backends.cuda.sdp_kernel
            if device.type == "cpu":
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=True, enable_mem_efficient=False
                ):
                    out = model.tfm.forward(
                        x=tfm_features.unsqueeze(0),
                        y=y_train.unsqueeze(0),
                        eval_pos=n_train,
                        task_type="reg",
                        checkpointing=True,
                    )
            else:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True, enable_math=False, enable_mem_efficient=True
                ):
                    out = model.tfm.forward(
                        x=tfm_features.unsqueeze(0),
                        y=y_train.unsqueeze(0),
                        eval_pos=n_train,
                        task_type="reg",
                        checkpointing=True,
                    )

        inv_order = torch.argsort((~train_mask).float(), stable=True)
        order = torch.argsort(inv_order, stable=True)

        # Use feature embeddings if available (better for link prediction)
        # encoder_out_full shape: [batch, seq_len, num_features+1, embed_dim]
        # We want feature columns (not label column), then mean pool over features
        if "encoder_out_full" in out:
            # Extract feature embeddings (exclude last column which is label)
            feature_embeds = out["encoder_out_full"][:, :, :-1, :]  # [1, n, f, d]
            # Mean pooling over features to get node embeddings
            node_embeds = feature_embeds.mean(dim=2).squeeze(0)[order]  # [n, d]
            return node_embeds
        else:
            # Fallback to original label column embeddings
            return out["encoder_embed"].squeeze(0)[order]

    class GraphPFNLinkPredictor(nn.Module):
        """GraphPFN-based link predictor."""

        def __init__(self, encoder: GraphPFN, embed_dim: int, hidden_dim: int):
            super().__init__()
            self.encoder = encoder
            self.link_scorer = LinkScorer(embed_dim, hidden_dim)
            self.node_head = NodeHead(embed_dim, hidden_dim)
            self.scenario_head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 4),  # 4 scenario classes
            )

        def encode(
            self, graph: dgl.DGLGraph, features: torch.Tensor, device: torch.device
        ) -> torch.Tensor:
            return graphpfn_encode(self.encoder, graph, features, device)

    def load_graphpfn_encoder(checkpoint_path: str, device: torch.device) -> GraphPFN:
        """Load pretrained GraphPFN encoder."""
        model = GraphPFN(edge_head=False)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Allow partial loads because graphpfn-v1.ckpt only contains adapter weights.
        model.load_state_dict(checkpoint["model"], strict=False)
        model.to(device)
        return model


# ============== ROLAND Baseline ==============


class ROLANDBaseline(nn.Module):
    """
    ROLAND baseline model (adapted from DeXposure).
    Temporal GNN with GCN + GRU for link prediction + edge weight + node prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.preprocess = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

        self.gru1 = nn.GRUCell(hidden_dim, hidden_dim)
        self.gru2 = nn.GRUCell(out_dim, out_dim)

        self.link_decoder = nn.Linear(out_dim, 1)
        self.weight_head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),
        )

        # Node prediction head (for TVL change prediction)
        self.node_head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),
        )

        self.dropout = dropout

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
        h1: Optional[torch.Tensor] = None,
        h2: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass.

        Returns:
            pred: Link predictions
            h1_new: Updated hidden state 1
            h2_new: Updated hidden state 2
            node_embed: Node embeddings for node prediction
        """
        # Preprocess
        h = self.preprocess(x)

        # GCN layer 1
        h = self.conv1(h, edge_index)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # GRU update 1
        if h1 is not None:
            h = self.gru1(h, h1)
        h1_new = h.clone()

        # GCN layer 2
        h = self.conv2(h, edge_index)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # GRU update 2
        if h2 is not None:
            h = self.gru2(h, h2)
        h2_new = h.clone()

        # Link prediction (Hadamard product)
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        h_hadamard = h_src * h_dst
        pred = self.link_decoder(h_hadamard).squeeze(-1)

        weight_pred = self.weight_head(h_hadamard).squeeze(-1)

        return pred, weight_pred, h1_new, h2_new, h  # Return node embeddings too


# ============== Auxiliary Loss Functions (7-component loss design) ==============


def compute_stats_loss(
    graph_pred_stats: Dict[str, torch.Tensor],
    graph_true_stats: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    L_stats: Graph statistics constraint loss (MSE).

    Enforces that predicted graph maintains structural properties:
    - Node count preservation
    - Edge density
    - Degree distribution moments
    """
    loss = torch.tensor(0.0, device=device)
    n_stats = 0

    for key in ["mean_degree", "density", "clustering"]:
        if key in graph_pred_stats and key in graph_true_stats:
            loss += F.mse_loss(graph_pred_stats[key], graph_true_stats[key])
            n_stats += 1

    return loss / max(n_stats, 1)


def compute_impute_loss(
    h: torch.Tensor,
    masked_indices: torch.Tensor,
    original_values: torch.Tensor,
    predicted_values: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    L_impute: Missing value imputation loss (SmoothL1).

    When edges/nodes are randomly masked during training,
    this loss measures reconstruction quality.
    """
    if masked_indices is None or len(masked_indices) == 0:
        return torch.tensor(0.0, device=device)

    return F.smooth_l1_loss(predicted_values, original_values)


def compute_scen_loss(
    scenario_logits: torch.Tensor,
    scenario_labels: torch.Tensor,
    device: torch.device,
    use_contrastive: bool = False,
) -> torch.Tensor:
    """
    L_scen: Scenario classification/contrastive loss.

    Classifies graph snapshots into scenario types:
    - Normal operation
    - Pre-shock
    - Shock
    - Post-shock recovery
    """
    if scenario_logits is None or scenario_labels is None:
        return torch.tensor(0.0, device=device)

    if use_contrastive:
        # Contrastive loss: similar scenarios should have similar embeddings
        # Simplified version using cosine similarity
        return torch.tensor(0.0, device=device)
    else:
        # Cross-entropy for scenario classification
        return F.cross_entropy(scenario_logits, scenario_labels)


def compute_smooth_loss(
    embeddings_t: torch.Tensor,
    embeddings_t_prev: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    L_smooth: Temporal smoothness regularization.

    Encourages embeddings to change gradually over time,
    preventing erratic predictions between consecutive timesteps.

    L_smooth = ||h_t - h_{t-1}||^2 / d
    """
    if embeddings_t_prev is None:
        return torch.tensor(0.0, device=device)

    # Temporal smoothness: L2 distance between consecutive embeddings
    diff = embeddings_t - embeddings_t_prev
    smooth_loss = (diff**2).mean()

    return smooth_loss


def compute_graph_stats(
    edge_src: torch.Tensor, edge_dst: torch.Tensor, num_nodes: int, device: torch.device
) -> Dict[str, torch.Tensor]:
    """Compute graph statistics for stats loss."""
    if len(edge_src) == 0:
        return {
            "mean_degree": torch.tensor(0.0, device=device),
            "density": torch.tensor(0.0, device=device),
            "clustering": torch.tensor(0.0, device=device),
        }

    # Mean degree
    degrees = torch.zeros(num_nodes, device=device)
    degrees.scatter_add_(
        0,
        edge_src.to(device),
        torch.ones_like(edge_src, dtype=torch.float, device=device),
    )
    mean_degree = degrees.mean()

    # Density
    max_edges = num_nodes * (num_nodes - 1)
    density = torch.tensor(len(edge_src) / max(max_edges, 1), device=device)

    # Simplified clustering coefficient (placeholder)
    clustering = torch.tensor(0.0, device=device)

    return {
        "mean_degree": mean_degree,
        "density": density,
        "clustering": clustering,
    }


def compute_graph_stats_from_pairs(
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_prob: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Compute expected graph statistics from edge probabilities."""
    if edge_src.numel() == 0:
        return {
            "mean_degree": torch.tensor(0.0, device=device),
            "density": torch.tensor(0.0, device=device),
            "clustering": torch.tensor(0.0, device=device),
        }

    # Expected out-degree
    degrees = torch.zeros(num_nodes, device=device)
    degrees.scatter_add_(0, edge_src, edge_prob)
    mean_degree = degrees.mean()

    # Expected density
    max_edges = num_nodes * (num_nodes - 1)
    density = edge_prob.sum() / max(max_edges, 1)

    # Placeholder clustering coefficient
    clustering = torch.tensor(0.0, device=device)

    return {
        "mean_degree": mean_degree,
        "density": density,
        "clustering": clustering,
    }


def compute_tvl_edge_weights(
    sizes_t: torch.Tensor,
    src_idx: torch.Tensor,
    y_exist: torch.Tensor,
    y_weight_log: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute TVL-based edge weights for loss reweighting.

    w_ij ∝ TVL_i * E_ij, where E_ij is edge weight for positive edges.
    For negative edges, use a unit edge factor to keep weights non-zero.
    """
    tvl_src = sizes_t[src_idx].clamp(min=0.0)
    edge_factor = torch.where(
        y_exist > 0.5,
        torch.expm1(y_weight_log).clamp(min=0.0),
        torch.ones_like(y_weight_log),
    )
    unnorm = tvl_src * edge_factor
    norm = unnorm.sum().clamp(min=EPS)
    return unnorm / norm


# ============== Training Functions ==============


class EmbeddingCache:
    """
    Cache for frozen encoder embeddings to avoid redundant computation.

    When the encoder is frozen, its outputs are deterministic - we can compute
    them once and reuse across epochs for 3-5x speedup.
    """
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        self.cache: Dict[str, torch.Tensor] = {}
        self._enabled = True

    def get_or_compute(self, sample, graph, features) -> torch.Tensor:
        """Get cached embedding or compute and cache it."""
        cache_key = sample.time_t

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Compute embedding
        with torch.no_grad():
            h = self.model.encode(graph, features, self.device)
        h = h.detach()

        # Cache it
        if self._enabled:
            self.cache[cache_key] = h

        return h

    def clear(self):
        """Clear the cache (e.g., when model weights change)."""
        self.cache.clear()

    def disable(self):
        """Disable caching (for fine-tuning mode)."""
        self._enabled = False
        self.cache.clear()


def train_graphpfn_epoch(
    model,
    pairs,
    optimizer,
    config,
    finetune_encoder: bool,
    prev_embeddings: Optional[Dict[str, torch.Tensor]] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    embedding_cache: Optional[EmbeddingCache] = None,
):
    """
    Train GraphPFN model for one epoch with 7-component loss.

    Loss components:
        1. L_edge (exist_loss): BCE for edge existence prediction
        2. L_link (weight_loss): SmoothL1 for edge weight prediction
        3. L_node (node_loss): SmoothL1 for node TVL change prediction
        4. L_stats (stats_loss): MSE for graph statistics constraint
        5. L_impute (impute_loss): SmoothL1 for masked value reconstruction
        6. L_scen (scen_loss): CE for scenario classification (if labels available)
        7. L_smooth (smooth_loss): Temporal smoothness regularization
    """
    model.train()
    if not finetune_encoder:
        model.encoder.eval()

    device = torch.device(config.device)

    # Loss accumulators for all 7 components
    total_exist, total_weight, total_node = 0.0, 0.0, 0.0
    total_stats, total_impute, total_scen, total_smooth = 0.0, 0.0, 0.0, 0.0
    total_samples = 0

    # Store embeddings for temporal smoothness (by date)
    current_embeddings: Dict[str, torch.Tensor] = {}

    # Determine if we should use AMP
    use_amp = scaler is not None and device.type == "cuda"

    for sample in pairs:
        graph = dgl.graph(
            (sample.edge_src_t, sample.edge_dst_t), num_nodes=len(sample.node_ids)
        )
        features = torch.tensor(sample.features_t, dtype=torch.float32)

        # === Encoder forward pass (with optional caching) ===
        if finetune_encoder:
            # Fine-tuning: compute with gradients
            if use_amp:
                with torch.cuda.amp.autocast():
                    h = model.encode(graph, features, device)
            else:
                h = model.encode(graph, features, device)
        else:
            # Frozen: use cache if available, otherwise compute once
            if embedding_cache is not None:
                h = embedding_cache.get_or_compute(sample, graph, features)
            else:
                with torch.no_grad():
                    h = model.encode(graph, features, device)
                h = h.detach()

        # Store embedding for temporal smoothness
        current_embeddings[sample.time_t] = h.detach().clone()

        # === Loss 3: L_node (node TVL change prediction) ===
        node_true = torch.tensor(sample.y_node, dtype=torch.float32, device=device)
        node_mask = torch.tensor(sample.node_mask, dtype=torch.bool, device=device)
        # Prepare edge info for neighbor-aware node prediction
        edge_src_t = torch.tensor(sample.edge_src_t, dtype=torch.long, device=device)
        edge_dst_t = torch.tensor(sample.edge_dst_t, dtype=torch.long, device=device)
        edge_weight_t = torch.tensor(sample.edge_weight_t, dtype=torch.float32, device=device)
        node_pred = model.node_head(h, edge_src_t, edge_dst_t, edge_weight_t)
        node_loss = (
            F.smooth_l1_loss(node_pred[node_mask], node_true[node_mask])
            if node_mask.any()
            else torch.tensor(0.0, device=device)
        )

        # Edge prediction setup
        src = torch.tensor(sample.pair_src, dtype=torch.long, device=device)
        dst = torch.tensor(sample.pair_dst, dtype=torch.long, device=device)
        y_exist = torch.tensor(sample.y_exist, dtype=torch.float32, device=device)
        y_weight = torch.tensor(sample.y_weight, dtype=torch.float32, device=device)
        weight_mask = torch.tensor(
            sample.weight_mask, dtype=torch.float32, device=device
        )
        # Baseline weight for residual learning (weight at time t)
        baseline_weight = torch.tensor(
            sample.baseline_weight if sample.baseline_weight is not None else np.zeros(len(sample.pair_src)),
            dtype=torch.float32, device=device
        )

        optimizer.zero_grad()

        logits, w_pred = model.link_scorer(h, src, dst)
        edge_prob = torch.sigmoid(logits)

        # === Loss 1: L_edge (edge existence) ===
        pos_weight = torch.tensor([config.neg_ratio], device=device)
        exist_loss = F.binary_cross_entropy_with_logits(
            logits, y_exist, pos_weight=pos_weight, reduction="mean"
        )

        # === Loss 2: L_link (edge weight) with RESIDUAL LEARNING ===
        # Model predicts residual (delta), target is y_weight - baseline_weight
        mask = weight_mask > 0.5
        if mask.any():
            residual_target = y_weight[mask] - baseline_weight[mask]
            weight_loss = F.smooth_l1_loss(w_pred[mask], residual_target, reduction="mean")
        else:
            weight_loss = torch.tensor(0.0, device=device)

        # === Loss 4: L_stats (graph statistics constraint) ===
        # Compare predicted edge probabilities with target graph structure
        edge_src_t = torch.tensor(sample.edge_src_t, dtype=torch.long, device=device)
        edge_dst_t = torch.tensor(sample.edge_dst_t, dtype=torch.long, device=device)
        if config.stats_loss_weight > 0.0:
            stats_pred = compute_graph_stats_from_pairs(
                src, dst, edge_prob, len(sample.node_ids), device
            )
            # Use the same pair set for true stats to keep scales consistent
            stats_true = compute_graph_stats_from_pairs(
                src, dst, y_exist, len(sample.node_ids), device
            )
            stats_loss = compute_stats_loss(stats_pred, stats_true, device)
        else:
            stats_loss = torch.tensor(0.0, device=device)

        # === Loss 5: L_impute (imputation - using node prediction as proxy) ===
        # For masked edges, use weight prediction error as imputation loss
        if config.impute_loss_weight > 0.0 and mask.any():
            impute_rand = (
                torch.rand(mask.sum(), device=device) < config.impute_mask_ratio
            )
            if impute_rand.any():
                impute_pred = w_pred[mask][impute_rand]
                impute_true = y_weight[mask][impute_rand]
                impute_loss = F.smooth_l1_loss(
                    impute_pred, impute_true, reduction="mean"
                )
            else:
                impute_loss = torch.tensor(0.0, device=device)
        else:
            impute_loss = torch.tensor(0.0, device=device)

        # === Loss 6: L_scen (scenario classification) ===
        # Placeholder - requires scenario labels in sample
        scen_loss = torch.tensor(0.0, device=device)
        if (
            config.scen_loss_weight > 0.0
            and getattr(sample, "scenario_label", None) is not None
        ):
            scen_logits = model.scenario_head(h.mean(dim=0, keepdim=True))
            scen_label = torch.tensor([sample.scenario_label], device=device)
            scen_loss = compute_scen_loss(scen_logits, scen_label, device)

        # === Loss 7: L_smooth (temporal smoothness) ===
        smooth_loss = torch.tensor(0.0, device=device)
        if config.smooth_loss_weight > 0.0:
            prev_time_t = sample.prev_time_t
            h_prev = None
            if prev_time_t and prev_time_t in current_embeddings:
                h_prev = current_embeddings[prev_time_t]
            elif prev_embeddings and prev_time_t and prev_time_t in prev_embeddings:
                h_prev = prev_embeddings[prev_time_t]
            if h_prev is not None and h_prev.shape == h.shape:
                smooth_loss = compute_smooth_loss(h, h_prev, device)

        # === Combined loss with all 7 components ===
        loss = (
            config.exist_loss_weight * exist_loss
            + config.weight_loss_weight * weight_loss
            + config.node_loss_weight * node_loss
            + config.stats_loss_weight * stats_loss
            + config.impute_loss_weight * impute_loss
            + config.scen_loss_weight * scen_loss
            + config.smooth_loss_weight * smooth_loss
        )

        # Backward pass with optional AMP scaling
        if use_amp:
            scaler.scale(loss).backward()
            # Unscale before gradient clipping
            scaler.unscale_(optimizer)
            if hasattr(config, 'gradient_clip_norm') and config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Gradient clipping
            if hasattr(config, 'gradient_clip_norm') and config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()

        # Accumulate all losses
        total_exist += exist_loss.item()
        total_weight += weight_loss.item()
        total_node += node_loss.item()
        total_stats += stats_loss.item()
        total_impute += impute_loss.item()
        total_scen += scen_loss.item()
        total_smooth += smooth_loss.item()
        total_samples += 1

    n = max(total_samples, 1)
    return {
        # Core prediction losses
        "exist_loss": total_exist / n,  # L_edge
        "weight_loss": total_weight / n,  # L_link
        "node_loss": total_node / n,  # L_node
        # Auxiliary losses
        "stats_loss": total_stats / n,  # L_stats
        "impute_loss": total_impute / n,  # L_impute
        "scen_loss": total_scen / n,  # L_scen
        "smooth_loss": total_smooth / n,  # L_smooth
    }, current_embeddings  # Return embeddings for next epoch's smoothness


def train_roland_epoch(
    model, pairs, optimizer, config, h1=None, h2=None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
):
    """Train ROLAND model for one epoch."""
    model.train()
    device = torch.device(config.device)
    total_loss, total_samples = 0.0, 0
    total_link, total_weight, total_node = 0.0, 0.0, 0.0

    # Loss weights
    node_loss_weight = getattr(config, "node_loss_weight", 0.4)
    weight_loss_weight = getattr(config, "weight_loss_weight", 1.8)

    # Determine if we should use AMP
    use_amp = scaler is not None and device.type == "cuda"

    for sample in pairs:
        # Build PyG edge index
        edge_index = torch.tensor(
            np.stack([sample.edge_src_t, sample.edge_dst_t]),
            dtype=torch.long,
            device=device,
        )
        x = torch.tensor(sample.features_t, dtype=torch.float32, device=device)

        # Reset hidden states if number of nodes changed
        num_nodes = x.size(0)
        if h1 is not None and h1.size(0) != num_nodes:
            h1, h2 = None, None

        edge_label_index = torch.tensor(
            np.stack([sample.pair_src, sample.pair_dst]),
            dtype=torch.long,
            device=device,
        )
        y_exist = torch.tensor(sample.y_exist, dtype=torch.float32, device=device)

        optimizer.zero_grad()

        pred, weight_pred, h1, h2, node_embed = model(
            x, edge_index, edge_label_index, h1, h2
        )
        h1, h2 = h1.detach(), h2.detach()

        # Link prediction loss
        link_loss = F.binary_cross_entropy_with_logits(pred, y_exist)

        # Edge weight prediction loss with RESIDUAL LEARNING
        y_weight = torch.tensor(sample.y_weight, dtype=torch.float32, device=device)
        weight_mask = torch.tensor(sample.weight_mask, dtype=torch.bool, device=device)
        baseline_weight = torch.tensor(
            sample.baseline_weight if sample.baseline_weight is not None else np.zeros(len(sample.pair_src)),
            dtype=torch.float32, device=device
        )
        if weight_mask.any():
            residual_target = y_weight[weight_mask] - baseline_weight[weight_mask]
            weight_loss = F.smooth_l1_loss(
                weight_pred[weight_mask], residual_target
            )
        else:
            weight_loss = torch.tensor(0.0, device=device)

        # Node prediction loss
        node_pred = model.node_head(node_embed).squeeze(-1)
        y_node = torch.tensor(sample.y_node, dtype=torch.float32, device=device)
        node_mask = torch.tensor(sample.node_mask, dtype=torch.bool, device=device)

        if node_mask.any():
            node_loss = F.smooth_l1_loss(node_pred[node_mask], y_node[node_mask])
        else:
            node_loss = torch.tensor(0.0, device=device)

        # Combined loss
        loss = (
            link_loss + weight_loss_weight * weight_loss + node_loss_weight * node_loss
        )

        # Backward pass with optional AMP scaling
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if hasattr(config, 'gradient_clip_norm') and config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Gradient clipping
            if hasattr(config, 'gradient_clip_norm') and config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()

        total_loss += loss.item()
        total_link += link_loss.item()
        total_weight += weight_loss.item()
        total_node += node_loss.item()
        total_samples += 1

    n = max(total_samples, 1)
    return (
        {
            "loss": total_loss / n,
            "link_loss": total_link / n,
            "weight_loss": total_weight / n,
            "node_loss": total_node / n,
        },
        h1,
        h2,
    )


# ============== Evaluation Functions ==============


def compute_recall_at_k(
    y_true: np.ndarray, y_score: np.ndarray, k_values: List[int] = [100, 500, 1000]
) -> Dict[str, float]:
    """Compute Recall@K for top-K predictions."""
    results = {}
    n_positive = int(y_true.sum())

    if n_positive == 0:
        return {f"recall@{k}": float("nan") for k in k_values}

    n_scores = len(y_score)
    if n_scores == 0:
        return {f"recall@{k}": float("nan") for k in k_values}

    # Get indices of top-K predictions
    top_indices = np.argsort(y_score)[::-1]

    for k in k_values:
        # Keep metric keys stable even when n_scores < k.
        k_eff = min(k, n_scores)
        top_k_indices = top_indices[:k_eff]
        true_positives_in_top_k = y_true[top_k_indices].sum()
        results[f"recall@{k}"] = float(true_positives_in_top_k / n_positive)

    return results


def compute_weighted_mae(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray
) -> float:
    """Compute Weighted MAE (weighted by true edge weights)."""
    if len(y_true) == 0 or weights.sum() == 0:
        return float("nan")

    # Normalize weights
    normalized_weights = weights / weights.sum()
    weighted_errors = normalized_weights * np.abs(y_true - y_pred)
    return float(weighted_errors.sum())


def evaluate_predictions(preds: List[Dict]) -> Dict[str, Any]:
    """Evaluate predictions and compute metrics."""
    exist_true, exist_score = [], []
    weight_true, weight_pred_list = [], []
    node_true, node_pred_list = [], []

    for item in preds:
        exist_prob = 1 / (1 + np.exp(-item["exist_logits"]))
        exist_true.append(item["y_exist"])
        exist_score.append(exist_prob)

        if item["weight_mask"].sum() > 0:
            mask = item["weight_mask"] > 0.5
            weight_true.append(item["y_weight"][mask])
            weight_pred_list.append(item["weight_pred"][mask])

        if item["node_mask"].any():
            node_true.append(item["y_node"][item["node_mask"]])
            node_pred_list.append(item["node_pred"][item["node_mask"]])

    exist_true = np.concatenate(exist_true)
    exist_score = np.concatenate(exist_score)

    # Exist metrics
    exist_metrics = {}
    if len(np.unique(exist_true)) > 1:
        exist_metrics["auprc"] = float(average_precision_score(exist_true, exist_score))
        exist_metrics["auroc"] = float(roc_auc_score(exist_true, exist_score))
        # Add Recall@K
        recall_at_k = compute_recall_at_k(exist_true, exist_score)
        exist_metrics.update(recall_at_k)
    else:
        exist_metrics["auprc"] = float("nan")
        exist_metrics["auroc"] = float("nan")
        exist_metrics["recall@100"] = float("nan")
        exist_metrics["recall@500"] = float("nan")
        exist_metrics["recall@1000"] = float("nan")

    # Weight metrics
    weight_metrics = {
        "mae": float("nan"),
        "rmse": float("nan"),
        "weighted_mae": float("nan"),
    }
    if weight_true:
        wt = np.concatenate(weight_true)
        wp = np.concatenate(weight_pred_list)
        weight_metrics["mae"] = float(mean_absolute_error(wt, wp))
        weight_metrics["rmse"] = float(np.sqrt(mean_squared_error(wt, wp)))
        # Weighted MAE (weighted by true edge weights)
        weight_metrics["weighted_mae"] = compute_weighted_mae(
            wt, wp, np.expm1(wt)
        )  # expm1 to get original scale from log1p

    # Node metrics
    node_metrics = {"mae": float("nan"), "rmse": float("nan")}
    if node_true:
        nt = np.concatenate(node_true)
        np_ = np.concatenate(node_pred_list)
        node_metrics["mae"] = float(mean_absolute_error(nt, np_))
        node_metrics["rmse"] = float(np.sqrt(mean_squared_error(nt, np_)))

    return {"exist": exist_metrics, "weight": weight_metrics, "node": node_metrics}


def evaluate_persistence_baseline(pairs: List[WeekPair]) -> Dict[str, Any]:
    """
    Evaluate naive persistence baseline: predict A_{t+h} = A_t, W_{t+h} = W_t.

    This baseline predicts that the network at t+h is identical to time t.
    It's a critical baseline for networks with high edge overlap (like DeFi's 98.5%).
    """
    exist_true, exist_score = [], []
    weight_true, weight_pred_list = [], []
    node_true, node_pred_list = [], []

    for pair in pairs:
        # Build edge set at time t for fast lookup
        edge_set_t = set(zip(pair.edge_src_t.tolist(), pair.edge_dst_t.tolist()))

        # Build edge weight lookup at time t
        edge_weight_map_t = {}
        for src, dst, w in zip(pair.edge_src_t, pair.edge_dst_t, pair.edge_weight_t):
            edge_weight_map_t[(int(src), int(dst))] = w

        # For each pair to predict, check if it existed at time t
        persist_exist_score = []
        persist_weight_pred = []

        for src, dst in zip(pair.pair_src, pair.pair_dst):
            edge_key = (int(src), int(dst))
            if edge_key in edge_set_t:
                persist_exist_score.append(1.0)  # Edge existed at t, predict it exists at t+h
                persist_weight_pred.append(edge_weight_map_t.get(edge_key, 0.0))
            else:
                persist_exist_score.append(0.0)  # Edge didn't exist at t, predict it doesn't exist
                persist_weight_pred.append(0.0)

        persist_exist_score = np.array(persist_exist_score, dtype=np.float32)
        persist_weight_pred = np.array(persist_weight_pred, dtype=np.float32)

        exist_true.append(pair.y_exist)
        exist_score.append(persist_exist_score)

        # Weight: only for edges that exist at both t and t+h
        if pair.weight_mask.sum() > 0:
            mask = pair.weight_mask > 0.5
            weight_true.append(pair.y_weight[mask])
            weight_pred_list.append(persist_weight_pred[mask])

        # Node: persistence predicts TVL change = 0
        if pair.node_mask.any():
            node_true.append(pair.y_node[pair.node_mask])
            node_pred_list.append(np.zeros(pair.node_mask.sum(), dtype=np.float32))

    exist_true = np.concatenate(exist_true)
    exist_score = np.concatenate(exist_score)

    # Exist metrics
    exist_metrics = {}
    if len(np.unique(exist_true)) > 1:
        exist_metrics["auprc"] = float(average_precision_score(exist_true, exist_score))
        exist_metrics["auroc"] = float(roc_auc_score(exist_true, exist_score))
        recall_at_k = compute_recall_at_k(exist_true, exist_score)
        exist_metrics.update(recall_at_k)
    else:
        exist_metrics["auprc"] = float("nan")
        exist_metrics["auroc"] = float("nan")
        exist_metrics["recall@100"] = float("nan")
        exist_metrics["recall@500"] = float("nan")
        exist_metrics["recall@1000"] = float("nan")

    # Weight metrics
    weight_metrics = {"mae": float("nan"), "rmse": float("nan"), "weighted_mae": float("nan")}
    if weight_true:
        wt = np.concatenate(weight_true)
        wp = np.concatenate(weight_pred_list)
        weight_metrics["mae"] = float(mean_absolute_error(wt, wp))
        weight_metrics["rmse"] = float(np.sqrt(mean_squared_error(wt, wp)))
        weight_metrics["weighted_mae"] = compute_weighted_mae(wt, wp, np.expm1(wt))

    # Node metrics: persistence predicts 0 change
    node_metrics = {"mae": float("nan"), "rmse": float("nan")}
    if node_true:
        nt = np.concatenate(node_true)
        np_ = np.concatenate(node_pred_list)  # All zeros
        node_metrics["mae"] = float(mean_absolute_error(nt, np_))
        node_metrics["rmse"] = float(np.sqrt(mean_squared_error(nt, np_)))

    return {"exist": exist_metrics, "weight": weight_metrics, "node": node_metrics}


def predict_graphpfn(model, pairs, config):
    """Generate predictions with GraphPFN model.

    Note: Model predicts weight RESIDUAL (delta from baseline).
    Final weight prediction = baseline_weight + model_output.
    """
    model.eval()
    device = torch.device(config.device)
    outputs = []

    with torch.no_grad():
        for sample in pairs:
            graph = dgl.graph(
                (sample.edge_src_t, sample.edge_dst_t), num_nodes=len(sample.node_ids)
            )
            features = torch.tensor(sample.features_t, dtype=torch.float32)
            h = model.encode(graph, features, device)
            # Neighbor-aware node prediction
            edge_src_t = torch.tensor(sample.edge_src_t, dtype=torch.long, device=device)
            edge_dst_t = torch.tensor(sample.edge_dst_t, dtype=torch.long, device=device)
            edge_weight_t = torch.tensor(sample.edge_weight_t, dtype=torch.float32, device=device)
            node_pred = model.node_head(h, edge_src_t, edge_dst_t, edge_weight_t).cpu().numpy()

            src = torch.tensor(sample.pair_src, dtype=torch.long, device=device)
            dst = torch.tensor(sample.pair_dst, dtype=torch.long, device=device)
            logits, w_residual = model.link_scorer(h, src, dst)

            # Residual learning: final prediction = baseline + residual
            baseline = sample.baseline_weight if sample.baseline_weight is not None else np.zeros(len(sample.pair_src))
            w_pred = w_residual.cpu().numpy() + baseline

            outputs.append(
                {
                    "time_t": sample.time_t,
                    "time_t1": sample.time_t1,
                    "node_ids": sample.node_ids,
                    "categories": sample.categories,
                    "sizes_t": sample.sizes_t,
                    "pair_src": sample.pair_src,
                    "pair_dst": sample.pair_dst,
                    "y_exist": sample.y_exist,
                    "y_weight": sample.y_weight,
                    "weight_mask": sample.weight_mask,
                    "y_node": sample.y_node,
                    "node_mask": sample.node_mask,
                    "exist_logits": logits.cpu().numpy(),
                    "weight_pred": w_pred,
                    "node_pred": node_pred,
                    "baseline_weight": baseline,
                }
            )

    return outputs


def predict_roland(model, pairs, config, h1=None, h2=None):
    """Generate predictions with ROLAND model.

    Note: Model predicts weight RESIDUAL (delta from baseline).
    Final weight prediction = baseline_weight + model_output.
    """
    model.eval()
    device = torch.device(config.device)
    outputs = []

    with torch.no_grad():
        for sample in pairs:
            edge_index = torch.tensor(
                np.stack([sample.edge_src_t, sample.edge_dst_t]),
                dtype=torch.long,
                device=device,
            )
            x = torch.tensor(sample.features_t, dtype=torch.float32, device=device)

            # Reset hidden states if number of nodes changed
            num_nodes = x.size(0)
            if h1 is not None and h1.size(0) != num_nodes:
                h1, h2 = None, None

            edge_label_index = torch.tensor(
                np.stack([sample.pair_src, sample.pair_dst]),
                dtype=torch.long,
                device=device,
            )

            pred, w_residual, h1, h2, node_embed = model(
                x, edge_index, edge_label_index, h1, h2
            )

            # Residual learning: final prediction = baseline + residual
            baseline = sample.baseline_weight if sample.baseline_weight is not None else np.zeros(len(sample.pair_src))
            w_pred = w_residual.cpu().numpy() + baseline

            # Node prediction
            node_pred = model.node_head(node_embed).squeeze(-1).cpu().numpy()

            outputs.append(
                {
                    "time_t": sample.time_t,
                    "time_t1": sample.time_t1,
                    "node_ids": sample.node_ids,
                    "categories": sample.categories,
                    "sizes_t": sample.sizes_t,
                    "pair_src": sample.pair_src,
                    "pair_dst": sample.pair_dst,
                    "y_exist": sample.y_exist,
                    "y_weight": sample.y_weight,
                    "weight_mask": sample.weight_mask,
                    "y_node": sample.y_node,
                    "node_mask": sample.node_mask,
                    "exist_logits": pred.cpu().numpy(),
                    "weight_pred": w_pred,
                    "node_pred": node_pred,
                    "baseline_weight": baseline,
                }
            )

    return outputs, h1, h2


def _summarize_metric_group(
    metrics_list: List[Dict[str, Any]], group: str, keys: List[str]
) -> Dict[str, Dict[str, float]]:
    """Summarize mean/std for metric group across folds."""
    summary = {}
    for key in keys:
        values = []
        for metrics in metrics_list:
            value = metrics.get(group, {}).get(key, float("nan"))
            if isinstance(value, float) and math.isnan(value):
                continue
            values.append(value)
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        else:
            summary[key] = {"mean": float("nan"), "std": float("nan")}
    return summary


def aggregate_fold_metrics(
    fold_results: List[Dict[str, Any]], horizons: List[int]
) -> Dict[str, Any]:
    """Aggregate fold metrics into mean/std summaries per horizon."""
    summary = {}
    for horizon in horizons:
        metrics_list = []
        for result in fold_results:
            if not result or "results" not in result:
                continue
            metrics = result["results"].get(f"h{horizon}")
            if metrics:
                metrics_list.append(metrics)
        if not metrics_list:
            continue
        summary[f"h{horizon}"] = {
            "n_folds": len(metrics_list),
            "exist": _summarize_metric_group(
                metrics_list,
                "exist",
                ["auprc", "auroc", "recall@100", "recall@500", "recall@1000"],
            ),
            "weight": _summarize_metric_group(
                metrics_list, "weight", ["mae", "rmse", "weighted_mae"]
            ),
            "node": _summarize_metric_group(metrics_list, "node", ["mae", "rmse"]),
        }
    return summary


def run_expanding_window_evaluation(
    config: ExperimentConfig,
    split_result: Dict[str, Any],
    date_to_snap: Dict[str, Dict],
    run_frozen: bool,
    run_dexposure_fm: bool,
    run_roland: bool,
    save_predictions: bool,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run expanding-window walk-forward evaluation across folds."""
    folds = split_result.get("folds", [])
    if not folds:
        return {"method": "expanding_window_walk_forward", "folds": [], "summary": {}}

    log_info(f"\n{'=' * 60}")
    log_info("EXPANDING WINDOW WALK-FORWARD EVALUATION")
    log_info(f"{'=' * 60}")
    log_info(f"  Folds: {len(folds)}")

    fold_entries = []
    model_fold_results = {
        "graphpfn_frozen": [],
        "dexposure_fm": [],
        "roland": [],
    }

    for fold in folds:
        fold_id = fold["fold_id"]
        train_snaps = [date_to_snap[d] for d in fold["train"] if d in date_to_snap]
        val_snaps = [date_to_snap[d] for d in fold["val"] if d in date_to_snap]
        test_snaps = [date_to_snap[d] for d in fold["test"] if d in date_to_snap]

        log_info(f"\n--- Fold {fold_id} ---")
        log_info(
            f"  Train: {fold['train'][0]} ~ {fold['train'][-1]} ({len(fold['train'])}w)"
        )
        log_info(f"  Val:   {fold['val'][0]} ~ {fold['val'][-1]} ({len(fold['val'])}w)")
        log_info(
            f"  Test:  {fold['test'][0]} ~ {fold['test'][-1]} ({len(fold['test'])}w)"
        )

        fold_entry = {
            "fold_id": fold_id,
            "train_range": f"{fold['train'][0]} ~ {fold['train'][-1]}",
            "val_range": f"{fold['val'][0]} ~ {fold['val'][-1]}",
            "test_range": f"{fold['test'][0]} ~ {fold['test'][-1]}",
        }

        if run_frozen:
            result = run_graphpfn_experiment(
                config,
                finetune=False,
                train_snaps=train_snaps,
                val_snaps=val_snaps,
                test_snaps=test_snaps,
                save_predictions=save_predictions,
                output_dir=output_dir,
                fold_id=fold_id,
            )
            if result:
                fold_entry["graphpfn_frozen"] = result
                model_fold_results["graphpfn_frozen"].append(result)

        if run_dexposure_fm:
            result = run_graphpfn_experiment(
                config,
                finetune=True,
                train_snaps=train_snaps,
                val_snaps=val_snaps,
                test_snaps=test_snaps,
                save_predictions=save_predictions,
                output_dir=output_dir,
                fold_id=fold_id,
            )
            if result:
                fold_entry["dexposure_fm"] = result
                model_fold_results["dexposure_fm"].append(result)

        if run_roland:
            result = run_roland_experiment(
                config,
                train_snaps=train_snaps,
                val_snaps=val_snaps,
                test_snaps=test_snaps,
                save_predictions=save_predictions,
                output_dir=output_dir,
                fold_id=fold_id,
            )
            if result:
                fold_entry["roland"] = result
                model_fold_results["roland"].append(result)

        fold_entries.append(fold_entry)

    summary = {}
    if model_fold_results["graphpfn_frozen"]:
        summary["graphpfn_frozen"] = aggregate_fold_metrics(
            model_fold_results["graphpfn_frozen"], config.forecast_horizons
        )
    if model_fold_results["dexposure_fm"]:
        summary["dexposure_fm"] = aggregate_fold_metrics(
            model_fold_results["dexposure_fm"], config.forecast_horizons
        )
    if model_fold_results["roland"]:
        summary["roland"] = aggregate_fold_metrics(
            model_fold_results["roland"], config.forecast_horizons
        )

    return {
        "method": "expanding_window_walk_forward",
        "n_folds": len(fold_entries),
        "folds": fold_entries,
        "summary": summary,
    }


# ============== Main Experiment Functions ==============


def run_graphpfn_experiment(
    config: ExperimentConfig,
    finetune: bool = True,
    train_snaps: Optional[List[Dict]] = None,
    val_snaps: Optional[List[Dict]] = None,
    test_snaps: Optional[List[Dict]] = None,
    save_predictions: bool = False,
    output_dir: Optional[Path] = None,
    fold_id: Optional[int] = None,
):
    """
    Run GraphPFN experiment (GraphPFN-Frozen or DeXposure-FM).

    Args:
        config: Experiment configuration
        finetune: Whether to fine-tune encoder (True) or freeze (False)
        train_snaps: Pre-loaded training snapshots (if None, will load from config)
        val_snaps: Pre-loaded validation snapshots
        test_snaps: Pre-loaded test snapshots
        save_predictions: Whether to save predictions to CSV
        output_dir: Output directory for predictions
    """
    if not GRAPHPFN_AVAILABLE:
        log_info("GraphPFN not available, skipping...")
        return None

    mode = "DeXposure-FM" if finetune else "GraphPFN-Frozen"
    model_name = "dexposure_fm" if finetune else "graphpfn_frozen"
    model_output_dir = None
    append_predictions = False
    if output_dir:
        if output_dir.name in {"frozen", "finetuned", "roland", "dexposure-fm", "deposure-fm"}:
            model_output_dir = output_dir
        else:
            model_output_dir = output_dir / model_name
        if fold_id is not None:
            model_output_dir = output_dir / f"{model_name}_fold{fold_id}"
        model_output_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"\n{'=' * 60}")
    log_info(f"GraphPFN Experiment ({mode})")
    log_info(f"{'=' * 60}")

    set_seed(config.seed)
    device = torch.device(config.device)

    # Load data if not provided
    if train_snaps is None:
        meta_category, category_list, category_to_idx = load_metadata(config.meta_path)
        network_data = load_network_data(config.data_path)
        all_dates = sorted(network_data.keys())

        snapshots = [
            build_snapshot(
                date, network_data[date], meta_category, category_to_idx, category_list
            )
            for date in all_dates
        ]
        snapshots = enrich_snapshots_with_historical_features(snapshots)

        # Split data by ratio (legacy mode)
        n = len(snapshots)
        train_end = int(n * config.train_ratio)
        val_end = int(n * (config.train_ratio + config.val_ratio))

        train_snaps = snapshots[:train_end]
        val_snaps = snapshots[train_end:val_end]
        test_snaps = snapshots[val_end:]
        log_info("  Using ratio-based split (legacy mode)")
    else:
        log_info("  Using strict temporal split (recommended)")

    log_info(
        f"  Train: {len(train_snaps)}, Val: {len(val_snaps) if val_snaps else 0}, Test: {len(test_snaps) if test_snaps else 0}"
    )

    # Load encoder ONCE. We reinitialize model heads per horizon for clean comparisons.
    encoder = load_graphpfn_encoder(config.checkpoint_path, device)
    embed_dim = encoder.tfm.embed_dim
    base_encoder_state = None
    if finetune:
        # Keep a CPU copy so we can reset the encoder between horizons without doubling GPU memory.
        base_encoder_state = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}

    # Build pairs for each horizon
    results = {}
    for horizon in config.forecast_horizons:
        log_info(f"\n--- Horizon h={horizon} ---")

        # Ensure per-horizon runs are comparable/reproducible (fresh head init, deterministic RNG).
        set_seed(config.seed)

        # Reset encoder to pretrained weights for each horizon (avoid cross-horizon leakage).
        if finetune and base_encoder_state is not None:
            encoder.load_state_dict(base_encoder_state, strict=False)

        # Create a fresh model head per horizon.
        model = GraphPFNLinkPredictor(encoder, embed_dim, config.hidden_dim).to(device)

        # Set encoder trainability
        for p in model.encoder.parameters():
            p.requires_grad = finetune

        # Optimizer (layer-wise LR when fine-tuning)
        if finetune:
            encoder_params = list(model.encoder.parameters())
            encoder_param_set = set(encoder_params)
            head_params = [p for p in model.parameters() if p not in encoder_param_set]
            optimizer = torch.optim.Adam(
                [
                    {"params": encoder_params, "lr": config.lr * 0.1},
                    {"params": head_params, "lr": config.lr},
                ],
                weight_decay=config.weight_decay,
            )
            log_info(
                f"  Using layer-wise LR: encoder={config.lr * 0.1:.1e}, head={config.lr:.1e}"
            )
        else:
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=config.lr,
                weight_decay=config.weight_decay,
            )

        # === Performance optimizations ===
        scaler = None
        if config.use_amp and device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()
            log_info("  AMP enabled (mixed precision training)")

        embedding_cache = None
        if not finetune and config.cache_frozen_embeddings:
            embedding_cache = EmbeddingCache(model, device)
            log_info("  Embedding cache enabled (frozen encoder)")

        train_pairs = build_week_pairs(
            train_snaps, config.neg_ratio, config.seed, horizon
        )
        val_pairs = (
            build_week_pairs(val_snaps, config.neg_ratio, config.seed, horizon)
            if val_snaps
            else []
        )
        test_pairs = (
            build_week_pairs(test_snaps, config.neg_ratio, config.seed, horizon)
            if test_snaps
            else []
        )

        log_info(
            f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}"
        )

        # Train with 7-component loss
        prev_embeddings = None
        metric_name = config.early_stop_metric
        best_metric = -float("inf")
        patience_counter = 0
        patience = config.early_stop_patience
        best_model_state = None

        for epoch in range(config.epochs):
            losses, prev_embeddings = train_graphpfn_epoch(
                model, train_pairs, optimizer, config, finetune, prev_embeddings,
                scaler=scaler, embedding_cache=embedding_cache,
            )
            # Print progress every epoch (7-component losses)
            loss_str = (
                f"exist={losses['exist_loss']:.4f}, weight={losses['weight_loss']:.4f}, "
                f"node={losses['node_loss']:.4f}"
            )
            aux_str = (
                f"stats={losses['stats_loss']:.4f}, impute={losses['impute_loss']:.4f}, "
                f"scen={losses['scen_loss']:.4f}, smooth={losses['smooth_loss']:.4f}"
            )
            weighted_parts = {
                "exist": config.exist_loss_weight * losses["exist_loss"],
                "weight": config.weight_loss_weight * losses["weight_loss"],
                "node": config.node_loss_weight * losses["node_loss"],
                "stats": config.stats_loss_weight * losses["stats_loss"],
                "impute": config.impute_loss_weight * losses["impute_loss"],
                "scen": config.scen_loss_weight * losses["scen_loss"],
                "smooth": config.smooth_loss_weight * losses["smooth_loss"],
            }
            weighted_total = sum(weighted_parts.values())
            if weighted_total > 0:
                contrib_str = (
                    "contrib%="
                    f"exist={weighted_parts['exist'] / weighted_total:.2%}, "
                    f"weight={weighted_parts['weight'] / weighted_total:.2%}, "
                    f"node={weighted_parts['node'] / weighted_total:.2%}, "
                    f"stats={weighted_parts['stats'] / weighted_total:.2%}, "
                    f"impute={weighted_parts['impute'] / weighted_total:.2%}, "
                    f"scen={weighted_parts['scen'] / weighted_total:.2%}, "
                    f"smooth={weighted_parts['smooth'] / weighted_total:.2%}"
                )
            else:
                contrib_str = "contrib%=nan"
            if val_pairs and (epoch + 1) % config.val_eval_every == 0:
                val_preds = predict_graphpfn(model, val_pairs, config)
                val_metrics = evaluate_predictions(val_preds)
                val_metric = val_metrics["exist"][metric_name]

                # Early stopping check
                if val_metric > best_metric:
                    best_metric = val_metric
                    patience_counter = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                    log_info(
                        f"  Epoch {epoch + 1}/{config.epochs}: {loss_str}, {aux_str}, "
                        f"{contrib_str}, val_{metric_name}={val_metric:.4f} ⭐ (best)"
                    )
                else:
                    patience_counter += 1
                    log_info(
                        f"  Epoch {epoch + 1}/{config.epochs}: {loss_str}, {aux_str}, "
                        f"{contrib_str}, val_{metric_name}={val_metric:.4f} (patience: {patience_counter}/{patience})"
                    )
                    if patience_counter >= patience:
                        log_info(f"  Early stopping triggered after {epoch + 1} epochs")
                        break
            else:
                log_info(
                    f"  Epoch {epoch + 1}/{config.epochs}: {loss_str}, {aux_str}, {contrib_str}"
                )

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            log_info(f"  Restored best model with val_{metric_name}={best_metric:.4f}")
            try:
                if model_output_dir:
                    best_model_path = model_output_dir / f"best_model_h{horizon}.pt"
                    torch.save({"model": model.state_dict()}, best_model_path)
                    log_info(f"  Saved best model to: {best_model_path}")
            except Exception as e:
                log_error(f"  Failed to save best model: {e}")

        # Test
        test_preds = predict_graphpfn(model, test_pairs, config)
        test_metrics = evaluate_predictions(test_preds)
        results[f"h{horizon}"] = test_metrics

        log_info(f"  Test AUPRC: {test_metrics['exist']['auprc']:.4f}")
        log_info(f"  Test AUROC: {test_metrics['exist']['auroc']:.4f}")
        log_info(f"  Test Recall@100: {test_metrics['exist'].get('recall@100', 'N/A')}")
        log_info(f"  Test Weight MAE: {test_metrics['weight']['mae']:.4f}")
        log_info(f"  Test Weighted MAE: {test_metrics['weight']['weighted_mae']:.4f}")
        log_info(f"  Test Node MAE: {test_metrics['node']['mae']:.4f}")

        # Save predictions if requested
        if save_predictions and model_output_dir:
            edges_path, nodes_path = save_predictions_csv(
                test_preds,
                model_output_dir,
                model_name,
                horizon,
                fold_id,
                append=append_predictions,
            )
            append_predictions = True
            log_info(f"  Predictions saved to: {edges_path.name}, {nodes_path.name}")

        # Save intermediate results after each horizon
        if model_output_dir:
            intermediate_result = {
                "model": mode,
                "results": results.copy(),
                "config": {
                    "finetune": finetune,
                    "epochs": config.epochs,
                    "seed": config.seed,
                },
                "status": "partial"
                if horizon != config.forecast_horizons[-1]
                else "complete",
            }
            intermediate_path = model_output_dir / "metrics_intermediate.json"
            with open(intermediate_path, "w") as f:
                json.dump(intermediate_result, f, indent=2)
            log_info(f"  Intermediate results saved to: {intermediate_path.name}")

        # Free horizon-specific model to reduce peak GPU memory when looping over horizons.
        del model, optimizer, embedding_cache, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "model": mode,
        "results": results,
        "config": {
            "finetune": finetune,
            "epochs": config.epochs,
            "seed": config.seed,
        },
    }

    if model_output_dir:
        metrics_path = save_metrics_json(result, model_output_dir)
        log_info(f"  Metrics saved to: {metrics_path.name}")

    return result


def run_roland_experiment(
    config: ExperimentConfig,
    train_snaps: Optional[List[Dict]] = None,
    val_snaps: Optional[List[Dict]] = None,
    test_snaps: Optional[List[Dict]] = None,
    save_predictions: bool = False,
    output_dir: Optional[Path] = None,
    fold_id: Optional[int] = None,
):
    """Run ROLAND baseline experiment."""
    log_info(f"\n{'=' * 60}")
    log_info("ROLAND Baseline Experiment")

    model_name = "roland"
    model_output_dir = None
    append_predictions = False
    if output_dir:
        model_output_dir = output_dir / model_name
        if fold_id is not None:
            model_output_dir = output_dir / f"{model_name}_fold{fold_id}"
        model_output_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"{'=' * 60}")

    set_seed(config.seed)
    device = torch.device(config.device)

    # Load data if not provided
    if train_snaps is None:
        meta_category, category_list, category_to_idx = load_metadata(config.meta_path)
        network_data = load_network_data(config.data_path)
        all_dates = sorted(network_data.keys())

        snapshots = [
            build_snapshot(
                date, network_data[date], meta_category, category_to_idx, category_list
            )
            for date in all_dates
        ]
        snapshots = enrich_snapshots_with_historical_features(snapshots)

        # Split data by ratio (legacy mode)
        n = len(snapshots)
        train_end = int(n * config.train_ratio)
        val_end = int(n * (config.train_ratio + config.val_ratio))

        train_snaps = snapshots[:train_end]
        val_snaps = snapshots[train_end:val_end]
        test_snaps = snapshots[val_end:]
        log_info("  Using ratio-based split (legacy mode)")
    else:
        log_info("  Using strict temporal split (recommended)")

    # Get input dimension
    input_dim = train_snaps[0]["features"].shape[1]

    log_info(
        f"  Train: {len(train_snaps)}, Val: {len(val_snaps) if val_snaps else 0}, Test: {len(test_snaps) if test_snaps else 0}"
    )

    results = {}
    for horizon in config.forecast_horizons:
        log_info(f"\n--- Horizon h={horizon} ---")

        # Create model
        model = ROLANDBaseline(input_dim, hidden_dim=64, out_dim=32).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        # AMP scaler for ROLAND training
        roland_scaler = None
        if config.use_amp and device.type == "cuda":
            roland_scaler = torch.cuda.amp.GradScaler()

        train_pairs = build_week_pairs(
            train_snaps, config.neg_ratio, config.seed, horizon
        )
        val_pairs = (
            build_week_pairs(val_snaps, config.neg_ratio, config.seed, horizon)
            if val_snaps
            else []
        )
        test_pairs = (
            build_week_pairs(test_snaps, config.neg_ratio, config.seed, horizon)
            if test_snaps
            else []
        )

        log_info(
            f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}"
        )

        # Train
        h1, h2 = None, None
        metric_name = config.early_stop_metric
        best_metric = -float("inf")
        patience_counter = 0
        patience = config.early_stop_patience
        best_model_state = None
        for epoch in range(config.epochs):
            losses, h1, h2 = train_roland_epoch(
                model, train_pairs, optimizer, config, h1, h2,
                scaler=roland_scaler,
            )
            weighted_parts = {
                "link": losses["link_loss"],
                "weight": config.weight_loss_weight * losses["weight_loss"],
                "node": config.node_loss_weight * losses["node_loss"],
            }
            weighted_total = sum(weighted_parts.values())
            if weighted_total > 0:
                contrib_str = (
                    "contrib%="
                    f"link={weighted_parts['link'] / weighted_total:.2%}, "
                    f"weight={weighted_parts['weight'] / weighted_total:.2%}, "
                    f"node={weighted_parts['node'] / weighted_total:.2%}"
                )
            else:
                contrib_str = "contrib%=nan"
            if val_pairs and (epoch + 1) % config.val_eval_every == 0:
                val_preds, _, _ = predict_roland(model, val_pairs, config, h1, h2)
                val_metrics = evaluate_predictions(val_preds)
                val_metric = val_metrics["exist"][metric_name]
                if val_metric > best_metric:
                    best_metric = val_metric
                    patience_counter = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                    tag = " ⭐ (best)"
                else:
                    patience_counter += 1
                    tag = f" (patience: {patience_counter}/{patience})"
                log_info(
                    f"  Epoch {epoch + 1}: loss={losses['loss']:.4f} "
                    f"(link={losses['link_loss']:.4f}, weight={losses['weight_loss']:.4f}, "
                    f"node={losses['node_loss']:.4f}), {contrib_str}, "
                    f"val_{metric_name}={val_metric:.4f}{tag}"
                )
                if patience_counter >= patience:
                    log_info(f"  Early stopping triggered after {epoch + 1} epochs")
                    break
            else:
                log_info(
                    f"  Epoch {epoch + 1}: loss={losses['loss']:.4f} "
                    f"(link={losses['link_loss']:.4f}, weight={losses['weight_loss']:.4f}, "
                    f"node={losses['node_loss']:.4f}), {contrib_str}"
                )

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            log_info(f"  Restored best model with val_{metric_name}={best_metric:.4f}")

        # Test
        test_preds, _, _ = predict_roland(model, test_pairs, config, h1, h2)
        test_metrics = evaluate_predictions(test_preds)
        results[f"h{horizon}"] = test_metrics

        log_info(f"  Test AUPRC: {test_metrics['exist']['auprc']:.4f}")
        log_info(f"  Test AUROC: {test_metrics['exist']['auroc']:.4f}")
        log_info(f"  Test Weight MAE: {test_metrics['weight']['mae']:.4f}")
        log_info(f"  Test Node MAE: {test_metrics['node']['mae']:.4f}")

        if save_predictions and model_output_dir:
            edges_path, nodes_path = save_predictions_csv(
                test_preds,
                model_output_dir,
                model_name,
                horizon,
                fold_id,
                append=append_predictions,
            )
            append_predictions = True
            log_info(f"  Predictions saved to: {edges_path.name}, {nodes_path.name}")

        # Save intermediate results after each horizon
        if model_output_dir:
            intermediate_result = {
                "model": "ROLAND",
                "results": results.copy(),
                "config": {"epochs": config.epochs, "seed": config.seed},
                "status": "partial"
                if horizon != config.forecast_horizons[-1]
                else "complete",
            }
            intermediate_path = model_output_dir / "metrics_intermediate.json"
            with open(intermediate_path, "w") as f:
                json.dump(intermediate_result, f, indent=2)
            log_info(f"  Intermediate results saved to: {intermediate_path.name}")

    result = {
        "model": "ROLAND",
        "results": results,
        "config": {"epochs": config.epochs, "seed": config.seed},
    }

    if model_output_dir:
        metrics_path = save_metrics_json(result, model_output_dir)
        log_info(f"  Metrics saved to: {metrics_path.name}")

    return result


def run_network_statistics(config: ExperimentConfig):
    """Compute network-level statistics for all snapshots."""
    log_info(f"\n{'=' * 60}")
    log_info("Computing Network Statistics")
    log_info(f"{'=' * 60}")

    meta_category, category_list, category_to_idx = load_metadata(config.meta_path)
    network_data = load_network_data(config.data_path)
    all_dates = sorted(network_data.keys())

    # Build snapshots
    snapshots = []
    for date in all_dates:
        snap = build_snapshot(
            date, network_data[date], meta_category, category_to_idx, category_list
        )
        snapshots.append(
            {
                "date": date,
                "edge_index": np.stack([snap["edge_src"], snap["edge_dst"]])
                if len(snap["edge_src"]) > 0
                else np.zeros((2, 0), dtype=np.int64),
                "num_nodes": len(snap["node_ids"]),
                "edge_weights": snap["edge_weight"],
                "node_sizes": snap["sizes"],
                "node_sectors": snap["categories"],
            }
        )

    # Compute statistics
    all_stats = compute_rolling_statistics(snapshots, category_list)

    # Convert to DataFrame
    df = pd.DataFrame(all_stats)

    # Save
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "network_statistics.csv", index=False)

    # Summary
    log_info(f"\nComputed statistics for {len(all_stats)} snapshots")
    log_info(f"Date range: {all_stats[0]['date']} to {all_stats[-1]['date']}")
    log_info("\nSummary statistics:")
    for col in ["density", "degree_gini", "tvl_hhi", "top_10_concentration"]:
        if col in df.columns:
            log_info(f"  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")

    return {"statistics": all_stats, "summary": df.describe().to_dict()}


# ============== Main ==============


def main():
    parser = argparse.ArgumentParser(description="DeXposure-FM Full Experiment Suite")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=[
            "all",
            "frozen",
            "dexposure-fm",
            "deposure-fm",  # Backward-compatible alias
            "roland",
            "persistence",
            "stats",
            "compare",
        ],
        help="Experiment mode: all, frozen, dexposure-fm (alias: deposure-fm), roland, persistence (naive baseline), stats, compare",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience on validation metric (default: 3)",
    )
    parser.add_argument(
        "--early-stop-metric",
        type=str,
        default="auprc",
        choices=["auprc", "auroc"],
        help="Metric for early stopping (default: auprc)",
    )
    parser.add_argument(
        "--val-eval-every",
        type=int,
        default=1,
        help="Evaluate on validation every N epochs (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    # Temporal split arguments (金融惯例: Expanding Window Walk-Forward)
    parser.add_argument(
        "--holdout-start",
        type=str,
        default="2025-01-01",
        help="Hold-out test set start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--min-train-weeks",
        type=int,
        default=104,
        help="Minimum training weeks (default: 104 = 2 years)",
    )
    parser.add_argument(
        "--val-weeks",
        type=int,
        default=24,
        help="Validation window size in weeks (default: 24, ensures h=12 has sufficient validation samples)",
    )
    parser.add_argument(
        "--test-weeks",
        type=int,
        default=8,
        help="Test window size per fold in weeks (default: 8)",
    )
    parser.add_argument(
        "--step-weeks",
        type=int,
        default=8,
        help="Step size for expanding window folds in weeks (default: 8)",
    )
    parser.add_argument(
        "--rolling",
        action="store_true",
        help="Run expanding window walk-forward evaluation (slower but more robust)",
    )
    parser.add_argument(
        "--save-predictions", action="store_true", help="Save predictions to CSV files"
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,4,8,12",
        help="Comma-separated forecast horizons (default: 1,4,8,12)",
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm (default: 1.0, set to 0 to disable)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )
    args = parser.parse_args()

    # Prevent concurrent runs to avoid GPU OOM
    global _run_lock_path
    _run_lock_path = GRAPHPFN_ROOT / ".run_full_experiment.lock"
    acquire_run_lock(_run_lock_path)
    atexit.register(release_run_lock)

    config = ExperimentConfig()
    config.epochs = args.epochs
    config.early_stop_patience = args.patience
    config.early_stop_metric = args.early_stop_metric
    config.val_eval_every = max(1, args.val_eval_every)
    config.seed = args.seed
    config.forecast_horizons = [int(h) for h in args.horizons.split(",")]
    config.gradient_clip_norm = args.gradient_clip_norm

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = Path("output") / timestamp

    mode_dir_map = {"dexposure-fm": "finetuned", "deposure-fm": "finetuned"}
    if args.mode == "all":
        output_dir = base_output_dir
    else:
        output_dir = base_output_dir / mode_dir_map.get(args.mode, args.mode)

    config.output_dir = str(output_dir)

    # Initialize logging and result management
    init_run_context(output_dir)

    log_info(f"{'=' * 60}")
    log_info("DeXposure-FM Experiment Suite")
    log_info(f"{'=' * 60}")
    log_info(f"Mode: {args.mode}")
    if args.mode == "all":
        log_info(f"Output base directory: {output_dir}")
    else:
        log_info(f"Output directory: {output_dir}")
    log_info(f"Device: {config.device}")
    log_info(f"Epochs: {config.epochs}")
    log_info(
        f"Early stopping: metric={config.early_stop_metric}, "
        f"patience={config.early_stop_patience}, "
        f"val_eval_every={config.val_eval_every}"
    )
    log_info(f"Forecast horizons: {config.forecast_horizons}")
    log_info(f"Seed: {config.seed}")

    # Save configuration
    save_result(
        "config",
        {
            "mode": args.mode,
            "epochs": config.epochs,
            "seed": config.seed,
            "forecast_horizons": config.forecast_horizons,
            "device": config.device,
            "output_dir": str(output_dir),
            "holdout_start": args.holdout_start,
            "min_train_weeks": args.min_train_weeks,
            "val_weeks": args.val_weeks,
            "test_weeks": args.test_weeks,
        },
    )

    set_seed(config.seed)

    # Wrap everything in try-except to ensure results are saved on error
    try:
        _run_experiments(args, config, output_dir)
    except Exception as e:
        log_error(f"Experiment failed with error: {e}")
        log_error(traceback.format_exc())
        save_result("error", str(e))
        save_result("traceback", traceback.format_exc())
        _result_manager.results["status"] = "failed"
        _result_manager._save()
        raise

    # Mark complete
    _result_manager.mark_complete()

    log_info(f"\n{'=' * 60}")
    log_info("EXPERIMENT COMPLETE")
    log_info(f"{'=' * 60}")
    log_info(f"Results saved to: {output_dir / 'experiment_results.json'}")
    log_info(f"Log file: {_logger.log_file}")


def _run_experiments(args, config: ExperimentConfig, output_dir: Path):
    """Run all experiments - separated for error handling."""

    # Load data once for all experiments
    log_info("\nLoading data...")
    meta_category, category_list, category_to_idx = load_metadata(config.meta_path)
    network_data = load_network_data(config.data_path)
    all_dates = sorted(network_data.keys())
    log_info(f"Loaded {len(all_dates)} snapshots: {all_dates[0]} to {all_dates[-1]}")

    # Build all snapshots
    log_info("Building snapshots...")
    all_snapshots = [
        build_snapshot(
            date, network_data[date], meta_category, category_to_idx, category_list
        )
        for date in all_dates
    ]
    all_snapshots = enrich_snapshots_with_historical_features(all_snapshots)

    # Expanding Window Walk-Forward Split (金融惯例)
    split_result = expanding_window_split(
        all_dates,
        holdout_start=args.holdout_start,
        min_train_weeks=args.min_train_weeks,
        val_weeks=args.val_weeks,
        test_weeks=args.test_weeks,
        step_weeks=args.step_weeks,
    )

    log_info(f"\n{'=' * 60}")
    log_info("EXPANDING WINDOW WALK-FORWARD SPLIT (金融惯例)")
    log_info(f"{'=' * 60}")
    log_info(f"  Rolling Folds: {split_result['n_folds']}")
    log_info(
        f"  Min Train: {args.min_train_weeks} weeks | Val: {args.val_weeks} weeks | Test: {args.test_weeks} weeks"
    )
    log_info(f"  Step Size: {args.step_weeks} weeks")
    if split_result["folds"]:
        log_info(
            f"\n  Fold 1:  Train {split_result['folds'][0]['train'][0]} ~ {split_result['folds'][0]['train'][-1]} ({len(split_result['folds'][0]['train'])}w)"
        )
        log_info(
            f"           Val   {split_result['folds'][0]['val'][0]} ~ {split_result['folds'][0]['val'][-1]} ({len(split_result['folds'][0]['val'])}w)"
        )
        log_info(
            f"           Test  {split_result['folds'][0]['test'][0]} ~ {split_result['folds'][0]['test'][-1]} ({len(split_result['folds'][0]['test'])}w)"
        )
        if len(split_result["folds"]) > 1:
            last = split_result["folds"][-1]
            log_info(
                f"  Fold {last['fold_id']}: Train ... ~ {last['train'][-1]} ({len(last['train'])}w)"
            )
            log_info(
                f"           Val   {last['val'][0]} ~ {last['val'][-1]} ({len(last['val'])}w)"
            )
            log_info(
                f"           Test  {last['test'][0]} ~ {last['test'][-1]} ({len(last['test'])}w)"
            )

    holdout = split_result["holdout"]
    log_info("\n  Hold-out (Final Evaluation):")
    log_info(
        f"    Train: {holdout['train'][0] if holdout['train'] else 'N/A'} ~ {holdout['train'][-1] if holdout['train'] else 'N/A'} ({len(holdout['train'])} weeks)"
    )
    log_info(
        f"    Val:   {holdout['val'][0] if holdout['val'] else 'N/A'} ~ {holdout['val'][-1] if holdout['val'] else 'N/A'} ({len(holdout['val'])} weeks)"
    )
    log_info(
        f"    Test:  {holdout['test'][0] if holdout['test'] else 'N/A'} ~ {holdout['test'][-1] if holdout['test'] else 'N/A'} ({len(holdout['test'])} weeks)"
    )
    log_info("    ⚠️  Hold-out test data is NEVER seen during training!")

    # 使用 holdout split 作为默认 (用于快速实验)
    date_splits = holdout

    # Compute and save data quality
    log_info("\nComputing data quality statistics...")
    data_quality = compute_data_quality(all_snapshots, network_data)
    with open(output_dir / "data_quality.json", "w") as f:
        json.dump(data_quality, f, indent=2)
    log_info(f"  Data quality saved to: {output_dir / 'data_quality.json'}")
    log_info(f"  Mean nodes/week: {data_quality['summary']['mean_nodes_per_week']:.1f}")
    log_info(f"  Mean edges/week: {data_quality['summary']['mean_edges_per_week']:.1f}")

    # Save data quality to result manager
    save_result("data_quality", data_quality)

    config_payload = {
        "mode": args.mode,
        "epochs": config.epochs,
        "seed": config.seed,
        "forecast_horizons": config.forecast_horizons,
        "device": config.device,
        "output_dir": str(output_dir),
        "holdout_start": args.holdout_start,
        "min_train_weeks": args.min_train_weeks,
        "val_weeks": args.val_weeks,
        "test_weeks": args.test_weeks,
    }

    def prepare_mode_output(mode_name: str) -> Path:
        """Switch output directory for per-mode outputs when running --mode all."""
        if args.mode != "all":
            return output_dir

        mode_output_dir = output_dir / mode_name
        init_run_context(mode_output_dir)
        config.output_dir = str(mode_output_dir)
        save_result("config", {**config_payload, "mode": mode_name, "output_dir": str(mode_output_dir)})
        with open(mode_output_dir / "data_quality.json", "w") as f:
            json.dump(data_quality, f, indent=2)
        save_result("data_quality", data_quality)
        return mode_output_dir

    # Get snapshots by split
    date_to_snap = {s["date"]: s for s in all_snapshots}
    train_snaps = [date_to_snap[d] for d in date_splits["train"] if d in date_to_snap]
    val_snaps = [date_to_snap[d] for d in date_splits["val"] if d in date_to_snap]
    test_snaps = [date_to_snap[d] for d in date_splits["test"] if d in date_to_snap]

    run_frozen = args.mode in ["all", "frozen"]
    run_dexposure_fm = args.mode in ["all", "dexposure-fm", "deposure-fm"]
    run_roland = args.mode in ["all", "roland"]
    run_models = run_frozen or run_dexposure_fm or run_roland

    rolling_results = None
    if args.rolling and run_models:
        log_info("\n[Rolling Evaluation] Starting expanding window evaluation...")
        try:
            rolling_results = run_expanding_window_evaluation(
                config=config,
                split_result=split_result,
                date_to_snap=date_to_snap,
                run_frozen=run_frozen,
                run_dexposure_fm=run_dexposure_fm,
                run_roland=run_roland,
                save_predictions=args.save_predictions,
                output_dir=output_dir,
            )
            save_task_result("rolling_evaluation", rolling_results)
            log_info("[Rolling Evaluation] Completed and saved.")
        except Exception as e:
            log_error(f"[Rolling Evaluation] Failed: {e}")
            save_result("rolling_evaluation_error", str(e))

    all_results = {
        "temporal_split": {
            "method": "expanding_window_walk_forward",
            "n_folds": split_result["n_folds"],
            "holdout": {
                "train_weeks": len(date_splits["train"]),
                "val_weeks": len(date_splits["val"]),
                "test_weeks": len(date_splits["test"]),
                "train_range": f"{date_splits['train'][0]} ~ {date_splits['train'][-1]}"
                if date_splits["train"]
                else "N/A",
                "val_range": f"{date_splits['val'][0]} ~ {date_splits['val'][-1]}"
                if date_splits["val"]
                else "N/A",
                "test_range": f"{date_splits['test'][0]} ~ {date_splits['test'][-1]}"
                if date_splits["test"]
                else "N/A",
            },
            "config": split_result["config"],
        },
        "data_quality_summary": data_quality["summary"],
    }
    if rolling_results:
        all_results["rolling_folds"] = rolling_results

    # Run experiments based on mode with proper error handling
    if args.mode in ["all", "frozen"]:
        mode_output_dir = prepare_mode_output("frozen")
        log_info("\n" + "=" * 60)
        log_info("[Task I] GraphPFN-Frozen - Starting...")
        log_info("=" * 60)
        try:
            result = run_graphpfn_experiment(
                config,
                finetune=False,
                train_snaps=train_snaps,
                val_snaps=val_snaps,
                test_snaps=test_snaps,
                save_predictions=args.save_predictions,
                output_dir=mode_output_dir,
            )
            if result:
                all_results["graphpfn_frozen"] = result
                save_task_result("graphpfn_frozen", result)
                log_info("[Task I] GraphPFN-Frozen - Completed and saved.")
        except Exception as e:
            log_error(f"[Task I] GraphPFN-Frozen failed: {e}")
            log_error(traceback.format_exc())
            save_result("graphpfn_frozen_error", str(e))

    if args.mode in ["all", "dexposure-fm", "deposure-fm"]:
        mode_output_dir = prepare_mode_output("finetuned")
        log_info("\n" + "=" * 60)
        log_info("[Task I] DeXposure-FM - Starting...")
        log_info("=" * 60)
        try:
            result = run_graphpfn_experiment(
                config,
                finetune=True,
                train_snaps=train_snaps,
                val_snaps=val_snaps,
                test_snaps=test_snaps,
                save_predictions=args.save_predictions,
                output_dir=mode_output_dir,
            )
            if result:
                all_results["dexposure_fm"] = result
                save_task_result("dexposure_fm", result)
                log_info("[Task I] DeXposure-FM - Completed and saved.")
        except Exception as e:
            log_error(f"[Task I] DeXposure-FM failed: {e}")
            log_error(traceback.format_exc())
            save_result("dexposure_fm_error", str(e))

    if args.mode in ["all", "roland"]:
        mode_output_dir = prepare_mode_output("roland")
        log_info("\n" + "=" * 60)
        log_info("[Task I] ROLAND Baseline - Starting...")
        log_info("=" * 60)
        try:
            result = run_roland_experiment(
                config,
                train_snaps=train_snaps,
                val_snaps=val_snaps,
                test_snaps=test_snaps,
                save_predictions=args.save_predictions,
                output_dir=mode_output_dir,
            )
            all_results["roland"] = result
            save_task_result("roland", result)
            log_info("[Task I] ROLAND - Completed and saved.")
        except Exception as e:
            log_error(f"[Task I] ROLAND failed: {e}")
            log_error(traceback.format_exc())
            save_result("roland_error", str(e))

    if args.mode in ["all", "persistence"]:
        mode_output_dir = prepare_mode_output("persistence")
        log_info("\n" + "=" * 60)
        log_info("[Task I] Persistence Baseline (Naive) - Starting...")
        log_info("=" * 60)
        try:
            # Evaluate persistence baseline on test pairs for each horizon
            persistence_results = {"model": "Persistence", "results": {}}
            for horizon in config.forecast_horizons:
                test_pairs_h = build_week_pairs(test_snaps, config.neg_ratio, config.seed, horizon)
                if test_pairs_h:
                    metrics = evaluate_persistence_baseline(test_pairs_h)
                    persistence_results["results"][f"h{horizon}"] = metrics
                    log_info(f"  h={horizon}: AUPRC={metrics['exist']['auprc']:.4f}, "
                             f"AUROC={metrics['exist']['auroc']:.4f}, "
                             f"Weight MAE={metrics['weight']['mae']:.4f}, "
                             f"Node MAE={metrics['node']['mae']:.4f}")
            all_results["persistence"] = persistence_results
            save_task_result("persistence", persistence_results)
            log_info("[Task I] Persistence Baseline - Completed and saved.")
        except Exception as e:
            log_error(f"[Task I] Persistence Baseline failed: {e}")
            log_error(traceback.format_exc())
            save_result("persistence_error", str(e))

    if args.mode in ["all", "stats"]:
        log_info("\n" + "=" * 60)
        log_info("[Stats] Network Statistics - Starting...")
        log_info("=" * 60)
        try:
            result = run_network_statistics(config)
            all_results["network_stats"] = result
            save_task_result("network_stats", result)
            log_info("[Stats] Network Statistics - Completed and saved.")
        except Exception as e:
            log_error(f"[Stats] Network Statistics failed: {e}")
            log_error(traceback.format_exc())
            save_result("network_stats_error", str(e))

    # Save consolidated results without overwriting ResultManager's experiment_results.json.
    all_results_path = output_dir / "all_results.json"
    with open(all_results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log_info(f"\nAll results saved to: {all_results_path}")

    # Print comparison table
    if args.mode in ["all", "compare"]:
        horizons = config.forecast_horizons
        col_width = 14
        sep_len = 30 + len(horizons) * (col_width + 1)

        log_info("\n" + "=" * sep_len)
        log_info("COMPARISON TABLE - Multi-step Forecasting Results")
        log_info("=" * sep_len)

        header_cols = [f"h={h} AUPRC" for h in horizons]
        log_info(
            f"{'Model':<30} " + " ".join([f"{col:<{col_width}}" for col in header_cols])
        )
        log_info("-" * sep_len)

        for name, result in all_results.items():
            if isinstance(result, dict) and "results" in result:
                values = [
                    result["results"]
                    .get(f"h{h}", {})
                    .get("exist", {})
                    .get("auprc", float("nan"))
                    for h in horizons
                ]
                log_info(
                    f"{result.get('model', name):<30} "
                    + " ".join([f"{v:<{col_width}.4f}" for v in values])
                )

    log_info("\nExperiment completed. Available modes:")
    log_info("  --mode frozen        : GraphPFN-Frozen (Task I)")
    log_info("  --mode dexposure-fm  : DeXposure-FM (Task I) (alias: deposure-fm)")
    log_info("  --mode roland        : ROLAND baseline (Task I)")
    log_info("  --mode persistence   : Persistence baseline (naive)")
    log_info("  --mode stats         : Network statistics computation")
    log_info("  --mode compare       : Print comparison table")
    log_info("  --mode all           : Run all experiments")


if __name__ == "__main__":
    main()
