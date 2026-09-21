#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization for DeXposure-FM

This script performs hyperparameter search using Optuna with:
- Bayesian optimization (TPE sampler)
- Pruning for early stopping
- Strict temporal train/val/test split

Usage:
    python run_optuna_search.py --n-trials 50
    python run_optuna_search.py --n-trials 100 --timeout 21600
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add project paths
GRAPHPFN_ROOT = Path(__file__).parent
sys.path.insert(0, str(GRAPHPFN_ROOT))

from run_full_experiment import (
    ExperimentConfig,
    set_seed,
    load_network_data,
    load_metadata,
    build_snapshot,
    build_week_pairs,
    train_graphpfn_epoch,
    predict_graphpfn,
    evaluate_predictions,
    GRAPHPFN_AVAILABLE,
)

if GRAPHPFN_AVAILABLE:
    from run_full_experiment import (
        load_graphpfn_encoder,
        GraphPFNLinkPredictor,
    )
    import torch


# ============== Strict Temporal Split ==============

def get_temporal_split(
    all_dates: list,
    train_end_year: int = 2024,
    test_start_year: int = 2025
) -> Dict[str, list]:
    """
    Create strict temporal split where test data is completely unseen.

    Args:
        all_dates: List of all snapshot dates (YYYY-MM-DD format)
        train_end_year: Last year for training (inclusive)
        test_start_year: First year for testing

    Returns:
        Dictionary with train_dates, val_dates, test_dates
    """
    train_dates = []
    val_dates = []
    test_dates = []

    for date in sorted(all_dates):
        year = int(date.split("-")[0])

        if year < train_end_year:
            train_dates.append(date)
        elif year == train_end_year:
            # Use last 20% of train_end_year as validation
            val_dates.append(date)
        else:  # year >= test_start_year
            test_dates.append(date)

    # Split val_dates: first 80% to train, last 20% to val
    if val_dates:
        split_idx = int(len(val_dates) * 0.8)
        train_dates.extend(val_dates[:split_idx])
        val_dates = val_dates[split_idx:]

    return {
        "train": train_dates,
        "val": val_dates,
        "test": test_dates,
    }


# ============== Objective Function ==============

def create_objective(
    snapshots: list,
    date_splits: Dict[str, list],
    base_config: ExperimentConfig,
    device: torch.device
):
    """Create Optuna objective function."""

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
        neg_ratio = trial.suggest_int("neg_ratio", 3, 10)
        exist_loss_weight = trial.suggest_float("exist_loss_weight", 0.5, 2.0)
        weight_loss_weight = trial.suggest_float("weight_loss_weight", 0.5, 2.0)
        node_loss_weight = trial.suggest_float("node_loss_weight", 0.1, 1.0)
        epochs = trial.suggest_int("epochs", 5, 15)

        # Create config
        config = ExperimentConfig(
            lr=lr,
            weight_decay=weight_decay,
            hidden_dim=hidden_dim,
            neg_ratio=neg_ratio,
            exist_loss_weight=exist_loss_weight,
            weight_loss_weight=weight_loss_weight,
            node_loss_weight=node_loss_weight,
            epochs=epochs,
            device=str(device),
        )

        try:
            # Get snapshots by date
            date_to_snap = {s["date"]: s for s in snapshots}

            train_snaps = [date_to_snap[d] for d in date_splits["train"] if d in date_to_snap]
            val_snaps = [date_to_snap[d] for d in date_splits["val"] if d in date_to_snap]

            if len(train_snaps) < 10 or len(val_snaps) < 2:
                return float("-inf")

            # Build pairs
            train_pairs = build_week_pairs(train_snaps, config.neg_ratio, base_config.seed, horizon=1)
            val_pairs = build_week_pairs(val_snaps, config.neg_ratio, base_config.seed, horizon=1)

            if not train_pairs or not val_pairs:
                return float("-inf")

            # Load model
            encoder = load_graphpfn_encoder(base_config.checkpoint_path, device)
            embed_dim = encoder.tfm.embed_dim
            model = GraphPFNLinkPredictor(encoder, embed_dim, config.hidden_dim).to(device)

            # Fine-tune
            for p in model.encoder.parameters():
                p.requires_grad = True

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay
            )

            # Train with pruning
            best_val_auprc = 0.0
            for epoch in range(config.epochs):
                train_graphpfn_epoch(model, train_pairs, optimizer, config, finetune_encoder=True)

                # Evaluate on validation
                val_preds = predict_graphpfn(model, val_pairs, config)
                val_metrics = evaluate_predictions(val_preds)
                val_auprc = val_metrics["exist"]["auprc"]

                if not np.isnan(val_auprc):
                    best_val_auprc = max(best_val_auprc, val_auprc)

                # Report for pruning
                trial.report(val_auprc, epoch)

                if trial.should_prune():
                    raise optuna.TrialPruned()

            return best_val_auprc

        except Exception as e:
            print(f"Trial failed with error: {e}")
            return float("-inf")

    return objective


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Search")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--study-name", type=str, default="dexposure_fm", help="Study name")
    parser.add_argument("--output-dir", type=str, default="output/optuna", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-end-year", type=int, default=2024, help="Last year for training")
    parser.add_argument("--test-start-year", type=int, default=2025, help="First year for testing")
    args = parser.parse_args()

    if not GRAPHPFN_AVAILABLE:
        print("GraphPFN not available. Please check installation.")
        sys.exit(1)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    base_config = ExperimentConfig(seed=args.seed)

    print("Loading data...")
    meta_category, category_list, category_to_idx = load_metadata(base_config.meta_path)
    network_data = load_network_data(base_config.data_path)
    all_dates = sorted(network_data.keys())

    print(f"Date range: {all_dates[0]} to {all_dates[-1]}")

    # Build snapshots
    snapshots = [
        build_snapshot(date, network_data[date], meta_category, category_to_idx, category_list)
        for date in all_dates
    ]

    # Get temporal split
    date_splits = get_temporal_split(
        all_dates,
        train_end_year=args.train_end_year,
        test_start_year=args.test_start_year
    )

    print(f"\nTemporal Split:")
    print(f"  Train: {len(date_splits['train'])} snapshots ({date_splits['train'][0] if date_splits['train'] else 'N/A'} to {date_splits['train'][-1] if date_splits['train'] else 'N/A'})")
    print(f"  Val: {len(date_splits['val'])} snapshots ({date_splits['val'][0] if date_splits['val'] else 'N/A'} to {date_splits['val'][-1] if date_splits['val'] else 'N/A'})")
    print(f"  Test: {len(date_splits['test'])} snapshots ({date_splits['test'][0] if date_splits['test'] else 'N/A'} to {date_splits['test'][-1] if date_splits['test'] else 'N/A'})")

    # Create study
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    # Create objective
    objective = create_objective(snapshots, date_splits, base_config, device)

    # Run optimization
    print(f"\nStarting Optuna optimization with {args.n_trials} trials...")
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    # Results
    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (val AUPRC): {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results
    results = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "timestamp": datetime.now().isoformat(),
        "temporal_split": {
            "train_dates": len(date_splits["train"]),
            "val_dates": len(date_splits["val"]),
            "test_dates": len(date_splits["test"]),
            "train_end_year": args.train_end_year,
            "test_start_year": args.test_start_year,
        },
    }

    with open(output_dir / "best_params.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save all trials
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": str(trial.state),
        })

    with open(output_dir / "all_trials.json", "w") as f:
        json.dump(trials_data, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    # Parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        print(f"\nParameter Importance:")
        for key, value in sorted(importance.items(), key=lambda x: -x[1]):
            print(f"  {key}: {value:.4f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
