# DeXposure-FM: Reproducing Experiments

This repo provides two main experiment suites:

- **Task I (multi-step forecasting)**: `run_full_experiment.py`
- **Task II (forward-looking risk analysis)**: `run_task2_model_based.py`

All commands below assume you are at the repo root and have run:

```
uv sync
```

If you did not fetch the dataset via Git LFS, download it first:

```
uv run python bin/download_dataset.py
```

## Task I: Multi-step forecasting (`run_full_experiment.py`)

Run a minimal CPU-friendly baseline (network statistics only):

```
uv run python run_full_experiment.py --mode stats
```

Run the full comparison suite (can be GPU-heavy):

```
uv run python run_full_experiment.py --mode compare
```

Key modes:

- `--mode frozen` : GraphPFN-Frozen (encoder frozen; train a small head)
- `--mode dexposure-fm` : DeXposure-FM (end-to-end fine-tuning)
- `--mode roland` : ROLAND temporal baseline
- `--mode stats` : network statistics only
- `--mode compare` : run Frozen + DeXposure-FM + ROLAND (recommended for paper tables)

Outputs are written under `output/` (timestamped runs by default).

## Task II: Forward-looking risk analysis (`run_task2_model_based.py`)

Run the full Task II suite:

```
uv run python run_task2_model_based.py --experiment all
```

Regenerate plots only (no model inference/training required):

```
uv run python run_task2_model_based.py --plot-only --output-dir output/task2_model_based
```

Available experiments (see `--help` for the full list):

- `forward_risk`
- `predictive_contagion`
- `early_warning`
- `sis_sensitivity` (minimal robustness check)

