# DeXposure-FM

DeXposure-FM is a time-series, graph foundation model for **measuring and
forecasting inter-protocol credit exposure on DeFi networks**. It supports:

- **Edge existence prediction** (link prediction)
- **Edge weight prediction** (exposure size)
- **Node TVL change prediction**
- **Macroprudential tools** (SIS / sector spillovers / contagion)

This repository is the **core runnable release**

Paper: [http://fengxianghe.github.io/paper/DeXposure-FM.pdf](http://fengxianghe.github.io/paper/DeXposure-FM.pdf)

Developers: [Aijie Shu](https://www.linkedin.com/in/aijie-shu-5420047a/), [Wenbin Wu](https://www.linkedin.com/in/wbwe/), [Gbenga Ibikunle](https://www.business-school.ed.ac.uk/staff/gbenga-ibikunle), [Fengxiang He](http://fengxianghe.github.io)

---

## Contents

- [Quickstart](#quickstart)
- [Environment & Hardware](#environment--hardware)
- [Dataset](#dataset)
- [Checkpoints](#checkpoints)
- [Training / Experiments / Tools](#training--experiments--tools)
- [CLI Reference](#cli-reference)
- [Outputs](#outputs)
- [Reproduction Notes](#reproduction-notes)
- [GraphPFN Base](#graphpfn-base)
- [FAQ](#faq)
- [License & Acknowledgments](#license--acknowledgments)

---

## Quickstart

**1) Install (uv)**

```
uv sync
```

**2) Dataset (Git LFS or script)**

```
git lfs install
git lfs pull
# or
uv run python bin/download_dataset.py
```

**3) Minimal CPU‑friendly run**

```
uv run python run_full_experiment.py --mode stats
```

**4) Macroprudential tools (Observed mode)**

```
uv run python run_macroprudential_tools.py observed \
  --date 2025-06-30 \
  --data-path data/historical-network_week_2025-07-01.json \
  --contagion \
  --output-dir output/macro-tools
```

You can also use the `Makefile` shortcuts (`make help`).

---

## Environment & Hardware

**Recommended**

- Python **3.12.9**
- CUDA **12.1**
- PyTorch **2.2.1+cu121**
- DGL **2.1.0+cu121**

**Recommended hardware**

- **H100** (full training / paper‑level runs)

**CPU‑only runnable**

- `run_full_experiment.py --mode stats`
- `run_macroprudential_tools.py observed`
- `run_task2_model_based.py --plot-only`

---

## Dataset

Dataset download helper: `bin/download_dataset.py`  
Default target directory: `data/`.

Key files:

- `data/historical-network_week_2020-03-30.json` (~1.1GB)
- `data/historical-network_week_2025-07-01.json` (~76MB)
- `data/meta_df.csv`
- `data/mapping/*.json`
- `data/network_data/*.csv`

---

## Checkpoints

This repository already includes the released weights under `checkpoints/`.

- Model card: `checkpoints/README.md`
- Metrics: `checkpoints/metrics-*.json`
- Weights:
  - `checkpoints/dexposure-fm-h1.pt`
  - `checkpoints/dexposure-fm-h4.pt`
  - `checkpoints/dexposure-fm-h8-h12.pt`
  - `checkpoints/graphpfn-frozen-all-horizons.pt`
  - `checkpoints/graphpfn-v1.ckpt`
  - `checkpoints/LimiX-16M.ckpt`

---

## Training / Experiments / Tools

### 1) Task I: Multi‑step forecasting (`run_full_experiment.py`)

**CPU stats only**

```
uv run python run_full_experiment.py --mode stats
```

**Full comparison (GPU)**

```
uv run python run_full_experiment.py --mode compare
```

### 2) Task II: Forward‑looking risk analysis (`run_task2_model_based.py`)

```
uv run python run_task2_model_based.py --experiment all
```

### 3) Macroprudential tools (`run_macroprudential_tools.py`)

**Observed**

```
uv run python run_macroprudential_tools.py observed --date 2025-06-30 \
  --data-path data/historical-network_week_2025-07-01.json
```

**Predict**

```
uv run python run_macroprudential_tools.py predict --date 2025-06-30 \
  --horizon 4 --device cuda \
  --data-path data/historical-network_week_2025-07-01.json
```

---

## CLI Reference

### `run_full_experiment.py`

| Flag | Default | Description |
|---|---|---|
| `--mode` | `all` | all / frozen / dexposure-fm / roland / persistence / stats / compare |
| `--output-dir` | `None` | Output directory |
| `--epochs` | `20` | Training epochs |
| `--patience` | `3` | Early stopping patience |
| `--early-stop-metric` | `auprc` | Early stop metric (auprc/auroc) |
| `--val-eval-every` | `1` | Validate every N epochs |
| `--seed` | `42` | Random seed |
| `--holdout-start` | `2025-01-01` | Test set start date |
| `--min-train-weeks` | `104` | Minimum training weeks |
| `--val-weeks` | `24` | Validation window (weeks) |
| `--test-weeks` | `8` | Test window (weeks) |
| `--step-weeks` | `8` | Rolling step (weeks) |
| `--rolling` | `False` | Walk‑forward evaluation |
| `--save-predictions` | `False` | Save CSV predictions |
| `--horizons` | `1,4,8,12` | Forecast horizons |
| `--gradient-clip-norm` | `1.0` | Gradient clipping |
| `--verbose` | `False` | Verbose logging |

### `run_task2_model_based.py`

| Flag | Default | Description |
|---|---|---|
| `--experiment` | `all` | all / forward_risk / predictive_contagion / early_warning / sis_sensitivity |
| `--epochs` | `20` | Training epochs |
| `--seed` | `42` | Random seed |
| `--device` | `None` | cpu/cuda |
| `--force-retrain` | `False` | Force retraining |
| `--frozen` | `False` | GraphPFN‑Frozen |
| `--output-dir` | `output/task2_model_based` | Output directory |
| `--quick` | `False` | Quick smoke‑test |
| `--forward-horizons` | `1,4,8,12` | forward_risk horizons |
| `--contagion-horizons` | `1,4,8,12` | predictive_contagion horizons |
| `--shared-model-h1` | `False` | Train h1 only and reuse |
| `--max-train-pairs` | `0` | Limit training pairs |
| `--max-forward-pairs` | `0` | Limit forward test pairs |
| `--max-contagion-samples` | `10` | Limit contagion samples |
| `--plot-only` | `False` | Plot only |
| `--no-plot` | `False` | Skip plotting |
| `--reuse-results` | `False` | Reuse cached results |

### `run_macroprudential_tools.py`

**Common flags**

| Flag | Default | Description |
|---|---|---|
| `--data-path` | `ExperimentConfig.data_path` | Network JSON |
| `--meta-path` | `ExperimentConfig.meta_path` | Metadata CSV |
| `--top-k` | `20` | SIS Top‑K |
| `--full-sis` | `False` | Full SIS output |
| `--spillover-matrix` | `False` | Full spillover matrix |
| `--contagion` | `False` | Contagion scenarios |
| `--output` | `""` | Output file |
| `--output-dir` | `output/macroprudential_tools` | Output directory |

**Observed**

| Flag | Required | Description |
|---|---|---|
| `--date` | Yes | Snapshot date (YYYY‑MM‑DD) |

**Predict**

| Flag | Default | Description |
|---|---|---|
| `--date` | Required | Anchor date |
| `--horizon` | Required | Forecast horizon (weeks) |
| `--edge-threshold` | `0.5` | Edge probability threshold |
| `--epochs` | `20` | Training epochs |
| `--seed` | `42` | Random seed |
| `--device` | `None` | cpu/cuda |
| `--frozen` | `False` | Frozen encoder |
| `--force-retrain` | `False` | Force retraining |
| `--model-cache-dir` | `output/model_cache` | Cache dir |
| `--cache-tag` | `None` | Cache tag |
| `--train-cutoff` | `""` | Train cutoff date |
| `--max-train-pairs` | `0` | Limit training pairs |

---

## Outputs

**Task I**

- `output/YYYY-MM-DD_HHMMSS/`
  - `frozen/`, `finetuned/`, `roland/`
  - `report.json`, `predictions/*.csv`, `*.log`

**Task II**

- `output/task2_model_based/`
  - `exp1_forward_risk.json`
  - `exp2_predictive_contagion.json`
  - `exp3_early_warning.json`
  - `fig_*.pdf`

**Macroprudential tools**

- `output/macroprudential_tools/observed_*.json`
- `output/macroprudential_tools/predict_*.json`

---

## Reproduction Notes

Additional commands and details:

- `docs/dexposure_fm_experiments.md`

---

## GraphPFN Base

If you want to reproduce the original GraphPFN baseline:

```
uv run bin/go.py exp/graphpfn-eval/finetune/raw/tolokers-2/tuning.toml --force
```

GraphPFN pretraining requires graph generation (see `bin/prior/README.md`).

---

## FAQ

**DGL CUDA not available?**  
Check CUDA version and the DGL wheel (default cu121).

**OOM / GPU memory issues?**  
Use `--quick`, `--max-train-pairs`, or `--frozen`.

**Large JSON loading is slow?**  
`ijson` streaming is enabled in `run_full_experiment.py`.

---

## License & Acknowledgments

Apache‑2.0.  
Third‑party components: TabICL, LimiX (see `NOTICE` and `LICENSES/`).
