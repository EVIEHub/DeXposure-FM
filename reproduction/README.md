# Four-horizon reconstruction release

Status: reproduction candidate. See [validation status](VALIDATION.md).
The new evaluation-only entrypoint has not yet completed an independent
four-horizon acceptance run. The metrics below are saved training-run results.

Release: `reconstruction-20260905`. This package supports evaluating the
September checkpoints and retraining the same implementation on the fixed
2025 holdout. It reports observed performance rather than promising a match
to every published number. Historical February weights remain at the model
repository root; see ../docs/V2_HISTORICAL_CHECKPOINT_AUDIT.md.

## Install and download

Use Linux x86-64, Python 3.12.9, PyTorch 2.2.1 with CUDA 12.1, and DGL 2.1.0
with CUDA 12.1. The repository uv.lock pins this environment. The September
h4/h8/h12 training used an A100 PCIe 40 GB. CPU evaluation is also supported,
but the locked CUDA environment is Linux-specific and CPU evaluation is slower.
The evaluation-only CPU path uses PyTorch's fused attention kernel (or the
existing query-chunked implementation when a mask is present) to avoid
materializing a full attention-score matrix. It does not modify the weights.
The training source and CUDA evaluation path are unchanged.

```bash
git clone --branch reconstruction-20260905 https://github.com/EVIEHub/DeXposure-FM.git
cd DeXposure-FM
uv sync --frozen
# If HF requests access, accept the model repository conditions and authenticate:
uv run huggingface-cli login
uv run python -m reproduction.prepare
```

The preparation command uses the exact revisions and hashes in release.json.
It downloads inputs to data/ and new models to checkpoints/reconstruction-20260905/.
It verifies existing files and refuses to replace a file with a different hash.
If that happens, move the conflicting file aside and run the command again.
No credential is embedded in this release.

## Evaluate the released checkpoints, without training

```bash
uv run python -m reproduction.evaluate --device cuda
```

This evaluates h1/h4/h8/h12 on the full holdout using the same preprocessing,
negative sampling, prediction and metric functions as the September training.
It strictly checks each checkpoint's horizon, schema and split. Outputs are:

- output/reconstruction-evaluation/metrics.json: newly computed metrics.
- output/reconstruction-evaluation/evaluation.json: signed numeric differences
  against expected_metrics.json (September) and paper_metrics.json (paper).
- A timestamped experiment log in the output directory.

Use `--save-predictions` to also export edge and node prediction CSVs. These
can be large. Use `--horizons 4` for a single horizon, `--device cpu` for CPU,
and a new `--output-dir` for each attempt. Existing output directories are
rejected to preserve previous results. A successful exit means the requested
evaluation finished with finite metrics; it does not mean the paper matched.
Hardware and software differences can affect floating-point results. Inspect
the recorded deltas, and report deviations rather than choosing only favorable
metrics. We have not established a multi-seed statistical equivalence bound.

## Retrain with the same source

```bash
uv run python -m reproduction.runtime \
  --mode dexposure-fm --holdout-start 2025-01-01 --val-weeks 24 \
  --epochs 20 --seed 42 --horizons 1,4,8,12 \
  --save-predictions --output-dir output/reconstruction-retrain
```

Run this on a GPU you have provisioned. It is not necessary for checkpoint
evaluation. runtime.py is byte-identical to the September training entrypoint;
its SHA-256 is d2e47cb6ce33aff1d8c1ecba0de3494f84c3596ee1cefb45e9e29825b2be4a0e.
Fresh training writes separate best_model_h1.pt, best_model_h4.pt,
best_model_h8.pt and best_model_h12.pt, plus metrics, feature schema and logs.
Preserve the full output directory. Use a new output path per run.

## Actual reconstruction settings

The executable runtime and fixed inputs govern this September rerun. The older
model card's example configuration is not the configuration of this release.

| Setting | Value |
| --- | --- |
| Snapshots | 283, 2020-03-23 through 2025-08-18 |
| Split | 226 train / 24 validation / 33 test snapshots |
| Test holdout | From 2025-01-01 |
| Test pairs h1/h4/h8/h12 | 32 / 29 / 25 / 21 |
| Optimizer | Adam, weight decay 0.0001 |
| Encoder / head learning rates | 0.00005 / 0.0005 |
| Seed / maximum epochs | 42 / 20 |
| Validation | Every epoch, select highest validation AUPRC, patience 3 |
| Negative-sampling setting | neg_ratio=5; sampling implementation in build_week_pairs/sample_negatives |
| Core loss weights | Edge existence 2.0; edge weight 0.5; node change 20.0 |
| Core losses | BCE with logits; SmoothL1 edge residual; SmoothL1 node change |
| Auxiliary loss weights | All four are 0 |
| Gradient clipping | Norm 1.0 |
| AMP | Enabled during CUDA training |

The sampled negative count can be limited by the available non-edges. Metrics
use the same sampling implementation as training; changing the candidate set
can change AUPRC and must be documented.

## Paper versus measured September results

Each cell is paper / September rerun, rounded here to three decimals. Full
precision rerun values are in expected_metrics.json. Higher AUROC/AUPRC is
better; lower MAE/RMSE is better.

| h | AUROC | AUPRC | Edge MAE | Edge RMSE | Node MAE | Node RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | .995 / .996 | .972 / .976 | 2.465 / 2.211 | 3.388 / 3.260 | .056 / .054 | .400 / .399 |
| 4 | .995 / .994 | .973 / .969 | 2.489 / 2.274 | 3.424 / 3.345 | .140 / .137 | .680 / .678 |
| 8 | .994 / .993 | .967 / .962 | 2.554 / 2.381 | 3.509 / 3.563 | .229 / .215 | .890 / .890 |
| 12 | .993 / .991 | .967 / .956 | 2.648 / 2.391 | 3.606 / 3.529 | .286 / .276 | 1.046 / 1.049 |

This release covers the DeXposure-FM rows of Task I. It does not certify the
baseline rows, Task II, or all paper figures. The paper's original results and
the September results are retained separately. A future paper revision can
cite the new results explicitly; this release does not alter arXiv v2.
