# DeXposure-FM sequential run, 19 September 2026

Release version: `fm-sequential-20260919-v1`
GitHub branch: `reproduce-sequential-20260919`
Hugging Face branch: `reproduce-sequential-20260919`

This release contains the h1, h4, h8 and h12 checkpoints, the frozen training
configuration and source, training logs, saved predictions, and result-reproduction
scripts. Task I evaluates the six forecasting metrics in Table 3. The downstream
release evaluates Table 4 and Figures 2–3 with a uniform TVL loss cap.

## Reproduce the metrics and figures

Clone the release branch and run the CPU replay command. Install
[uv](https://docs.astral.sh/uv/) if needed. The Hugging Face model uses automatic
access approval; accept the model access conditions and authenticate with
`uvx --from huggingface-hub hf auth login` before the first download.

```bash
git clone --branch reproduce-sequential-20260919 https://github.com/EVIEHub/DeXposure-FM.git
cd DeXposure-FM
PYTHONHASHSEED=0 MPLBACKEND=Agg uv run --no-project \
  --with-requirements reproduction/sequential_20260919/requirements-replay.txt \
  python reproduction/sequential_20260919/reproduce.py --download
```

The script checks input and prediction SHA-256 values, recomputes all 24 Task I
metrics from saved predictions, rebuilds the observed/predicted/realized graphs,
and runs the corrected contagion evaluator on 107 forecast origins and 321 shock
scenarios. It checks the recomputed metrics against this release and exports
Figures 2–3 as PNG and PDF. This command replays saved predictions; it does not
retrain the model or run fresh neural-network inference.

Outputs are written to `output/sequential-20260919/evaluation/`:

- `task1_metrics.json`
- `exp2_predictive_contagion.json`
- `verification.json`
- `figures/fig_contagion_comparison.{png,pdf}`
- `figures/fig_contagion_advantage.{png,pdf}`

## Checkpoints and training records

The immutable HF revision is recorded in [release.json](release.json).
Under `runs/20260919_fm_sequential_h1_h4_h8_h12/`:

- `checkpoints/main2025_v2_hremaining/dexposure-fm-h{1,4,8,12}.pt` contains the four released checkpoints.
- `checkpoints/main2025_v2_hremaining/SHA256SUMS` verifies their contents.
- `checkpoints/main2025_v2_hremaining_train/` contains per-stage model and optimizer
  states, histories, saved predictions, the frozen selection, and `source.tar.gz`.
- `logs/` contains the training log and input checksums.

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    'EVIEHub/DeXposure-FM',
    'runs/20260919_fm_sequential_h1_h4_h8_h12/checkpoints/main2025_v2_hremaining/dexposure-fm-h4.pt',
    revision='fm-sequential-20260919-v1',
)
```

## Training procedure

A single parameter configuration is used for all horizons. Training proceeds
h1 → h4 → h8 → h12, carrying the selected model and matching Adam, AMP and RNG
state into the next stage. Each stage selects its checkpoint using validation
AUPRC. Selection is frozen before evaluating the 2025 test period.

| Setting | Value |
|---|---|
| Seed | 42 |
| Train / validation / test snapshots | 226 / 24 / 33 |
| Test start | 2025-01-01 holdout boundary |
| Head / encoder learning rate | 0.0005 / 0.00005 |
| Existence / edge weight / node loss weights | 2 / 0.5 / 20 |
| Negative-to-positive sampling ratio | 5:1 |
| Maximum epochs per stage / early-stop patience | 40 / 5 |
| Completed epochs, h1/h4/h8/h12 | 40 / 6 / 6 / 8 |
| Selected epoch, h1/h4/h8/h12 | 39 / 1 / 1 / 3 |

Frozen files: [plan.json](plan.json), [split.json](split.json),
[selection_frozen.json](selection_frozen.json), [training.log](training.log).
Training implementation: `cloud/train_src/sequential_fm.py`,
`cloud/train_src/tune_fm.py`, and `cloud/train_src/run_full_experiment.py`.
The recorded cloud launch is `bash cloud/train.sh bash cloud/fm_sequential_run.sh`;
the cloud controller's credentials and artifact repository must be configured for
the operator's account. The CPU replay command above only needs the public release.

## Task I results

Regression errors use the training evaluator's log-scale targets.

| Horizon | Source | AUPRC | AUROC | Edge MAE | Edge RMSE | Node MAE | Node RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| h1 | Paper | 0.972 | 0.995 | 2.465 | 3.388 | 0.056 | 0.400 |
| h1 | This run | 0.973 | 0.995 | 2.219 | 3.248 | 0.053 | 0.400 |
| h4 | Paper | 0.973 | 0.995 | 2.489 | 3.424 | 0.140 | 0.680 |
| h4 | This run | 0.971 | 0.995 | 2.219 | 3.269 | 0.142 | 0.677 |
| h8 | Paper | 0.967 | 0.994 | 2.554 | 3.509 | 0.229 | 0.890 |
| h8 | This run | 0.968 | 0.994 | 2.337 | 3.461 | 0.216 | 0.896 |
| h12 | Paper | 0.967 | 0.993 | 2.648 | 3.606 | 0.286 | 1.046 |
| h12 | This run | 0.965 | 0.993 | 2.360 | 3.533 | 0.275 | 1.061 |

These results are close to the paper's Table 3 values, with differences across
metrics. Exact values and comparisons are included in the result files.

## Corrected stress-test results

Received losses are capped at `max(0, TVL)` for every node, including zero TVL.
The same rule is applied to the persistence baseline, predicted graph and realized
graph. The worst 20% is selected using the corrected baseline errors, pooled over
shock scenarios within each horizon. Positive ΔMAE indicates lower model error.

| Horizon | ΔMAE overall (pp) | ΔMAE worst 20% (pp) | Win rate, worst 20% |
|---|---:|---:|---:|
| h1 | -0.116 | +0.173 | 70.0% |
| h4 | -0.094 | +0.221 | 61.1% |
| h8 | -0.013 | +0.625 | 93.3% |
| h12 | -0.153 | +0.011 | 76.9% |

Persistence has lower overall error. The model has lower mean error in the
baseline's worst-20% subset; h12 is nearly tied. These numbers use the corrected
simulator, so they are reported separately from the paper's original Table 4.
The release is a single-seed run, and the 2025 benchmark had been inspected in
prior experiments.

[Figure 2](results/downstream_zero_tvl_fixed_20260921/figures/fig_contagion_comparison.png)
| [Figure 3](results/downstream_zero_tvl_fixed_20260921/figures/fig_contagion_advantage.png)

## Evaluation record

`results/downstream_validation_20260921/` records the initial downstream checks
and diagnostic substitutions. `results/downstream_zero_tvl_fixed_20260921/`
records the corrected three-graph evaluation, per-scenario outputs and checksums.
Historical scripts preserve their original paths; use `reproduce.py` as the
portable entrypoint.

Table 5's eight data-summary values matched at the printed precision. Figure 4's
225 rendered matrix cells matched. Figure 5's saved comparisons and plots are
included. Figure 6 requires separate pre-event models and is not an output of
this 2025-holdout run.

Boundary and training-state tests:

```bash
uv run --no-project --with numpy python paper/tests/test_contagion_zero_tvl.py
# In the training environment:
python -m unittest paper.tests.test_fm_sequential
```
