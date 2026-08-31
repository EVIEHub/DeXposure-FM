# DeXposure-FM arXiv v2 reproduction record

Target paper: arXiv:2602.03981v2, revised 30 June 2026.

## Evidence status

- The code in this repository runs Task I and Task II for horizons 1, 4, 8,
  and 12 weeks.
- `checkpoints/metrics-v2-paper.json` contains the values printed in the v2
  paper. The older `metrics-h1.json` belongs to a different h=1 run.
- The original h=1 and h=12 fine-tuned checkpoint provenance is incomplete.
- The current `dexposure-fm-h1.pt` is from an earlier run whose metrics do not
  equal the v2 h=1 table row.
- The file `dexposure-fm-h8-h12.pt` is an h=8 checkpoint with a legacy name. It
  must not be used as an exact h=12 checkpoint.
- A new run that produces all four checkpoints is a v2 reconstruction. It is
  not recovery of byte-identical original weights unless hashes from the
  original run are found.

## Fixed inputs

| Object | Path | SHA-256 |
| --- | --- | --- |
| Main network, 283 weekly snapshots | `data/historical-network_week_2020-03-30.json` | `aa330bbb8fbf99719fc85d49625d7df7bd68f2b042f5806ded080cec99bad3f8` |
| Spillover figure network | `data/historical-network_week_2025-07-01.json` | `d77920a7212847dd3cbfbbaabc8622c25e6523d2e7b2006d79d71b85851e1d0b` |
| Institution metadata | `data/meta_df.csv` | `a8306889fc4473972e843d8e847c9db68776cb86014746f1029fee574e254305` |
| GraphPFN base checkpoint | `checkpoints/graphpfn-v1.ckpt` | `5543a25b07b4be523490b7ef14adbb6c7a9ea763d93a4ef7a22584c0cee9a76d` |

Do not use a 171-snapshot copy of the main network that ends on 2023-06-26.
That file cannot evaluate the paper's 2025 holdout.

## Fixed run

```bash
uv sync --frozen
uv run python bin/download_dataset.py
bash run_v2_reproduction.sh
```

The Hugging Face repository is gated. Accept its access conditions and run
`hf auth login` before downloading the fixed inputs.

The script fixes horizons 1, 4, 8, and 12, holdout start 2025-01-01,
validation window 24 weeks, 20 epochs, and seed 42.

## Task I values printed in v2

| Model | h | AUROC | AUPRC | Edge MAE | Edge RMSE | Node MAE | Node RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeXposure-FM | 1 | .995 | .972 | 2.465 | 3.388 | .056 | .400 |
| DeXposure-FM | 4 | .995 | .973 | 2.489 | 3.424 | .140 | .680 |
| DeXposure-FM | 8 | .994 | .967 | 2.554 | 3.509 | .229 | .890 |
| DeXposure-FM | 12 | .993 | .967 | 2.648 | 3.606 | .286 | 1.046 |
| Frozen GraphPFN | 1 | .988 | .938 | 3.260 | 4.383 | .059 | .401 |
| Frozen GraphPFN | 4 | .988 | .940 | 3.189 | 4.049 | .142 | .682 |
| Frozen GraphPFN | 8 | .987 | .938 | 3.169 | 4.064 | .215 | .896 |
| Frozen GraphPFN | 12 | .986 | .936 | 3.136 | 4.130 | .324 | 1.056 |
| ROLAND | 1 | .961 | .865 | 3.240 | 4.264 | .060 | .403 |
| ROLAND | 4 | .962 | .868 | 3.242 | 4.162 | .141 | .684 |
| ROLAND | 8 | .961 | .867 | 3.213 | 4.180 | .221 | .895 |
| ROLAND | 12 | .961 | .866 | 3.195 | 4.177 | .279 | 1.058 |
| Persistence | 1 | .763 | .604 | 2.487 | 4.296 | .057 | .403 |
| Persistence | 4 | .782 | .635 | 2.372 | 4.138 | .138 | .685 |
| Persistence | 8 | .749 | .580 | 2.618 | 4.400 | .213 | .899 |
| Persistence | 12 | .762 | .603 | 2.541 | 4.304 | .272 | 1.065 |

## Task II values printed in v2

| Horizon | Mean contagion change | Worst-20 change | Scenario win rate |
| ---: | ---: | ---: | ---: |
| 1 | -0.95 | +3.04 | 100% |
| 4 | -1.71 | +1.42 | 83% |
| 8 | -0.17 | +2.60 | 100% |
| 12 | -0.55 | +2.21 | 83% |

## Acceptance rule

A reconstructed run matches the v2 tables only when every reported metric
rounds to the same displayed value and all four checkpoint files are nonempty.
If a value differs, publish the result as a non-matching reconstruction. Do not
replace the v2 release silently.
