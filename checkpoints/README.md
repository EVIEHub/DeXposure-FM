---
license: apache-2.0
language:
  - en
tags:
  - graph-neural-network
  - defi
  - financial-networks
  - link-prediction
  - time-series
  - credit-exposure
  - pytorch
library_name: pytorch
pipeline_tag: graph-ml
datasets:
  - custom
metrics:
  - auprc
  - auroc
  - mae
  - rmse
---

# DeXposure-FM

DeXposure-FM forecasts weekly credit-exposure graphs between DeFi protocols.
The model predicts edge existence, edge exposure weight, and node TVL change.

Paper: [arXiv:2602.03981v2](https://arxiv.org/abs/2602.03981)

Code: [EVIEHub/DeXposure-FM](https://github.com/EVIEHub/DeXposure-FM)

## Important v2 artifact status

The v2 manuscript was revised on 30 June 2026.
`metrics-v2-paper.json` records the manuscript values. The older
`metrics-h1.json` belongs to a different h=1 run. The hosted checkpoints are not
a complete archive of all four original v2 fine-tuned weights.

| Hosted file | Proven horizon | v2 evidence status |
| --- | ---: | --- |
| `dexposure-fm-h1.pt` | 1 | Earlier run. Its recorded AUPRC 0.978 and AUROC 0.996 do not equal the v2 h=1 row. |
| `dexposure-fm-h4.pt` | 4 | Metrics agree with the v2 h=4 row. |
| `dexposure-fm-h8-h12.pt` | 8 | Legacy filename. Use as h=8 only, not as an exact h=12 checkpoint. |
| `graphpfn-frozen-all-horizons.pt` | 1, 4, 8, 12 | Frozen baseline checkpoint. |

The exact original h=12 fine-tuned checkpoint is not currently hosted. A new
four-horizon run must be labelled a **v2 reconstruction** unless an original
checkpoint hash is recovered.

## v2 reported Task I results

These are the values printed in arXiv v2. They describe the paper experiment,
not every checkpoint currently present in this repository.

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

ROLAND and Persistence baseline rows are recorded in
[`docs/V2_REPRODUCIBILITY.md`](https://github.com/EVIEHub/DeXposure-FM/blob/main/docs/V2_REPRODUCIBILITY.md).

## Download

```bash
huggingface-cli download EVIEHub/DeXposure-FM --local-dir checkpoints
```

The repository is gated. Accept its access conditions and authenticate before
downloading.

## Reproduce the v2 experiment

```bash
git clone https://github.com/EVIEHub/DeXposure-FM.git
cd DeXposure-FM
uv sync --frozen
uv run python bin/download_dataset.py
bash run_v2_reproduction.sh
```

The fixed experiment uses horizons 1, 4, 8, and 12, holdout start 2025-01-01,
validation window 24 weeks, 20 epochs, and seed 42.

A run matches the paper only if every metric rounds to the same displayed value
and all four reconstructed checkpoints are present and nonempty.

## Intended use

- Research on weekly DeFi exposure-network forecasting.
- Reproduction of the paper's graph forecasting and macroprudential experiments.
- Offline risk monitoring and stress-test research with human review.

## Out of scope

- Real-time trading or automated financial decisions.
- Price prediction.
- Claims about non-DeFi networks without a separate validation study.

## Limitations

- Weekly snapshots can miss intra-week changes.
- On-chain data can be delayed, incomplete, or missing protocol context.
- Forecast performance can change under a new market regime.
- Current hosted fine-tuned weights do not provide all four original v2
  checkpoints.

## License

Apache-2.0. See the repository `NOTICE` and `LICENSES/` files for third-party
components.
