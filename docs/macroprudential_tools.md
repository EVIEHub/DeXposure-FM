# Macroprudential Tools (Observed + Predictive)

This repo includes a small CLI that turns DeXposure snapshots into **usable macroprudential monitoring outputs**:

- **Systemic Importance Score (SIS)**: protocol-level importance ranking from network centrality + tail exposures + size
- **Sector spillovers**: sector-to-sector cross-exposure matrix + spillover concentration (HHI)
- **Contagion stress tests**: DebtRank-style propagation losses under counterfactual shocks

Crucially, it supports **two modes**:

1. **Observed mode**: compute tools on the observed exposure graph at time \(t\), \(G_t\).
2. **Predictive mode (forecast-then-measure)**: use DeXposure-FM to forecast a future graph \(\hat G_{t+h}\), then compute the same tools on the predicted graph.


## Setup notes

- Large network files can be heavy to load. `run_full_experiment.py` uses `ijson` for
  streaming JSON parsing to reduce memory usage.

## INTERNAL NOTES | Contagion loss cap

Received contagion losses are capped at `max(0, TVL)` for every creditor,
including nodes with exactly zero TVL. Zero-TVL nodes remain in the graph and
in exposure-share denominators, but cannot accumulate or propagate losses.
Loss removed by the cap is not redistributed. Existing missing-to-zero input
handling is unchanged; this does not establish that missing TVL is economically
zero.

Earlier simulator versions applied the cap only when TVL was positive. That
made zero and arbitrarily small positive TVL produce discontinuous aggregate
losses. After this correction, recompute observed, predicted, and realized
graphs with the same rule, including baseline errors and the worst-20% subset.
Do not combine corrected predictions with old baseline or realized losses, or
label corrected results as an exact reproduction of the printed paper table.

The boundary regression check covers all three simulator entry points:

```
uv run --no-project --with numpy python paper/tests/test_contagion_zero_tvl.py
```


## Observed mode

Compute SIS/spillovers (and optionally contagion) on a specific snapshot:

```
uv run python run_macroprudential_tools.py observed \
  --date 2025-06-30 \
  --data-path data/historical-network_week_2025-07-01.json \
  --contagion \
  --output-dir output/macro-tools
```

Outputs a JSON file like `output/macro-tools/observed_2025-06-30.json`.


## Predictive mode (forecast-then-measure)

Forecast \(\hat G_{t+h}\) and compute the same tool outputs on:

- observed \(G_t\)
- predicted \(\hat G_{t+h}\)
- actual \(G_{t+h}\) (if present in the dataset, used for evaluation)

Example:

```
uv run python run_macroprudential_tools.py predict \
  --date 2025-06-30 \
  --horizon 4 \
  --data-path data/historical-network_week_2025-07-01.json \
  --device cuda \
  --contagion \
  --output-dir output/macro-tools
```

The script will **train or load a cached model** in `output/model_cache` by default. Use:

- `--frozen` for faster head-only training (GraphPFN encoder frozen)
- `--force-retrain` to ignore caches
- `--model-cache-dir` / `--cache-tag` to control cache isolation


## Output structure (high-level)

Both modes write a JSON payload with:

- `snapshot` / `snapshots`: per-graph summaries containing:
  - `risk_metrics` (HHI, density, mean SIS, spillover HHI, etc.)
  - `sis_top` (top-K SIS ranking)
  - `spillover` (total + concentration; optionally full matrix)
- `contagion` (optional): per-scenario losses on observed/predicted/actual graphs
