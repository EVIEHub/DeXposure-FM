# September four-horizon reconstruction

Prerelease status: reproduction candidate. Five unit checks and all pinned
input hashes pass. The new evaluation-only command has not yet completed an
independent full acceptance run. See
[validation status](https://github.com/EVIEHub/DeXposure-FM/blob/reconstruction-20260905/reproduction/VALIDATION.md).

This release provides a fixed code and model package for independent evaluation
of DeXposure-FM on the 2025 holdout. It is a new reconstruction release, not a
replacement for the historical artifacts or a claim that every arXiv v2 table
and figure has been reproduced.

- Four separately identified checkpoints for h1, h4, h8 and h12.
- Immutable Hugging Face revisions and SHA-256 checks for weights and inputs.
- The exact September training entrypoint, training manifest and measured metrics.
- An evaluation-only command, optional prediction CSV export, and numerical
  comparisons against both the September rerun and the paper.
- Historical root checkpoints retained. Their incomplete weight-to-result
  mapping is documented in the historical audit.

Start with [the reproduction guide](https://github.com/EVIEHub/DeXposure-FM/blob/reconstruction-20260905/reproduction/README.md).
The [model artifacts](https://huggingface.co/EVIEHub/DeXposure-FM/tree/035c8cfa240ddf6a4c579e6e597df3427e542482/reconstructions/20260905)
include training logs and outputs. The Hugging Face repository may require
accepting its access conditions and signing in before download.

The September results are mixed relative to the paper. For example, h12 AUPRC
is 0.956405 versus the paper's 0.967, while h12 edge MAE is 2.390661 versus
2.648. All six metrics for all four horizons are disclosed in the guide.
No statistical equivalence bound has been established.
