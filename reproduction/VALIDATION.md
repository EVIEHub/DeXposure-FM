# Validation status

This is a reproduction candidate, not an end-to-end certified release.

## Completed checks

- Four September checkpoints were trained, uploaded, downloaded locally, and
  verified against their SHA-256 values. The training metrics and logs are
  retained in the pinned Hugging Face release directory.
- The four files downloaded from the public model repository match the
  reconstruction hashes. All eight local files in release.json pass the
  preparation command's hash checks.
- Five unit checks pass with Python 3.12 and PyTorch 2.2.1 on macOS. They check
  metric deltas/non-finite values, corrupted input rejection, byte-identical
  training source, the pinned base-model loader, and CPU fused attention
  against math attention on small tensors.
- The training entrypoint's command-line help runs successfully.

## Not yet completed

The new evaluation-only entrypoint has not completed all four horizons in an
independent acceptance run. Its macOS CPU run has passed data preparation,
the 283-snapshot and 226/24/33 split checks, checkpoint metadata checks, strict
weight loading, and construction of the 32 h1 test pairs. Full prediction and
the 24 metric comparisons remain unverified through this new entrypoint.

The unmodified math-attention CPU attempt stopped at the first 11,085-node
test graph. A later run uses PyTorch CPU fused attention to reduce memory
allocation. This is not evidence that either run reproduced the full metrics.
An earlier terminated attempt had no recorded exit cause; it must not be
presented as a diagnosed out-of-memory failure.

expected_metrics.json contains the completed September training runs' saved
test metrics, not results from the new acceptance run. The locked Linux CUDA
installation has not been independently installed and tested from this tag.
Do not interpret passing unit tests as a complete clean-environment reproduction.

## Acceptance evidence still required

Run the documented preparation and evaluation commands from this fixed tag.
Retain evaluation.json, metrics.json, the terminal log and, if requested,
prediction CSVs. Report all 24 numerical differences against the September
results and the paper. Investigate material deviations instead of changing
the test split or choosing only favorable results.

The present release does not certify baseline rows, Task II, every figure,
or statistical equivalence to arXiv v2.
