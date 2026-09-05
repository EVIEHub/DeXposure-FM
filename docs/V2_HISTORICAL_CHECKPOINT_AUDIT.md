# Historical EVIEHub checkpoint audit

## Verified objects

- arXiv 2602.03981v1 and v2 Table 3 have identical numeric rows, including all four DeXposure-FM horizons. This establishes unchanged reported results, not a checkpoint-to-evaluation identity proof.
- Original public model release: EVIEHub/DeXposure-FM revision e0c9667dea5ef07df49ad8cfad413ba19f4ad00e, uploaded 2026-02-03, titled DeXposure-FM v1.0.
- Current inspected revision: 5a5867805e001d14aa68dea45b797a62dc49ede4. All four originally released weight blobs have unchanged SHA-256.
- Local copies in checkpoints/dexposure-fm-release match the original public SHA-256 values and load using torch.load(weights_only=True).

| File | SHA-256 |
| --- | --- |
| dexposure-fm-h1.pt | d867e66834f54d48ae4ca3d042faf28e04712316b580587e8837121ef76c6e30 |
| dexposure-fm-h4.pt | ce8ea6a0e9b6fd966233fc2b79a697e22bcf29c9b5bc89cd1d4398c709e70250 |
| dexposure-fm-h8-h12.pt | a7773e85a66a8914822edd8e172f9e9b7bff2a076a94ca970dea630efb63b030 |
| graphpfn-frozen-all-horizons.pt | 114183835be3de3689150fc2a21a3a941072500b1117db957cf02374dba6a14d |

## Historical result provenance

Local Git commit 340dac4a41ebaf2221470a901e8ceeaa06726a0d records that paper metrics were taken from these GPU experiment files:

- h1 and h4: output/2026-01-26_160546/finetuned/metrics_intermediate.json
- h8 and h12: output/2026-01-27_084004/finetuned/metrics_intermediate.json

This is a historical commit statement. The original run directories were not located in the current workspace scan.

The February HF release already contains the corresponding metrics in metrics-h4.json (both h1 and h4) and metrics-h8-h12.json (both h8 and h12). They are not September additions. Two reported values differ from ordinary three-decimal rounding: historical h1 edge RMSE 3.3874919145213123 rounds to 3.387, while the paper prints 3.388; historical h8 AUROC 0.9934984300089452 rounds to 0.993, while the paper prints 0.994. Therefore these are closely corresponding historical result records, not an exact all-cell match.

metrics-h1.json instead reports AUPRC 0.977961656036245 and edge MAE 2.090425968170166, unlike paper h1 AUPRC 0.972 and MAE 2.465. The release config labels this h1 model as trained January 28, versus January 26 for h4 and January 27 for h8/h12. Dates are config assertions, not independent training logs.

## Weight contents and overwrite evidence

All released checkpoints have only a top-level model state dictionary, without an embedded horizon, run ID, data split, or evaluation hash.

- h1 has 345 tensor entries and a node head with input dimension 576 and an additional hidden layer.
- h4 and h8-h12 each have 343 tensor entries and a node head with input dimension 192.
- h8-h12 contains one model state dictionary, not separate h8 and h12 models.

Historical run_full_experiment.py at commits 340dac4 and 54ae9dd saves each horizon's best state to the same output_dir/best_model.pt. Under a completed sequential multi-horizon run using that code and directory, later horizons overwrite earlier saved weights. This explains how multiple metric rows can survive while only one checkpoint remains. The exact deployed source and export operation have not been independently recovered.

Consequently h4 is a strong historical candidate. h8-h12 may be the last saved horizon (h12 if the run completed in order 8,12), or an earlier snapshot. The current loader's legacy h8 mapping is not provenance proof. Exact horizon assignment requires an original export/log record or evaluation with the historical protocol.

## Recovery boundary

The original public release and byte-identical local weights are located. The paper's historical result-run identifiers are located. A complete set of four independently identified paper-result checkpoints is not established. The September reconstruction must remain separately identified.

Next targeted recovery objects are the two January run directories, especially best_model.pt, prediction files, full logs, and any checkpoint export command. If those are unavailable, evaluate the unchanged public candidates under the recovered historical data and evaluation configuration; matching filenames alone is insufficient.

## Sources

- https://arxiv.org/html/2602.03981v1
- https://arxiv.org/html/2602.03981v2
- https://huggingface.co/EVIEHub/DeXposure-FM/tree/e0c9667dea5ef07df49ad8cfad413ba19f4ad00e
- Local Git commits 340dac4a41ebaf2221470a901e8ceeaa06726a0d, 54ae9dd, and 54d65cfdb4287610115ce767545d7d4c1f973c05.
