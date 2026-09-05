"""Evaluate the released weights using the original rerun evaluator; no training."""
import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from functools import partial

from reproduction.prepare import ROOT, sha256, verify_files


def cpu_attention(module, qkv, q, kv, attn_mask):
    """Use PyTorch's CPU fused kernel without materializing full attention scores."""
    import torch
    if attn_mask is not None:
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True,
                                           enable_mem_efficient=False):
            return module.compute_attention_by_torch_batched(qkv, q, kv, attn_mask)
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False,
                                       enable_mem_efficient=False):
        return type(module).compute_attention_by_torch(module, qkv, q, kv, attn_mask)


def metric_comparison(observed, reference):
    rows = []
    for horizon, metrics in observed.items():
        for group, names in [('exist', ['auroc', 'auprc']), ('weight', ['mae', 'rmse']), ('node', ['mae', 'rmse'])]:
            for name in names:
                value = float(metrics[group][name])
                if not math.isfinite(value):
                    raise ValueError(f'Non-finite {horizon}/{group}/{name}')
                target = float(reference[horizon][group][name])
                rows.append(dict(horizon=horizon, group=group, metric=name,
                                 observed=value, reference=target, delta=value-target))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--horizons', default='1,4,8,12')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--output-dir', default='output/reconstruction-evaluation')
    parser.add_argument('--save-predictions', action='store_true')
    args = parser.parse_args()
    horizons = [int(h) for h in args.horizons.split(',')]
    if not horizons or len(set(horizons)) != len(horizons) or not set(horizons) <= {1,4,8,12}:
        parser.error('--horizons must be a unique subset of 1,4,8,12')
    os.chdir(ROOT)
    manifest = verify_files()
    from reproduction import runtime as r
    import torch
    if not r.GRAPHPFN_AVAILABLE or r.dgl is None:
        raise RuntimeError('GraphPFN/DGL is unavailable; install the documented Linux environment')
    if args.device == 'cuda' and not r.check_dgl_cuda_available():
        raise RuntimeError('CUDA-enabled PyTorch and DGL are required for --device cuda')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    r.init_run_context(output)
    config = r.ExperimentConfig()
    config.device = args.device
    config.forecast_horizons = horizons
    config.output_dir = str(output)
    import ijson
    meta, categories, indices = r.load_metadata(config.meta_path)
    schema = r.build_feature_schema(config.meta_path, categories)
    # Build one weekly snapshot at a time instead of retaining the 1 GB raw JSON.
    snapshots = []
    with open(config.data_path, 'rb') as handle:
        for date, network in ijson.kvitems(handle, 'data'):
            snapshots.append(r.build_snapshot(date, network, meta, indices, categories))
            if len(snapshots) % 25 == 0:
                r.log_info(f'Prepared {len(snapshots)}/283 snapshots')
    del network
    snapshots.sort(key=lambda snapshot: snapshot['date'])
    dates = [snapshot['date'] for snapshot in snapshots]
    if len(dates) != 283:
        raise ValueError(f'Expected 283 snapshots, got {len(dates)}')
    split = r.expanding_window_split(dates, holdout_start='2025-01-01',
        min_train_weeks=104, val_weeks=24, test_weeks=16, step_weeks=16)['holdout']
    if {k:len(v) for k,v in split.items()} != {'train':226,'val':24,'test':33}:
        raise ValueError('Unexpected split')
    snapshots = r.enrich_snapshots_with_historical_features(snapshots)
    r.log_info('Historical features prepared')
    test = [s for s in snapshots if s['date'] in split['test']]
    del snapshots
    expected = json.loads((ROOT/'reproduction/expected_metrics.json').read_text())
    paper = json.loads((ROOT/'reproduction/paper_metrics.json').read_text())
    results = {}
    for horizon in horizons:
        r.log_info(f'Loading h={horizon} checkpoint')
        r.set_seed(config.seed)
        path = ROOT/f'checkpoints/reconstruction-20260905/dexposure-fm-h{horizon}.pt'
        payload = torch.load(path, map_location='cpu', weights_only=True)
        provenance = payload['provenance']
        if payload['horizon'] != horizon or payload['artifact_status'] != 'complete':
            raise ValueError('Wrong horizon or incomplete checkpoint')
        if provenance['split'] != split or provenance['seed'] != config.seed:
            raise ValueError('Checkpoint split/seed mismatch')
        if payload['feature_schema'] != schema:
            raise ValueError('Feature schema mismatch')
        encoder = r.load_graphpfn_encoder(config.checkpoint_path, torch.device(config.device))
        model = r.GraphPFNLinkPredictor(encoder, encoder.tfm.embed_dim, config.hidden_dim).to(config.device)
        model.load_state_dict(payload['model'], strict=True)
        if args.device == 'cpu':
            for module in model.modules():
                if isinstance(module, r.MHA):
                    module.compute_attention_by_torch = partial(cpu_attention, module)
        pairs = r.build_week_pairs(test, config.neg_ratio, config.seed, horizon)
        r.log_info(f'h={horizon}: prepared {len(pairs)} test pairs')
        if len(pairs) != 33-horizon:
            raise ValueError('Unexpected number of test pairs')
        predictions = []
        for index, pair in enumerate(pairs, 1):
            r.log_info(f'h={horizon}: predicting pair {index}/{len(pairs)}, nodes={len(pair.node_ids)}')
            predictions.extend(r.predict_graphpfn(model, [pair], config))
        results[f'h{horizon}'] = r.evaluate_predictions(predictions)
        metric_comparison(results, expected)
        (output/'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
        if args.save_predictions:
            r.save_predictions_csv(predictions, output, 'DeXposure-FM', horizon)
        r.log_info(f'h={horizon}: {results[f"h{horizon}"]}')
        del model, encoder, payload, pairs, predictions
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    record = dict(config=asdict(config), split=split, release=manifest,
                  torch_version=str(torch.__version__), device=args.device,
                  cpu_attention_backend='pytorch-flash' if args.device == 'cpu' else None,
                  runtime_sha256=sha256(ROOT/'reproduction/runtime.py'),
                  versus_rerun=metric_comparison(results, expected),
                  versus_paper=metric_comparison(results, paper))
    (output/'evaluation.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print('Evaluation finished. Inspect evaluation.json deltas; no automatic paper-match claim is made.')


if __name__ == '__main__':
    main()
