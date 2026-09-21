"""INTERNAL NOTES: oracle substitutions diagnose saved forecasts, not new model scores.
Run with PYTHONHASHSEED=0 uv run --no-project --with numpy python <this file>.
All shocks and common-node sets stay fixed. No training or test-set tuning.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib, importlib.util, json, os
import numpy as np

OUT = Path(__file__).resolve().parent
AUDIT = OUT.parent
RUN = AUDIT.parent
REPO = AUDIT.parents[3]
ROOT = RUN / 'hf_download/runs' / RUN.name
TRAIN = ROOT / 'checkpoints/main2025_v2_hremaining_train'
spec = importlib.util.spec_from_file_location('task2', REPO / 'archive/code/run_task2_model_based.py')
TASK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TASK)
reference = json.loads((AUDIT / 'paper_common_nodes_all/exp2_predictive_contagion.json').read_text())
manifest = {r: h for h, r in (s.split('  ', 1) for s in (TRAIN/'tuning_manifest.sha256').read_text().splitlines())}

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n')

def graph(a, logits, weights, nodes):
    p = {**a, 'exist_logits': logits, 'weight_pred': weights, 'node_pred': nodes}
    return TASK.array_snap_to_dict_snap(TASK.reconstruct_network_from_predictions(p, SimpleNamespace(**a, features_t=None)))

def stats(a):
    valid = a['node_mask'][a['pair_src']] & a['node_mask'][a['pair_dst']]
    yes = a['y_exist'] > .5
    chosen = a['exist_logits'] > 0
    tp, fp, fn = valid & yes & chosen, valid & ~yes & chosen, valid & yes & ~chosen
    wp = np.expm1(np.maximum(a['weight_pred'].astype(float), 0))
    wt = np.expm1(np.maximum(a['y_weight'].astype(float), 0))
    top = np.argsort(np.where(valid & chosen, wp, -1))[-5:][::-1]
    return dict(tp=int(tp.sum()), fp=int(fp.sum()), fn=int(fn.sum()),
        pred_weight=float(wp[valid & chosen].sum()), actual_weight=float(wt[valid & yes].sum()),
        fp_weight=float(wp[fp].sum()), tp_pred_weight=float(wp[tp].sum()), tp_true_weight=float(wt[tp].sum()),
        top_predicted_edges=[dict(source=a['node_ids'][a['pair_src'][i]], target=a['node_ids'][a['pair_dst'][i]],
          predicted=wp[i], actual=wt[i], false_positive=not bool(yes[i])) for i in top])

assert os.environ.get('PYTHONHASHSEED') == '0'
results = {}
for h in [1, 4, 8, 12]:
    old = {x['time_t']: x for x in reference['horizons'][f'h={h}']['samples']}
    rows, edges = [], []
    for f in sorted((TRAIN/f'predictions/h{h}').glob('week_*.npz')):
        assert hashlib.sha256(f.read_bytes()).hexdigest() == manifest[f.relative_to(ROOT).as_posix()]
        with np.load(f, allow_pickle=False) as z: a = {k:z[k] for k in z.files}
        for k in ['time_t','time_t1']: a[k] = str(a[k].item())
        for k in ['node_ids','categories']: a[k] = a[k].tolist()
        truth_logits = np.where(a['y_exist'] > .5, 20., -20.)
        no_fp_logits = np.where(a['y_exist'] > .5, a['exist_logits'], -20.)
        true_tp_weights = np.where(a['y_exist'] > .5, a['y_weight'], a['weight_pred'])
        variants = {
          'original': (a['exist_logits'], a['weight_pred'], a['node_pred']),
          'true_node_tvl': (a['exist_logits'], a['weight_pred'], a['y_node']),
          'remove_false_positive_edges': (no_fp_logits, a['weight_pred'], a['node_pred']),
          'true_topology_pred_weights': (truth_logits, a['weight_pred'], a['node_pred']),
          'true_tp_weights_keep_fp': (a['exist_logits'], true_tp_weights, a['node_pred']),
          'remove_fp_true_tp_weights': (no_fp_logits, a['y_weight'], a['node_pred']),
          'true_all_edges_pred_tvl': (truth_logits, a['y_weight'], a['node_pred']),
          'perfect': (truth_logits, a['y_weight'], a['y_node']),
        }
        nets = {k:graph(a,*v) for k,v in variants.items()}
        pred_nodes = list(nets['original']['nodes'])
        order = sorted(pred_nodes, key=lambda n:nets['original']['nodes'][n]['tvlUsd'], reverse=True)
        # Match archived NumPy argsort tie ordering exactly.
        arr = TASK.reconstruct_network_from_predictions(a, SimpleNamespace(**a,features_t=None))
        order = [arr['node_ids'][i] for i in np.argsort(arr['sizes'])[::-1]]
        scenarios = [('Top Protocol Shock',order[:1],.5), ('Top 5 Protocols Shock',order[:5],.3),
          ('Bridge Sector Shock',[n for n in pred_nodes if 'bridge' in nets['original']['nodes'][n]['category'].lower()],1.)]
        e = stats(a)
        e.update(time_t=a['time_t'], predicted_total_tvl=sum(n['tvlUsd'] for n in nets['original']['nodes'].values()),
          actual_total_tvl=sum(n['tvlUsd'] for n in nets['perfect']['nodes'].values()))
        edges.append(e)
        for name, shocked, frac in scenarios:
            losses = {k:TASK.simulate_contagion(net, shocked, frac)['total_loss_pct'] for k,net in nets.items()}
            ref = old[a['time_t']]['scenarios'][name]
            np.testing.assert_allclose(losses['original'], ref['predicted_t_h']['total_loss_pct'],rtol=0,atol=1e-9)
            np.testing.assert_allclose(losses['perfect'], ref['actual_t_h']['total_loss_pct'],rtol=0,atol=1e-9)
            rows.append(dict(time_t=a['time_t'],time_t1=a['time_t1'],scenario=name,shocked_nodes=shocked,
              actual=losses['perfect'],baseline=ref['observed_t']['total_loss_pct'],losses=losses))
    actual = np.array([r['actual'] for r in rows])
    baseline = np.array([r['baseline'] for r in rows])
    be = abs(baseline-actual)
    tail = np.argsort(be)[::-1][:int(np.ceil(.2*len(rows)))]
    summary = {}
    for k in variants:
        p = np.array([r['losses'][k] for r in rows]); error = abs(p-actual)
        summary[k] = dict(mae=float(error.mean()),bias=float((p-actual).mean()),delta_mae=float(be.mean()-error.mean()),
          worst20_mae=float(error[tail].mean()),worst20_delta=float((be[tail]-error[tail]).mean()),
          by_scenario={s:float(np.mean([abs(r['losses'][k]-r['actual']) for r in rows if r['scenario']==s])) for s,_,_ in scenarios})
    totals = {k:sum(e[k] for e in edges) for k in ['tp','fp','fn','pred_weight','actual_weight','fp_weight','tp_pred_weight','tp_true_weight']}
    totals.update(precision=totals['tp']/(totals['tp']+totals['fp']),recall=totals['tp']/(totals['tp']+totals['fn']),
      fp_weight_fraction=totals['fp_weight']/totals['pred_weight'],pred_to_actual_weight=totals['pred_weight']/totals['actual_weight'])
    results[f'h{h}'] = dict(n=len(rows),baseline_mae=float(be.mean()),summary=summary,edge_totals=totals)
    dump(OUT/f'h{h}_details.json',dict(rows=rows,edge_stats=edges))
    dump(OUT/'summary.json',results)
    print('HORIZON',h,json.dumps(results[f'h{h}']),flush=True)
print('PASS: all 321 original and actual scenario losses match prior audit to 1e-9 percentage points.',flush=True)
