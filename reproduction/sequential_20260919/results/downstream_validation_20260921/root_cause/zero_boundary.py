"""INTERNAL NOTES: isolate the simulator's zero-TVL boundary; no model tuning.
PYTHONHASHSEED=0 uv run --no-project --with numpy python <this file>
"""
from pathlib import Path
from types import SimpleNamespace
import inspect, importlib.util, json, os
import numpy as np

OUT=Path(__file__).resolve().parent
AUDIT=OUT.parent
RUN=AUDIT.parent
REPO=AUDIT.parents[3]
ROOT=RUN/'hf_download/runs'/RUN.name
TRAIN=ROOT/'checkpoints/main2025_v2_hremaining_train'
spec=importlib.util.spec_from_file_location('task2',REPO/'archive/code/run_task2_model_based.py')
TASK=importlib.util.module_from_spec(spec);spec.loader.exec_module(TASK)
original=TASK.simulate_contagion
# Diagnostic implementation: the same cap applies even when TVL is exactly zero.
# This changes the target simulator too, so its errors are not paper-comparable.
source=inspect.getsource(original)
old='''                if tvl_c > 0:
                    losses[creditor] = min(losses[creditor], tvl_c)
'''
new='''                losses[creditor] = min(losses[creditor], max(0.0, tvl_c))
                if tvl_c > 0:
'''
assert source.count(old)==1
scope=dict(TASK.__dict__);exec(source.replace(old,new),scope)
consistent_cap=scope['simulate_contagion']

def simulate_demo(cap, tvl):
    graph={'nodes':{'debtor':{'tvlUsd':100.},'creditor':{'tvlUsd':tvl}},
       'edges':[{'source':'creditor','target':'debtor','weight':1.}]}
    return cap(graph,['debtor'],.5)['total_loss_pct']
demo={str(tvl):{'original':simulate_demo(original,tvl),'consistent_cap':simulate_demo(consistent_cap,tvl)} for tvl in [0.,1e-6]}
assert demo['0.0']['original']==100.
assert abs(demo['1e-06']['original']-50.)<1e-5
assert demo['0.0']['consistent_cap']==50.
assert os.environ.get('PYTHONHASHSEED')=='0'
result={'toy_self_check':demo,'horizons':{},'warning':'Oracle and modified-simulator results are diagnostic only, not corrected model scores or reproductions.'}
for h in [1,4,8,12]:
    prev=json.loads((OUT/f'h{h}_details.json').read_text())
    by_date={}
    for r in prev['rows']:by_date.setdefault(r['time_t'],[]).append(r)
    rows=[];counts=[]
    for f in sorted((TRAIN/f'predictions/h{h}').glob('week_*.npz')):
        with np.load(f,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        for k in ['time_t','time_t1']:a[k]=str(a[k].item())
        for k in ['node_ids','categories']:a[k]=a[k].tolist()
        pair=SimpleNamespace(**a,features_t=None)
        pred=TASK.array_snap_to_dict_snap(TASK.reconstruct_network_from_predictions(a,pair))
        true=TASK.array_snap_to_dict_snap(TASK.reconstruct_actual_network(a,pair))
        true_zero={n for n,v in true['nodes'].items() if v['tvlUsd']==0}
        current_zero={n for n,s,m in zip(a['node_ids'],a['sizes_t'],a['node_mask']) if m and s==0}
        # Node dictionaries copied; no changes to original predictions or stored artifacts.
        def zeroed(ids):return {**pred,'nodes':{n:{**v,'tvlUsd':0.} if n in ids else v.copy() for n,v in pred['nodes'].items()}}
        variants={'zero_actual_zero_nodes':zeroed(true_zero),'zero_observed_zero_nodes':zeroed(current_zero),
          'zero_observed_zero_symbol_nodes':zeroed({n for n in current_zero if not n.isdecimal()})}
        counts.append(dict(time_t=a['time_t'],actual_zero=len(true_zero),actual_zero_pred_positive=sum(pred['nodes'][n]['tvlUsd']>0 for n in true_zero),
          observed_zero=len(current_zero),observed_zero_but_future_positive=len(current_zero-true_zero),
          symbol_examples=[dict(id=n,predicted=pred['nodes'][n]['tvlUsd'],actual=true['nodes'][n]['tvlUsd']) for n in ['WBTC','WETH','USDT','USDC'] if n in pred['nodes']]))
        for r in by_date[a['time_t']]:
            frac={'Top Protocol Shock':.5,'Top 5 Protocols Shock':.3,'Bridge Sector Shock':1.}[r['scenario']]
            losses={k:original(net,r['shocked_nodes'],frac)['total_loss_pct'] for k,net in variants.items()}
            cp=consistent_cap(pred,r['shocked_nodes'],frac)['total_loss_pct']
            ct=consistent_cap(true,r['shocked_nodes'],frac)['total_loss_pct']
            rows.append({**r,'zero_variants':losses,'consistent_cap_predicted':cp,'consistent_cap_actual':ct})
    baseline=np.array([abs(r['baseline']-r['actual']) for r in rows]);tail=np.argsort(baseline)[::-1][:int(np.ceil(.2*len(rows)))]
    summaries={}
    for k in variants:
        errors=np.array([abs(r['zero_variants'][k]-r['actual']) for r in rows])
        summaries[k]=dict(mae=float(errors.mean()),delta_mae=float(baseline.mean()-errors.mean()),worst20_delta=float((baseline[tail]-errors[tail]).mean()))
    summaries['consistent_cap_both_graphs']=dict(mae=float(np.mean([abs(r['consistent_cap_predicted']-r['consistent_cap_actual']) for r in rows])),
        mean_target_shift=float(np.mean([r['consistent_cap_actual']-r['actual'] for r in rows])))
    result['horizons'][f'h{h}']=summaries
    (OUT/f'h{h}_zero_details.json').write_text(json.dumps(dict(rows=rows,counts=counts),indent=2)+'\n')
    (OUT/'zero_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('HORIZON',h,json.dumps(summaries),flush=True)
print('PASS zero-boundary self-check and all 321 fixed-shock cases.',flush=True)
