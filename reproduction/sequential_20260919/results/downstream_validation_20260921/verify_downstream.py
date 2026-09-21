"""INTERNAL NOTES: saved-prediction audit; no training and no cloud operations.
Run with uv run --no-project --with numpy --with matplotlib --with networkx --with ijson python ...
"""
from pathlib import Path
from types import SimpleNamespace
import csv, hashlib, importlib.util, json, math, os, sys, subprocess
import numpy as np
import ijson
import scipy  # Required by NetworkX PageRank; fail instead of silent zero fallback.

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[3]
RUN=OUT.parent
ROOT=RUN/'hf_download/runs'/RUN.name
TRAIN=ROOT/'checkpoints/main2025_v2_hremaining_train'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

TASK=module('task2',REPO/'archive/code/run_task2_model_based.py')
SPILL=module('spill',REPO/'DeXposure_FM_V2/scripts/make_spillover_matrix_figure.py')
META=SPILL._load_meta_category(ROOT/'data/meta_df.csv')

def compact(date,snap):
    ids=[]; sizes=[]; seen=set()
    for n in snap['nodes']:
        if not isinstance(n,dict) or n.get('id') is None:continue
        nid=str(n['id']).strip()
        if not nid or nid in seen:continue
        seen.add(nid); ids.append(nid); sizes.append(float(n.get('size') or 0))
    ix={n:i for i,n in enumerate(ids)}; edges={}
    for e in snap['links']:
        if e.get('source') is None or e.get('target') is None or e.get('size') is None:continue
        s,d=str(e['source']).strip(),str(e['target']).strip();w=float(e['size'])
        assert math.isfinite(w) and w>=0
        if s in ix and d in ix and w>0:edges[(s,d)]=edges.get((s,d),0)+w
    keys=sorted(edges)
    w=np.array([edges[k] for k in keys],dtype=np.float32)
    return dict(date=date,node_ids=ids,sizes=np.array(sizes,dtype=np.float32),
      categories=[META.get(n,'other') for n in ids],edge_src=np.array([ix[k[0]] for k in keys],dtype=np.int64),
      edge_dst=np.array([ix[k[1]] for k in keys],dtype=np.int64),edge_weight=w)

def self_check():
    q=SimpleNamespace(node_mask=np.array([True,True,False]),pair_src=np.array([0,1,0]),pair_dst=np.array([1,0,2]),
      node_ids=['a','b','c'],categories=['x','y','z'],sizes_t=np.array([10.,20.,30.]),features_t=None,
      y_exist=np.array([1,0,1]),y_weight=np.log1p([5.,0.,7.]),y_node=np.zeros(3))
    p={'exist_logits':np.array([2.,-2.,2.]),'weight_pred':np.log1p([5.,0.,7.]),'node_pred':np.zeros(3)}
    n=TASK.reconstruct_network_from_predictions(p,q)
    assert n['node_ids']==['a','b'] and len(n['edge_src'])==1
    d=TASK.array_snap_to_dict_snap(n);assert abs(d['edges'][0]['weight']-5)<1e-9
    assert abs(TASK.simulate_contagion({'nodes':{'a':{'tvlUsd':100}},'edges':[]},['a'],.5)['total_loss_pct']-50)<1e-12
    print('SELF_CHECK passed: masks, edge log conversion, isolated shock',flush=True)

self_check()
import networkx as nx
_original_pagerank=nx.pagerank
def checked_pagerank(*args,**kwargs):
    try:return _original_pagerank(*args,**kwargs)
    except Exception as error:raise SystemExit(f'PageRank failed: {error}')
nx.pagerank=checked_pagerank
# Bind all predictions and data to the frozen run manifests before reading NPZ data.
manifest={rel:d for d,rel in (x.split('  ',1) for x in (TRAIN/'tuning_manifest.sha256').read_text().splitlines())}
files=sorted((TRAIN/'predictions').glob('h*/week_*.npz'))
assert len(files)==107
for f in files:assert sha(f)==manifest[f.relative_to(ROOT).as_posix()]
input_hashes={Path(rel).name:d for d,rel in (x.split('  ',1) for x in (ROOT/'logs/input_manifest.sha256').read_text().splitlines())}
for f in (ROOT/'data').iterdir():
    if f.is_file():assert sha(f)==input_hashes[f.name],f
print('INPUT_HASHES passed',flush=True)
SPLIT=json.loads((TRAIN/'split.json').read_text());DATES=SPLIT['test']
KEEP=set(DATES)|{'2022-04-25','2022-05-02','2022-05-23','2022-10-24','2022-10-31','2022-11-21'}
SNAPS={};STATS={};RAW_SPILL=None
with (ROOT/'data/historical-network_week_2020-03-30.json').open('rb') as f:
    for date,snap in ijson.kvitems(f,'data',use_float=True):
        if date in KEEP:
            c=compact(date,snap);SNAPS[date]=c
            STATS[date]={'tvl':float(c['sizes'].astype(float).sum()),'nodes':len(c['node_ids']),'edges':len(c['edge_src'])}
        if date=='2025-06-30':RAW_SPILL=snap
assert set(DATES)<=SNAPS.keys()
print('DATA loaded',len(SNAPS),'snapshots',flush=True)
CURRENT={}; common_filter=False; coverage=[]
original_convert=TASK.array_snap_to_dict_snap

def convert(snap,edge_weight_is_log=True):
    d=original_convert(snap,edge_weight_is_log)
    if common_filter and 'features' not in snap and snap.get('date') in CURRENT:
        valid=CURRENT[snap['date']]
        d={**d,'nodes':{k:v for k,v in d['nodes'].items() if k in valid},
           'edges':[e for e in d['edges'] if e['source'] in valid and e['target'] in valid]}
    return d
TASK.array_snap_to_dict_snap=convert

def pairs(_snapshots,neg,seed,horizon):
    global CURRENT
    out=[];CURRENT={}
    for f in sorted((TRAIN/f'predictions/h{horizon}').glob('week_*.npz')):
        with np.load(f,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        for k in ['time_t','time_t1']:a[k]=str(a[k].item())
        for k in ['node_ids','categories']:a[k]=a[k].tolist()
        snap=SNAPS[a['time_t']];actual=SNAPS[a['time_t1']]
        assert a['node_ids']==snap['node_ids']
        np.testing.assert_array_equal(a['sizes_t'],snap['sizes'])
        assert DATES.index(a['time_t1'])-DATES.index(a['time_t'])==horizon
        valid=set(actual['node_ids'])&set(a['node_ids']); CURRENT[a['time_t']]=valid
        assert {n for n,m in zip(a['node_ids'],a['node_mask']) if m}==valid
        # Cross-check that saved targets reconstruct the raw realized graph on common nodes.
        truth=TASK.reconstruct_actual_network(a,SimpleNamespace(**a,features_t=None))
        expected={(actual['node_ids'][s],actual['node_ids'][d]):float(w) for s,d,w in zip(actual['edge_src'],actual['edge_dst'],actual['edge_weight']) if actual['node_ids'][s] in valid and actual['node_ids'][d] in valid}
        saved={(truth['node_ids'][s],truth['node_ids'][d]):float(w) for s,d,w in zip(truth['edge_src'],truth['edge_dst'],truth['edge_weight'])}
        assert set(expected)==set(saved)
        np.testing.assert_allclose([saved[k] for k in expected],[math.log1p(expected[k]) for k in expected],rtol=1e-6,atol=1e-6)
        a.update(features_t=None,edge_src_t=snap['edge_src'],edge_dst_t=snap['edge_dst'],
                 edge_weight_t=np.array([math.log1p(float(w)) for w in snap['edge_weight']],dtype=np.float32))
        obj=SimpleNamespace(**a); obj.saved_prediction=a;out.append(obj)
    assert len(out)==33-horizon
    print('PAIRS cross-checked',horizon,len(out),flush=True)
    return out
TASK.build_week_pairs=pairs
TASK.predict_graphpfn=lambda model,pairs,config:[p.saved_prediction for p in pairs]
config=SimpleNamespace(neg_ratio=5,seed=42)
models={h:h for h in [1,4,8,12]}
all_results={}
for mode,n,restrict in [('legacy_default10',10,False),('paper_common_nodes_all',0,True)]:
    common_filter=restrict;dest=OUT/mode;dest.mkdir(exist_ok=True)
    r=json.loads((dest/"exp2_predictive_contagion.json").read_text()) if (dest/"exp2_predictive_contagion.json").exists() else TASK.run_predictive_contagion(config,models,[],META,horizons=list(models),max_samples_per_horizon=n,output_dir=dest)
    for h in models:
        rows=r['horizons'][f'h={h}']['samples']
        assert all(len(x['scenarios'])==3 for x in rows)
    all_results[mode]={h:v['advantage_overall'] for h,v in r['horizons'].items()}
    write(OUT/'partial_comparison.json',all_results)
common_filter=False
risk=json.loads((OUT/"exp1_forward_risk.json").read_text()) if (OUT/"exp1_forward_risk.json").exists() else TASK.run_forward_risk_prediction(config,models,[],META,horizons=list(models),max_pairs_per_horizon=0,output_dir=OUT)
# Figure 5 uses squared Pearson correlation, not the regression R-squared score.
figure5={}
for k,printed in [('tvl_hhi',.001),('density',.841),('mean_sis',.700),('spillover_index',.000)]:
    pred=np.array([m[k] for h in risk['horizons'].values() for m in h['predicted_metrics']])
    act=np.array([m[k] for h in risk['horizons'].values() for m in h['actual_metrics']])
    assert np.isfinite(pred).all() and np.isfinite(act).all()
    figure5[k]={'paper_pearson_r_squared':printed,'new_pearson_r_squared':float(np.corrcoef(pred,act)[0,1]**2),
      'mae':float(np.mean(np.abs(pred-act))),'regression_r_squared':float(1-np.sum((act-pred)**2)/np.sum((act-act.mean())**2)),'n':len(act)}
# Table 5: event-window anchors are explicitly defined in the original event helper.
events={}
for name,start,end,printed in [('Terra/Luna','2022-04-25','2022-05-23',[261.9,166.3,-36.5,9.3]),('FTX','2022-10-24','2022-11-21',[119.9,175.1,46.1,-1.8])]:
    pre,post=STATS[start],STATS[end]
    vals=[pre['tvl']/1e9,post['tvl']/1e9,(post['tvl']/pre['tvl']-1)*100,(post['edges']/pre['edges']-1)*100]
    events[name]={'anchor':start,'end':end,'paper':printed,'new':vals,'match_one_decimal':[round(v,1)==p for v,p in zip(vals,printed)],'raw_counts':[pre,post]}
# Figure 4: compare the run's two pinned datasets at the paper's observed date.
with (ROOT/'data/historical-network_week_2025-07-01.json').open('rb') as f:
    july=next(s for d,s in ijson.kvitems(f,'data',use_float=True) if d=='2025-06-30')
matrices=[]
for s in [RAW_SPILL,july]:
    e=SPILL._compute_sector_exposure(s,META,min_edge_weight=0)
    sectors=SPILL._select_sectors(e,max_sectors=15)
    if 'other' not in sectors:sectors.append('other')
    matrices.append({'sectors':sectors,'matrix':SPILL._build_matrix(e,sectors)})
write(OUT/'figure4_matrix.json',{'date':'2025-06-30','long_history':matrices[0],'july_input':matrices[1],'equal':matrices[0]==matrices[1]})
# Regenerate original figures without touching manuscript assets.
TASK.plot_all_figures(output_dir=OUT,forward_risk_results=risk,contagion_results=r)
write(OUT/'comparison.json',{'run':RUN.name,'table4':all_results,'figure5':figure5,'table5':events,
 'figure6':{'status':'not_eligible_for_this_run','reason':'Training data include both 2022 event windows; separate pre-event models required.'},
 'protocol':{'legacy_default10':'Unmodified archived simulator/evaluator, 10 seeded test origins per horizon; baseline retains original node set.',
 'paper_common_nodes_all':'All 107 origins, three shocks each; all networks restricted to common nodes. Other simulator rules unchanged.',
 'sampled_candidates':'Saved predictions include all future-positive pairs plus 5:1 sampled negatives; not exhaustive all-pairs graph inference.',
 'shock_selection':'Original helper selects Top-N nodes from predicted TVL and uses the same set in all three graphs.',
 'hash_seed':os.environ.get('PYTHONHASHSEED'),'new_training':False},
 'source_hashes':{'task2':sha(REPO/'archive/code/run_task2_model_based.py'),'spillover':sha(REPO/'DeXposure_FM_V2/scripts/make_spillover_matrix_figure.py'), 'selection':sha(TRAIN/'selection_frozen.json'),'adapter':sha(__file__)}})
print('COMPLETE',OUT/'comparison.json',flush=True)
