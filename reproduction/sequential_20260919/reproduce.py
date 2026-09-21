"""Reproduce saved-prediction Task I metrics and corrected Table 4 / Figures 2-3."""
import argparse
import ast
from pathlib import Path
import json
import shutil
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle',default='output/sequential-20260919/bundle')
parser.add_argument('--output-dir',default='output/sequential-20260919/evaluation')
parser.add_argument('--download',action='store_true')
args=parser.parse_args()
release=json.loads((Path(__file__).resolve().parent/'release.json').read_text())
if args.download:
    from huggingface_hub import HfApi,hf_hub_download
    repo_id,revision=release['hf_repo'],release['hf_revision']
    run_prefix='runs/'+release['run']
    train_prefix=run_prefix+'/checkpoints/main2025_v2_hremaining_train/'
    wanted=['plan.json','selection_frozen.json','split.json','tuning_manifest.sha256']
    files=[train_prefix+n for n in wanted]+[run_prefix+'/logs/input_manifest.sha256']
    files.extend(x.path for x in HfApi().list_repo_tree(repo_id,path_in_repo=train_prefix+'predictions',revision=revision,recursive=True) if x.path.endswith('.npz'))
    for remote in files:
        local=Path(args.bundle)/remote.removeprefix(run_prefix+'/')
        local.parent.mkdir(parents=True,exist_ok=True)
        if not local.exists():
            cached=hf_hub_download(repo_id,remote,revision=revision)
            shutil.copyfile(cached,local)
    for name in ['historical-network_week_2020-03-30.json','historical-network_week_2025-07-01.json','meta_df.csv']:
        local=Path(args.bundle)/'data'/name;local.parent.mkdir(parents=True,exist_ok=True)
        if not local.exists():
            cached=hf_hub_download(repo_id,'data/'+name,revision=revision)
            shutil.copyfile(cached,local)

from pathlib import Path
from types import SimpleNamespace
import csv, hashlib, importlib.util, json, math, os, sys, subprocess
import numpy as np
import ijson
import scipy  # Required by NetworkX PageRank; fail instead of silent zero fallback.

OUT=Path(args.output_dir).resolve();OUT.mkdir(parents=True,exist_ok=True)
REPO=Path(__file__).resolve().parents[2]
RUN=Path(args.bundle).resolve()
ROOT=RUN
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


assert os.environ.get('PYTHONHASHSEED')=='0', 'Set PYTHONHASHSEED=0 for deterministic propagation order.'
# Use the training core's original metric functions without importing CUDA dependencies.
from sklearn.metrics import average_precision_score,roc_auc_score,mean_absolute_error,mean_squared_error
from typing import Any,Dict,List
names={'compute_recall_at_k','compute_weighted_mae','evaluate_predictions'}
tree=ast.parse((REPO/'cloud/train_src/run_full_experiment.py').read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert len(functions)==len(names)
exec(compile(ast.Module(body=functions,type_ignores=[]),'training_metric_functions','exec'),globals())
expected=json.loads((Path(__file__).resolve().parent/'task1_metrics.json').read_text())
metrics={}
for h in models:
    ps=pairs([],5,42,h)
    metrics[f'h{h}']=evaluate_predictions([p.saved_prediction for p in ps])
    for group,keys in [('exist',['auprc','auroc']),('weight',['mae','rmse']),('node',['mae','rmse'])]:
        for key in keys:
            np.testing.assert_allclose(metrics[f'h{h}'][group][key],expected[f'h{h}'][group][key],rtol=0,atol=1e-6)
    print('TASK_I_MATCH',h,flush=True)
write(OUT/'task1_metrics.json',metrics)
common_filter=True
result=TASK.run_predictive_contagion(config,models,[],META,horizons=list(models),max_samples_per_horizon=0,output_dir=OUT)
reference=json.loads((Path(__file__).resolve().parent/'results/downstream_zero_tvl_fixed_20260921/summary.json').read_text())['modes']['paper_common_nodes_all']
for h in models:
    got=result['horizons'][f'h={h}']['advantage_overall']
    for key in ['mae_model_all','mae_baseline_all','delta_mae_all','delta_mae_worst','win_rate_worst']:
        np.testing.assert_allclose(got[key],reference[f'h={h}'][key],rtol=0,atol=1e-9)
result['evaluation_note']='Uniform TVL cap | 2025 hold-out | 107 origins / 321 scenarios'
TASK.plot_contagion_comparison(result,OUT)
TASK.plot_contagion_advantage(result,OUT)
write(OUT/'verification.json',{'status':'passed','mode':'saved_prediction_replay','horizons':[1,4,8,12],'task1_metric_tolerance':1e-6,'task2_metric_tolerance':1e-9,'n_origins':107,'n_scenarios':321,'release':release})
print('PASS: 24 Task I metrics and corrected Table 4 reproduced; Figures 2-3 exported.',flush=True)
