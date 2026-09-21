"""INTERNAL NOTES: inspect event table using the archived event evaluator's node universe."""
import json,ijson,ast,math
from typing import Dict,List,Tuple
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent; repo=P.parents[3];root=P.parent/'hf_download/runs'/P.parent.name
EPS=1e-8
source=(repo/'archive/code/run_full_experiment.py').read_text(); tree=ast.parse(source)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['node_features','build_snapshot']],type_ignores=[]),'legacy_snapshot','exec'))
wanted={'2022-04-25','2022-05-16','2022-05-23','2022-10-24','2022-11-14','2022-11-21'};snaps={}
with (root/'data/historical-network_week_2020-03-30.json').open('rb') as f:
 for d,s in ijson.kvitems(f,'data',use_float=True):
  if d in wanted:snaps[d]=build_snapshot(d,s,{}, {'Unknown':0},['Unknown'])
results={}
for name,anchor,prev,end,paper in [('Terra/Luna','2022-04-25','2022-05-16','2022-05-23',[261.9,166.3,-36.5,9.3]),('FTX','2022-10-24','2022-11-14','2022-11-21',[119.9,175.1,46.1,-1.8])]:
 a,b=snaps[anchor],snaps[end];pre=float(a['sizes'].sum());preedges=len(a['edge_src']);vs={}
 for mode,allowed in [('all',set(b['node_ids'])),('common_with_previous_week',set(snaps[prev]['node_ids'])),('common_with_anchor',set(a['node_ids']))]:
  post=float(np.sum([v for n,v in zip(b['node_ids'],b['sizes']) if n in allowed],dtype=np.float64))
  edges=sum(b['node_ids'][s] in allowed and b['node_ids'][d] in allowed for s,d in zip(b['edge_src'],b['edge_dst']))
  v=[pre/1e9,post/1e9,(post/pre-1)*100,(edges/preedges-1)*100]
  vs[mode]={'metrics':v,'rounded':[round(x,1) for x in v],'matches':[round(x,1)==y for x,y in zip(v,paper)],'post_edges':edges}
 results[name]={'anchor':anchor,'previous_week':prev,'end':end,'pre_edges':preedges,'paper':paper,'variants':vs}
print(json.dumps(results,indent=2));(P/'table5_event_universe.json').write_text(json.dumps(results,indent=2)+'\n')
