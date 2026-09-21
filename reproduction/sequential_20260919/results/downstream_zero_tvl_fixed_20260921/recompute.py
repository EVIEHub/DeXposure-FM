"""INTERNAL NOTES: recompute Table 4 / Figures 2-3 with one loss-cap rule.
PYTHONHASHSEED=0 MPLBACKEND=Agg uv run --no-project --with numpy --with scipy \
  --with networkx --with matplotlib --with ijson python <this file>
Reuses the previous audited saved-prediction loader; no training or postprocessing.
"""
from pathlib import Path
import ast
import hashlib
import json
import os

assert os.environ.get('PYTHONHASHSEED') == '0'
previous = Path(__file__).resolve().parent.parent/'downstream_validation_20260921'
adapter = previous/'verify_downstream.py'
# Execute only the existing loader setup, stopping before its evaluation loop.
# __file__ remains this script, so the adapter writes to the new output directory.
tree = ast.parse(adapter.read_text(), filename=str(adapter))
stop = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == 'all_results' for t in n.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(adapter), 'exec'), globals())

for directory in [previous, previous/'root_cause']:
    for line in (directory/'SHA256SUMS').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        assert sha(directory/relative) == digest, relative
print('OLD_ARTIFACTS_UNCHANGED passed', flush=True)

fixed_simulator = TASK.simulate_contagion
calls = 0
def checked_simulator(snap, *args, **kwargs):
    global calls
    result = fixed_simulator(snap, *args, **kwargs)
    assert np.isfinite(result['total_loss_pct'])
    assert -1e-9 <= result['total_loss_pct'] <= 100+1e-9
    calls += 1
    return result
TASK.simulate_contagion = checked_simulator

all_results = {}
for mode, samples, restrict in [('paper_common_nodes_all', 0, True), ('legacy_default10', 10, False)]:
    common_filter = restrict
    dest = OUT/mode
    dest.mkdir(exist_ok=True)
    result = TASK.run_predictive_contagion(config, models, [], META,
        horizons=list(models), max_samples_per_horizon=samples, output_dir=dest)
    old = json.loads((previous/mode/'exp2_predictive_contagion.json').read_text())
    summaries = {}
    for h in models:
        key = f'h={h}'
        entries = result['horizons'][key]['samples']
        assert [(r['time_t'],r['time_t1']) for r in entries] == [(r['time_t'],r['time_t1']) for r in old['horizons'][key]['samples']]
        diagnostic = json.loads((previous/'root_cause'/f'h{h}_zero_details.json').read_text())
        diagnostic = {(r['time_t'],r['scenario']):r for r in diagnostic['rows']}
        baselines, predicted, actual = [], [], []
        for row in entries:
            assert len(row['scenarios']) == 3
            for name, scenario in row['scenarios'].items():
                diag = diagnostic[row['time_t'],name]
                p = scenario['predicted_t_h']['total_loss_pct']
                a = scenario['actual_t_h']['total_loss_pct']
                # Independent prior oracle-script implementation of the same rule.
                np.testing.assert_allclose([p,a], [diag['consistent_cap_predicted'],diag['consistent_cap_actual']], rtol=0,atol=1e-9)
                baselines.append(scenario['observed_t']['total_loss_pct'])
                predicted.append(p); actual.append(a)
        b,p,a = map(np.array,(baselines,predicted,actual))
        be, pe = abs(b-a), abs(p-a)
        tail = np.argsort(be)[::-1][:int(np.ceil(.2*len(be)))]
        aggregate = result['horizons'][key]['advantage_overall']
        np.testing.assert_allclose([aggregate['mae_model_all'],aggregate['mae_baseline_all'],aggregate['delta_mae_all'],aggregate['delta_mae_worst']],
            [pe.mean(),be.mean(),be.mean()-pe.mean(),(be[tail]-pe[tail]).mean()],rtol=0,atol=1e-12)
        summaries[key] = aggregate
    all_results[mode] = summaries
    print('CHECKED', mode, json.dumps(summaries), flush=True)
    if restrict:
        result['evaluation_note'] = 'Uniform TVL cap | 2025 hold-out | 107 origins / 321 scenarios'
        TASK.plot_contagion_comparison(result,OUT)
        TASK.plot_contagion_advantage(result,OUT)

assert calls == 3*(321+120), calls
write(OUT/'summary.json', dict(run=RUN.name, loss_rule='Received losses capped at max(0, TVL) for every creditor, including TVL=0.',
    missing_value_policy='Existing missing-to-zero input convention retained; no claim that missing TVL is economically zero.',
    zero_node_policy='Zero-TVL nodes remain in incoming-exposure allocation but cannot accumulate or propagate loss; capped loss is not redistributed.',
    model_postprocessing='None; saved node and edge predictions unchanged.',
    worst20='Re-selected from corrected baseline absolute errors within each horizon.',
    paper_comparability='Changed simulator and target losses; numerical comparison with printed Table 4 is not exact reproduction.',
    modes=all_results, simulator_calls=calls, new_training=False,
    hashes={'loader':sha(adapter),'runner':sha(__file__), 'task2':sha(REPO/'archive/code/run_task2_model_based.py'),
      'archive_macro':sha(REPO/'archive/code/dexposure_fm/macroprudential_tools.py'),
      'cloud_macro':sha(REPO/'cloud/train_src/dexposure_fm/macroprudential_tools.py'),
      'frozen_selection':sha(TRAIN/'selection_frozen.json'),'prediction_manifest':sha(TRAIN/'tuning_manifest.sha256')}))
print('PASS: both protocols, all three graphs, independent loss/aggregate checks, no oracle substitutions.',flush=True)
