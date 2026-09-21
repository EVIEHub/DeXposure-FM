"""Render corrected Figures 2-3 from saved results and check bar bounds.
MPLBACKEND=Agg uv run --no-project --with numpy --with matplotlib python <this file>
"""
from pathlib import Path
import importlib.util
import json
import hashlib

OUT = Path(__file__).resolve().parent.parent
REPO = OUT.parents[3]
source = REPO/'archive/code/run_task2_model_based.py'
spec = importlib.util.spec_from_file_location('task2',source)
task = importlib.util.module_from_spec(spec);spec.loader.exec_module(task)
result = json.loads((OUT/'paper_common_nodes_all/exp2_predictive_contagion.json').read_text())
result['evaluation_note'] = 'Uniform TVL cap | 2025 hold-out | 107 origins / 321 scenarios'
plt = task._import_matplotlib()
save = plt.savefig
checked = []
def save_checked(path,*args,**kwargs):
    fig = plt.gcf()
    assert len(fig.axes) == 4
    for ax in fig.axes:
        lower,upper = ax.get_ylim()
        for bar in ax.patches:
            assert lower <= min(bar.get_y(),bar.get_y()+bar.get_height())
            assert max(bar.get_y(),bar.get_y()+bar.get_height()) <= upper
    checked.append(str(Path(path).relative_to(OUT)))
    return save(path,*args,**kwargs)
plt.savefig = save_checked
task.plot_contagion_comparison(result,OUT)
task.plot_contagion_advantage(result,OUT)
(OUT/'plot_verification.json').write_text(json.dumps(dict(bar_bounds_passed=checked,
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2)+'\n')
print('PASS: all bars fit within shared axis bounds in both figures.')
