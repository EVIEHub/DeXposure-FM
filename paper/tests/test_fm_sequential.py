"""CPU check of real Adam-state inheritance, early stopping, and interrupted resume."""
from dataclasses import dataclass
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from cloud.train_src import tune_fm as t
from cloud.train_src import sequential_fm as s


@dataclass
class Config:
    forecast_horizons: object = None
    lr: float = .0005
    exist_loss_weight: float = 2.
    weight_loss_weight: float = .5
    node_loss_weight: float = 20.
    weight_decay: float = .01
    use_amp: bool = False
    epochs: int = 40
    data_path: str = 'test-data'
    meta_path: str = 'test-meta'
    checkpoint_path: str = 'test-base'
    seed: int = 42


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1)
        self.head = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.head(self.encoder(x))


class SequentialTest(unittest.TestCase):
    def test_inheritance_and_resume(self):
        plan = {'experiment_kind': 'sequential_horizon_training', 'rank_mode': 'validation_auprc',
                'early_stop_patience': 2}
        candidate = {'name': 'baseline', 'lr': .0005, 'encoder_lr': .00005,
                     'exist_loss_weight': 2., 'weight_loss_weight': .5, 'node_loss_weight': 20.}
        observed, counter = [], [0]
        interrupt_at = [None]

        def build(*_):
            torch.manual_seed(42)
            random.seed(42)
            return Model()

        def train(model, pairs, optimizer, config, *_args, **_kwargs):
            counter[0] += 1
            if counter[0] == interrupt_at[0]:
                raise RuntimeError('test interruption')
            observed.append({'weight': model.encoder.weight.detach().clone(),
                             'steps': [int(v['step']) for v in optimizer.state.values()],
                             'loss_weight': config.exist_loss_weight})
            optimizer.zero_grad()
            loss = model(torch.rand(1, 1)).square().sum()
            loss.backward()
            optimizer.step()
            return {'total': loss.item()}, None

        def evaluate(_):
            # Epoch two wins; two subsequent failures stop each stage at four.
            score = [.8, .9, .7, .6][(counter[0] - 1) % 4]
            return {'exist': {'auprc': score, 'auroc': score},
                    'weight': {'mae': 1., 'rmse': 1.}, 'node': {'mae': 1., 'rmse': 1.}}

        core = SimpleNamespace(build_checkpoint_provenance=lambda **kw: kw,
                               train_graphpfn_epoch=train, predict_graphpfn=lambda *_: None,
                               evaluate_predictions=evaluate)
        with tempfile.TemporaryDirectory() as tmp, patch.object(t, 'build_model', build):
            root = Path(tmp)
            def run(directory, horizon=1, parent=None, resume=False, cand=candidate):
                return t.train_trial(core, Config(), cand, horizon, [], [], None, {}, {},
                                     directory, 40, plan, resume=resume, parent=parent)
            summary = run(root / 'h1')
            self.assertEqual((summary['selected_epoch'], summary['completed_epochs']), (2, 4))
            parent = root / 'h1/best_model_h1.pt'
            parent_state = torch.load(parent, weights_only=False)
            self.assertTrue(all(int(v['step']) == 2 for v in parent_state['optimizer']['state'].values()))
            counter[0] = 0
            observed.clear()
            next_candidate = dict(candidate)
            h4 = run(root / 'h4', 4, parent, cand=next_candidate)
            self.assertTrue(torch.equal(observed[0]['weight'], parent_state['model']['encoder.weight']))
            self.assertEqual(observed[0]['steps'], [2, 2, 2, 2])
            self.assertEqual(observed[0]['loss_weight'], 2.)
            self.assertEqual(h4['parent_sha256'], t.sha256(parent))
            self.assertEqual(h4['cumulative_selected_epochs'], 4)
            final = torch.load(root / 'h4/best_model_h4.pt', weights_only=False)
            self.assertTrue(all(int(v['step']) == 4 for v in final['optimizer']['state'].values()))
            # Repeat the same second stage, interrupt after epoch two, then resume.
            counter[0], interrupt_at[0] = 0, 3
            with self.assertRaisesRegex(RuntimeError, 'test interruption'):
                run(root / 'resume', 4, parent, cand=next_candidate)
            # Simulate interruption during separate history/best artifact writes.
            (root / 'resume/history.json').write_text('[]')
            (root / 'resume/best_model_h4.pt').write_bytes(b'incomplete write')
            counter[0], interrupt_at[0] = 2, None
            resumed = run(root / 'resume', 4, parent, True, next_candidate)
            self.assertEqual(resumed, h4)
            last = torch.load(root / 'resume/resume_h4.partial.pt', weights_only=False)
            uninterrupted = torch.load(root / 'h4/resume_h4.partial.pt', weights_only=False)
            for key in last['model']:
                self.assertTrue(torch.equal(last['model'][key], uninterrupted['model'][key]))
            # Reject accidentally joining a checkpoint to a different predecessor.
            with self.assertRaisesRegex(ValueError, 'different stage parent'):
                run(root / 'resume', 4, None, True, next_candidate)

    def test_plan_and_controller_binding(self):
        plan = t.read_json(s.PLAN_PATH)
        s.validate_plan(plan)
        manifest = t.read_json(t.ROOT / 'cloud/preflight_manifest.json')
        binding = manifest['bindings']['main2025_fm_sequential']
        self.assertEqual(binding['epochs'], plan['epochs'])
        self.assertEqual(binding['horizons'], plan['horizons'])
        plan['stages'] = {'8': dict(plan['parameters'], exist_loss_weight=4.)}
        with self.assertRaisesRegex(ValueError, 'overrides are not allowed'):
            s.validate_plan(plan)
        del plan['stages']
        plan['parameters']['exist_loss_weight'] = 6.
        with self.assertRaisesRegex(ValueError, 'parameter selection'):
            s.validate_plan(plan)


if __name__ == '__main__':
    unittest.main()
