"""Small, offline checks for the validation-only tuning and recovery contract."""
from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cloud.train_src import tune_fm as tuning
from cloud import preflight
from cloud import verify_v2_checkpoint


def metrics(ap=.97, auc=.995, edge_mae=2.4, edge_rmse=3.4, node_mae=.2, node_rmse=.8):
    return {"exist": {"auprc": ap, "auroc": auc},
            "weight": {"mae": edge_mae, "rmse": edge_rmse},
            "node": {"mae": node_mae, "rmse": node_rmse}}


class TuningTests(unittest.TestCase):
    def test_host_preflight_then_real_entry_reaches_data_loading(self):
        import torch

        class ReachedDataLoading(Exception):
            pass

        def load_metadata(_path):
            raise ReachedDataLoading("startup completed; stop before loading real data")

        core = SimpleNamespace(
            DGL_CUDA_AVAILABLE=True,
            ExperimentConfig=lambda **kwargs: SimpleNamespace(meta_path="unused"),
            load_metadata=load_metadata,
        )
        manifest = tuning.read_json(ROOT / "cloud/preflight_manifest.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cloud/train_src").mkdir(parents=True)
            for relative in ("cloud/preflight_manifest.json", "cloud/train_src/run_full_experiment.py"):
                (root / relative).write_bytes((ROOT / relative).read_bytes())
            input_checkpoint = root / "checkpoints/graphpfn-v1.ckpt"
            input_checkpoint.parent.mkdir()
            input_checkpoint.write_bytes(b"input checkpoint placeholder")
            preflight._check_output_parents(root, preflight._output_specs(manifest, "main2025_fm_tuning"))
            train, release = root / tuning.TRAIN_REL, root / tuning.RELEASE_REL
            self.assertTrue((train / "finetuned").is_dir())
            self.assertFalse(any(p.is_file() for directory in (train, release) for p in directory.rglob("*")))
            with patch.object(tuning, "ROOT", root), patch.object(sys, "path", sys.path.copy()), \
                    patch.object(sys, "argv", ["tune_fm.py", "--run"]), \
                    patch.dict(sys.modules, {"run_full_experiment": core,
                                             "verify_v2_checkpoint": verify_v2_checkpoint}), \
                    patch.object(torch.cuda, "get_device_properties",
                                 return_value=SimpleNamespace(total_memory=40 * 1024**3)):
                with self.assertRaises(ReachedDataLoading):
                    tuning.main()
            self.assertTrue((train / "plan.json").is_file())
            self.assertTrue((train / "source.tar.gz").is_file())
            self.assertIn("ReachedDataLoading", (train / "failure.txt").read_text())
            self.assertEqual(input_checkpoint.read_bytes(), b"input checkpoint placeholder")

    def test_output_setup_accepts_only_directory_trees(self):
        for scaffold in (False, True):
            with self.subTest(scaffold=scaffold), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                if scaffold:
                    for relative in (tuning.TRAIN_REL / ".empty/nested", tuning.RELEASE_REL):
                        (root / relative).mkdir(parents=True)
                train, release = tuning.prepare_output_dirs(root)
                self.assertEqual((train, release), (root / tuning.TRAIN_REL, root / tuning.RELEASE_REL))
                self.assertTrue((train / "finetuned").is_dir())
                tuning.prepare_output_dirs(root)

    def test_prepare_output_dirs_allows_resume_trial_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trial = root / tuning.TRAIN_REL / "trials/h1/baseline"
            trial.mkdir(parents=True)
            (trial / "resume_h1.partial.pt").write_bytes(b"ckpt")
            (root / tuning.RELEASE_REL).mkdir(parents=True)
            self.assertTrue(tuning.has_trial_artifacts(root))
            with self.assertRaises(tuning.OutputDirectoryError):
                tuning.prepare_output_dirs(root)
            train, release = tuning.prepare_output_dirs(root, allow_existing=True)
            self.assertEqual(train, root / tuning.TRAIN_REL)
            self.assertTrue((trial / "resume_h1.partial.pt").is_file())
            self.assertTrue((train / "finetuned").is_dir())

    def test_load_screen_summaries_requires_every_candidate(self):
        plan = {"screen_epochs": 6, "candidates": [{"name": "baseline"}, {"name": "exist_weight_4"}]}
        with tempfile.TemporaryDirectory() as temporary:
            train = Path(temporary)
            first = train / "trials/h1/baseline"
            first.mkdir(parents=True)
            tuning.write_json(first / "summary.json", {
                "candidate": "baseline", "horizon": 1, "completed_epochs": 6,
                "rank": [3, -0.01, -0.01], "status": "screened"})
            self.assertIsNone(tuning.load_screen_summaries(train, 1, plan))
            second = train / "trials/h1/exist_weight_4"
            second.mkdir(parents=True)
            tuning.write_json(second / "summary.json", {
                "candidate": "exist_weight_4", "horizon": 1, "completed_epochs": 6,
                "rank": [2, -0.2, -0.02], "status": "screened"})
            summaries = tuning.load_screen_summaries(train, 1, plan)
            self.assertEqual([item["candidate"] for item in summaries], ["baseline", "exist_weight_4"])

    def test_output_setup_rejects_existing_files_links_and_special_paths(self):
        cases = (
            (tuning.TRAIN_REL / "trial/weights.pt", "file"),
            (tuning.RELEASE_REL / ".hidden", "empty_file"),
            (tuning.TRAIN_REL, "file"),
            (tuning.RELEASE_REL, "file"),
            (Path("checkpoints"), "file"),
            (tuning.TRAIN_REL, "directory_link"),
            (tuning.RELEASE_REL, "dangling_link"),
            (Path("checkpoints"), "directory_link"),
            (tuning.TRAIN_REL / "nested/link", "directory_link"),
            (tuning.TRAIN_REL / "nested/link", "file_link"),
            (tuning.TRAIN_REL / "nested/link", "dangling_link"),
            (tuning.TRAIN_REL / "pipe", "fifo"),
        )
        for relative, kind in cases:
            with self.subTest(path=relative, kind=kind), tempfile.TemporaryDirectory() as temporary:
                root, outside = Path(temporary) / "run", Path(temporary) / "outside"
                outside.mkdir()
                sentinel = outside / "sentinel"
                sentinel.write_bytes(b"do not alter")
                path = root / relative
                path.parent.mkdir(parents=True)
                if kind in {"file", "empty_file"}:
                    path.write_bytes(b"original" if kind == "file" else b"")
                elif kind == "fifo":
                    os.mkfifo(path)
                else:
                    target = {"directory_link": outside, "file_link": sentinel,
                              "dangling_link": outside / "missing"}[kind]
                    path.symlink_to(target)
                with patch.object(tuning, "ROOT", root), \
                        patch.object(sys, "argv", ["tune_fm.py", "--run"]):
                    with self.assertRaises(tuning.OutputDirectoryError):
                        tuning.main()
                self.assertEqual(sentinel.read_bytes(), b"do not alter")
                if kind in {"file", "empty_file"}:
                    self.assertEqual(path.read_bytes(), b"original" if kind == "file" else b"")
                elif kind.endswith("link"):
                    self.assertTrue(path.is_symlink())
                self.assertFalse((root / tuning.TRAIN_REL / "failure.txt").exists())
                self.assertFalse((outside / "failure.txt").exists())

    def test_rejected_attempt_preserves_previous_failure_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failure = root / tuning.TRAIN_REL / "failure.txt"
            failure.parent.mkdir(parents=True)
            failure.write_bytes(b"original failure evidence")
            with patch.object(tuning, "ROOT", root), \
                    patch.object(sys, "argv", ["tune_fm.py", "--run"]):
                with self.assertRaises(tuning.OutputDirectoryError):
                    tuning.main()
            self.assertEqual(failure.read_bytes(), b"original failure evidence")
            self.assertFalse((root / tuning.RELEASE_REL).exists())

    def test_unreadable_output_tree_fails_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / tuning.TRAIN_REL
            train.mkdir(parents=True)
            with patch.object(tuning, "ROOT", root), \
                    patch.object(sys, "argv", ["tune_fm.py", "--run"]), \
                    patch.object(Path, "iterdir", side_effect=PermissionError("denied")):
                with self.assertRaises(tuning.OutputDirectoryError):
                    tuning.main()
            self.assertFalse((train / "failure.txt").exists())
            self.assertFalse((root / tuning.RELEASE_REL).exists())

    def test_rank_protects_all_six_metrics(self):
        baseline = metrics()
        better = metrics(.98, .996, 2.3, 3.3, .19, .79)
        tradeoff = metrics(.99, .999, 2.3, 3.3, .19, .81)
        self.assertGreater(tuning.validation_rank(better, baseline), (0, 0))
        self.assertLess(tuning.validation_rank(tradeoff, baseline), (0, 0))
        self.assertEqual(tuning.validation_rank(baseline, baseline), (0, 0))

    def test_paper_surplus_rank_counts_rounding_beats(self):
        printed = [0.972, 0.995, 2.465, 3.388, 0.056, 0.400]
        short = metrics(0.9724, 0.9954, 2.464, 3.387, 0.055, 0.399)
        beat = metrics(0.973, 0.996, 2.4, 3.3, 0.05, 0.39)
        worse_ap = metrics(0.970, 0.996, 2.4, 3.3, 0.05, 0.39)
        self.assertEqual(tuning.paper_surplus_rank(short, printed)[0], 4)
        self.assertEqual(tuning.paper_surplus_rank(beat, printed)[0], 6)
        self.assertGreater(tuning.paper_surplus_rank(beat, printed),
                           tuning.paper_surplus_rank(worse_ap, printed))
        plan = {"rank_mode": "paper_surplus", "paper_printed": {"1": printed}}
        self.assertEqual(tuning.rank_metrics(beat, metrics(), plan, 1),
                         tuning.paper_surplus_rank(beat, printed))

    def test_gate_rejects_changed_metric(self):
        plan = tuning.read_json(tuning.PLAN_PATH)
        tuning.require_agreement(metrics(), metrics(), plan)
        with self.assertRaises(ValueError):
            tuning.require_agreement(metrics(.975), metrics(), plan)
        with self.assertRaises(ValueError):
            tuning.validation_rank(metrics(float("nan")), metrics())

    def test_approved_regression_tolerance_does_not_relax_classification(self):
        plan = tuning.read_json(tuning.PLAN_PATH)
        expected = metrics(.9755985342838878, .9956923252856781,
                           2.2023587226867676, 3.2257222335342948,
                           .053542621433734894, .39885670917238936)
        replay = metrics(.9755989513791486, .9956922856958119,
                         2.2022950649261475, 3.225658225376408,
                         .053544264286756516, .39886098683343474)
        tuning.require_agreement(replay, expected, plan)
        original = {k: v for k, v in plan.items() if k != "regression_agreement_atol"}
        with self.assertRaisesRegex(ValueError, "weight.mae"):
            tuning.require_agreement(replay, expected, original)
        for group, metric in tuning.METRICS:
            changed = metrics()
            delta = 2e-5 if group == "exist" else 2e-4
            changed[group][metric] += delta
            with self.subTest(group=group, metric=metric), self.assertRaises(ValueError):
                tuning.require_agreement(changed, metrics(), plan)
        # The original relative tolerance still applies for larger magnitudes.
        tuning.require_agreement(metrics(edge_mae=100.0005), metrics(edge_mae=100), plan)
        with self.assertRaises(ValueError):
            tuning.require_agreement(metrics(edge_mae=100.002), metrics(edge_mae=100), plan)
        for key in ("agreement_atol", "regression_agreement_atol", "agreement_rtol"):
            changed = copy.deepcopy(plan)
            changed[key] *= 2
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                tuning.validate_plan(changed)

    def test_bound_plan_and_existing_contracts(self):
        manifest = tuning.read_json(ROOT / "cloud/preflight_manifest.json")
        tuning.validate_plan(tuning.read_json(tuning.PLAN_PATH))
        for command in ("bash cloud/fm_tune_run.sh", "bash cloud/fm_eval_paper_run.sh",
                        "bash cloud/main2025_v2_remaining_run.sh --all",
                        "bash cloud/main2025_v2_remaining_run.sh"):
            preflight._validate_binding(manifest, command)
        changed = copy.deepcopy(manifest)
        changed["bindings"]["main2025_fm_tuning"]["input_fetch"]["revision"] = "0" * 40
        with self.assertRaises(preflight.PreflightError):
            preflight._validate_binding(changed, "bash cloud/fm_tune_run.sh")

    def test_new_search_uses_validation_without_rewriting_old_replay_failure(self):
        plan = copy.deepcopy(tuning.read_json(tuning.PLAN_PATH))
        with tempfile.TemporaryDirectory() as tmp:
            train = Path(tmp)
            gate = {"status": "not_run", "policy": "validation_reference_only",
                    "reason": "historical_test_agreement_not_required_for_new_search"}
            references = {str(h): metrics() for h in plan["horizons"]}
            for h in plan["horizons"]:
                path = train / "baseline" / f"dexposure-fm-h{h}.pt"
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(f"baseline {h}".encode())
                plan["baseline_sha256"][str(h)] = tuning.sha256(path)
            tuning.write_json(train / "baseline_validation.json", references)
            tuning.write_json(train / "baseline_gate.json", gate)
            tuning.verify_baseline_evidence(train, plan)
            tuning.write_json(train / "baseline_gate.json", {**gate, "status": "passed"})
            with self.assertRaisesRegex(ValueError, "must not claim"):
                tuning.verify_baseline_evidence(train, plan)
            tuning.write_json(train / "baseline_gate.json", gate)
            references.pop("12")
            tuning.write_json(train / "baseline_validation.json", references)
            with self.assertRaisesRegex(ValueError, "Incomplete validation"):
                tuning.verify_baseline_evidence(train, plan)
            references["12"] = metrics()
            tuning.write_json(train / "baseline_validation.json", references)
            (train / "baseline/dexposure-fm-h4.pt").write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                tuning.verify_baseline_evidence(train, plan)
            legacy = {k: v for k, v in plan.items() if k != "baseline_policy"}
            rows = {str(h): {"metrics": metrics(), "expected": metrics()} for h in plan["horizons"]}
            rows["4"]["metrics"]["weight"]["mae"] -= .000182
            tuning.write_json(train / "baseline_gate.json", {"status": "passed", "horizons": rows})
            with self.assertRaisesRegex(ValueError, "weight.mae"):
                tuning.verify_baseline_evidence(train, legacy)

    def test_new_search_only_touches_test_pairs_after_selection_freeze(self):
        text = Path(tuning.__file__).read_text()
        tree = ast.parse(text)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        source = ast.get_source_segment(text, function)
        before, after = source.split('selection_sha = sha256(train / "selection_frozen.json")')
        self.assertNotIn('groups["test"]', before)
        self.assertNotIn('require_agreement(', source)
        self.assertNotIn('baseline_expected', source)
        self.assertIn('groups["test"]', after)
        self.assertIn('"status": "not_run"', before)
        manifest = tuning.read_json(ROOT / "cloud/preflight_manifest.json")
        self.assertFalse(any(p["path"].endswith("/baseline_predictions")
                             for p in preflight._output_specs(manifest, "main2025_fm_tuning")))

    def test_trial_has_no_test_or_paper_input_and_preserves_resume_state(self):
        tree = ast.parse(Path(tuning.__file__).read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_trial")
        used = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
        self.assertFalse({"test_pairs", "test_results", "paper_printed"} & used)
        arguments = {arg.arg for arg in function.args.args}
        self.assertFalse({"test_pairs", "test_snaps"} & arguments)
        source = ast.get_source_segment(Path(tuning.__file__).read_text(), function)
        for fragment in ('optimizer.load_state_dict', 'scaler.load_state_dict', 'restore_rng(saved["rng"])'):
            self.assertIn(fragment, source)

    def test_controller_keeps_four_horizons_and_nested_artifact_verification(self):
        controller = (ROOT / "cloud/vast_api_train.sh").read_text()
        runner = (ROOT / "cloud/vast_api_remote_run.sh").read_text()
        self.assertIn('BINDING_NAME" = main2025_fm_tuning', controller)
        self.assertIn('--verify "$run_root" --source-sha256 "$SOURCE_SHA"', controller)
        self.assertIn('"$TRAIN_REL/tuning_manifest.sha256"', controller)
        self.assertIn('"$TRAIN_REL/tuning_manifest.sha256"', runner)
        self.assertIn('"$TRAIN_REL/finetuned/best_model_h1.pt"', runner)
        for path in ("cloud/fm_tune_run.sh", "cloud/vast_api_train.sh", "cloud/vast_api_remote_run.sh"):
            subprocess.run(["bash", "-n", str(ROOT / path)], check=True)

    def test_plan_contains_every_horizon_and_baseline_identity(self):
        plan = tuning.read_json(tuning.PLAN_PATH)
        for h in (1, 4, 8, 12):
            digest = plan["baseline_sha256"][str(h)]
            self.assertEqual(len(digest), 64)
            int(digest, 16)
        self.assertEqual(plan["max_train_seconds"], 86400)
        self.assertEqual(plan["max_offer_dph"], .81)

    def test_resume_preserves_optimizer_and_all_rng_streams(self):
        import numpy as np
        import torch
        from dataclasses import dataclass

        @dataclass
        class Config:
            lr: float = .01
            weight_decay: float = .0001
            use_amp: bool = False
            epochs: int = 20
            seed: int = 42
            device: str = "cpu"
            forecast_horizons: object = None
            exist_loss_weight: float = 2.
            weight_loss_weight: float = .5
            node_loss_weight: float = 20.
            data_path: str = "unused"
            meta_path: str = "unused"
            checkpoint_path: str = "unused"

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(1, 1)
                self.head = torch.nn.Linear(1, 1)

        def build(_core, _cfg):
            tuning.random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            return Model()

        def epoch(model, _pairs, optimizer, *_args, **_kwargs):
            noise = tuning.random.random() + np.random.rand() + torch.rand(()).item()
            optimizer.zero_grad()
            loss = (model.head(model.encoder(torch.ones(1, 1))) - noise).square().sum()
            loss.backward()
            optimizer.step()
            return {"loss": loss.item()}, None

        def predict(model, _pairs, _cfg):
            value = sum(p.detach().abs().sum().item() for p in model.parameters())
            return metrics(.95 + .01 / (1 + value), .99, 2 + value, 3 + value, .1 + value, .2 + value)

        core = SimpleNamespace(train_graphpfn_epoch=epoch, predict_graphpfn=predict,
                               evaluate_predictions=lambda m: m,
                               build_checkpoint_provenance=lambda **kwargs: kwargs)
        candidate = dict(tuning.read_json(tuning.PLAN_PATH)["candidates"][0])
        plan = {"rank_mode": "validation_vs_baseline"}
        with tempfile.TemporaryDirectory() as tmp, patch.object(tuning, "build_model", build), patch.object(tuning, "log"):
            direct, resumed = Path(tmp) / "direct", Path(tmp) / "resumed"
            args = (core, Config(), candidate, 1, [], [], metrics(), {}, {})
            tuning.train_trial(*args, direct, 20, plan)
            tuning.train_trial(*args, resumed, 6, plan)
            tuning.train_trial(*args, resumed, 20, plan, resume=True)
            left = torch.load(direct / "resume_h1.partial.pt", weights_only=False)
            right = torch.load(resumed / "resume_h1.partial.pt", weights_only=False)
            self.assertEqual(left["epoch"], right["epoch"])
            for key in left["model"]:
                torch.testing.assert_close(left["model"][key], right["model"][key], rtol=0, atol=0)
            self.assertEqual(left["best_rank"], right["best_rank"])
            self.assertEqual(left["best_epoch"], right["best_epoch"])


if __name__ == "__main__":
    unittest.main()
