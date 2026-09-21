"""Offline checks for the pinned EVIEHub v1.0 2025-holdout eval."""
from __future__ import annotations

import subprocess
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cloud" / "train_src"))
from cloud import preflight
from cloud.train_src import eval_paper as evaluation


class EvalPaperTests(unittest.TestCase):
    def test_plan_pins_original_v1_release_and_shared_h8_h12_file(self):
        plan = evaluation.read_json(evaluation.PLAN_PATH)
        evaluation.validate_plan(plan)
        self.assertEqual(plan["paper_revision"], evaluation.PAPER_REVISION)
        self.assertEqual(plan["paper_files"]["8"], plan["paper_files"]["12"])
        self.assertEqual(plan["paper_files"]["8"]["name"], "dexposure-fm-h8-h12.pt")
        self.assertEqual(plan["max_train_seconds"], 14400)

    def test_detects_current_and_legacy_node_heads(self):
        current = {
            "node_head.net.0.weight": torch.zeros(256, 576),
            "node_head.net.3.weight": torch.zeros(128, 256),
            "node_head.net.5.weight": torch.zeros(1, 128),
        }
        legacy = {
            "node_head.net.0.weight": torch.zeros(256, 192),
            "node_head.net.2.weight": torch.zeros(1, 256),
        }
        self.assertEqual(evaluation.detect_layout(current), "current")
        self.assertEqual(evaluation.detect_layout(legacy), "legacy_self")
        with self.assertRaises(ValueError):
            evaluation.detect_layout({"node_head.net.0.weight": torch.zeros(256, 64)})

    def test_binding_is_eval_only_and_reuses_2025_inputs(self):
        manifest = evaluation.read_json(ROOT / "cloud/preflight_manifest.json")
        preflight._validate_binding(manifest, "bash cloud/fm_eval_paper_run.sh")
        binding = manifest["bindings"]["main2025_fm_eval_paper"]
        self.assertEqual(binding["epochs"], 0)
        self.assertEqual(
            binding["input_fetch"],
            manifest["bindings"]["main2025_v2_repeat20"]["input_fetch"],
        )
        script = (ROOT / "cloud/fm_eval_paper_run.sh").read_text()
        self.assertIn("cloud/train_src/eval_paper.py --run", script)
        self.assertNotIn("tune_fm.py --run", script)

    def test_controller_skips_v2_provenance_and_hashes_eval_manifest(self):
        controller = (ROOT / "cloud/vast_api_train.sh").read_text()
        runner = (ROOT / "cloud/vast_api_remote_run.sh").read_text()
        self.assertIn('BINDING_NAME" = main2025_fm_eval_paper', controller)
        self.assertIn("cloud/train_src/eval_paper.py", controller)
        self.assertIn('BINDING_NAME" != main2025_fm_eval_paper', controller)
        self.assertIn("bash cloud/fm_eval_paper_run.sh", runner)
        self.assertIn('"$TRAIN_REL/eval_manifest.sha256"', controller)
        self.assertIn('"$TRAIN_REL/eval_manifest.sha256"', runner)
        for path in ("cloud/fm_eval_paper_run.sh", "cloud/vast_api_train.sh",
                     "cloud/vast_api_remote_run.sh"):
            subprocess.run(["bash", "-n", str(ROOT / path)], check=True)

    def test_plan_rejects_split_h8_h12_files(self):
        plan = evaluation.read_json(evaluation.PLAN_PATH)
        plan["paper_files"]["12"] = {
            "name": "dexposure-fm-h12.pt",
            "sha256": "0" * 64,
        }
        with self.assertRaisesRegex(ValueError, "h8 and h12"):
            evaluation.validate_plan(plan)


if __name__ == "__main__":
    unittest.main()
