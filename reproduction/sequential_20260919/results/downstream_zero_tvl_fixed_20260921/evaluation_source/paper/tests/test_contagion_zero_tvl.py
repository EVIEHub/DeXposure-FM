"""Run: uv run --no-project --with numpy python paper/tests/test_contagion_zero_tvl.py"""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
PATHS = (
    "archive/code/run_task2_model_based.py",
    "archive/code/dexposure_fm/macroprudential_tools.py",
    "cloud/train_src/dexposure_fm/macroprudential_tools.py",
)


class ContagionZeroTVLTests(unittest.TestCase):
    def test_zero_and_small_positive_tvl_use_the_same_loss_cap(self):
        for path in PATHS:
            spec = importlib.util.spec_from_file_location("contagion_test", ROOT / path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for tvl in (0.0, 1e-9, 1e-6, 0.004, 5.0, 100.0):
                with self.subTest(path=path, creditor_tvl=tvl):
                    graph = {
                        "nodes": {"debtor": {"tvlUsd": 100.0}, "creditor": {"tvlUsd": tvl}},
                        "edges": [{"source": "creditor", "target": "debtor", "weight": 1.0}],
                    }
                    result = module.simulate_contagion(graph, ["debtor"], shock_fraction=0.5)
                    expected_loss = 50.0 + min(50.0, tvl)
                    self.assertAlmostEqual(result["total_loss"], expected_loss)
                    self.assertAlmostEqual(result["total_loss_pct"], 100 * expected_loss / (100 + tvl))
                    if tvl == 0:
                        self.assertEqual(result["distressed_nodes"], ["debtor"])
            with self.subTest(path=path, case="all_zero"):
                result = module.simulate_contagion(
                    {"nodes": {"a": {"tvlUsd": 0.0}}, "edges": []}, ["a"]
                )
                self.assertEqual(result["total_loss"], 0.0)
                self.assertEqual(result["total_loss_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
