"""Run with python -m unittest reproduction.test_release."""
import json
import math
import unittest
from unittest.mock import patch
from reproduction import prepare
from reproduction.evaluate import metric_comparison, cpu_attention


class ReleaseTests(unittest.TestCase):
    def test_cpu_fused_attention_matches_math(self):
        import torch
        from lib.limix.model.layer import MultiheadAttention
        torch.manual_seed(42)
        module = MultiheadAttention(32, 2, dropout=0)
        for mask in (None, torch.zeros(7, 9)):
            q = torch.randn(1, 7, 2, 16)
            kv = torch.randn(1, 9, 2, 2, 16)
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True,
                                               enable_mem_efficient=False):
                expected = module.compute_attention_by_torch(None, q, kv, mask)
                actual = cpu_attention(module, None, q, kv, mask)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_comparison_detects_improvement_regression_and_invalid_metric(self):
        reference = json.loads((prepare.ROOT/'reproduction/paper_metrics.json').read_text())
        observed = json.loads(json.dumps(reference))
        observed['h1']['exist']['auprc'] += 0.01
        observed['h4']['weight']['mae'] += 0.1
        rows = metric_comparison(observed, reference)
        self.assertEqual(len(rows), 24)
        self.assertAlmostEqual(rows[1]['delta'], 0.01)
        self.assertAlmostEqual(rows[8]['delta'], 0.1)
        observed['h1']['node']['mae'] = math.nan
        with self.assertRaises(ValueError):
            metric_comparison(observed, reference)

    def test_bad_file_hash_is_rejected(self):
        with patch.object(prepare, 'sha256', return_value='wrong'):
            with self.assertRaises(ValueError):
                prepare.verify_files()

    def test_runtime_is_the_training_source(self):
        manifest = json.loads((prepare.ROOT/'reproduction/training_manifest.json').read_text())
        self.assertEqual(prepare.sha256(prepare.ROOT/'reproduction/runtime.py'),
                         manifest['shared_contract']['core_training_file_sha256'])

    def test_limited_base_loader_rejects_wrong_bytes_without_network(self):
        import lib.tfm
        with patch('lib.tfm.Path.is_file', return_value=True), \
             patch('lib.tfm.Path.read_bytes', return_value=b'wrong weights'), \
             patch('lib.tfm.hf_hub_download', side_effect=AssertionError('unexpected network')):
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                lib.tfm.load_tfm('LimiX', {})


if __name__ == '__main__':
    unittest.main()
