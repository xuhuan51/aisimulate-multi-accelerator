import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from sglang_simulator.simulation.sglang.model_config import C_MambaPoolHook, C_ModelConfigHook


class ModelConfigMetadataTests(TestCase):
    def make_config(self, path, draft=False):
        class ModelConfig:
            def _find_quant_modelslim_config(self):
                return {"original": True}

        C_ModelConfigHook.hook(ModelConfig)
        config = ModelConfig()
        config.model_path = path
        config.is_draft_model = draft
        return config

    def test_cpu_mock_retains_format_without_reading_weight_entries(self):
        with TemporaryDirectory() as path:
            description = Path(path, "quant_model_description.json")
            description.write_text("unread weight-loader payload")
            config = self.make_config(path)
            with patch.dict(os.environ, {"SGLANG_USE_CPU_ENGINE": "1"}):
                self.assertEqual(config._find_quant_modelslim_config(), {"quant_method": "modelslim"})

    def test_cpu_mock_preserves_absent_and_draft_metadata(self):
        with TemporaryDirectory() as path, patch.dict(os.environ, {"SGLANG_USE_CPU_ENGINE": "1"}):
            self.assertIsNone(self.make_config(path)._find_quant_modelslim_config())
            Path(path, "quant_model_description.json").touch()
            self.assertIsNone(self.make_config(path, draft=True)._find_quant_modelslim_config())

    def test_non_cpu_calls_original_reader(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.make_config("unused")._find_quant_modelslim_config(), {"original": True})


class MambaPayloadTests(TestCase):
    def make_pool(self, device, cpu):
        class Pool:
            def __init__(self, **kwargs):
                self.received = kwargs
                self.device = kwargs["device"]

        C_MambaPoolHook.hook(Pool)
        with patch.dict(os.environ, {"SGLANG_USE_CPU_ENGINE": str(cpu)}):
            return Pool(size=257, device=device, cache_params="logical-shape")

    def test_cpu_payload_keeps_capacity_and_logical_device(self):
        pool = self.make_pool("cpu", 1)
        self.assertEqual(pool.received, {"size": 257, "device": "meta", "cache_params": "logical-shape"})
        self.assertEqual(pool.device, "cpu")

    def test_other_devices_keep_native_constructor(self):
        for device, cpu in [("cpu", 0), ("cuda", 1)]:
            with self.subTest(device=device, cpu=cpu):
                self.assertEqual(self.make_pool(device, cpu).received["device"], device)
