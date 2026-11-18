import os
import unittest
from unittest.mock import patch

from sglang_simulator.simulation.sglang.runtime_adapter import configure_cpu_multiprocessing


class CpuStartMethodTests(unittest.TestCase):
    def test_default_uses_the_native_start_method(self):
        with patch.dict(os.environ, {}, clear=True), patch("multiprocessing.set_start_method") as setter:
            configure_cpu_multiprocessing()
        setter.assert_not_called()

    def test_forkserver_preloads_cpu_workers(self):
        with patch.dict(os.environ, {
            "SGLANG_SIMULATOR_CPU_START_METHOD": "forkserver", "SGLANG_USE_CPU_ENGINE": "1"
        }, clear=True), patch("multiprocessing.get_all_start_methods", return_value=["spawn", "forkserver"]), \
                patch("multiprocessing.set_forkserver_preload") as preload, \
                patch("multiprocessing.set_start_method") as setter:
            configure_cpu_multiprocessing()
        preload.assert_called_once_with(["sglang_simulator.simulation.sglang.cpu_worker_preload"])
        setter.assert_called_once_with("forkserver", force=True)

    def test_invalid_method_is_rejected(self):
        with patch.dict(os.environ, {"SGLANG_SIMULATOR_CPU_START_METHOD": "fork"}, clear=True):
            with self.assertRaisesRegex(ValueError, "must be spawn or forkserver"):
                configure_cpu_multiprocessing()

    def test_forkserver_requires_cpu_execution(self):
        with patch.dict(os.environ, {"SGLANG_SIMULATOR_CPU_START_METHOD": "forkserver"}, clear=True):
            with self.assertRaisesRegex(ValueError, "requires SGLANG_USE_CPU_ENGINE=1"):
                configure_cpu_multiprocessing()

    def test_forkserver_requires_platform_support(self):
        with patch.dict(os.environ, {
            "SGLANG_SIMULATOR_CPU_START_METHOD": "forkserver", "SGLANG_USE_CPU_ENGINE": "1"
        }, clear=True), patch("multiprocessing.get_all_start_methods", return_value=["spawn"]):
            with self.assertRaisesRegex(ValueError, "unavailable on this platform"):
                configure_cpu_multiprocessing()
