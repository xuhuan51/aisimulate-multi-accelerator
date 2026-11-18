import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class WorkerBootstrapTests(unittest.TestCase):
    def run_bootstrap(self, code, cpu=True):
        install = Mock()
        torch = SimpleNamespace(cuda=SimpleNamespace(get_device_capability=Mock()))
        hooks = SimpleNamespace(install_simulator_hooks=install)
        modules = {"torch": torch, "sglang_simulator.simulation.sglang.hook_bootstrap": hooks}
        with patch.dict(os.environ, {"SGLANG_SIMULATOR_BOOTSTRAP": "1", "SGLANG_USE_CPU_ENGINE": "1" if cpu else "0"}, clear=True), \
                patch.dict(sys.modules, modules), patch.object(sys, "argv", ["-c"]), \
                patch.object(sys, "orig_argv", ["python", "-c", code], create=True):
            path = Path(__file__).parents[1] / "src/usercustomize.py"
            exec(compile(path.read_text(), str(path), "exec"), {"__name__": "usercustomize_test"})
        return install, torch

    def test_resource_tracker_keeps_its_standard_library_startup(self):
        install, _ = self.run_bootstrap("from multiprocessing.resource_tracker import main;main(7)")
        install.assert_not_called()

    def test_spawn_worker_installs_hooks_before_unpickling(self):
        install, torch = self.run_bootstrap("from multiprocessing.spawn import spawn_main;spawn_main()")
        install.assert_called_once_with()
        self.assertEqual(torch.cuda.get_device_capability(), (10, 0))

    def test_physical_runtime_does_not_install_cpu_hooks(self):
        install, _ = self.run_bootstrap("from multiprocessing.spawn import spawn_main;spawn_main()", cpu=False)
        install.assert_not_called()
