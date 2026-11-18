from types import SimpleNamespace
from unittest import TestCase

from sglang_simulator.simulation.manager import ConfigManager
from sglang_simulator.simulation.sglang.runtime_adapter import canonical_worker_type
from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.time_predictor.aisimulate import make_perf_config


class NativeWorkerIdentityTests(TestCase):
    def test_native_role_survives_canonical_predictor_configuration(self):
        for mode in ("null", "prefill", "decode"):
            with self.subTest(mode=mode):
                scheduler = SchedulerConfig(
                    tp_size=16, dp_size=4, ep_size=16,
                    backend_version="qualified-runtime",
                    worker_type=canonical_worker_type(mode),
                )
                role = "aggregated" if mode == "null" else mode
                config = make_perf_config(
                    SimpleNamespace(model_path="/model"),
                    SimpleNamespace(name="ascend910c"),
                    scheduler,
                    {"worker_type": role},
                )
                self.assertEqual(config["worker_type"], role)
                self.assertEqual(config["tp"], 4)
                self.assertEqual(config["attention_dp"], 4)

    def test_conflicting_role_cannot_reuse_an_aggregated_identity(self):
        scheduler = SchedulerConfig(worker_type="decode")
        with self.assertRaisesRegex(ValueError, "worker_type"):
            make_perf_config(None, None, scheduler, {"worker_type": "aggregated"})

    def test_saved_native_server_args_preserve_role(self):
        for mode in (None, "null", "prefill", "decode"):
            with self.subTest(mode=mode):
                scheduler = ConfigManager._parse_server_args({"disaggregation_mode": mode}, "sglang")
                self.assertEqual(scheduler.worker_type, canonical_worker_type(mode))

    def test_unknown_native_role_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "disaggregation mode"):
            canonical_worker_type("unknown")
