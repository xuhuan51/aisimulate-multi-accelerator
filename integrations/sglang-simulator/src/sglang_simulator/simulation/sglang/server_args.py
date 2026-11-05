# SPDX-License-Identifier: Apache-2.0
# Modified for AISimulate multi-accelerator workload replay.
# Upstream attribution is retained in LICENSE and source-manifest.json.
"""Provide declared target capability during GPU-oriented argument resolution."""


class ServerArgsAdapter:
    @staticmethod
    def install():
        # Direct installation also covers callers that import ServerArgs before
        # the simulator Engine. Class-construction hooks cannot cover that case.
        from sglang.srt.server_args import ServerArgs as target

        from sglang_simulator.simulation.sglang.runtime_adapter import runtime_flavor

        if runtime_flavor() == "ascend":
            return

        if getattr(target.resolve_once, "_simulator_target_capability", False):
            return
        original_resolve = target.resolve_once

        def resolve(self):
            from sglang_simulator.simulation.manager import ConfigManager

            accelerator = (
                ConfigManager._get_raw_config()
                .get("platform", {})
                .get("accelerator", {})
            )
            capability = accelerator.get("cuda_compute_capability")
            if capability is None or self.device != "cpu":
                return original_resolve(self)
            if (
                not isinstance(capability, list)
                or len(capability) != 2
                or any(type(n) is not int or n < 0 for n in capability)
            ):
                raise ValueError("cuda_compute_capability must be [major, minor]")
            import torch

            # DSA's native resolution queries CUDA even for CPU simulation.
            # Scope this replacement to configuration; no GPU is initialized.
            original_capability = torch.cuda.get_device_capability
            torch.cuda.get_device_capability = lambda device=None: tuple(capability)
            try:
                return original_resolve(self)
            finally:
                torch.cuda.get_device_capability = original_capability

        resolve._simulator_target_capability = True
        target.resolve_once = resolve
