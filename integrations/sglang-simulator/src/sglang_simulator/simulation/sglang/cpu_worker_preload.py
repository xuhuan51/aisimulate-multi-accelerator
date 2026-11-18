# SPDX-License-Identifier: Apache-2.0
# Modified for AISimulate multi-accelerator workload replay.
# Upstream attribution is retained in LICENSE and source-manifest.json.
"""Preload CPU scheduler imports once in the simulator's fork server."""
import gc
import os

if os.environ.get("SGLANG_USE_CPU_ENGINE") != "1":
    raise RuntimeError("Simulator worker preload requires the CPU engine")

from sglang_simulator.simulation.sglang.hook_bootstrap import install_simulator_hooks

install_simulator_hooks()

import torch
from sglang.srt.managers import (  # noqa: F401
    data_parallel_controller,
    detokenizer_manager,
    scheduler,
)
from sglang.srt.models.registry import ModelRegistry  # noqa: F401

torch.set_num_threads(1)
gc.freeze()
