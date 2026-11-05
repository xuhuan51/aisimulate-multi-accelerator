# SPDX-License-Identifier: Apache-2.0
# Modified for AISimulate multi-accelerator workload replay.
# Upstream attribution is retained in LICENSE and source-manifest.json.
from importlib import import_module

from sglang_simulator.time_predictor.base import (
    InferTimePredictor,
    ScheduleBatch,
    ScheduleRequest,
)

__all__ = (
    "AIConfiguratorTimePredictor",
    "AISimulateTimePredictor",
    "InferTimePredictor",
    "MLTimePredictor",
    "ReplayTimePredictor",
    "ScheduleBatch",
    "ScheduleRequest",
)


def __getattr__(name):
    modules = {
        "AIConfiguratorTimePredictor": "aiconfigurator",
        "AISimulateTimePredictor": "aisimulate",
        "MLTimePredictor": "ml",
        "ReplayTimePredictor": "replay",
    }
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{modules[name]}"), name)
