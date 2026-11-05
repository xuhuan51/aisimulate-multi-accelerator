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
