# SPDX-License-Identifier: Apache-2.0
# Modified for AISimulate multi-accelerator workload replay.
# Upstream attribution is retained in LICENSE and source-manifest.json.
from abc import ABC, abstractmethod


class BaseBenchmarkRunner(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def benchmark(self) -> dict:
        pass

    @abstractmethod
    def flush_cache(self):
        pass

    @abstractmethod
    def shutdown(self):
        pass
