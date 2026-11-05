# SPDX-License-Identifier: Apache-2.0
# Modified for AISimulate multi-accelerator workload replay.
# Upstream attribution is retained in LICENSE and source-manifest.json.
from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkConfig:
    request_rate: float = float("inf")
    max_concurrency: Optional[int] = None
    ignore_request_timestamp: bool = False
