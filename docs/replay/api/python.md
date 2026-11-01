<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python Replay API

The Python API compiles public configuration into a data-only runner contract.
Its `ReplaySpec` is not the native Rust JSON structure with the same name.
Use the compiler rather than translating fields between them by hand.

## Public configuration to runner

```python
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper.replay import ReplayOutputRequirements

config = CorePredictionConfig.from_yaml("prediction.yaml")
spec = prediction_to_replay_spec(config)
factory = EngineReplayRunnerFactory()
factory.capabilities().require_compatible(spec)
runner = factory.create(worker_id=0)
try:
    report = runner.run(
        spec,
        output_requirements=ReplayOutputRequirements(
            include_raw_report=True,
            capture_per_request=True,
        ),
    )
finally:
    runner.close()
```

The configuration above is the same core YAML used by `predict --stack engine`.
`CorePredictionConfig` does not accept optional Router/Planner top-level sections;
the CLI resolves and compiles those through their selected configuration adapters.
The compiler signature is:

```python
prediction_to_replay_spec(
    config: CorePredictionConfig,
    *,
    adapter_specs: dict[str, AdapterReplaySpec] | None = None,
    afd_performance_model: AFDPerformanceModel | None = None,
    execution_mode: str = "offline",
) -> ReplaySpec
```

Compilation resolves concrete traffic, topology, capacity and per-role estimator
identities. `ForwardPassEstimatorSpec.config` is the sole estimator identity;
its convenience properties are projections. Do not independently edit a
model/backend/version property after materialization. Configuration and
capability validation precede native execution but do not guarantee measured
accuracy or availability of every timing query.

## Inputs and reports

The [runner ABI](../../adapters/runner-abi.md) defines `ReplaySpec`,
`BackendDeploymentSpec`, `RunnerFactory`, `RunnerCapabilities`, and `Runner`.
`spec.workload` contains concrete trace or synthetic inputs; `spec.concurrency`
is a closed-loop cap or `None` for open loop. `spec.goal` carries evaluation
requirements. Adapter data and runtime hooks are serialized values, not live
policy objects.

`ReplayReport.metrics` supplies numeric values used by scoring. Non-power
metrics must be finite numbers; only designated power fields can be null.
`metadata` retains provenance and optional native detail. Engine runner raw
reports are retained under `metadata["native_report"]` when requested; do not assume
that every optional runner exposes identical metadata. Agentic results retain
`agentic_qualification: functional_only` and phase/snapshot/profile evidence.
See [result definitions](../../getting-started/understand-results.md) before
comparing latency or throughput across runs.

`ReplayOutputRequirements` defaults to summary output. It can request a raw
report, per-request capture, memory diagnostics or performance diagnostics.
These controls retain additional evidence without changing the workload.
`capture_performance_diagnostics=True` also publishes
`metadata["native_report"]["forward_pass_batches"]` from completed native forwards.
Each row identifies the serving role, logical worker, phase, and either an
attention DP rank or the whole logical worker (`dp_rank: null`). The histogram
and request-instance counters describe actual scheduled batches, excluding
queued requests, idle ranks, and unfinished forwards. Whole-worker batch sizes
sum the ranks in one committed grouped forward; padded tensor shapes do not
inflate these counts. The mean batch size is `request_instances / passes`, and
the forward-duration-weighted mean is `request_duration_ms / duration_ms` when
duration is positive. Worker duration is counted once per grouped forward.
These are forward observations, not periodic occupancy gauges or client
concurrency. Prefill, decode, and mixed forwards have separate rows; preparation
epochs are excluded when the runtime starts a new measured report.
Telemetry fields are part of the shared contract, but the built-in Engine
JSON runner currently rejects `capture_telemetry=True`; the native Rust
observer is a separate interface. Analytical EPD does not provide raw native
or per-request EPD reports. Large capture can require substantial memory.

## Lifecycle and errors

One runner is reusable within one worker. The production Engine factory loads
`aisimulate._runtime` lazily and has no Dynamo dependency. A missing extension
raises `RunnerUnavailableError`; an invalid runtime or unsupported runtime
combination raises the runner's validation error. `close()` is currently a
no-op for this built-in runner; consumers must still call it for other runners.

Low-level calls do not automatically provide the CLI's subprocess host-memory
supervision. The caller owns resource budgets, output publication and failure
handling, especially for long agentic profiles. Use
[local resource controls](../../reference/local-resources.md) when executing
through the CLI or guarded Sweeper. The native JSON bridge and private runner
lowering helpers are implementation interfaces, not replacements for this API.

Source: [compiler](../../../python/aisimulate/src/aisimulate/compiler.py),
[runner](../../../python/aisimulate/src/aisimulate/runner.py), and
[contract definitions](../../../python/aisimulate/src/aisimulate/sweeper/replay.py).
