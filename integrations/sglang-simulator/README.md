# SGLang Simulator

For the dataclass-based Ascend SGLang fork, see
[Ascend scheduler/KV adapter](docs/ascend-adapter.md). Select it explicitly with
`SGLANG_SIMULATOR_RUNTIME=ascend` and `SGLANG_SIMULATOR_SGLANG_ROOT=/path/to/ascend-sglang`.
The adapter executes scheduler metadata on CPU; NPU kernels and performance
accuracy require separate validation.

SGLang Simulator reuses SGLang's scheduler and cache implementation while
replacing model forward execution with a latency predictor. It supports
timestamped trace replay, synthetic workloads, hierarchical cache simulation,
and serving-compatible metrics without loading model weights.

See the [SGLang Simulator advanced-feature guide](../../docs/docs/advanced_features/sglang_simulator.mdx)
for the user-facing setup and serving workflow.

## Compatibility

The multiprocess integration uses this checkout's SGLang source and requires its
`attn_dp_size` and distributed bootstrap interfaces. Runtime capability checks
reject incompatible installations. Each simulated rank records the actual source
path and SGLang version in `metrics.json`.

On Linux, CPU simulations can share worker imports with
`SGLANG_SIMULATOR_CPU_START_METHOD=forkserver`. The default is `spawn`.
The fork server preloads scheduler and model-registry modules, including for
nested DP launchers. This option requires `SGLANG_USE_CPU_ENGINE=1`.

The CPU mock model retains the ModelSlim format marker while leaving per-weight
packing metadata to the replaced weight loader. Its Mamba/GDN state payloads use
meta tensors with the native shapes and dtypes. Native CPU slot allocators,
request-to-state mappings and logical cache capacities still govern scheduling.
The latency predictor receives the original model path and explicit precision
settings. Physical recurrent-state computation and payload transfer require
device validation.

Distributed runs stream complete forward records to `iteration.jsonl` as they
execute. Profile finalization writes request and metric summaries. An interrupted
run can leave a partial iteration log, so check its completion report before
using the run as a serving result.

## Requirements

- A compatible SGLang checkout. The simulator uses the SGLang source from the
  same monorepo checkout.
- A local model directory containing model configuration files. Tokenizer files
  are also required unless tokenizer initialization is disabled.
- Predictor data for AISimulate, AIConfigurator, ML, or replay mode.

Use an official SGLang image matching the checkout when validating GPU and
runtime compatibility.

## Installation

From the SGLang repository:

```bash
pip install -e tools/sglang-simulator
```

The simulator does not install or pin a second `sglang` package. Run it from a
checkout whose `python/sglang` package is available on `PYTHONPATH`, or from a
matching official SGLang image.

AIConfigurator is optional. Install it separately when using the
`aiconfigurator` predictor. The `aic` extra pins AIConfigurator to the exact release
validated with the simulator so upstream API changes cannot silently alter an
installation. In a clean virtual environment, install the extra with:

```bash
pip install -e "tools/sglang-simulator[aic]"
```

In an existing SGLang image, install the same pin without dependency resolution
to avoid replacing its NumPy/CUDA stack:

```bash
pip install --no-deps "aiconfigurator==0.10.0"
```

Upgrade this pin only after rerunning the AIC predictor and compatibility tests.

### AISimulate

The `aisimulate` predictor calls the public
`RustForwardPassPerfModel.best_available()` API in a persistent subprocess. Set
`predictor.python_executable` to a Python 3.11+ environment with AISimulate
installed. The SGLang environment can remain on Python 3.10; it does not need
AIConfigurator for this path. The example below uses the local `aiperf` environment;
edit that interpreter path on another machine.

Use [`examples/sim_configs/aisimulate_parallel.json`](examples/sim_configs/aisimulate_parallel.json)
for a TP8 / attention DP4 / EP8 CPU simulation. The example's A100 / SGLang 0.5.10
identity selects bundled performance data; set hardware, model and backend version
to the actual target for accuracy comparisons. `predictor.config` passes canonical
AISimulate options through, including `systems_paths`, `estimation_mode`, and
`fallback_policy`. Topology comes from `scheduler` and conflicting overrides fail.

Run from the repository root in the `sglang` environment:

```bash
python test/manual/sglang_simulator/verify_parallel.py \
  --predictor aisimulate --entrypoint http --scenario mixed \
  --output-dir /tmp/sglang-aisimulate-http
```

For your own server, use the same configuration with `--max-total-tokens` set to
the target's per-rank KV capacity. AISimulate currently requires `OFFLINE` mode
and an explicit KV capacity; automatic sizing still belongs to the legacy AIC
memory path. The manual check uses a small capacity of 4096 tokens.

For each forward, the coordinator sends one batch per attention DP group, including
idle groups, to AISimulate. Its whole-forward milliseconds become logical seconds;
no separate compute or communication cost is added. Predictor query time and host
CPU execution wall time are excluded; modeled host costs can be supplied through
`request_timing` below. `iteration.jsonl` records FPM inputs, predictions
and native padding; `metrics.json` records predictor provenance and query counts.
Native padding is not an explicit input to the current AISimulate public API, so
the adapter records it without adding fictitious requests. This integration does
not fix AISimulate's existing MoE or DP imbalance modeling limitations.

#### AISimulate PD replay

The installed runner also accepts `engine.mode: disaggregated` through
`aisimulate predict --stack sglang-simulator`. It launches native Prefill
and Decode cohorts, with independent canonical timing identities,
KV capacities and recurrent-state slot counts. Global client concurrency is
released by Decode client completion. Both roles record the same client-send time.

Each role's `parallelism.replicas` sets its cohort count. With
`pd_routing: replica_round_robin`, requests rotate over replicas and then over
each replica's Attention-DP ranks. All cohorts share one client-concurrency cap
and one virtual event clock. Cohort directories retain their own state pools,
runtime identity and forward records. Combined Decode metrics restore each
cohort's global client-time origin before computing throughput and latency.

For Qwen3.5, configure `kv_cache.recurrent_state` on each worker, matching
attention TP, KV page size and state geometry. Use offline requests arriving
at zero, prefix caching disabled, and an explicit bandwidth transfer model.
KV pages are sent per prompt chunk, with final GDN/convolution state and CPU
auxiliary metadata on the last chunk. Target payload bytes come from canonical
model geometry. The payload tensors use meta storage during CPU replay.

Install this package in the AISimulate environment, set
`SGLANG_SIMULATOR_PYTHON` to the selected SGLang interpreter, and run

```bash
SGLANG_SIMULATOR_RUN_ROOT=/tmp/sglang-pd-replays \
python -m aisimulate predict --stack sglang-simulator \
  -c qwen-pd.json --format json --output-dir /tmp/qwen-pd-result
```

`SGLANG_SIMULATOR_NATIVE_ARGS` can select a resolved launch template from the
target serving runtime. The replay applies each role's topology and cache settings.
The evidence directory contains per-role requests, iterations, state pools,
runtime source hashes, canonical predictor provenance, shared admission events
and chunk transfer records. Reported service metrics use the Decode client stream.
Bandwidth-only predictions retain their configured transport assumption.
Single-pair native replay has passed request-level validation. Multi-cohort
execution is under CPU validation. Cross-TP state repartition, measured transfer
providers and gateway response timing remain pending native validation.

To compare with the existing GLM-5.2/B300 paired GPU forward recordings without
rerunning GPU work:

```bash
python test/manual/sglang_simulator/verify_aisimulate_accuracy.py \
  --output-dir /tmp/sglang-aisimulate-forward-accuracy
```

The script defaults to the sibling AISimulate checkout's local SGLang 0.5.19
profiles and two measured TP8 deployments (DP2/EP2 and DP8/EP1), each with three
repeats, no MTP, and overlap disabled. Use `--help` to change paths. It pairs ranks
by iteration, reconstructs geometry from per-request lengths, and writes JSON/CSV
errors with input and native-library hashes. These are fixed observed forward
errors; they do not establish end-to-end TTFT, TPOT, or throughput accuracy.

The native scheduler can also be checked against these serving measurements:

```bash
python test/manual/sglang_simulator/verify_aisimulate_e2e.py \
  --case dp2-ep2-mtp-off-paired --mode closed-loop \
  --output-dir /tmp/sglang-aisimulate-e2e-dp2
```

Use `--case dp8-ep1-mtp-off` for the second deployment, or `--mode arrivals` for
recorded-arrival replay. The default is three repeats. Closed-loop mode maintains
32 outstanding requests and starts the next one after logical client completion; it does
not count time waiting for a client slot as request latency. `BenchmarkConfig`'s
`max_concurrency` is supported for multiprocess runs with infinite request rate
and `ignore_request_timestamp=True`. Recorded-arrival replay has no concurrency cap.

The harness reconstructs synthetic token IDs with measured lengths and the
page-aligned common prefix, and checks request totals, cache hits, capacity and
completion. Input, CPU and output costs default to zero; pass
`--request-timing-config /path/to/timing.json` to model them. This file contains
the `request_timing` object described below, without an outer wrapper. The scheduler is this
checkout's version, while the reference service/profile version is 0.5.19; both
identities remain in the artifacts. For DSA models, the example declares
`platform.accelerator.cuda_compute_capability: [10, 3]` for B300, used only while
resolving arguments that otherwise query a local CUDA device.

## Quick start

The maintained tests define the supported first-version scope:

- [`test/test_simulation_sglang_runner.py`](test/test_simulation_sglang_runner.py):
  direct Python use of the repository-level
  [`SGLangBenchmarkRunner`](../../benchmark/simulator/bench_runner.py);
- [`test/test_simulation_sglang_serving.py`](test/test_simulation_sglang_serving.py):
  server plus benchmark-client use through the HTTP serving path with AIC, ML,
  and replay predictors and ShareGPT or timestamped traffic;
- [`test/test_simulation_offline_blocking.py`](test/test_simulation_offline_blocking.py):
  equivalent logical results in `OFFLINE` and `BLOCKING` modes;
- [`test/test_simulation_cache_hit_ratio.py`](test/test_simulation_cache_hit_ratio.py):
  reusable-prefix accounting and cache-tier hit metrics across repeated runs.

From `tools/sglang-simulator`:

```bash
python3 -m pytest -q test/test_simulation_sglang_runner.py
python3 -m pytest -q test/test_simulation_sglang_serving.py
```

Read these tests as the minimal maintained examples for constructing a dataset,
running a benchmark, starting a simulator server, sending programmatic, ShareGPT,
or timestamped traffic, comparing execution modes, and collecting request,
latency, throughput, and prefix-cache metrics.

## Serving mode

Choose a fresh output directory and export it in the server terminal before
starting the server:

```bash
export SGLANG_USE_CPU_ENGINE=1
export CUDA_VISIBLE_DEVICES=""
export SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE
export SIMULATOR_OUTPUT_DIR=/tmp/sglang-simulator-serving-001
test ! -e "$SIMULATOR_OUTPUT_DIR"
export SGLANG_SIMULATOR_OUTPUT_DIR="$SIMULATOR_OUTPUT_DIR"

python3 -m sglang_simulator.simulation.sglang.launch_server \
  --model-path /absolute/path/to/model \
  --sim-config-path /absolute/path/to/simulator.json \
  --port 30000
```

In the benchmark terminal, export the same output directory before sending
timestamped traffic with the simulator-aware benchmark adapter:

```bash
cd /path/to/sglang
export SIMULATOR_OUTPUT_DIR=/tmp/sglang-simulator-serving-001
export SGLANG_SIMULATOR_OUTPUT_DIR="$SIMULATOR_OUTPUT_DIR"

python3 benchmark/simulator/bench_serving.py \
  --simulator-mode offline \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model /absolute/path/to/model \
  --dataset-name autobench \
  --dataset-path /absolute/path/to/trace.jsonl \
  --use-trace-timestamps \
  --num-prompts 100 \
  --profile \
  --output-file "$SIMULATOR_OUTPUT_DIR/benchmark.json"
```

The server and benchmark are separate processes, so exporting
`SGLANG_SIMULATOR_OUTPUT_DIR` in the server terminal does not configure the
benchmark terminal. The benchmark adapter reads `metrics.json` from this path
after profiling and uses those server-side logical-time metrics for its serving
table and output file. If the benchmark points at another directory, it may show
unrelated stale metrics or client wall-clock values. Use the same fresh path in
both terminals for every run.

Set the deployment topology under `scheduler` in `--sim-config-path`; it takes
precedence over parallel CLI options. `tp_size` is the total rank count, `dp_size`
is attention DP, and `ep_size` is expert parallelism. TP8 / DP4 / EP8 launches
eight CPU scheduler workers with native Gloo metadata groups. Each attention group
has TP2; MoE uses TP1 / EP8. Model weights and accelerator kernels are mocked.
Ordinary request-replica DP remains 1. The current multiprocess scope is one host,
`OFFLINE`, PP1 / CP1, with overlap, speculative decoding and HiCache disabled.

Other server options are normal SGLang command-line arguments. For direct Python
integration, see
[`test_simulation_sglang_runner.py`](test/test_simulation_sglang_runner.py); for the
process/HTTP path, see
[`test_simulation_sglang_serving.py`](test/test_simulation_sglang_serving.py).

## Simulation modes

| Mode | Behavior |
|---|---|
| `OFFLINE` | Advances the simulator's logical clock without sleeping. |
| `BLOCKING` | Sleeps for predicted forward and visible L2-to-L1 load latency. |

Use server-side simulator metrics for comparisons. Client wall-clock duration is
not the simulated timeline in `OFFLINE` mode. When using the benchmark adapter,
make sure its `SGLANG_SIMULATOR_OUTPUT_DIR` matches the server's output directory
so the printed table and `benchmark.json` are sourced from the current run's
`metrics.json`.

## Configuration

A simulator configuration has three main sections and optional `request_timing`:

```json
{
  "platform": {
    "accelerator": {"name": "h20_sxm"},
    "disk_read_bandwidth_gb": 8,
    "disk_write_bandwidth_gb": 8,
    "memory_read_bandwidth_gb": 64,
    "memory_write_bandwidth_gb": 64,
    "num_device_per_node": 1
  },
  "predictor": {
    "name": "replay",
    "database_path": "/absolute/path/to/replay_table.json"
  },
  "scheduler": {
    "tp_size": 4,
    "ep_size": 4,
    "dp_size": 1,
    "pp_size": 1,
    "cp_size": 1,
    "cp_style": "none",
    "data_type": "BF16",
    "kv_cache_data_type": "BF16",
    "backend_name": "sglang"
  }
}
```

- `platform` describes the simulated accelerator and storage bandwidth.
- `predictor` selects forward-latency prediction.
- `scheduler` describes the real target deployment topology and backend metadata.
  `tp_size`, `ep_size`, `dp_size`, `pp_size`, and `cp_size` are modeled values;
  they do not launch physical workers. For AIConfigurator, `tp_size` is converted
  to attention TP after removing modeled DP and CP, while `cp_size` is passed as
  AIConfigurator context parallelism. `cp_style` uses the AIConfigurator values
  such as `none`, `allgather`, `ulysses`, or `ring`. Decode-only `dcp_size` has no
  separate AIConfigurator field and is not modeled yet.

### Prefix-cache accuracy

Prefix-cache hit accuracy is highly sensitive to `max_total_tokens`. It controls
the simulated device KV-cache capacity and participates in hierarchical host-cache
sizing, so a mismatch changes eviction timing and device, host, and storage hit
attribution. For deployment-faithful results, copy `max_total_num_tokens=N` from
the real SGLang server startup log and launch the simulator with
`--max-total-tokens N`. Avoid relying on a separately estimated capacity when
comparing the simulator with production traces.

Supported predictors:

| Predictor | Purpose |
|---|---|
| `aiconfigurator` | Operator and module performance-database estimation. |
| `ml` | A trained sklearn-compatible 18-feature latency model. |
| `replay` | Exact or nearest-neighbor batch-composition replay. |

Relative predictor paths are resolved from the simulator configuration location.
Environment variables in paths use `${NAME}` syntax.

### Request stages and closed-loop feedback

The multiprocess `OFFLINE` simulator (TP > 1, normal scheduler loop) models:

```text
client send -> transport -> tokenizer queue/service -> dispatch
  -> scheduler receive -> native batch selection -> CPU pre -> forward -> CPU post
  -> output queue/service -> transport -> client token -> client completion
  -> release concurrency slot -> optional think time -> next client send
```

Add this optional object to the simulator JSON. All durations are milliseconds;
these zero defaults are placeholders, not hardware measurements:

```json
"request_timing": {
  "input_transport_ms": 0,
  "tokenize_ms": 0,
  "tokenize_per_token_ms": 0,
  "tokenizer_workers": 1,
  "dispatch_ms": 0,
  "scheduler_pre_ms": 0,
  "scheduler_pre_per_request_ms": 0,
  "scheduler_post_ms": 0,
  "scheduler_post_per_request_ms": 0,
  "output_ms": 0,
  "output_per_token_ms": 0,
  "output_workers": 1,
  "output_transport_ms": 0,
  "completion_ms": 0,
  "client_think_ms": 0,
  "stream_interval": 1
}
```

Tokenization service costs `tokenize_ms + input_tokens * tokenize_per_token_ms`.
Output service costs `output_ms + emitted_tokens * output_per_token_ms` per chunk.
The first token is emitted immediately; subsequent chunks contain up to
`stream_interval` tokens, with the last partial chunk flushed on completion.
The tokenizer and output pools are separate shared resources; their worker counts
control serialization. Transport and completion delays do not occupy these pools.
CPU pre/post costs are per scheduler iteration and rank, with a per-request term;
TP/DP costs are not summed as though every rank ran serially. Forward cost remains
the predictor's responsibility. This does not implement `event_loop_overlap`.

The coordinator processes client/input/output events during GPU intervals. Native
schedulers see ready inputs only at their next receive boundary. Closed-loop
concurrency spans client send through client completion, including the final
response delay; client slot waiting and think time are outside request latency.
The output pipeline is drained before writing metrics and resetting a run.

`GenericRequest.custom_params["timing"]` can override input/output duration fields
for individual requests. For diagnostic replay, set `created_time` and
`scheduler_ready_time` in seconds relative to the same monotonic origin. A recorded
ready time bypasses the input cost model, with unknown intermediate stages left
null; it is supported only with recorded arrivals, not closed-loop loading.
The old `queue_start` input is accepted as an alias for `scheduler_ready_time`.
HTTP callers put these fields in `sampling_params.custom_params.simulation`.

`request.jsonl` now includes a `timeline` with input stage timestamps, first
forward, server/client token timestamps, output chunks, and both completion times.
TTFT/TPOT are measured at logical client token reception; E2E and concurrency end
at client completion. Server TTFT and individual host-stage metrics are reported
separately. Simulator execution wall time never substitutes for target host costs.

Use measured service durations as inputs; queue waiting emerges from shared
resources and arrivals. A measured client-to-scheduler total already contains
waiting, and should not be inserted as tokenizer service time. The two pools and
affine CPU costs are explicit approximations; shared tokenizer/output event loops,
OS scheduling, and CPU interference are not automatically profiled or reproduced.

The deterministic CPU acceptance check uses synthetic durations and a 100 ms
forward; it checks TTFT=172 ms, TPOT=108 ms, E2E=293 ms and total time=913 ms for
three C1 requests, twice on the same eight workers:

```bash
python test/manual/sglang_simulator/verify_request_timing.py \
  --timing modeled --output-dir /tmp/sglang-request-timing-modeled
python test/manual/sglang_simulator/verify_request_timing.py \
  --timing zero --output-dir /tmp/sglang-request-timing-zero
```

### Measure request timing inputs

The request timing collector wraps a **real** SGLang server. It keeps the native
model and scheduling logic and adds no GPU synchronization. Use the same model,
parallel configuration, CPU allocation and tokenizer as the target deployment.
Launch it in the real serving environment, without the simulator's CPU-engine
or bootstrap flags, and with the intended GPUs visible:

```bash
python -m sglang_simulator.tools.request_timing serve \
  --trace-dir ./request-timing/traces \
  --model-path /path/to/model --tp 8 \
  --disable-overlap-schedule --port 30000
```

This first collector supports one node, PP1, one tokenizer and one detokenizer,
ordinary streaming text generation, and no speculative decoding or dynamic batch
tokenization. TP, attention DP and EP may exceed one. The output directory must
be new or empty. Server launch remains separate from the client:

```bash
python -m sglang_simulator.tools.request_timing bench \
  --base-url http://127.0.0.1:30000 --concurrency 1 \
  --num-requests 64 --prompt-words 128,1024,4096 --output-tokens 32 \
  --output ./request-timing/client-c1.json

python -m sglang_simulator.tools.request_timing report \
  --traces ./request-timing/traces --client ./request-timing/client-c1.json \
  --output-dir ./request-timing/calibration-c1
```

`bench --requests-jsonl workload.jsonl` instead uses every supplied request once;
each line contains `text` **or** `input_ids`, plus optional `sampling_params`.
Keep text and token-ID workloads separate. Generated prompt sizes are word counts;
the collector measures actual input token counts. Four warmup requests are excluded
by default. Repeat at the target concurrency (e.g. `--concurrency 32`) with a new
client output name, and report that run into a separate directory to inspect
batch-size effects. Match the real workload's lengths and streaming settings.

The report writes:

- `request_timing.json`: a plain timing object, usable with the existing
  `test/manual/sglang_simulator/verify_aisimulate_e2e.py --request-timing-config`,
  or as the value of `request_timing` in the simulator config.
- `report.json`: sample counts, p50/p90/p99, nonnegative affine fits and fit MAE,
  per-scheduler process statistics, omitted fields and trace/config provenance.

Tokenization service uses active wall time, including native tokenizer threads.
Scheduler, dispatch and API processing use thread CPU time to exclude blocking
GPU/IPC waits. Async wrappers measure individual coroutine execution slices, so
other tasks running during an `await` are excluded. Full wall durations are also
reported for diagnosis. Scheduler ranks are sampled individually, never summed.
Output cost combines amortized detokenization, API handling and JSON encoding,
including the observed rate of HTTP chunk coalescing. A single input/batch size
produces only a constant estimate, not an invented per-token/request slope.

Transport is **omitted by default**: client-to-API and serialization-to-client
intervals include event-loop waiting. `report --include-delivery-residuals` can
export these as workload-specific approximations only for C1 with matching
host/boot monotonic clocks. The dispatch-to-scheduler residual remains diagnostic:
it already includes waiting for the scheduler's next receive boundary. Missing
stages, failed requests, unfinished scheduler iterations and inconsistent final
token counts fail export rather than producing a partial calibration.

These estimates do not isolate host work inside `run_batch`, and the simulator's
single output pool approximates several real processes. Instrumentation itself
has overhead. Check the resulting TTFT/TPOT against a separate uninstrumented run;
collecting parameters alone does not establish improved prediction accuracy.

## Collect baselines and train an ML predictor

Use [collect_baseline](src/sglang_simulator/tools/collect_baseline/README.md) to capture request traces
and synchronized batch steps from a real SGLang server. Then use
[train_ml_time_predictor](src/sglang_simulator/tools/train_ml_time_predictor/README.md) to train an
18-feature latency model. Installing the package provides `sglang-simulator-collect`
and `sglang-simulator-train`, including a small collected training example.

## Workload formats

The Autobench trace format uses timestamps in milliseconds:

```json
{"prompt":[1,2,3],"prompt_len":3,"output_len":1,"timestamp":200}
```

Random and ShareGPT workloads are also supported by the runner API and serving
benchmark paths.

## Validation

Run the CPU compatibility and unit tests from the repository root:

```bash
pip install -e tools/sglang-simulator
python3 -m pytest -q tools/sglang-simulator/test/test_simulation_sglang_runner.py
python3 -m pytest -q tools/sglang-simulator/test/test_simulation_sglang_serving.py
```

Run the two files as separate pytest commands because the runner test installs
process-global simulator hooks and state.

Run repository checks before submitting:

```bash
git ls-files -z tools/sglang-simulator | \
  xargs -0 env SKIP=no-commit-to-branch pre-commit run --files
```

Runtime changes should also be validated in a matching official SGLang image
with both `OFFLINE` and `BLOCKING` modes. Predictor changes should report
step-level error, and scheduler or cache changes should compare request latency,
throughput, and prefix-cache reuse against measured traces.
