<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Introduction
Data collection is a standalone process for collecting the database for aiconfigurator. By default, you don't have to collect the data by yourself.
Small versions of database will not introduce huge perf difference. Say, you can use 1.0.0rc3 data of trtllm on h200_sxm and deploy the generated
configs with Dynamo + trtllm 1.0.0rc4 worker.

If you want to go through the process, you can try belowing commands. However, you need to prepare the env by yourself such as installing a specific trtllm version.
This process is not well verified, you need to debug sometimes.

For a framework-version or GPU-platform upgrade, follow the
[Collector Upgrade Playbook](../../../docs/perf-model/collector/upgrade.md).
The repo-tracked [`aic-auto-collect`](../.claude/skills/aic-auto-collect/SKILL.md)
skill applies that workflow during long, resumable collection runs.
SGLang 0.5.14 Hopper/Blackwell follow-up work must also consult the
platform alignment ledger, a local campaign record kept outside the repo
(`docs/perf_database/sglang-0.5.14-hopper-blackwell-ledger.md` in the
campaign workspace).

# Preparation

SGLang 0.5.14 standalone MLA rows distinguish kernel compute from KV storage.
On B200/B300, `trtllm_mla` quantizes Q/K/V internally when the KV cache is FP8,
so those rows use `mla_dtype=fp8, kv_cache_dtype=fp8`. On H200, the absorbed
576-dimensional FA3 path uses BF16 compute, including when KV storage is FP8.
BF16 inputs to the collector do not determine the kernel's compute precision.
Compute-precision labels are restricted to the audited `trtllm_mla`, `fa3`,
and `triton` backends; an unknown backend raises instead of receiving a BF16 label.
Legacy Blackwell rows labeled `bfloat16/fp8` need fresh measurements with the
correct label; do not copy their timings into a second precision key.

Before collecting the data, make sure you own the whole node and no interfierence happens.
Next, please enable persistent-mode and lock frequency of the node. Make sure the cooling system of the node is working well.
```bash
sudo nvidia-smi -pm 1
```
```bash
sudo nvidia-smi -ac yyy,xxx
```
xxx, yyy frequency can be queried by nvidia-smi -q -i 0, refer to the Max Clocks part, xxx is SM frequency, yyy is Memory frequency.
A script to set frequency:
```
#!/bin/bash

# Run nvidia-smi query and extract SM and Memory frequencies from Max Clocks
sm_freq=$(nvidia-smi -q -i 0 | grep -A 4 "Max Clocks" | grep "SM " | grep -o "[0-9]\+ MHz" | grep -o "[0-9]\+")
mem_freq=$(nvidia-smi -q -i 0 | grep -A 4 "Max Clocks" | grep "Memory " | grep -o "[0-9]\+ MHz" | grep -o "[0-9]\+")

# Check if frequencies were successfully extracted
if [ -z "$sm_freq" ] || [ -z "$mem_freq" ]; then
    echo "Error: Could not extract SM or Memory frequency from Max Clocks."
    exit 1
fi

# Generate the command
echo "sudo nvidia-smi -ac $mem_freq,$sm_freq"
```
Prepare a clean env with the target framework and nccl lib installed.

# Collect comm data
```bash
export PATH=$PATH:${NCCL_TEST_BIN_PATH}/
network/collect_comm.sh #all_reduce data will be collected using default trtllm backend
network/collect_comm.sh --all_reduce_backend vllm #all_reduce data will be collected using vllm backend
network/collect_comm.sh --all_reduce_backend vllm --device xpu #all_reduce data will be collected using vllm backend on XPU
```
Today we only collect intra-node comm. This script will collect custom allreduce data for trtllm within a node.
It will also collect nccl allreduce, all_gather, all2all, reduce_scatter using nccl.
The standalone scripts stage `nccl_perf.txt`, `oneccl_perf.txt`, and
`custom_allreduce_perf.txt`. Collector finalization converts accepted output
to parquet under
`python/aisimulate/src/aisimulate_core/systems/data/<system>/comm/<backend>/<version>/`.

# Model-centric cases and healing runs

The model-centric case layer (introduced as "collector v2") is how collection
coverage is declared, and it is unchanged under Collector V3 — V3 adds
op-centric *management* on top (per-family runtime pins, provenance sidecars,
declared reuse; see `docs/perf_database/collector-v3-op-centric-design.md`),
it does not replace how cases are declared or healed. A healing run collects
cases for a specific model/GPU pair instead of running every op bucket and
hoping the support matrix improves. Use `--model-cases-full` when you want
the case YAML to define a full model-centric run. Omitting all model-case
flags runs the backend registry directly without a model-centric case plan.

```bash
# Heal one model on one GPU type.
python3 collect.py --backend sglang \
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 \
  --gpu b200_sxm

# Inspect the resolved model plan without collecting.
python3 collect.py --backend sglang \
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 \
  --gpu b200_sxm \
  --plan-only

# Select the architecture case file directly.
python3 collect.py --backend trtllm \
  --model-architecture Qwen3MoeForCausalLM \
  --gpu b200_sxm \
  --plan-only

# Full model-centric run: aggregate base ops plus every model case YAML file.
python3 collect.py --backend trtllm --model-cases-full

# Raw backend registry run: no model-centric case plan.
python3 collect.py --backend trtllm

# Ephemeral subset filter (healing): substring match, repeatable.
python3 collect.py --backend trtllm --ops moe --case-filter "tp=4"
```

Case files (see `cases/README.md` for the full authoring guide):

```text
cases/base_ops/<op>.yaml               — shared sweep recipes (interpolation grid)
cases/models/<architecture>_cases.yaml — op activation + model shape values
cases/capabilities.yaml                — positive hardware floors (dtype/op -> min SM)
cases/denylist.yaml                    — hang/node-killer cases only
model_cases.py                         — resolves which ops a model plan collects
capabilities.py                        — generation-time capability/denylist filter
```

The plan is one equation: cases = dedup(base grid ∪ model shapes), then
intersected with hardware capability floors and minus the hang denylist.

TRT-LLM MLA precision and GPT-OSS TP profile collection evidence is recorded
in [the September 17 campaign](trtllm/gym-profiles-20260917.md).

## Whole-forward FPM campaign

Use the dedicated `python3 -m collector.fpm_forward` entry point for vLLM
whole-forward campaigns. The compatibility route
`python3 collector/collect.py --ops fpm_forward` remains supported and must be
selected alone. Both routes require a resolved model, `--gpu`, and
`--fpm-max-gpus`. The campaign derives a latency-blind prefill/decode
design from the model's AIC attention cases, asks Generator for three rendered
artifacts per cell — a keepalive Pod, LeaderWorkerSet, or Grove PodCliqueSet
manifest plus `fpm_env.sh` and a thin engine-launch `run.sh` — stages them with
its own in-pod runtime `fpm_exec.sh` (etcd lifecycle, result gate and checker,
completion barrier, exit-code policy), and publishes the formal table only
after every frozen cell passes.
For multinode runs, select the installed controller with
`--fpm-orchestrator lws|grove`. Start with `--plan-only` or a one-cell
`--smoke --limit 1` run.

```bash
python3 -m collector.fpm_forward \
  --model-path nvidia/GLM-5.2-NVFP4 \
  --gpu b200_sxm \
  --fpm-max-gpus 8 \
  --plan-only
```

For profile-based FPM collection, the profile's `backend_version` is the dataset
identity label. It may be a custom string such as `my-vllm-patch-3` and need not
match the installed vLLM package version. The collector retains the actual
container version separately in `collector-provenance.json` under
`runtime.backend_version`; runtime capability, benchmark-schema, topology and
precision checks still apply. An optional Dynamo version is provenance only and
does not select FPM templates or resolve the backend version. See the
[onboarding version contract](../../../docs/perf-model/fpm-self-service/implementation.md#plan-preview-and-explicitly-execute)
for guided setup, omitted-version detection inside the target image, and
publication identity.

The low-level entry point also accepts `--fpm-backend-version LABEL` for dataset
identity and `--fpm-runtime-backend-version VERSION` for an independently observed
package-version pin. Guided onboarding supplies that pin after its image
preflight. The low-level collector does not launch this separate CPU preflight:
without an observed pin it can collect timings and record actual package
provenance, but it cannot select audited version-specific memory/execution hooks.
Supply the observed pin or accepted runtime instrumentation when those hooks are
required. The op-level collector's runtime manifest and version-routing rules
below are unchanged.

### Incremental Ascend attention coverage

`python -m collector.sglang.ascend_curve_append --operation attention
--extend-attention-points --output-dir NEW_DIRECTORY OLD_CAMPAIGN NEW_CAMPAIGN`
adds previously unmeasured coordinates to an existing attention curve. Runtime,
hardware, collector source, timing protocol and serving setup must match. Any
duplicate physical coordinate is rejected, including a faster remeasurement;
the existing latency and observation hash are preserved. Multiple collection
identities are joined through explicit `coverage_sources` in the provenance
sidecar so the native reader still receives one verified identity per curve.
The default command continues to accept only disjoint whole curves. Both inputs
must already have passed their normal exporter checks; this command does not
turn diagnostic or held-out measurements into qualified calibration data.

### Ascend vector runtime and transport precision

`sglang/ascend_vector.py` accepts the audited SGLang commit
`baa9befa381511cd4d56f83980951e5a45fffc6d`. For the earlier measured commit
`76f9213a411018547f4fd6a75f36feaa4d6bed58`, collection supports the verified
`rope_attention` and `split_qk_norm` boundaries. Other roles on that revision
produce a recorded failure requesting a serving audit. Freeze a plan containing
the requested roles before collection so its exporter can verify complete
coverage. Input layout, numerical checks and timing qualification still apply.

The vector exporter records `0.5.21.dev223+g76f9213a4` for that earlier commit
and `0.5.22` for the newer commit. Publish each runtime into its corresponding
data directory using its measured image and ABI identity.
An optional `framework_commit` in the frozen vector plan pins the expected
runtime. Collection and export reject a different observed commit.

The DeepEP collector verifies actual INT8 hidden states and FP32 scales before
dequantization, then records both dtypes for every rank. The exporter rejects
explicit precision mismatches and incomplete precision pairs. Historical rows
without dtype fields retain the earlier collector's INT8/FP32 checked contract.
BF16 dispatch observations require a separately qualified transport and compute
path before publication through this INT8 contract.

Combine outputs also receive an explicit BF16 check and a per-rank dtype record.
Value equality alone accepts tensors with different dtypes, as documented by
[PyTorch](https://docs.pytorch.org/docs/2.10/generated/torch.equal.html).
Explicit non-BF16 combine records are rejected by the exporter. Historical
records without this field retain their existing contract but lack this new
explicit dtype evidence.

Generation-time admission may omit a topology only when a concrete AIC
size-vs-capacity estimate proves its configured token envelope cannot fit.
Missing performance data, bootstrap architectures, and structural estimator
errors remain runnable so the target runtime supplies the observation.

Model identity is resolved from a local checkpoint/config, the packaged AIC
config cache, or Hugging Face. A real model architecture that is not registered
by AIC uses a config-derived bootstrap template; an unresolved or empty config
fails before planning. When the checkpoint exists only inside the runtime Pod,
pass its real local metadata explicitly with
`--fpm-model-config /path/to/config.json` (a directory containing `config.json`
is also accepted). The resolved config snapshot and SHA-256 are frozen into the
collection plan.

Generator and Collector share the Dynamo-native benchmark result schema
version through the rendered `fpm_env.sh`; the staged `fpm_exec.sh` validates
that contract in-pod without a Collector-side text patch. Formal
`fpm_forward_perf.parquet` output uses metadata schema v6 and records
`collector_attempt_id`, `runtime_run_id`, `runtime_grid_digest`, and
`kv_seed_regime` on every row. The seed regime is `real_kv`, `fake_fallback`,
an approved `skip:<reason>`, `legacy`, or `n/a`; non-benign warm-up skips
remain `fake_fallback` rather than acceptable evidence. Republishing the same
attempt is idempotent. If a cell already exists under a different attempt or
runtime identity, first publisher wins: that entire cell is skipped while
non-overlapping cells in the same publication still land.

The default publication gate requires every frozen cell to pass.
`--fpm-publish-partial` explicitly publishes passed cells, records the exact
`missing_cells` in the checkpoint, and still returns a nonzero incomplete-run
result. The database checkpoint also records
`skipped_first_publisher_wins`. A validated completed schema-v6 database is
terminal on resume even if raw cell artifacts have since been pruned.
`run-manifest.json` schema v2 is invocation-scoped: its `attempts` include the
attempt ID, timestamps, final status, and partial phase timings for failed
attempts; a no-op resume writes no historical attempts.

## Failure philosophy: observe, don't predict

There is no declarative expected-failure layer. A case that cannot run on the
current framework version fails fast and is recorded in `errors_<module>.json`
and `collection_summary_<backend>.json` with a classification and a
(model, dtype) group label. The summary aggregates failures by group: a whole
group failing is a fix-me signal (collector bug, unverified combo, framework
gap), not something to tolerate or auto-skip. Missing points are tolerated
downstream by interpolation and fallback. Escalation rules — when a failure
deserves a denylist entry, a registry `unverified` marker, or a capability
floor — live in `.claude/rules/collector/failure_handling.md`.

`--gpu <type>` resolves the SM version from
`src/aisimulate_core/systems/<gpu>.yaml`; use `--sm <version>` on an
unregistered GPU. Without either, the local device capability is detected
automatically.

To add a new architecture, create one `cases/models/<architecture>_cases.yaml`
file. To add a new model in an existing architecture, add the model path to
that architecture's `model_paths` list. Add shared op sweeps to the matching
`cases/base_ops/<op>.yaml` file. Add a new op collector only when the existing
ops cannot generate the needed data points.

# Version Management

## Overview

Each backend (trtllm, vllm, sglang) has a **registry** (`registry.py`) that maps ops to collector modules, and a **version resolver** (`version_resolver.py`) that picks the right module at runtime. Individual collector files declare their compatibility via `__compat__`. The current collector framework versions and runtime images are declared in `framework_manifest.yaml`.

```text
framework_manifest.yaml — current collector framework versions and images (schema v2)
framework_manifest.py   — manifest loader/validator/resolver
op_catalog.py         — table→family identity map (op_backend_catalog.yaml)
model_cases.py       — model-centric model/SM case-plan resolver
registry.py          — declares which module handles which version range
version_resolver.py  — routes runtime version → module (packaging.version)
collect.py/collect_ops — validates __compat__ and fails incompatible ops
__compat__           — per-file metadata declaring supported framework versions
cases/               — model-centric case manifests and SM exceptions
wideep/              — WideEP collector namespace for special images/runtimes
wideep/*/registry.py — WideEP-only ops appended when the v2 plan requests them
network/             — collective communication collectors and Slurm network jobs
```

`framework_manifest.yaml` is `schema_version: 2`: each framework entry has a
`default` runtime (`version` + `images`) plus an optional `families:` map of
per-family overrides. Resolution for a given `(framework, op)` walks
`table → catalog family → families[family] or default` — exactly one runtime
per op, always. `families:` overrides require the op catalog
(`op_backend_catalog.yaml`); without it, family-scoped pins fail closed rather
than silently falling back to `default`.

WideEP runtimes (`wideep_sglang`, `wideep_trtllm`) are flattened into peer
framework entries rather than nested under a `wideep:` key, so the same
per-op resolution logic applies uniformly. Each WideEP entry adds
`base_framework` (the stock framework it forks from), `data_backend` (which
table namespace its output is written under), and `collector_dir`.
`collect.py` requires the exact public version for a run and rejects
mixed-pin selections across ops.

Image digests are policy, not decoration: the rule is syntactic — any image
reference containing `/` must be pinned with `@sha256:<64 hex>`; only bare
internal image names (no `/`, e.g. `deepseek-v4-blackwell`) are exempt. In
practice that means every public-registry reference is digest-pinned.
When bumping a version pin, re-fetch and update the digest for every image
variant touched — a stale digest silently keeps pulling the old image. Always
pin the manifest-list (index) digest, never a platform-child digest —
`docker buildx imagetools inspect <image>` prints the index digest; a child
digest pins one architecture and breaks the other platforms.

Use `collector.framework_manifest.resolve_op_runtime(framework, op)` to
resolve a single op's runtime, and
`collector.framework_manifest.validate_resolution()` to check the whole
manifest against every backend registry at once — it returns a list of error
strings (empty = valid) and is the fail-closed CI gate asserting every
registered op resolves to exactly one pinned runtime. See
`docs/perf_database/collector-v3-op-centric-design.md` §4 for the full
design.

## File Naming Convention

- **No version fork**: `collect_{op}.py` (no suffix)
- **Version forks**: `collect_{op}_v1.py`, `collect_{op}_v2.py`, ... (sequential)
- The number is just an ordering key; the actual version range lives in `__compat__`

## `__compat__` Format

Every collector file must declare `__compat__` after the license header:

```python
# SPDX-FileCopyrightText: ...
# SPDX-License-Identifier: Apache-2.0

__compat__ = "trtllm>=1.1.0"
```

The format is `<framework><constraints>` using PEP 440-style operators:
- `"trtllm>=1.1.0"` — open-ended (latest version)
- `"trtllm>=0.21.0,<1.1.0"` — bounded range
- `"sglang>=0.5.5"` — sglang example

Pre-release ordering is respected: `1.1.0rc2 < 1.1.0 < 1.1.0.post1`.

## Registry Format

Each entry in `registry.py` is an `OpEntry` dataclass (defined in `collector/registry_types.py`).
Exactly one of `module` (unversioned) or `versions` (versioned) must be provided — this is
validated at construction time.

```python
from collector.registry_types import OpEntry, VersionRoute

# Unversioned (no fork):
OpEntry(op="gemm", module="collector.trtllm.collect_gemm", get_func="...", run_func="...")

# Versioned (has forks) — VersionRoutes in descending min_version order:
OpEntry(op="myop", get_func="...", run_func="...", versions=(
    VersionRoute("X.Y.Z", "collector.<backend>.collect_myop_v2"),
    VersionRoute("0.0.0", "collector.<backend>.collect_myop_v1"),
))
```

The resolver picks the first `VersionRoute` where `min_version <= runtime_version`.

## Adding a New Op

1. Create `collector/<backend>/collect_myop.py`
2. Add `__compat__ = "<backend>>=X.Y.Z"` after the license header
3. Export `get_myop_test_cases()` and `run_myop()` (or `run_myop_torch()`)
4. Add an entry to `collector/<backend>/registry.py`:
   ```python
   OpEntry(op="myop", module="collector.<backend>.collect_myop", get_func="get_myop_test_cases", run_func="run_myop")
   ```
5. Run `pytest tests/unit/collector/ -m unit` to verify registry integrity

## Handling a Framework API Change

When upstream framework `X.Y.Z` changes an API that a collector depends on:

1. Rename the original file to `collect_{op}_v1.py` (if not already versioned)
2. Create `collect_{op}_v2.py` with the new API calls
3. Add `__compat__ = "<backend>>=X.Y.Z"` to the new file
4. Add `__compat__` upper bound to the old file if it doesn't have one (e.g. change `">=1.1.0"` to `">=1.1.0,<X.Y.Z"`)
5. Convert the registry entry from unversioned to versioned (or prepend a new `VersionRoute`):
   ```python
   # Before (unversioned):
   OpEntry(op="gemm", module="collector.trtllm.collect_gemm", ...)

   # After (versioned — all forks carry explicit _vN suffix):
   OpEntry(op="gemm", versions=(
       VersionRoute("X.Y.Z", "collector.<backend>.collect_gemm_v2"),
       VersionRoute("0.0.0", "collector.<backend>.collect_gemm_v1"),
   ), ...)
   ```
6. Run tests to validate

## Runtime Behavior

At collection time:
1. `collect.py` reads the backend registry
2. `build_collections()` resolves each op to the correct module for the detected framework version
3. If the resolved module declares `__compat__`, it is validated against the runtime version — mismatches fail that op explicitly and are recorded in the error summary
4. Unsupported versions (no matching entry) are skipped with a warning

## Tests

```bash
pytest tests/unit/collector/ -m unit
```

Covers: version parsing (PEP 440 with rc/post), `__compat__` constraint evaluation, module routing, and structural validation of all three registries.

# Collect gemm/attention/moe data/etc.

## Smoke Test

Use `--smoke` to quickly verify the collector runs end-to-end. It randomly samples 4 test cases per op instead of running the full suite.

```bash
# Smoke test for sglang
python3 collect.py --backend sglang --smoke

# Smoke a specific op
python3 collect.py --backend trtllm --ops moe --smoke
```

## Power Monitoring (Optional)

The collector supports GPU power monitoring during kernel execution using NVML. This feature is optional and disabled by default.

### Enable Power Monitoring
```bash
# Basic power monitoring
python3 collect.py --backend trtllm --measure_power

# With custom minimum duration (default: 1.0s)
python3 collect.py --backend trtllm --measure_power --power_test_duration_sec 2.0
```

### Options
- `--measure_power`: Enable NVML-based power monitoring (samples at 100ms intervals)
- `--power_test_duration_sec`: Minimum test duration for accurate power readings (default: 1.0s)

### Output
When power monitoring is enabled, performance CSV files will include additional columns:
- `power`: Average power consumption during kernel execution (Watts)
- `power_limit`: GPU power management limit (Watts)

**Example output:**
```csv
framework,version,device,op_name,kernel_source,gemm_dtype,m,n,k,latency,power,power_limit
TRTLLM,1.2.0,NVIDIA H200 SXM,gemm,torch_flow,bfloat16,1024,4096,4096,0.234,523.4,700.0
```

### Requirements
Power monitoring requires:
- `pynvml` Python package: `pip install pynvml`
- NVML support (NVIDIA drivers)

If unavailable, a warning is logged and execution continues without power data.

### Notes
- Power monitoring adds minimal overhead (<1%)
- Kernel iterations are automatically adjusted to meet minimum duration for accurate measurements
- Backward compatible: without `--measure_power`, CSVs remain unchanged

## CUDA Graph Fallback Support

The `benchmark_with_power` helper function now supports graceful fallback to eager execution when CUDA graph capture fails. This is particularly useful for complex operations like MOE (Mixture of Experts) with large batch sizes.

### Features
- **Automatic fallback**: When `allow_graph_fail=True`, CUDA graph capture failures trigger eager execution instead of raising exceptions
- **Power measurement in both paths**: Power monitoring works correctly in both graph replay and eager execution modes
- **Memory safety**: Automatic `torch.cuda.empty_cache()` call on graph capture failure to prevent memory fragmentation
- **Transparency**: Results include `used_cuda_graph` flag to indicate which execution path was used

### Usage Example
```python
from helper import benchmark_with_power

def my_kernel():
    # Your kernel code here
    moe.forward(hidden_states, logits)

# Use benchmark_with_power with fallback support
with benchmark_with_power(
    device=device,
    kernel_func=my_kernel,
    num_warmups=3,
    num_runs=6,
    repeat_n=1,
    allow_graph_fail=True,  # Enable graceful fallback
) as results:
    latency = results["latency_ms"]
    power_stats = results["power_stats"]  # Available in both paths

    # Check which execution path was used
    if not results["used_cuda_graph"]:
        print("CUDA graph capture failed, used eager execution")
```

### When to Use
- **Complex operations**: MOE, dynamic memory patterns, or operations that may not be graph-compatible
- **Large batch sizes**: When graph capture may fail due to memory constraints
- **Development/debugging**: To ensure collection continues even if graph capture fails

### Backward Compatibility
- Default behavior unchanged: `allow_graph_fail=False` maintains existing behavior
- Existing collectors work without modifications
- Only opt-in when needed for specific use cases

## For TensorRT-LLM
### Optional( Read if collecting for mxfp4 kernels)
To collect performance data for mxfp4 kernels used in GPTOSS models, depending on the version of TensorRT-LLM, you might need to manually install triton-kernels. If your version is **>= 1.3.0rc2**, nothing needs to be done and you can run the collection process directly. For any version before **1.3.0rc2**. Follow the below instructions to install triton-kernels and expose them to TensorRT-LLM
1. Build and install Triton (tested with the commit below):
```bash
git clone https://github.com/triton-lang/triton.git
cd triton
# Specific commit verified with TensorRT-LLM
git checkout f3067cd3bd0c29065fa4ecdb724b6f29cbabea5f
pip install -r python/requirements.txt # build-time dependencies
pip install wheel build
python3 setup.py bdist_wheel # if this step fails due cuda related error, you might need to set CUDA_HOME following the optional step
pip install ./dist/*.whl
```
2. (Optional) You may need to set CUDA_HOME env variable to successfully build triton
```bash
export CUDA_HOME=/usr/local/cuda
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
python3 setup.py bdist_wheel
```
3. Expose the Triton kernels to TensorRT-LLM The kernels are not packaged in the wheel, so set the environment variable TRITON_ROOT to your Triton clone:
```bash
export TRITON_ROOT=/local/user/triton
# TensorRT-LLM expects the kernels at:
#   $TRITON_ROOT/python/triton_kernels
```
### Run collection for all available operations
```bash
python3 collect.py --backend trtllm
```
For trtllm, the whole collecting process takes about 30 gpu-hours. On 8-gpu, it takes 3-4 hours.
Please note that the whole process will report a lot of missing datapoints with errors. But it's okay. Our system is kindof robust to fair amount of missing data.
Once everything is done, you might see mutliple xxx.txt files under the same folder. Refer to src/aisimulate_core/systems/ folder to prepare the database including
how many files are needed accordingly.

## Resume Collection (Checkpoint)

Checkpoint files are always written so an interrupted run can be resumed later:

```bash
# Normal run (writes checkpoint to .collector_checkpoint/)
python3 collect.py --backend trtllm

# Resume an interrupted run (skips already-attempted tasks)
python3 collect.py --backend trtllm --resume

# Custom checkpoint directory
python3 collect.py --backend trtllm --resume --checkpoint-dir /path/to/checkpoints
```

A task is marked **done** once it is attempted (success or failure).
Only tasks that never finished are re-queued on `--resume`;
`--resume-retry-failed` additionally re-runs previously failed tasks.
Running without `--resume` always starts fresh (overwrites old checkpoint).

## For SGLang

Use the pinned stock image, `lmsysorg/sglang:v0.5.14` (or its `-cu130`
variant), for normal SGLang collection.

```bash
python3 collect.py --backend sglang
```

This collects the stock SGLang ops, including:

- GEMM operations (FP8, BF16, INT8, INT4)
- MLA (Multi-head Latent Attention) for context and generation
- MLA BMM (Batch Matrix Multiplication) operations
- MoE (Mixture of Experts) operations
- Normal attention operations

Large-EP local expert compute is available as the separately routed `moe_ep`
operator under `wideep/sglang/collect_deepep_moe.py`; run it explicitly in its
pinned SGLang 0.5.10 image. WideEP MLA is not registered because its legacy
wrapper now reaches the stock 0.5.14-only module implementation.

### DeepEP multi-node collector
For models with DeepEP MoE, inter-node communication data requires a separate
multi-node setup. It is collected by the standalone
`wideep/sglang/collect_moe_a2a.py` collector, launched one Slurm job per
world size by `network/slurm/submit_moe_a2a.sh` — see section 4 of
`network/slurm/README.md`. The older manual log-scraping pipeline under
`wideep/sglang/deepep/` is deprecated (its README carries the retirement
story); do not add new data through it.

# Supporting a new large-EP (WideEP) model

Large-EP MoE uses stock `moe_perf` for modeled local expert compute and
`moe_a2a_perf` for measured dispatch/combine communication. To support model X:

1. **Declare the shapes.** Add the model to
   `cases/models/<Architecture>_cases.yaml` (new architecture: one new file;
   new model in an existing architecture: append to `model_paths`) and mark
   its `model_case_values.moe` rows with `wideep: true`. Correlated MoE
   dimensions (hidden/inter size, topk, expert count) stay together in one row;
   never cross another model's values. The `wideep` flag activates standalone
   A2A shape population.

2. **Ensure stock `moe_perf` coverage.** Local compute is modeled as a
   uniformly balanced stream over EP-local experts, using TP=1 stock MoE
   coordinates. Cross-version stock-table reuse is allowed.

3. **Collect `moe_a2a` across nodes.** From `network/slurm/`:
   ```bash
   bash submit_moe_a2a.sh                      # sglang DeepEP HT+LL, one job per world size
   bash submit_trtllm_alltoall.sh              # trtllm NVLink alltoall (single/multi-GPU NVL)
   ```
   The comm tables have no model column — a new model whose
   (hidden_size, topk, num_experts) tuple is already covered needs no new
   runs; a new physical shape lands in the sweep automatically once declared
   in step 1.

4. **Publish with sidecars.** Finalized communication parquet goes into
   `.../<system>/comm/<backend>/<version>/moe_a2a_perf.parquet` together
   with its `collection_meta.yaml` entry — never a parquet without its
   provenance. The per-world `moe_a2a` outputs need the cross-job merge
   procedure in `network/slurm/README.md` section 4.3.

## MoE table units and caveats

`moe_a2a_perf` records latency in microseconds; `load_moe_a2a_data` converts
leaves to milliseconds. Stock `moe_perf` retains its existing timing contract.

Stock `moe_perf` also supports optional Boolean `default_eligible` selection metadata. The [Core API contract](../../../docs/perf-model/configuration.md#choosing-a-forward-pass-api) defines automatic and exact-source selection. Finalization validates non-null Boolean flags and the named source required by `false`, excludes the flag from measurement identity, and preserves existing annotations when merging a legacy collection that omits the column. Collectors do not infer eligibility from kernel labels or row order.

**Legacy-overwrite caveats.** A new-schema row replaces a legacy-adapted
leaf only at the *same* key, and the legacy adapters derive their node/EP
geometry rather than reading it:

- Legacy sglang DeepEP comm tables carry no `ep_size`; the adapter assumes
  `ep_size = node_num * 8` (8-GPU HGX fleets). New `moe_a2a` worlds
  therefore overwrite those leaves only when collected at 8 GPUs/node — a
  4-GPU/node (GB200) world with the same node count keys differently and
  **coexists** with the legacy data instead of replacing it.
- The legacy trtllm alltoall table (GB200 NVL4) carries no `num_nodes`
  column; the adapter derives `node_num = max(1, moe_ep_size // 4)`. New
  rows overwrite only at 4 GPUs/node; a row with an explicit `num_nodes`
  column is honored as written.

**`log_perf` freezes the CSV header from the first row**, so optional
columns are all-or-nothing per file. In particular, power columns exist in a
staging CSV only if the very first logged row had them: flipping
`--measure_power` across a `--resume` of the same staging file appends rows
whose fields no longer match the frozen header — misaligned columns or empty
cells that fail at load time. Collect any one file with a single power
setting end to end. This property is also why the `moe_a2a` collector ships
**no power column at all** (owner ruling during PR review): its low-latency
timing covers one round-trip rather than a per-phase region, and meaningful
per-phase power would need winning-config re-runs — a measurement-method
design for hardware, not a column to fake. The loaders treat absent (or
null) power as 0.0.

# Test
Rebuild and install the new aiconfigurator. Please make sure you have your new system definition file prepared. It's src/aisimulate_core/systems/xxx.yaml

# Validate the correctness
Today, we have limited method to validate the database. You can try tools/sanity_check to validate the database a little bit. But it highly depends on your understanding
of the GPU system and kernel optimization.

# Known Issues

## Experimental Ascend serving observations

`sglang/ascend_trace.py` records a dedicated live SGLang Ascend service through
its `/start_profile` API. It runs an unprofiled control and independent profiled
requests, validates token/cache/retraction counts, and preserves rank-specific
device kernel shapes, dtypes, layouts and durations. Cache flushing requires
the explicit `--allow-flush-cache` flag. Run from the repository root:

```bash
python3 python/aisimulate/collector/sglang/ascend_trace.py \
  --container glm52-united-v0522 --model glm-5 \
  --output /tmp/ascend-observations --repeats 3 --allow-flush-cache
python3 python/aisimulate/collector/sglang/ascend_trace_report.py \
  /tmp/ascend-observations --expected-ranks 16
```

The default request is exactly 8192 input and 1024 output tokens, routed to
attention DP0; only the first three forwards are profiled. `--input-tokens`,
`--output-tokens`, `--dp-rank`, `--steps`, and `--repeats` specify that scope.
Outputs are experimental JSON and compressed profiler evidence, not canonical
performance tables. The report retains physical identities and marks groups
with missing repetitions or CV above 10% unqualified. Completed exports alone
do not prove complete forward capture: require model-specific per-rank kernel
counts before accepting a campaign. CPU spans, communication summaries, and
their nested kernels must not be added together as forward latency.

`sglang/ascend_quant_probe.py` runs framework-dispatched W8A8 probes on an idle
NPU. It requires SGLang commit `baa9befa381511cd4d56f83980951e5a45fffc6d`,
the corresponding Ascend image, and `TASK_QUEUE_ENABLE=1`. The default
`--family quant` covers static/dynamic linear, grouped matmul and dynamic
quantization. `--family attention` covers BF16 LightningIndexer and sparse
attention at 8192/8704/9216 context lengths. It warms eager execution and graph
replay, measures NPU events, and compares sampled outputs with CPU references.
Inputs, weights, balanced expert routing and attention scaling are synthetic
controls. Decode attention probes use eight active sequences; serving graph
padding can include idle sequences, so these are distinct cases.

Source contracts for probe inputs, all relative to `python/sglang/` at that
commit: ModelSlim weight/scales use the builder in
`srt/layers/quantization/modelslim/schemes/modelslim_w8a8_int8.py:36`;
GMM expert counts, cumulative group lists, output dtype and per-token scales
follow `srt/hardware_backend/npu/moe/matmul.py:23` and
`srt/hardware_backend/npu/quantization/moe_methods.py:681`;
indexer Q/K/weights, cumulative Q lengths, KV lengths, paged block table,
TND/PA_BSND layouts and top-k2048 follow
`srt/layers/attention/dsa/dsa_npu_indexer.py:244` and `:354`;
sparse attention Q/K/V, RoPE, indices, lengths, layouts and mode flags follow
`srt/hardware_backend/npu/attention/ascend_backend.py:1471`.
The probes call installed implementations without copying vendor kernel code.

Neither entry point registers a production model plan or publishes FPM rows.
Host/driver identity, routing distribution, active sequence lengths and graph
padding remain required inputs for a future coordinated consumer integration.

`sglang/ascend_hccl.py` collects unquantized HCCL collectives with `torchrun`.
The pinned SGLang runtime dispatches AllReduce through
`srt/distributed/device_communicators/npu_communicator.py:23`; AllGather and
ReduceScatter follow `srt/distributed/parallel_state.py:1297` and `:1124`,
with PyTorch aliases selected in `srt/distributed/utils.py:21` at the same
commit. This does not benchmark DeepEP's fused dispatch/combine path.
For example, inside the allocated Ascend container:

```bash
TASK_QUEUE_ENABLE=0 HCCL_BUFFSIZE=650 python3 -m torch.distributed.run --nproc_per_node=2 \
  collector/sglang/ascend_hccl.py --ranks-per-node 2 --output /tmp/hccl-2.json \
  --serving-setup /path/to/serving-setup.json
```

The collector records input bytes per rank, full signed-pattern checks before
and after timing, every rank's NPU-event and host timings, and the maximum rank
duration per repetition. Eager and graph samples remain separate; graph replay
is warmed independently. Match the serving environment: the recorded United
uses queue mode 0 and a 650-MB HCCL buffer. Process-group options come from
SGLang's `get_torch_distributed_pg_options`; the environment and resolved HCCL
options are preserved and cannot be merged across measurement protocols.
`--serving-setup` applies a recorded `sglang-runtime-setup/v1` environment and
each local rank's observed CPU mask before importing the runtime. Capture the
launcher exports and live scheduler CPU masks; worker process-title rewriting
can obscure `/proc/<worker>/environ`. The exporter rejects missing setup,
environment mismatches and CPU affinity mismatches. Runs without a setup remain
controls and cannot be published as serving-profile data.
`--gc-policy enabled` preserves Python GC and records
its duration during each timing interval; `disabled` collects before timing and
temporarily disables GC to diagnose harness-induced pauses. The choice is
retained in measurement provenance. Graph/eager or GC protocols cannot be merged
silently. `ascend_hccl_export.py` recomputes quality statistics, reports every
unqualified point, rejects duplicate keys and incomplete correctness checks,
and writes `hccl_perf.parquet` plus `provenance.json` into a fresh directory.
Its runtime manifest requires the immutable image/source, CANN/HCCL build,
driver versions, hardware and collector source digest. The native HCCL reader
uses the exact system selector documented in
[hardware extensions](../../../docs/perf-model/extending.md#hardware-system-yaml).

`ascend_recording_server.py` is a dedicated state-collection launcher. It calls
the installed SGLang CLI and installs `ascend_state_capture.py` in spawned workers.
The observer retains actual per-layer expert counts across graph replay; it does
not substitute a balanced routing distribution. Its records preserve the physical
input buffer separately from the counts consumed by grouped matmul. The default
copies count tensors; optional `AISIM_CAPTURE_COPY_COUNTS=0` retains references
and fails if the runtime reuses their storage. Both modes are state observations,
not qualified performance measurements.

With `AISIM_CAPTURE_INPUTS=1`, the same launcher also installs
`ascend_input_capture.py` and writes `inputs-rank-*.jsonl` beside the routing
records. It observes linear, MLA BMM, indexer and sparse-attention calls, including
shapes, strides, dtypes, storage formats and static quant-bias selection. Small
sequence-length tensors are read after actual graph replay; large activations
and attention buffers are represented by metadata only. These records distinguish
logical active requests from the physical padded graph geometry. The optional
observer preserves custom-op overload attributes such as `.default`; its
instrumented service must not supply performance timings.

`ascend_state_requests.py` drives bounded prefill/early-decode captures, and
`ascend_state_sweep.py` waits until all dedicated requests have produced their
first token before arming steady decode capture. The latter is currently scoped
to the verified United attention-DP2 layout and its8k KV budget. The control file
must be written atomically. Do not use the recording service's instrumented
request latency to qualify prediction accuracy. `ascend_state_report.py` rejects
missing ranks/layers, invalid count vectors and inconsistent storage geometry,
then deduplicates exact invocation geometries while retaining their observations.

`ascend_deepep_cases.py`, `ascend_deepep.py`, and `ascend_deepep_export.py`
form an experimental, explicit Ascend transport campaign. The producer invokes
the installed SGLang dispatchers at commit
`baa9befa381511cd4d56f83980951e5a45fffc6d`:
`srt/layers/moe/token_dispatcher/deepep.py:611` (normal dispatch),
`:696` (normal combine), `:840` (low-latency dispatch), and `:900`
(low-latency combine). The runtime supplies its default DeepEP strategies,
workspace sizing and quantization selection. The plan records hidden6144,
256 experts, top8, 32 maximum low-latency tokens per rank, and contiguous
16-rank node placement. The payload represents BF16 hidden states and router
IDs/weights in the actual dispatcher API; synthetic routes are explicitly
balanced or concentrated in the first eight experts. They are not observations
of the checkpoint's routing distribution.

Dispatch transports INT8 activations with FP32 scales; combine transports BF16.
Every valid dispatched element and every combined output is checked before
and after timing. Each captured graph contains one transport call. Rank
barriers separate calls outside the timed region; events surround the actual
invocation/replay. Captured graphs remain alive while communication handles
may refer to their storage. Graph no-op controls must fail numerical checks.
Each repeated sample averages64 individually timed calls, then publication
takes the median of per-sample rank maxima. CV above10% is excluded, with the
original samples retained. Failed executions remain classified observations;
they never produce timing rows, and the full frozen plan must be accounted for.

The coordinated native consumer is `AscendDeepEp` and
`moe/sglang/0.5.22/ascend_deepep_perf.parquet`. Its exact dimensions are phase,
dispatch/combine, global ranks, ranks per node, routing distribution and
eager/graph execution. Only tokens per rank interpolate within measured bounds.
It uses the requested hardware/backend/version and rejects missing coverage;
there is no CUDA data substitution or analytic fallback. Normal-mode graph
execution is not a declared serving branch. This primitive measures transport
boundaries, excluding expert compute and whole-forward stream overlap.

The Ascend vector role `logits_finalize` measures the operations after the
logits collective: BF16 `[TP, requests, vocabulary/TP]` is reordered into
request-major logits, then passed through the installed
`LogitsProcessor._copy_logits_to_buffer`. The source contract at the same pinned
SGLang commit is `srt/layers/logits_processor.py:1061-1080,1153-1182`.
Eager calls allocate FP32 output; decode graph calls copy into the shared FP32
buffer described by `model_executor/runner/decode_cuda_graph_runner.py:419-427,992`.
The campaign uses the United32-request graph capacity and checks the entire
vocabulary for sampled requests against independently reordered CPU values.
The collective and LM-head multiplication remain separate measurements. The
existing `router_topk` measurement already includes its BF16-to-FP32 input
conversion; do not count a second router cast.

`ascend_pd_cases.py` and `ascend_pd.py` collect isolated request handoffs with
the installed MemFabric1.2.1 SDMA implementation. They use the real
`NPUMLATokenToKVPool` factory (`hardware_backend/npu/memory_pool_npu.py:579,851`)
and MLA rank mapping (`disaggregation/common/conn.py:1269-1348`) at the pinned
SGLang revision. The source request goes through
`AscendKVManager.register_buffer_to_engine` and `send_kvcache`
(`disaggregation/ascend/conn.py:127-142,272-405`). Only the transfer-facing
manager fields are populated: PP1/DCP1, MLA, no hybrid cache, the actual buffer
addresses/strides, and the framework-selected allocator policy. Control-plane
bootstrap and request admission are outside the measured boundary.

The reference pool has53248 usable tokens plus one128-token sentinel page,
78 layers and three BF16 components of widths512/64/128. Source and destination
attention TP4/8 mappings are separate keys. A source sends to each assigned
destination sequentially, as in the transfer worker; independent source ranks
run concurrently. All32 participants exist even when only one DP group's
request is active. The collector records the actual mapping and full wire-byte
count, and hashes every transferred byte on every receiving replica before
and after timing. A no-transfer control must fail. Continuous pages and seeded
fully fragmented pages have separate identities. These observations do not
model contention between distinct requests or metadata/bootstrap latency.

## NFS File Locking (Worker Deadlock)

**Symptom**: Collection stalls after a few test cases with no error messages.

**Cause**: `fcntl.flock()` doesn't work reliably on NFS. Workers deadlock when writing to shared output files.

**Solution**: Use `/tmp/` for output files, then copy results after collection.

# Support Matrix

Browse the [Legacy AIC Support Matrix](https://ai-dynamo.org/aisimulate/support-matrix/)
for CLI compatibility coverage, or inspect the
[per-system CSV files](../src/aisimulate_core/systems/support_matrix/).
For strict-native estimator coverage, use the
[FPE Support Matrix](https://ai-dynamo.org/aisimulate/fpe-support-matrix/).

## SGLang ordinary MLA context modules

For uniform-projection checkpoints, `run_attention_torch(..., ordinary_mla=True)`
collects the complete context attention module into
`mla_context_module_perf.txt`. It invokes SGLang's real `prepare_qkv_latent`
and creates fresh `AttentionInputs` on every warmup and timed iteration, so
Q/KV down-projection is included. Prefill is timed eagerly. The loaded projection
weights must match the requested single `gemm_type`; mixed BF16/NVFP4 modules
are rejected instead of mislabeled.

The CLI flag `--ordinary-mla` requires `--mode context --attn-type mla`.
Other combinations fail during argument parsing, before collection starts.
If any dispatch fails, the CLI reports its traceback, finishes the remaining
dispatches, and exits with status 1 so partial collection is not reported as success.

Pass `--target-tp-size 4` (or `8`) to derive the per-rank head count from the
model. It must be positive and divide the model's native head count. If
`--num-heads` is also supplied, it must equal native heads divided by TP.
Without an explicit TP size, ordinary MLA derives TP separately for each
per-rank head count. Python callers pass the actual per-rank head count and
`target_tp_size`. Set
`--chunked-prefill-size 16384` on the CLI, or `chunked_prefill_size=16384`
on `run_mla_module()` or `run_mla_module_worker()`, to reproduce a serving
chunk limit of 16,384 tokens. The worker forwards the limit into its subprocess.
Explicit limits must be positive for one-shot module collection.
Omitting the option keeps SGLang's default. The existing
wide-EP and DSA paths keep their separate behavior. The ordinary context table
uses the existing consumer schema; this does not add mixed-precision modeling.

## TRT-LLM MoE autotuning failures

TRT-LLM MoE collection requires either a loaded tactic cache or at least one
successful autotuning attempt before measuring a tuning-enabled case. Ordinary
failures may try smaller eligible token counts; if every attempt fails, the
case raises before timing or publishing rows. CUDA accelerator errors propagate
immediately. The successful collection path and historical profile values are
unchanged by this failure guard.
