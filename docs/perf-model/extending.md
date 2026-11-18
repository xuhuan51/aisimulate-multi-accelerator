<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Extend performance models

AISimulate owns the application, model definitions, performance data, and native
estimator. Start from the [development environment](../../DEVELOPMENT.md).
All paths below are relative to the repository root unless linked otherwise.
The `aisimulate_core` source namespace is retained inside the `aisimulate`
wheel; installing or rebuilding a separate AIConfigurator package is unnecessary.

For whole-forward FPM onboarding, start with the [self-service config/profile workflow](fpm-self-service/implementation.md#onboard-with-an-agent),
which supports direct interpolation without an analytical model class. The
[SOL-assisted FPM checks](#sol-assisted-fpm-checks) describe the
additional operation, memory and native SOL tests needed when adding an analytical
model for `method: sol`. Neither FPM route requires per-operation GPU timing collection.

## Choose the smallest extension

| What changed? | Start here |
|---|---|
| A new architecture name with an existing operation pipeline | Map the architecture to an existing family in `python/aisimulate/src/aisimulate_core/sdk/common.py`. |
| A new layer composition or model family | Add a registered class under `python/aisimulate/src/aisimulate_core/sdk/models/` and map its architecture. |
| Existing operations need additional measured shapes or a backend version | Add collector cases and collect the missing performance-data cells. |
| A genuinely new operation or execution contract | Extend the native operator and wire contract, then the Python model description and collection path. |

An architecture mapping alone does not establish data coverage, quantization,
KV-memory correctness, Replay support, or prediction accuracy. Validate the
specific model/backend/hardware cell and the public workflow it will serve.

## 1. Resolve and register the model

The [models package](../../python/aisimulate/src/aisimulate_core/sdk/models/README.md) owns
registry-based model construction. Resolution reads model configuration,
resolves the architecture to a family, then selects the `@register_model`
class. `models/blocks/` contains reusable composition helpers and must not
register model classes.

For an existing family, inspect
[`common.py`](../../python/aisimulate/src/aisimulate_core/sdk/common.py) and the selected model
class before adding `ARCHITECTURE_TO_MODEL_FAMILY` entries. Verify layer counts,
attention/KV heads, head dimensions, quantization defaults, and any custom
Hugging Face configuration fields. A model name resembling another model is
not sufficient evidence that their operation pipelines match.

For a new family, implement `BaseModel.create(...)`, register the family, and
build its context/generation operations using the existing family-specific
examples. See the [registry extension examples](../../python/aisimulate/src/aisimulate_core/sdk/models/README.md#adding-a-new-model).
Reuse the shared MoE block builder where its contract applies, including the
separate large-EP registration described there.

Mamba2 kernels and NemotronH hybrid model descriptions already exist in
[`operations/mamba.py`](../../python/aisimulate/src/aisimulate_core/sdk/operations/mamba.py) and
[`models/nemotron_h.py`](../../python/aisimulate/src/aisimulate_core/sdk/models/nemotron_h.py).
They are useful references, not evidence that every Mamba variant or topology
is supported. In particular, AFD partitioning has separate restrictions below.

## 2. Extend the operation contract when necessary

Python describes model work; Rust computes per-operation latency, energy,
and SOL values. Keep that single-oracle boundary when adding an operation:

1. Implement the native operator in
   [`crates/core/src/perfmodel/operators/`](../../crates/core/src/perfmodel/operators)
   and any table loader in
   [`perf_database/`](../../crates/core/src/perfmodel/perf_database).
   Add an independently justified numerical test, including unsupported and
   missing-data behavior.
2. Add the typed Python operation in
   [`sdk/operations/`](../../python/aisimulate/src/aisimulate_core/sdk/operations). Follow its
   existing construction, weight-sizing, and engine-backed table-view
   conventions. Do not add a Python interpolation or performance-query oracle.
3. Extend `_to_opspec` in
   [`sdk/engine.py`](../../python/aisimulate/src/aisimulate_core/sdk/engine.py), the native
   [`Op` representation](../../crates/core/src/perfmodel/operators/op.rs),
   and the [`engine specification`](../../crates/core/src/perfmodel/engine/spec.rs).
   Preserve positional enum compatibility; a schema-breaking change requires
   coordinated versioning and consumer updates.
4. Wire the operation into the model pipeline and validate its parameter
   conversion. Run the existing
   [OpSpec coverage test](../../python/aisimulate/tests/unit/sdk/test_opspec_coverage.py) and
   [single-oracle contract](../../python/aisimulate/tests/cross_package/test_single_oracle_contract.py).
5. Add a representative estimator parity case when a model reaches the new
   path. Follow the [parity README](../../crates/core/parity_tests/perfmodel/README.md)
   and review the numerical evidence before updating goldens.

Read the applicable rules linked from [AGENTS.md](../../AGENTS.md) before
changing collector or generator code. The current operator instructions in
this page use the unified repository layout.

## 3. Collect only the missing data

Use the [Collector README](../../python/aisimulate/collector/README.md) and the collector rules
required by `AGENTS.md`. Cases live under `python/aisimulate/collector/cases/`;
model-specific cases belong under `models/`, and reusable operation shapes
under `base_ops/`. For a new collector operation, follow the
[collector operation runbook](../../python/aisimulate/.claude/skills/aic-collector-op-development/SKILL.md).

Run the intended backend/runtime on the intended hardware, retain the effective
model and runtime identity, and finalize accepted staging output as Parquet
with the required collection/reuse metadata. The canonical data root is:

```text
python/aisimulate/src/aisimulate_core/systems/data/<system>/<family>/<backend>/<version>/
```

Do not relabel another backend version's data as newly measured. Update query
version/support metadata only when the new cell meets its collection and
consumer contracts. For whole-forward profiles, use the separate
[FPM self-service overview](fpm-self-service/README.md).

## 4. Validate the intended public path

### Hardware system YAML

System definitions live in
`python/aisimulate/src/aisimulate_core/systems/<system>.yaml`. Keep the unit of
one simulator device consistent across memory, compute, topology and collected
operator identities. Register the name in `SupportedSystems` and provide its
`data_dir`; this registers hardware metadata, not a measured model/backend cell.

| YAML field | Meaning and calibration requirement |
| --- | --- |
| `data_dir` | Per-system data root relative to the systems directory; an empty data root does not establish operation coverage. |
| `gpu.mem_capacity` | Bytes exposed per runtime device; record whether this is physical or runtime-available capacity. |
| `gpu.mem_bw` | Bytes/second per device; specify whether a benchmark counts reads, writes, or both. |
| `gpu.mem_bw_empirical_scaling_factor` | Additional bandwidth multiplier; use 1.0 when the selected bandwidth already includes measured utilization. |
| `gpu.mem_empirical_constant_latency` | Additional memory-operation latency in seconds; omission uses the native zero default and does not mean fixed latency was measured as zero. |
| `gpu.bfloat16_tc_flops`, `gpu.int8_tc_flops` | Per-device arithmetic coefficients, counting a multiply-add as two operations. Normally hardware peaks; a measured calibration must label its shapes and effective throughput explicitly and cannot claim a strict physical SOL lower bound. |
| `gpu.fp32_flops`, `gpu.fp8_tc_flops`, `gpu.fp4_tc_flops` | Define only supported, sourced coefficients. FP32 means scalar/vector throughput, not a TF32 matrix benchmark. Missing compute modes remain explicit errors. |
| `gpu.sm_version`, `gpu.power` | CUDA architecture identifier and per-device TDP respectively; omit when inapplicable or unverified. |
| `node.num_gpus_per_node` | Runtime devices per physical node; a board may expose multiple devices. The historical field name also applies to NPUs. |
| `node.intra_node_bw`, `node.inter_node_bw` | Single-direction bytes/second per device for the measured topology and concurrency. Do not substitute host NIC bandwidth for an NPU fabric. |
| `node.p2p_latency` | Seconds of fixed P2P protocol latency. Describe whether CPU submission is included. The native P2P formula uses inter-node bandwidth and omits this constant in SOL mode. |
| `node.pcie_bw`, `node.num_gpus_per_rack`, `node.inter_rack_bw` | Optional measured or sourced PCIe and rack properties; omit unknown values. |
| `misc.nccl_mem`, `misc.nccl_version`, `misc.other_mem` | Existing NCCL and serving-memory inputs; do not relabel HCCL observations as NCCL or invent reservations to make memory estimation pass. |
| `misc.hccl_version` | Selects the exact HCCL provenance directory at `data/<system>/comm/hccl/<value>/hccl_perf.parquet`; optional. Different driver/runtime identities use different directories. |

HCCL tables retain exact dtype, operation, rank count, ranks per node, contiguous
rank placement and eager/graph execution identities. The payload axis is input
bytes per rank; AllGather output is larger by the group size, while ReduceScatter
input is the complete send tensor. Qualified points require full-payload checks
before and after timing on every rank, at least three samples and CV at most 10%.
Only measured payload brackets interpolate; missing tuples and out-of-range
payloads fail explicitly. Malformed files, duplicate physical keys and mixed
runtime identities are errors. HCCL data is never relabeled as NCCL or borrowed
from another transport. Declaring the directory does not qualify a model or
establish collective coverage.

The NPU DSA collector publishes `attention/sglang/<version>/ascend_attention_perf.parquet`.
Its typed native `AscendAttention` operation accepts these fields:

| Field | Meaning |
| --- | --- |
| `name`, `scale_factor` | Operation label and number of identical invocations. |
| `key.operation` | `lightning_indexer` or `sparse_attention`, matching the measured NPU kernel. |
| `key.phase` | `prefill` or `decode`; sequence length means new tokens for prefill and full KV length for decode. |
| `key.local_heads` | Per-rank heads: 32 for the indexer, or 4/8/16 for sparse attention. |
| `key.execution_mode` | Exact `eager` or `graph` dispatch; these tables never substitute for one another. |
| `graph_buckets` | Strictly increasing capture sizes for graph decode. Empty for other execution paths. |

This primitive currently describes BF16 TND queries and PA_BSND caches with
page size 128, sparse top-k 2048, MLA rank 512, RoPE width 64 and 32×128 indexer
heads. Cache views retain NCDHW format 30. Decode uses a block-table width of
1407; prefill derives its width from the KV length. The pool has at least 417
pages, including the sentinel page. Other geometries require separate collection
and an explicit contract extension.

The runtime batch is the number of active requests per attention rank. Graph
padding selects a measured capture bucket, while padded KV lengths remain zero.
The native table interpolates KV length and active batch only within that same
bucket. Prefill partitions data by the exact active request count, interpolating
new tokens per request and prefix length within that partition. Published data
currently covers one request; collecting additional batch partitions does not
qualify them until their numerical and source checks pass. Out-of-range coordinates and missing brackets are coverage
errors; neither another runtime nor an analytic estimate fills the gap. Data is
indexed once on first use, and repeated queries access bounded axis curves.
An operator table does not by itself complete model composition or validate
whole-forward prediction accuracy. Production models still use
`ForwardPassPerfModel::best_available` as their construction interface.

The `AscendBmm` primitive reads `mla_bmm/sglang/<version>/ascend_bmm_perf.parquet`.
It distinguishes the strided query projection, transpose-BMM prefill output,
and custom `batch_matmul_transpose` decode output. BF16 views use the observed
storage format and strides, including the query's 192-wide slice of a 256-wide
head and the transposed key weight. Contiguous substitutes are rejected.

| Field | Meaning |
| --- | --- |
| `name`, `scale_factor` | Operation label and number of identical invocations. |
| `key.role` | `query_nope`, `output_prefill`, or `output_decode`; each selects its exact measured kernel and stride contract. |
| `key.local_heads` | Per-rank attention heads, currently 4/8/16. |
| `key.execution_mode` | Exact `eager` or `graph` execution. |
| `key.active_experts` | Optional active-expert count for an explicitly supplied workload. Interpolates the single-expert and balanced measured curves at the same received-token count. Omission preserves the synthetic distribution curve. |
| `token_multiple` | Round physical input rows up to this multiple; use 1 for unpadded prefill and the configured capture multiple for decode. |

Only measured token brackets interpolate. The declared decode collection axis
currently covers 1–32 rows, matching the United deployment's per-DP request
limit; this is a deployment coverage limit, not a hardware maximum. Large
custom-decode invocations rejected by the runtime remain in the failed campaign
evidence. Extending the deployment's batch limit requires collecting that path.

The `ascend910c` definition is a scoped hardware calibration for Atlas 800T A3
with 16 runtime-visible chips per node. Its packaged
[calibration notes](../../python/aisimulate/src/aisimulate_core/systems/data/ascend910c/README.md)
describe effective compute coefficients, same-SuperPod topology, omitted fields,
and unsupported prediction paths. Hardware microbenchmarks are not GLM-5.2
ModelSlim operator profiles. Keep raw runs and accuracy results under
`benchmarks/evidence/`, and validate the existing loader, topology selection,
precision resolution and missing-data behavior before claiming integration.

### Public workflow checks

The `AscendMoeChain` primitive reads `moe/sglang/<version>/ascend_moe_chain_perf.parquet`.
One observation contains GMM1, grouped quantized SwiGLU and GMM2 with dependent
inputs. Dispatch, combine, router and shared experts remain separate. Weight
storage is NZ INT8, channel scales BF16 and token scales FP32. The loader checks
the complete expert count vector against the declared distribution; a buffer's
allocated rows never substitute for its valid received rows.

| Field | Meaning |
| --- | --- |
| `name`, `scale_factor` | Operation label and number of identical routed layers. |
| `key.phase` | `prefill` uses a buffer sized to valid received rows; `decode` uses the declared physical capacity. |
| `key.local_experts` | 16 or 8 resident experts, with hidden size 6144 and intermediate size 2048. |
| `key.distribution` | Explicit synthetic `balanced` or `single_expert` count profile; neither represents observed checkpoint routing automatically. |
| `key.execution_mode` | Exact `eager` or `graph` execution. |
| `input_token_multiple` | Round input rows to the physical capture multiple before applying the routing ratio. |
| `token_numerator`, `token_denominator` | Convert rounded runtime rows to received rows with ceiling. Model composition must supply its explicit routing/parallelism assumption. |

The initial grid measures decode capacities 4096 with 16 local experts and
8192 with 8. Only the former is currently verified against a live EP16 dispatch;
the latter remains a standalone input-buffer case until distributed validation.
One-token balanced and single-expert inputs are physically identical and reuse
one observation with two explicit profile rows. Zero received rows have no
qualified chain timing yet; nonzero out-of-range requests fail. Synthetic
distribution curves enable functional modeling but do not qualify routing skew,
MoE communication overlap, resident runtime overhead or whole-forward accuracy.

With `key.distribution = "balanced"`, an explicit `key.active_experts` selects
an active-weight working-set estimate. For fixed receive volume, the
single-expert curve supplies the fixed component and the balanced curve supplies
the endpoint with `min(received_tokens, local_experts)` active experts. Intermediate
counts linearly interpolate between these endpoints in Rust and report an
estimated source. Counts must be positive and cannot exceed either received
tokens or resident experts. Both curves must have matching table provenance.
The resident weight inventory continues to include every local expert.

The operation JSON accepts an omitted active count for existing callers.
EngineSpec schema 37 gates the added positional key field in compiled payloads.
Schema 38 additionally carries the optional Decode routing workload and the
bound request-count vector used by synchronized attention DP.
Ordinary model composition continues to use its supplied routing assumptions.
Workload-aware callers must obtain active counts from their routing model or
qualified observations. Per-layer and per-rank distribution binding requires
separate whole-model validation. Independent Decode distribution measurements
are recorded under `benchmarks/evidence/`.

The `AscendVector` primitive uses `vector/sglang/<version>/ascend_vector_perf.parquet`.
Its versioned boundary contract covers ModelSlim RMSNorm with bias, fused Q/K
normalization, FP32 indexer LayerNorm with BF16 conversions, fused interleaved
RoPE, cache writes, SwiGLU, routing top-k, casts, concatenation, embedding and
the routed/shared-output operations. Each role owns its exact dtype, tensor
stride, storage offset and kernel signature. Fused boundaries include their
internal operations; do not also charge those internals as separate operators.

| Field | Meaning |
| --- | --- |
| `name`, `scale_factor` | Operation label and number of identical invocations. |
| `key.role` | Source-qualified logical boundary, such as `add_rms_bias`, `rope_attention`, or `cache_scatter`. |
| `key.dimensions` | Role-specific physical geometry; only declared and measured tuples are supported. |
| `key.execution_mode` | Exact `eager` or `graph` execution. Cache indices are INT64 for eager and INT32 for graph. |
| `token_axis` | `tokens` or `requests`, selecting the runtime row count before sharding. |
| `token_split` | Divide the selected row count with ceiling for the operation's input shard. |
| `token_multiple` | Round the sharded row count up to the configured physical padding multiple. |

The initial cache-write grid uses 53,376 physical token slots and distinct valid
write locations. It has no separate latency model for repeated padding writes
to sentinel slot zero. Resident weights and RoPE tables have explicit storage
sizes; KV storage belongs to the cache model, not the write operation. Published
rows carry the checked invocation signature hash and the runtime identity.
Only measured row-count brackets interpolate; unsupported geometry, execution
or precision fails. These warm-buffer measurements are functional support,
with whole-forward accuracy still uncalibrated.

Rebuild the editable package after native or packaging changes:

```bash
uv sync --project python/aisimulate --extra dev
source python/aisimulate/.venv/bin/activate
python -m pytest -c python/aisimulate/pytest.ini   python/aisimulate/tests/unit/sdk/test_opspec_coverage.py   python/aisimulate/tests/cross_package/test_single_oracle_contract.py
```

Also run the focused tests for the changed model, collector, and native
operator. On macOS, use the local pytest guidance in `AGENTS.md`.

Check the exact model/system/backend/version with `aiconfigurator cli support`
from this environment, then run the intended `aisimulate predict` or
`recommend` configuration. Keep model revision, precision, topology, workload,
performance-data identity, and reports with the test results. If deployment
artifacts are in scope, exercise their generator path too.

Keep these outcomes separate in the change description: model construction,
required-data coverage, native execution, replay/generator compatibility, and
accuracy against measured hardware. A support check or numerical parity test
alone does not prove end-to-end predictive accuracy.

### AFD Operation Partitioning Compatibility

Attention-FFN Disaggregated (AFD) estimate mode has one additional maintenance contract beyond the normal aggregated and P/D-disaggregated paths. [`python/aisimulate/src/aisimulate_core/sdk/afd_partition.py`](../../python/aisimulate/src/aisimulate_core/sdk/afd_partition.py) splits a model's `context_ops` / `generation_ops` into A-worker and F-worker pools by operation name. When adding a new model family or new operation, make sure the generated operation names can be classified by the AFD partitioner.

The current AFD partitioning contract is:

1. **A-worker ops**: operations that belong to the embedding / attention side, such as `embedding`, `add_norm_1`, attention norms, `qkv`, MLA, BMM, RoPE, attention kernels, and projection GEMMs.
2. **F-worker ops**: operations that belong to the FFN / MoE side, such as router GEMMs, dense `ffn` / `mlp` ops, routed expert ops, shared expert ops, and activation / gate / up / down GEMMs.
3. **Boundary ops**: operations that naturally sit at the A/F boundary, such as `add_norm_2`, FFN/MoE/MLP norms, `logits_gemm`, `reduce_add`, and `*_combine`. These default to the A-worker and can be moved to the F-worker with the AFD boundary placement option.
4. **Skipped model-internal communication ops**: communication or dispatch operations already represented by the AFD communication model, such as `CustomAllReduce`, `P2P`, `NCCL`, TP all-gather / reduce-scatter, and MoE dispatch ops. These should not be counted again in either compute pool because AFD adds its own cross-pool and intra-pool communication through `AFDTransfer`, `AFDFAllGather`, `AFDFReduceScatter`, and `AFDCombine`.
5. **Overlap ops**: an `OverlapOp` can stay atomic only when every non-skipped inner op belongs to the same side. If an overlap group spans the A/F boundary, the partitioner must fail or be extended to split / rebuild that overlap explicitly.
6. **Layer families that need explicit rules**: Mamba and GDN layers are not covered by the current attention/FFN partition rules. Until a dedicated partitioning rule, operation model, and communication / memory accounting are added, the partitioner raises an explicit `AFDPartitionError` for these ops instead of falling back to an unknown-op side assignment.

If a new operation cannot be classified, do not rely on an unknown-op fallback for production use. Update `python/aisimulate/src/aisimulate_core/sdk/afd_partition.py` with an explicit classification rule and add a focused unit test in `tests/unit/sdk/test_afd_partition.py`.

## SOL-assisted FPM checks

Direct self-service can use an unregistered architecture with a verified
profile; SOL-assisted FPM requires an analytical model and support in the native
[`fpm_sol` evaluator](../../crates/core/src/perfmodel/operators/fpm_sol.rs).
An ordinary operator or a `SOL_FULL` diagnostic does not automatically implement
that evaluator's contract.

Before timing collection, record the immutable checkpoint configuration,
actual layer types/counts, attention/KV heads and dimensions, expert geometry,
precision, sharding, runtime kernels, and context/cache layout. Validate:

1. Both context and generation operation lists are nonempty and lower to native
   `OpSpec` values with correct per-rank dimensions and repetition.
2. Weight inventory stays resident even when execution skips work. Cache sizing
   and its inverse agree across window, compression, and page boundaries; a
   one-token slope is insufficient for a nonlinear cache.
3. `database_mode: SOL`, `estimation_mode: op_level`, and `fallback_policy: deny`
   produce finite analytical diagnostics for the intended shapes and phases.
4. Every retained operation also works through FPM's own SOL evaluator.
5. After a reviewed measurement pair is available, explicit
   `estimation_mode: fpm_interpolation`, `method: sol`, and `fallback_policy: deny`
   construct and query the exact model/deployment cell. Keep this distinct from
   the earlier data-free analytical check.

The model keeps its phase operations as `sol_ops` inside the whole-forward
operation, and its weight inventory supplies `weight_bytes`. Do not duplicate
phase replacement, interpolation, or SOL formulas in Python.

## Add a GEMM precision profile

Add the correct memory/compute coefficients to the precision enum and inspect
its consumer graph before claiming support. The native
[`GEMM_QUANT_UTIL_LEVEL`](../../crates/core/src/perfmodel/operators/gemm.rs)
and MoE equivalent own cross-profile efficiency levels. Python's level views
are exported from the native runtime; do not hand-maintain a second level table.
Add a measured or explicitly inferred level for a genuinely new profile,
with its derivation beside the native entry. Validation must admit only the
relations enabled by the chosen transfer policy and available reference data.

Exercise SILICON failure without data, permitted HYBRID/EMPIRICAL transfer,
restrictive-policy failure, reference ordering, and eventual own-data precedence.
`fp8_static` additionally requires its overhead tables. Checkpoint-to-precision
resolution and hardware compatibility are separate contracts; reachability of an
estimate does not prove that the backend can execute the checkpoint.
