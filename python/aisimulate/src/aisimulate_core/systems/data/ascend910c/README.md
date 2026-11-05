# Ascend 910C hardware calibration

The experimental `vector/sglang/0.5.21.dev223+g76f9213a4` dataset contains
160 NPU25 measurements of attention RoPE and split Q/K normalization from
SGLang commit `76f9213a411018547f4fd6a75f36feaa4d6bed58`.
It covers 112 eager and 48 captured-graph points. Two unstable eager points
are excluded. Each curve uses one frozen collection protocol. The original
0.5.22 dataset remains the comparison baseline.
The version requires explicit `allow_unlisted_version=True` for experimental
SDK queries. Other operator families still need matching-runtime coverage
before this version can support a complete GLM forward prediction.
The 160 exact native queries and 27 separately collected interpolation cases
are recorded under
`benchmarks/evidence/ascend910c-glm52-w8a8/production/operator-repair-20261104/`.

`ascend910c.yaml` describes one runtime-visible `Ascend910_9382` chip in an
Atlas 800T A3 node. A node exposes **16 chips on eight dual-chip boards**.
Memory capacity and throughput are per chip, never per board or per node.

The serving-memory fields use32 worker snapshots from two current-model United
runs with SGLang0.5.22/CANN9.1.0.B243/driver26.1.1. `misc.other_mem=937753092`
is the largest observed pre-KV reservation after subtracting the independently
reconciled52,471,997,948-byte resident model inventory. It includes runtime
allocations and allocator slack; graph/execution allocations use SGLang's
separate static-fraction headroom. The existing SGLang backend adds its20%
other-memory margin. NCCL is absent, so its compatibility bucket is zero;
this is not an HCCL-memory measurement under an NCCL label. Evidence and scope
are in `benchmarks/evidence/ascend910c-glm52-w8a8/production/static-runtime-memory-20261024.json`.
Other topologies have not received independent runtime-memory calibration.

Detailed measurement reports live in each version directory's `metadata/`
subdirectory. Only performance tables and their standard `collection_meta.yaml`
belong directly in the version directory, so strict provenance admission does
not mistake reports for unregistered tables.

This directory contains an experimental source-qualified linear table under
`gemm/sglang/0.5.22/ascend_linear_perf.parquet`: 1,378 stable measurements from
1,382 attempted cases, with detailed reports under `metadata/`.
It distinguishes static W8A8, dynamic W8A8, BF16, the indexer weighting projection's
FP32-input/BF16-weight/FP32-output path, bias, storage format and
eager/graph execution. Native queries use exact keys and interpolate only inside
measured token brackets. Four unstable measurements are excluded and retained
in provenance. Missing curves and out-of-bracket queries fail explicitly.

The native GLM ModelSlim graph composes these tables, including21 indexer
layers,3 dense layers and75 MoE layers. It retains the actual static/dynamic
precision split, shared RoPE storage, retained MLA views and per-phase HCCL
boundaries. Queries assume uniform attention-DP load and the selected balanced
synthetic routing profile. Measured query bounds and uncalibrated whole-serving
accuracy remain distinct from functional forward prediction.
The MoE chain table adds 225 profile rows from 217 stable physical observations
(220 attempted) under `moe/sglang/0.5.22/ascend_moe_chain_perf.parquet`.
Balanced and single-expert count distributions are explicit synthetic inputs;
their identical one-token cases reuse one measurement. Each timed invocation
contains GMM1, quantized SwiGLU and GMM2 with NZ INT8 weights, BF16 channel
scales and FP32 token scales. Eight weight copies rotate. All attempted cases
pass sampled chain/quantization checks before and after actual execution; eight
graph-noop controls fail as expected. Three unstable observations are excluded.
Native queries reproduce every published row. These data exclude dispatch and
combine and do not qualify real routing skew or a whole deployment's performance.

DeepEP adds125 stable dispatch/combine records from two-node32/19 and local16
campaigns. World16 balanced normal M1..1024 and graph low-latency M1..32 are
covered. Small world32 balanced graph batches remain outside qualified
transport coverage. Normal-mode M2048 fails runtime tiling and has no timing
row. Source, full-payload checks, graph-noop controls and all excluded samples
remain in the reports.

The vector family adds842 stable observations from848 cases under
`vector/sglang/0.5.22/ascend_vector_perf.parquet`. It distinguishes18 logical
boundaries and28 physical geometries, including ModelSlim normalization bias,
FP32 indexer LayerNorm, strided RoPE and format-30 cache writes. All numerical
checks pass;18 no-op graph controls fail as expected. Native queries reproduce
every published row. Cache indices use INT64 in eager and INT32 in graph mode;
the current grid writes distinct valid slots and does not separately qualify
repeated sentinel writes. Six unstable timings remain in provenance. The logits
boundary includes the post-collective reordering and FP32 output conversion.

The attention family contains 1,517 qualified measurements from 1,576 attempts
under `attention/sglang/0.5.22/ascend_attention_perf.parquet`. It covers the
BF16 lightning indexer and sparse attention, with exact local-head and
eager/graph identities. Active requests and padded capture rows are separate;
padded KV lengths are zero. Two graph-noop negative controls failed as expected.
The 58 unstable timings and one numerical execution failure remain in provenance;
the latter has no timing row. Native queries reproduce all 1,517 published rows.
Prefill partitions by active batch with equal query lengths and a total budget
of 8,192 new tokens. Decode includes lower capture-bucket edges 9, 17 and 25.
The isolated failed input produces one zero output in eager and graph despite
an FP32/FP64 CPU reference near -0.0673; the original reproducer is retained.
Interpolation at a missing coordinate does not qualify that input's numerical
behavior. Missing interpolation brackets still fail explicitly.
These results qualify table loading and execution, not whole-model accuracy.

`misc.hccl_version` now selects `9.1.0-b243-drv26.1.1-nodes31-32-v3`.
Its 449 qualified rows combine 342 single-node npu1-32 observations with 107
new npu1-31/npu1-32 measurements. The latter cover 16 ranks with 8 per node and
32 ranks with 16 per node. Both nodes use driver 26.1.1 and the same pinned image,
collector and serving setup. All 160 new cases pass full-payload numerical checks
before and after actual eager/captured execution; 53 unstable cases remain in
provenance. An additional 64 stable local observations cover 192/384 MiB input
payloads. The combined 544 attempts include 95 unstable observations.
Native queries using the installed YAML reproduce all 449 published rows.
Some small eager payloads, especially in the 32-rank group, lack measured
brackets and fail explicitly. These are collective measurements; actual PD KV
transport, packing and overlap still require separate integration.

The MLA BMM family contains 220 qualified rows from 222 attempts under
`mla_bmm/sglang/0.5.22/ascend_bmm_perf.parquet`. Query BMM preserves the strided
192-wide view within each 256-wide head and the transposed BF16 key weight.
Prefill output uses `npu_transpose_batchmatmul`; decode output uses the custom
`batch_matmul_transpose` destination-buffer API. The collector rotates 64 weight
copies and checks every timed output against sampled CPU references before and
after measurement. All cases pass; three graph-noop controls fail as expected.
Two unstable rows remain excluded. Native lookup reproduces all 220 table rows.
Decode collection is scoped to the deployment's 32 requests per attention rank.
The first overbroad grid's custom-kernel failures at large decode token counts
remain in the raw evidence. This does not establish a universal kernel limit.

The system-specific query-version override pins SGLang to `0.5.22`; the Ascend
runtime belongs to a separate compatibility domain and does not promote the
CUDA systems' fleet-wide `next` version. This pin is not a model coverage claim.
Registering the hardware name or loading this table does not qualify GLM-5.2,
SGLang-Ascend, ModelSlim INT8 MoE,
whole-forward prediction, or replay accuracy. No NVIDIA communication data or
backend version is inherited. The existing NCCL-based memory estimator also
requires a separate HCCL integration; `misc` deliberately has no invented NCCL
version, communication-memory reservation, or serving-memory allowance.

Linear collection uses the pinned image below, driver 26.1.1, 64 independent
weight copies and 128 calls per timing sample. All attempted cases pass sampled
CPU-reference checks through the actual eager invocation or captured replay.
Matched M=8 controls on nodes 31 and 32 differ by at most 4.4% in paired means.
Working-set sensitivity reaches 9.4%. Earlier linear grids failed independent
held-out accuracy checks, including kernel-tiling discontinuities. The current
mixed-precision grid has verified execution and native table lookup, but its
interpolation, whole-forward and serving accuracy remain uncalibrated. Functional
bring-up takes priority over prediction-accuracy tuning. Original observations and intentional graph-noop failures are
under `benchmarks/evidence/ascend910c-glm52-w8a8/production/linear*`.

The retained calibration manifest is
`benchmarks/evidence/ascend910c-glm52-w8a8/hardware-calibration.json` in the
source checkout. It identifies raw samples, hashes, units, aggregation,
timing boundaries, runtime image, and qualification limits. Measurements were
made on 2026-10-20 and 2026-10-21. Runtime image:
`quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443`.

- HBM bandwidth counts both reads and writes for a 512 MiB copy. The bandwidth
  is already measured, so the additional empirical multiplier is 1.0.
- BF16 and INT8 compute entries are effective throughput of measured
  4096 × 4096 × 4096 matrix operations, aggregated across all 16 chips. They
  occupy the existing compute-coefficient fields but are **not theoretical
  peaks**. SOL using these entries is an analytical calibration, not a proven
  physical lower bound. INT8 input quantization and fused MoE are not included.
- Intra-node bandwidth uses eight concurrent cross-board HCCL ping-pong pairs;
  inter-node bandwidth uses 16 concurrent pairs between `npu1-32` and `npu1-19`.
  Both report SuperPod ID 1. Inter-node numbers apply to this observed fabric,
  not Ethernet/RDMA links between arbitrary nodes or different SuperPods.
- Bandwidth uses the median per-pair throughput at 256 MiB, in bytes/second,
  with each full-payload round trip divided by two. P2P latency uses small
  messages with native calls queued before device execution. It includes HCCL
  protocol work, not CPU submission gaps or solely wire propagation.
- A bandwidth plus constant model cannot reproduce protocol transitions:
  held-out cross-node messages of 1–128 MiB have at most 1.61% error in this
  campaign, but the 256 KiB point has about 58% error. Use collected HCCL
  operation curves before claiming accurate small-message serving predictions.
- Historical hardware calibration used local driver 26.1.1 and peer driver
  25.5.0; both report HAL 7.35.23. Revalidate affected measurements after a
  driver/runtime/topology change. Results are not qualified
  for more than the two measured nodes or for loaded/co-tenant hardware.

Unmeasured scalar FP32, FP8, FP4, PCIe bandwidth, TDP, memory fixed latency,
rack size and serving-memory overhead are omitted. CUDA `sm_version` is not
applicable. Missing compute modes retain the existing explicit error behavior;
do not derive their throughput from BF16 or INT8.
