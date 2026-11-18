<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance-model configuration

## Choosing a forward-pass API

Use `RustForwardPassPerfModel.best_available(config)` from Python or
`ForwardPassPerfModel::best_available(config)` from Rust. The canonical
`ForwardPassPerfModelConfig` owns model, hardware, backend, topology, data
policies, a required immutable `worker_type`, and the complete nested
`estimator_config`. Worker roles are `prefill`, `decode`, and `aggregated`.
Topology carries two optional context-parallel knobs next to `tp`, `pp`,
`attention_dp`, `moe_tp_size`, and `moe_ep_size`: `cp_size` (prefill context
parallelism, SGLang `--attn-cp-size` / vLLM `-pcp`; extra attention ranks, so
`tp * attention_dp * cp_size == moe_tp_size * moe_ep_size`) and `dcp`
(decode context parallelism, vLLM `-dcp` / SGLang `--dcp-size`; stripes the
decode KV cache across the existing TP ranks and adds no GPUs). Both default to
one and stay out of the serialized identity when unset.

```python
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

config = ForwardPassPerfModelConfig(
    model="Qwen/Qwen3-32B",
    system="h200_sxm",
    backend="vllm",
    worker_type="aggregated",
    tp=2,
    estimation_mode="auto",
    fallback_policy="deny",
    estimator_config={
        "features": {"attention_kv_weight": 1.0},
        "fpm_regression": {
            "sampling": {"bins_per_axis": [4, 16], "max_observations": 128},
            "min_observations": 5,
            # Opt into periodic rebuilding; omission leaves it disabled.
            "fit": {"rebuild_interval": 4096},
        },
        "correction": {"enabled": True},
    },
)
model = RustForwardPassPerfModel.best_available(config)
print(model.diagnostics()["provenance"])
```

### Engine identity controls

The canonical configuration also carries quantization overrides and `attention_backend`, `moe_backend`, `moe_kernel_source` (default absent), `enable_eplb` (default `false`), and `wideep_num_slots` (default absent). These controls reach model construction, KV memory sizing, and replay provenance. EPLB/slots and nondefault MoE backend or kernel-source selection require an MoE model. Collected FPM interpolation cannot represent EPLB, slots, MoE backend, or kernel-source overrides; it rejects an explicit incompatible request and is skipped during automatic selection for those identities.

`moe_kernel_source` selects an exact, nonblank collected `kernel_source` label for fused MoE compute. It is distinct from the existing `moe_backend` graph/backend control; source labels are not backend aliases and are preserved without trimming. `None` keeps the existing default source-selection policy, including eligible low-latency NVFP4 selection. `SILICON` reads only the requested source's table; `EMPIRICAL` derives its estimate from that same source; `HYBRID` may fall back to empirical estimation within that source, but does not substitute a different source. Missing source data remains an error. An explicit `moe_torch_flow_min_latency` requires gated NVFP4 and at most 128 tokens after attention-DP gathering. Pure-roofline `SOL` remains table-independent and does not claim measured support for the requested source.

`moe_perf.parquet` may include Boolean selection metadata `default_eligible`. An absent column preserves legacy automatic selection; when present, every value must be a non-null Boolean. A `false` row requires a nonblank string `kernel_source`, preserved exactly, and is available only through that named source. It cannot enter automatic standard or low-latency grids, including empirical cross-shape and cross-quant reference selection. Among eligible rows, existing source priority and first-row precedence remain unchanged. Default table views, coverage, and readiness use those same eligible grids; raw Parquet enumeration retains all measured rows. Malformed eligibility metadata is an invalid-data error, never a missing-data fallback. The flag is not a measurement identity dimension. Collector finalization preserves it during merges, including existing annotations when a legacy recollection omits the column; genuinely new legacy keys remain eligible.

An explicit source is rejected for dense graphs, MegaMoE modules, large-EP expert-compute graphs, and any constructed timing phase with no compatible fused MoE operator. It is also incompatible with whole-forward FPM, including the legacy Task `forward_model='fpm'` rewrite. Invalid graph/source combinations fail as invalid configuration rather than triggering estimator fallback. An untrained `fpm_regression` model remains not-ready; retaining a source in its configuration is not evidence of source-specific prediction support.

AFD regular companions currently reject exact-source requests through their legacy estimator and fixed timing paths. The external-FPM companion forwards the source to canonical validation, which rejects the incompatible FPM request. These controls describe standalone AISimulate behavior, not downstream Dynamo planner integration.

`nextn` remains compute-side identity. Expected accepted draft tokens are a
simulator workload assumption, supplied separately by the unified CLI as
`engine.nextn_accepted`; they do not tune the estimator.

### Selection and fallback

`estimation_mode` defaults to `auto`; `fallback_policy` defaults to `deny`.
Auto always searches `op_level`, `fpm_interpolation`, then `fpm_regression`,
including when fallback is denied. For an explicit mode, deny permits only
that estimator; allow tries the requested estimator first, then the remaining
estimators in the same global priority order. Each native mode tries the
ordered `systems_paths` before moving to another estimator. Omitted roots preserve
configured SDK discovery (or the systems-path environment override when the SDK
uses its packaged default); an explicit `default` entry selects the packaged root.
The resolved paths are shared by preflight, construction and capacity estimation.
Selection occurs
at construction; queries do not silently switch estimators on a data-domain error.

Invalid caller configuration does not trigger fallback. A constructed regression
may be unready: nonempty queries return `None` until their selected workload
store has a usable fit. Offline prediction/recommendation reject an untrained
regression instead of fabricating a latency. The current aggregated regression
retains its four workload stores; dedicated roles retain one each.

The returned provenance records the requested and selected modes, failed
selection attempts, effective backend version, data policy, selected root,
and complete estimator configuration. Its resolved config pins the selected
mode with deny so saved replay input repeats that selection.

### Speculative decoding configuration

Use the same `best_available(config)` constructor and returned model for
speculative iteration costs. The canonical `speculation` field carries the
existing SDK scheme identity:

| Field | Default | Meaning |
| --- | --- | --- |
| `speculation` | `null` | Optional speculative scheme; omit it for the existing autoregressive or legacy `nextn` configuration. |
| `speculation.kind` | Required | Registered scheme name: `none`, `mtp`, `ngram`, `eagle3`, `dflash`, `dspark`, or `draft_model`. A valid `none` configuration normalizes to disabled speculation. Scheme, model, and backend capability checks still apply. |
| `speculation.params` | `{}` | Scheme-specific parameters, validated by the existing scheme implementation. Unknown parameters are rejected. |
| `speculation.draft_model_path` | `null` | Draft checkpoint path or model ID. Required and verified for standalone `draft_model`, which builds its graph from that checkpoint. |
| `speculation.draft_config` | `null` | Parsed draft checkpoint configuration retained as identity evidence. Learned schemes can reuse resolved geometry; this field does not replace `draft_model_path` for standalone `draft_model`. |

For example, Python/JSON prompt lookup retains its existing shape:
`{"kind": "ngram", "params": {"num_speculative_tokens": 2}}`. MTP uses
`{"kind": "mtp", "params": {"depth": 2}}`. Legacy `nextn` input remains
supported; do not combine a nonzero `nextn` with `speculation`. The generic cost
API uses each scheme's parameter validation rather than a shared 1–5 draft-token
limit. The existing n-gram contract retains vLLM and 1–5 draft tokens. Replay's
backend, scheme, and scheduling limits remain separate.

Rust callers now construct the public struct instead of the former
`ForwardPassSpeculationConfig::Ngram` enum variant:

```rust
use aisimulate_core::{
    BackendKind, DatabaseMode, EstimationMode, ForwardPassPerfModel,
    ForwardPassPerfModelConfig, ForwardPassSpeculationConfig, ForwardPassWorkerType,
};

let mut config = ForwardPassPerfModelConfig::new(
    "Qwen/Qwen3-8B", "h200_sxm", BackendKind::Vllm, ForwardPassWorkerType::Decode,
);
config.estimation_mode = EstimationMode::OpLevel;
config.database_mode = DatabaseMode::Sol;
config.speculation = Some(ForwardPassSpeculationConfig {
    kind: "ngram".into(),
    params: [("num_speculative_tokens".into(), 2.into())].into_iter().collect(),
    draft_model_path: None,
    draft_config: None,
});
let model = ForwardPassPerfModel::best_available(config)?;
```

This is a Rust source migration; existing n-gram JSON and legacy `nextn`
configuration remain readable. OP timing and analytical SOL reuse the same
speculative graph: choose `estimation_mode="op_level"` and set `database_mode`
to the desired database policy, or `"SOL"` for analytical timing. SOL is not a
new estimator mode, and individual operations may lack SOL support. FPM's
`estimator_config.fpm_interpolation.method="sol"` instead controls interpolation
of measured whole-forward data; it does not select analytical-only timing.

Query conventions remain unchanged. For OP/SOL,
`estimate_forward_pass_time_ms()` consumes already-packed `ForwardPassMetrics`
decode counters: for `B` real requests, verification width `v`, and per-request
KV length `s`, use `num_decode_requests = B*v` and
`sum_decode_kv_tokens = B*v*s`. Their ratio remains `s`; the query does not apply
the width again. `predict_decode_latency_total()` wraps the same counters.
`static_phase_latency()` instead takes the real request batch and applies the
configured verification width internally. This OP telemetry convention is
distinct from the proposed speculative FPM mapping, which remains unsupported.

The canonical constructor currently supports explicit speculative schemes with
OP/SOL timing. FPM interpolation and regression are unsupported candidates for
these schemes. An explicit FPM request with `fallback_policy="deny"` fails;
`auto` or an explicit mode with `fallback_policy="allow"` can select a supported
OP candidate while preserving the complete speculative identity. They do not
drop speculation to obtain an autoregressive estimate. FPM profiles remain
autoregressive-only. Lower-level FPM hybrid code does not imply support through
the canonical ForwardPassMetrics query API.

Acceptance assumptions and scheduler seeds belong to the prediction/Replay
layer, not this cost configuration. The returned model continues to estimate
iteration latency; its constructor and query signatures are unchanged. For
prediction and AgentX, the [registered-scheme Replay input](../replay/engine/speculation.md#registered-schemes)
adds acceptance and a seed while retaining this cost identity. That path uses
the existing linear 1–5-token executor and requires fixed KV capacity; broader
cost-model support does not imply broader scheduler support.

### Estimator controls

`estimator_config` is passed intact through the Python facade, CLI, Sweeper,
and Replay, then parsed and validated in Rust. Unknown fields report their
nested paths. The supported namespaces are:

- `op_level.ascend`: measured Ascend910C/SGLang composition controls below.
- `features`: `attention_kv_weight`, `prefill_attention_pair_weight`, and
  `ffn_token_weight`, each defaulting to 1.0. These currently affect regression
  only; positive finite values are required when regression is constructed.
- `fpm_regression`: independent `sampling`, `min_observations` (5), and `fit`.
  The default fit kind is `standardized_nnls` (also accepted as `linear`),
  with a free intercept and nonnegative slopes. `spline` selects an additive
  piecewise-linear fit with learned knots and nonnegative segment slopes.
  Optional `fit.linear` controls fitted axes, signed slopes, and lazy updates;
  omission preserves the existing linear behavior and serialized defaults.
  `singular_ridge_scale` defaults to `1e-9` and applies to the shared linear
  fit only when retrying a singular equation. `rebuild_interval` defaults to JSON `null` / Python
  `None`, disabling periodic rebuilding. A positive integer opts into
  rebuilding after that many retained-sample mutations.
- `correction`: `enabled` (true), independent `sampling`, `min_observations`
  (5), `factor_bounds` (min 0.5, max 2.0), and the existing `max_num_tokens`
  (8192), `max_batch_size` (512), and `max_kv_tokens` (2000000) ranges.
- `fpm_interpolation.method`: `auto` (default), `sol`, `direct`, or `direct_simplex`. Rust selects
  SOL for a registered architecture, or direct interpolation for a verified
  architecture without a registered class when a profile is supplied. Without
  a profile, auto retains SOL. Explicit SOL requires a registered analytical
  model. Both direct methods require a profile and support registered architectures.
- `fpm_interpolation.collect_coverage`: `false` (default). Opt in to bounded
  evidence from actual direct-FPM lookups, as described below. It requires
  explicit `estimation_mode: fpm_interpolation`, a resolved direct method, and
  `fallback_policy: deny`. It is omitted from normalized serialization when false.

For `system="ascend910c"` with SGLang, Rust resolves these defaults and retains
explicit overrides in saved engine identity. Other systems/backends reject
explicit Ascend controls.

| Field under `estimator_config.op_level.ascend` | Default | Behavior |
| --- | --- | --- |
| `routing` | `balanced` | Uses the measured synthetic balanced expert-count profile. Other distributions are not yet supported by the complete model composition. |
| `prefill_expert_workload` | omitted | Observed expert assignment counts for DeepEP Prefill compute, with an `observation_id`, matching `quantization_sha256` and positive global `input_tokens`. GLM uses `shape_policy: anchored` and75layer vectors of256counts, each conserving8times `input_tokens`. Qwen uses `shape_policy: measured` and60layer vectors of512counts, each conserving10times `input_tokens`, on TP2/DP8/EP16, TP4/DP4/EP16 or TP8/DP2/EP16. Counts are nonnegative and each expert count is at most `input_tokens`. The selected model checks its own geometry and descriptor. Native operators price the resulting per-layer critical-rank compute shapes. |
| `decode_expert_workload` | omitted | Optional observed TP4/DP4/EP16 Graph Decode routing workload. GLM uses bucket8 and75layers, Qwen uses bucket4 and60layers. Qwen Prefill may use historical costs or the explicitly selected current FP32 path. Requires matching ModelSlim SHA256, framework commit, sample SHA256, input length, history policy and16-rank received-token/active-expert samples per layer. Rust prices rank costs from measured expert endpoints, selects each layer's critical rank and interpolates sequence positions. Unmatched request vectors retain balanced costs. |
| `tensor_moe_routing_profile` | unset | Explicit GLM tensor-MoE Decode calibration with a model/backend identity, evidence hash, 75 per-layer expert probabilities and padding IDs, and a measured `(physical_tokens, active_experts, latency_ms)` grid. Requires Graph execution and matching EP1 geometry. Rust counts real and padding tokens separately, interpolates within the grid, and rejects uncovered queries. Prefill retains its existing measured path. |
| `shared_expert_overlap` | `false` | Adds shared-expert and routed-expert costs serially, matching the observed single-stream GLM eager and Graph paths on the pinned 910C runtime. Explicit `true` models parallel branches using `max(sum(routed), sum(shared))`; Tensor-MoE prefill retains serial composition. |
| `decode_execution_mode` | `graph` | `graph` selects captured decode profiles and padding buckets; `eager` selects eager profiles with attention-TP token alignment. Missing eager identities fail explicitly. Prefill remains eager. This does not control scheduler overlap or CPU launch overhead. |
| `qwen_graph_small_ops` | `false` | Qwen-only Graph Decode coverage switch. `true` also requires measured Graph contracts for embedding, GQA prepare/cache/attention/gate, and logits finalization. Missing or historical eager rows fail contract validation; there is no proxy fallback. `false` preserves the earlier partial Graph composition and its explicitly reported small-role proxies. Prefill is unchanged. |
| `qwen_eager_ep_pair` | `false` | Select one measured `expert_transport_pair` boundary per Qwen FP32 TP-scatter eager Decode layer. Requires serial expert execution. The cost is an unprofiled dispatch-followed-by-combine measurement at EP16/cap16, with expert compute charged separately. This compound observation provides total communication timing while per-stage attribution and the effect of intervening expert work remain unverified. It replaces both stage timing nodes without synthetic zero-cost rows or a total-forward residual. Prefill and Graph retain their own selected boundaries. |
| `qwen_eager_host_cpu_bound` | `false` | Apply independently measured exclusive CPU API component means to Qwen FP32 eager Decode at TP4/DP4/EP16. Six source-defined components are multiplied by architectural call counts for each measured rank. Existing `Overlap` composition takes the maximum of the rank CPU paths and the device path. Telemetry binds CPU samples to their physical attention-DP owner, including unequal request counts and idle DP ranks. Same-rank mixed Prefill/Decode is outside the measured CPU contract and fails explicitly. Requires serial isolated-kernel timing and zero residual overhead. Host curves use logical requests per DP, strict measured bounds, and relative standard error of the mean at most 10% with at least ten samples. The initial contract is specific to the pinned Node20 host/runtime and requests 1 through 3. This is a CPU/device lower bound. Per-operation launch dependencies and final device tail remain validation gaps. |
| `qwen_eager_host_cpu_transfer` | `false` | Explicitly allow the TP4 CPU reference to price TP2/DP8 or TP8/DP2 on the same 16-rank Qwen runtime. Requires `qwen_eager_host_cpu_bound`. CPU lookup identities retain the source TP4 contract. Device operators retain target-topology costs, and physical CPU ranks are grouped by the target attention TP. The assumption is that the measured CPU API costs transfer across these layouts. Target-topology validation is required and the assumption is recorded in execution metadata. |
| `qwen_prefill_current_roles` | `[]` | Explicitly select current eager Prefill costs for `gdn_conv`, `gdn_scan`, `gdn_state_io`, `gqa_attention`, `expert_dispatch`, `routed_experts`, and/or `normal_transport` at TP2/DP8/EP16, TP4/DP4/EP16 and TP8/DP2/EP16. Unselected Prefill roles retain their historical cost contracts. With `qwen_runtime_contract: historical`, a nonempty selection requires Graph Decode. With `modelslim_fp32_tp_scatter`, it supports eager or Graph Decode and requires all four of `gdn_conv`, `gdn_scan`, `gdn_state_io`, `gqa_attention` so the recurrent state and GQA kernels use current costs. Decode retains the requested runtime contract. Current Prefill convolution uses history0 for the first chunk and history1 for initialized state, selected from the request prefix. Selecting `routed_experts` uses a distinct measured INT8-input/BF16-output expert-chain reference contract. Its global-proxy token axis represents `ceil(tokens / EP16) * top10` received rows. BF16 Normal input quantization remains unpriced and is reported explicitly. Selecting `normal_transport` requires the complete `prefill_expert_workload`. It replaces dispatch/count-metadata and combine with separate BF16 H4096 top10 EP16 ready-input measurements indexed by sent tokens and critical receive-core work. The native mapping preserves expert order across48 AIV cores and assumes equal source-rank fractions within each captured expert total. Full-rank Prefill telemetry supplies the actual TP-padded SUM_LEN buffer, including idle peers. Static queries retain their declared uniform DP geometry. Both send-size and concentration must lie inside qualified measured slices. Heldout layout and source-skew validation remains necessary. Other current Prefill operators retain their existing axes. Qualified coordinates are required. The selected set and remaining historical costs are reported in execution assumptions. |
| `qwen_runtime_contract` | `historical` | `modelslim_fp32_tp_scatter` selects the audited Qwen ModelSlim runtime with FP32 recurrent state, TP-padded attention, TP-scattered MoE, 60 attention reduce-scatter calls and 61 all-gather calls in each Prefill or Decode forward, and dispatch capacity16. Decode execution mode selects eager or Graph costs. Prefill uses eager collectives and requires FP32 eager operator costs. Supported topologies are TP2/DP8/EP16, TP4/DP4/EP16 and TP8/DP2/EP16. EP16 expert-compute and transport measurements can be reused across these attention topologies at identical sent-token coordinates within the same eager or Graph execution mode. Attention operators retain their topology-specific contracts and missing token coordinates fail. The historical setting selects the original BF16 eager data contract. Each role requires matching measured data. |
| `decode_timing_scope` | `isolated_kernels` | `runtime_layers` replaces decoder-layer kernel sums with measured, numerically checked real layer invocations, including eager host dispatch, stream management and communication boundaries. Requires exact-topology profiles for eager TP16/ADP4/EP16/DCP1 or TP32/ADP32/EP32/DCP1 and measured request-count brackets. Layer profiles retain their measured internal execution dependencies. Idle DP populations need their own curves; padding does not count as a real request. With `shape_policy: anchored`, unequal counts within one TP padding bucket reuse each rank's homogeneous-population curve; MAX_LEN populations crossing buckets use the maximum active request count as their physical-population anchor. KV context transfers use the attention-only delta described below. All transfers are `estimated`. SUM_LEN cross-bucket populations, request-count extrapolation, other topologies, graph execution and additive residual calibration are rejected. Prefill and embedding/final output operations retain their existing timing. |
| `enable_dp_lm_head` | `true` | Matches SGLang's output-layer grouping. `true` retains attention-local TP; `false` uses global TP, sums actual output-request counts across attention DP, and includes DP hidden-state reduction and TP logits exchange. Set separately for each worker role to match its launch flags. |
| `enable_tp_lm_head_all_to_all` | `false` | With global LM head, uses measured all-to-all only for equal nonzero row counts with attention DP equal to global TP; other batches use all-gather. Local logits cast is measured; additional reordering/scatter and CPU synchronization remain unqualified. This path is reported as estimated composition. |
| `eager_runtime_overhead` | omitted | Optional additive per-forward residual calibrated from unprofiled forward telemetry. Requires `calibration_id`, `operator_contract`, `attention_tp`, `attention_dp`, `moe_ep_size`, `prefill_ms`, and `decode_ms`. Timings must be finite and nonnegative. The topology must match the canonical identity; currently restricted to SGLang 0.5.22 and eager execution. |
| `eager_runtime_overhead.operator_contract` | no compatible default | Must be `glm52_eager_metadata/v2`, binding an explicitly supplied calibration to the composition that includes measured CPU-to-NPU sequence-length transfers. Missing or older identities fail; remove the whole residual setting for operator-only prediction. |
| `eager_runtime_overhead.decode_per_extra_active_rank_ms` | `0` | Optional finite nonnegative decode residual per additional attention-DP rank with real requests. Uses the actual synchronized rank vector before padding, not batch size or total configured DP. Fit separately from the base `decode_ms` and retain independent validation. |
| `kv_transfer_page_layout` | `contiguous` | Selects the measured PD page pattern: `contiguous` or `fragmented`. Both role models must agree; applies when `engine.kv_transfer.type: measured`. |
| `shape_policy` | `measured` | `measured` requires measured interpolation brackets. `anchored` preserves covered queries and estimates missing numeric coordinates using same-identity boundary measurements and analytical work ratios. It never substitutes hardware, precision, kernel, topology or graph capture buckets. Both PD roles must agree. |

Qwen measurement tables can declare `latency_statistic` as `mean` or `median`.
Legacy device rows use `median` and retain the raw-CV limit of 10%.
An explicit arithmetic mean uses relative standard error `cv / sqrt(sample_count)`
with the same 10% limit and at least ten observations. Raw CV and sample counts
remain unchanged. Mean uncertainty describes the recorded sampling run. A separate
launch or workload comparison is still needed to assess systematic error.
The exclusive-host-CPU contract always stores means.

`runtime_layers` is a coarser operator boundary, not an additive correction to
whole-forward measurements. Its table contains the isolated invocation of each
of eight structural layer types. The engine combines them using the audited
GLM layer counts, replaces all covered timings, and retains the covered
operations for resident-weight accounting and explicit mixed-round kernel composition. The scope includes real Python
and runtime work around kernels; it must not be described as pure device-kernel
time. With `shape_policy: measured`, both request and KV coordinates must have
measured interpolation brackets. With `shape_policy: anchored`, the engine can
extend the KV coordinate in either direction within the sparse-attention regime,
while retaining the measured request population, padding, topology and quantization
identity. Each structural layer type chooses its nearest measured KV coordinate
inside its applicable request brackets, then computes:

```text
layer(new_KV) = measured_layer(layer_anchor_KV)
              + sparse_attention(new_KV) - sparse_attention(layer_anchor_KV)
              + indexer_delta_if_this_layer_computes_an_index
layers(new_KV) = sum(layer_count * layer(new_KV))
```

For unequal eager Decode populations crossing attention-TP padding buckets,
MAX_LEN materializes the largest token buffer on every rank. The layer query
therefore uses the largest active request count, while retaining the number of
real active ranks and each rank's KV context. Logical request counts in the
scheduler, logits, KV allocation, and reported batch statistics are unchanged.
This is an anchored estimate: a padded request has no real KV sequence, so a
homogeneous measured population is not an exact timing identity or a guaranteed
latency bound. The corresponding measured request and active-rank brackets must
exist; missing data still fails. SUM_LEN does not materialize that shared buffer
and cannot use this transfer across padding buckets.

Collection identities are checked separately for each exact TP/ADP/EP topology.
Adding a TP32 table does not reuse TP16 observations or permit mixed runtime
identities inside one topology. Numerical coverage must include all global TP
ranks: at least 16 times nine checks for TP16 and 32 times nine for TP32.
The sparse-attention delta uses 64 total heads divided by the local attention TP;
the replicated indexer retains its 32-head identity. An enabled topology still
fails when its actual active-rank, request, or KV coordinates lack qualified data.

The eight representative layer IDs are 0, 1, 2, 3, 4, 5, 6 and 77; only
0, 1, 2 and 6 compute an index. Their multiplicities yield exactly 78
sparse-attention and 21 indexer calls, evaluated by the canonical Rust operator
model with the same active and padded batch. Independent, quality-qualified
captures may have different actual KV lengths; those measured coordinates and
latencies remain unchanged. Layers with an existing KV bracket use that bracket
directly. The
anchor and requested context must both be at least the model's DSA top-k of 2048, so the extension cannot
cross that execution boundary. Dense projections, expert work, host dispatch and
collectives retain the measured invocation boundary at that request population;
they are not scaled with context length. This is a modeled context transfer,
always reported as `estimated`, not a new measurement or a fitted whole-forward
residual. Missing attention data, request brackets or layer identities still fail
explicitly. This also covers the early decode steps whose context is slightly
shorter than the captured profile after its warmup iterations. No request-count
or topology extrapolation is implied, and this opt-in scope does not establish
complete PD search coverage or end-to-end serving accuracy. Changed token content
can change expert routing; independent workload validation remains necessary.
Layer profiles use the model-selected experts from the captured inputs, bind the
ModelSlim descriptor SHA256, and expose `decode_routing:
captured_model_selected_top8` in execution assumptions. The `routing` option
continues to describe the independent kernel tables used by prefill; these
layer observations are not labeled as synthetic balanced expert routing.

`prefill_expert_workload` represents the observed input distribution.
For each MoE layer, contiguous expert placement partitions its
observed assignments according to the requested EP16 or EP32 topology. The
largest rank's assignment fraction determines the routed compute token count
from the actual global padded token volume. Native operators still perform
all timing interpolation and extrapolation. Each layer keeps its own shared
expert overlap boundary, and weight accounting remains unchanged. The complete
observed counts and their identity survive canonical configuration serialization.
Tensor-MoE rejects this option; runtime-layer decode is unchanged.

Qwen's historical expert table uses a global-token coordinate whose balanced
rank-local receive count is `ceil(global_tokens / EP) * 10`. Its observed
workload maps the actual attention-DP token volume through each layer's largest
rank receive fraction to that coordinate. Dispatch and Combine retain their
own measured coordinates. Shared-expert and Decode compute keep their existing
workload mappings. The Qwen historical expert table measures INT8 input with
precomputed scales. Normal BF16 input quantization remains a separate uncovered
cost. Complete count vectors and their observation identity remain in the
canonical configuration for independent workload validation.

This option models only the critical rank's compute volume. The existing
balanced within-rank compute kernel profiles and maximum-sender communication
profiles remain explicit assumptions. It neither scales a measured Combine wait
nor counts that wait on top of the late rank's compute. Profiles must come from
the same weights and representative input distribution; matching the quantization
descriptor alone does not establish matching weights. Exclude validation prompts
and seeds when producing the workload observations. A new prompt distribution,
expert placement or transport bottleneck requires separate validation; this
option alone does not establish a serving accuracy bound.

`decode_expert_workload` binds recorded routing work to the GLM or Qwen Graph expert
chain through the canonical performance-model configuration. Each sample records
`request_counts`, `sequence_length`, and `layers` of 16 rank records with
`received_tokens` and `active_experts`. GLM records75layers and conserves256
assignments from four eight-token Graph buffers with top8 routing. Qwen
records60layers and conserves160assignments from sixteen single-token
EP sources with top10 routing, including the TP padding observed in the capture.
`samples_sha256` hashes
the compact JSON sample array in the declared field order. The model checks
`quantization_sha256` against its ModelSlim descriptor and requires
TP4/DP4/EP16/DCP1 with isolated-kernel Graph Decode. Qwen's receive-volume
curves require H4096/I1024,32local experts and2560physical buffer rows.
The Qwen option supports both historical Prefill costs and the explicit
`qwen_runtime_contract: modelslim_fp32_tp_scatter` Prefill path with matching
`qwen_prefill_current_roles`. The observed workload applies only to Graph Decode.
Changing the Prefill runtime contract preserves the Graph Decode operator specs
when `qwen_graph_small_ops: true` and all workload identity guards match.
Its independent-rank
telemetry path passes the complete request vector into expert pricing so skewed
and cross-rank mixed work does not select a uniform captured routing sample.

Rust computes each rank's expert-chain cost from the single-expert and balanced
curves at fixed receive volume, takes the maximum within each layer, and sums
the model's layer costs. Repeated samples at one sequence coordinate are averaged.
Sequence coordinates interpolate within each exact request-count vector.
`history_policy: deny` rejects positions outside the sampled range.
`history_policy: hold_boundary` explicitly retains the boundary cost there.
Unmatched request vectors use the existing balanced expert cost. Prefill and
mixed-mode work retain their existing paths. Workload-derived costs report an
estimated source. Preserve the routing source and prompt-distribution identity
alongside these inputs and validate the resulting throughput and latency.

`eager_runtime_overhead` preserves the raw operator estimates and adds a separately
named, `estimated` runtime contribution once per nonempty forward, including
synchronized attention-DP batches. It is never multiplied by request count, and
empty batches cost zero. Its residual may include host submission, synchronization,
and gaps in operator coverage; it is not a measurement of isolated CPU time. Keep
the referenced raw observations, calibration rule, software/environment identity,
and held-out validation with the calibration ID. No default overhead is inferred
from hardware/model names or copied to unmeasured parallel strategies. Static
diagnostics, replay, and search use the same serialized control.

The eager DCP=1 composition queries CPU INT32 sequence-length transfer profiles
for each of the 78 attention calls and 21 indexer refreshes. Prefill uses request
counts; decode uses padded rows, including the synchronized attention-DP layout.
Graph decode does not incur these copies. These measurements do not include the
attention/indexer kernels and do not add resident model weights. The source
contract is SGLang `baa9befa381511cd4d56f83980951e5a45fffc6d`,
`hardware_backend/npu/attention/ascend_backend.py::forward_sparse` and
`layers/attention/dsa/dsa_npu_indexer.py::forward_npu`.

Residuals fitted before these transfers were modeled absorb at least some of
their cost. They are incompatible with the updated composition. Do not merely
add the new contract string to old coefficients. Historical calibrated reports
remain reproducible with their original frozen simulator image; root-cause
validation of the updated model must omit `eager_runtime_overhead` entirely.
No new empirical residual profile is supplied with this correction.

For a nonempty decode group with `active` real attention-DP ranks, the optional
rank term gives `decode_ms + (active - 1) * decode_per_extra_active_rank_ms`.
Padded empty ranks do not increase this cost. Prefill uses only `prefill_ms`.
Static per-rank diagnostics assume the configured ranks are uniformly active;
request-vector estimates and replay use their actual rank occupancy. The default
zero slope preserves previous constant-overhead profiles. This models steady
execution residuals, not first-use compilation, gateway buffering, or KV setup.
EngineSpec schema 34 records the active attention-DP population and bounded
population transfer for runtime layers. This distinguishes real requests from padded idle rows. Rebuild binary engine
caches from older schema versions rather than reusing them.

SGLang replay uses synchronized attention-DP timing for supported Ascend
aggregated deployments as well as disaggregated workers. Every committed pass
passes all rank metrics, including idle ranks, into the same native estimator.
This prevents a single active rank from being priced as a uniformly occupied
DP group. Pure prefill and pure decode retain their existing measured paths.
Cross-rank mixed prefill/decode rounds require `shape_policy: anchored` and
report `estimated` provenance. Each rank retains its own attention phase and KV
length. Any prefill rank selects eager SUM_LEN padding and normal DeepEP expert
dispatch for the whole group, including ranks doing decode. These rounds use
the independent kernel composition; `runtime_layers` expands its covered kernels
instead of applying a pure-decode layer profile to a different communication
path. This composition does not include unmeasured eager layer overhead and is
not a claim of mixed-round accuracy. `shape_policy: measured` rejects the
unqualified mixed boundary. Prefill and decode within the same rank remain
unsupported; this matches the validated deployment's disabled mixed chunking.
The canonical Python model's `estimate_ascend_dp_operations(rank_metrics)` returns
the selected critical rank's `(name, latency_ms, energy_wms, source)` breakdown
from the same Rust group query. Include every DP rank, including idle ranks.
This is diagnostic evidence for that forward, not an alternate estimator or a
measurement of host scheduling time. Unsupported hardware/estimators reject it.

The GLM adapter requires the local checkpoint's `config.json` and
`quant_model_description.json`. It validates the full ModelSlim precision
partition, including unquantized indexer/router/logits weights, and uses BF16
KV with 128-token physical pages. Attention-local TP1/2/4/8/16 combines with
either global EP16/32 and MoE-TP1, or global MoE-TP16/32 and EP1. Attention-DP
equals global width divided by local TP. PP1/CP1 and no speculation/EPLB remain
required. Native memory checks filter infeasible layouts for each workload.
Batch sizes are per attention-DP replica, with uniform load assumed across
replicas. Graph decode uses qualified buckets from 8/16/24/32, divisible by
local TP. Eager decode rounds tokens up to local attention TP before tensor and
expert partitioning, following the pinned MLP synchronization path. Attention
keeps active requests separate from padded zero-length KV rows and queries
measurements for that layout; unpadded eager data is a separate plane.
For TP8/ADP2, one active request therefore supplies eight matrix rows and one
valid KV sequence. Tensor-MoE with ADP1 and no A2A requires no MLP synchronization
and retains unpadded eager tokens. It never relabels captured
measurements as eager measurements. DCP2/4/8/16 may divide local TP in aggregated prediction; it uses
separately measured storage/attention/merge chains and logical KV blocks of
`128 * dcp`. Eager DCP decode has no qualified profiles and is rejected.
PD with DCP is unsupported by the pinned runtime.

For a graph-disabled reference run, set
`estimator_config.op_level.ascend.decode_execution_mode: eager` and run SGLang
with `--cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled`
and `--disable-overlap-schedule`. The latter matches the
blocking native scheduler model. Shared-expert device execution is controlled
separately by `shared_expert_overlap`, whose default serial composition matches
the pinned GLM traces. An explicit parallel setting uses the corresponding
branch composition. Model execution
mode selection alone does not qualify end-to-end accuracy or include unmeasured
host launch overhead.

For SGLang PD workers with attention DP greater than one and DCP1, native
execution gathers the committed batch metrics for every rank before estimating
the shared forward duration. Idle ranks contribute zero active tokens. The
Ascend operator model aligns each rank's tokens to attention TP and selects
the pinned runtime's MAX_LEN or SUM_LEN padding policy. Prefill uses SUM_LEN;
eager decode uses MAX_LEN when twice the aligned token sum is at least the
maximum aligned length times DP. Captured decode uses a shared graph bucket.
Global dense/tensor-MoE work uses the resulting global buffer; attention
retains each rank's active sequences. SUM_LEN gather/combine uses the measured
all-reduce branch. Expert receive counts assume balanced routing unless an
observed `prefill_expert_workload` is supplied for prefill compute, and
DeepEP time uses the measured maximum-sender profile; a nonuniform transport
profile is not inferred from it. Mixed prefill/decode groups remain unsupported
by this synchronized timing path. Aggregated execution retains its prior model.

Direct forward-pass queries with multiple rank entries require the complete DP
vector, including idle entries. A single entry retains the historical uniform
rank estimate. Missing padded attention measurements continue to fail explicitly.
Per-operation evidence describes the critical rank; it is not a sum of energy
across all ranks or nodes.

EngineSpec binary schema 31 adds conditional DP communication branches, following
schema 30's attention eager-alignment policy. Rebuild
the native extension together with the Python model composition; older binary
payloads are rejected rather than decoded using the new positional layout.

These tables support workload-dependent shape queries, not just 8192-input,
1024-output requests. The default `shape_policy: measured` interpolates only
inside measured bounds. Sparse Attention prefill prefix interpolation weights
the anchors by the number of causally selected query/key pairs, capped at 2048
keys per query. Exact table entries retain their measured values. When both
prefix anchors are saturated, their measured variation is interpolated by
prefix distance. Indexer and DCP-chain tables retain their own interpolation.
Anchored estimates use the same sparse-prefix weights when a bracket requires
extrapolating a missing inner coordinate or batch partition.
`shape_policy: anchored` extends numeric axes while
retaining the original rows. Attention/indexer estimates use causal query/key
work, the sparse top2048 limit, HBM traffic and the calibrated BF16 rate;
DCP also accounts for storage and query/output communication. The measured
boundary latency is split into a fixed component and a remaining component
scaled by the analytical work ratio. Attention/indexer and DCP derive the fixed
component from the boundary and its nearest measured neighbor in analytical
work space, bounded between zero and the smaller latency. The full latency of
the smallest collected shape is not a launch cost: using it as a floor would
charge a small query for all the compute of a much larger collected query.
With only one distinct work anchor, the fixed component is unidentifiable;
these attention estimates scale the whole boundary latency and remain explicitly
estimated. Exact measured shapes and the strict measured-range policy are
unchanged. Other operator families retain their existing measured-curve floors.
Eager prefill batch interpolation may evaluate a ragged grid's missing inner
coordinates this way. Graph buckets remain categorical and must be measured.

Linear/BMM/MoE extensions use compute and weight/activation traffic;
vector work scales with token count. HCCL uses the same measured collective
identity and its topology's latency/bandwidth calibration. HCCL identities
support `bfloat16`, `float16`, `float32`, `int32` and `int64`, with the payload
axis expressed as input bytes per rank. Eight `int64` metadata fields occupy
64bytes per rank. Their measured dtype and payload range are preserved through
lookup, independently of floating-point model-activation collectives. PD keeps page
rounding and scales page work within the same sender mapping and page layout.
DeepEP's measured dispatch bounds remain enforced: its observed runtime
tiling failures are not made executable by enabling extrapolation.
The policy is part of the canonical identity, saved configurations and PD
metadata; extrapolated compute results carry the `estimated` source tag.
`fallback_policy: deny` still prevents switching estimators; it does not
disable an explicitly selected op-level shape policy.

An accepted topology or sufficient KV capacity does not establish measured
coverage. Numerical execution, extrapolation and table lookup checks do not
establish whole-serving accuracy. Long-context estimates can be substantially
less accurate than nearby measured points, especially across runtime tiling
changes. Use `measured` to require the original data domain.

EngineSpec schema35 adds the optional tensor-MoE routing profile. Schema29 added request-based HCCL payload sizing and separate
resident-copy counts for shared RoPE storage and retained MLA weight views.
Recompile older binary specs from their canonical configuration.

The top-level `fpm_profile` contains the complete profile dictionary: pinned
model revision, architecture, context length, expert count, deployment precision
and topology, cache geometry, memory evidence, and provenance. A profile requires
an explicit literal `backend_version` that matches its selected deployment;
slot aliases and omitted versions are rejected. This profile schema does not
declare recorded DCP, so combining `fpm_profile` with an explicit `dcp` is
rejected; measured DCP profiles without `fpm_profile` retain their existing route.
Profile/schema and precision conflicts fail before estimator fallback. Omitted precision fields are filled
from the profile and preserved in the resolved canonical configuration.

The canonical SDK and `predict` accept SGLang PP1 profiles for measured forward
timing. SGLang prediction requires a linear cache and recorded `runtime_memory`
from the matching native worker. Its fixed cache allocation is reused only with
the exact recorded scheduler envelope and memory fraction. Automatic SGLang
cache budgeting from component byte estimates remains unsupported. The
self-service collector and profile-based Sweeper enumeration retain their vLLM
backend restriction.

Each deployment may declare `worker_type: prefill`, `decode`, or `aggregated`.
The canonical construction request selects only the matching role's precision,
scheduler envelope, cache geometry and memory evidence. Separate P/D deployments
may share hardware and topology while retaining different resources. Historical
deployments without this field keep their shared-resource behavior and omit the
field on serialization; they cannot coexist with role-specific deployments at
the same hardware/runtime/topology identity. The Python memory adapters accept
`worker_type` to select the same role, defaulting to `aggregated` for existing
callers. An absent or different explicit role is an error, not a fallback to
another role's capacity.

Runtime normalization and construction verify the profile architecture against
checkpoint `config.json` from the local model path or pinned remote revision
before interpolation selection. Missing or malformed architecture metadata and
mismatched declarations fail explicitly. This check reads configuration only:
no weights or analytical graph are required. An architecture with verified
metadata remains valid for direct interpolation without a registered analytical
class. Schema-only profile and application configuration parsing remain lightweight.

For measured-only timing, pass `estimation_mode="fpm_interpolation"`,
`fallback_policy="deny"`, and
`estimator_config={"fpm_interpolation": {"method": "direct"}}` together with
`fpm_profile`. Direct interpolation requires `database_mode="SILICON"`, emits
whole-forward native operations without SOL operations, and never constructs
an analytical graph. Profile resource estimates and memory planning do not
require timing data or a native timing model.

Direct readiness requires genuine measurements for the request's `worker_type`:
prefill requires prefill rows, decode requires decode rows, and aggregated
requires both phases. Querying an absent phase still fails explicitly. Construction
continues to later systems roots when a required phase is unavailable. Query
coverage still needs an exact point or supported interpolation. Cross-KV prefill
uses the nearest same-batch lower and upper KV curves that both cover the
requested token count, without a KV distance limit. SOL's site-distance guard
does not apply to this direct bracket. See the [self-service coverage rules](fpm-self-service/implementation.md#choose-the-model-execution-route).

Explicit `method: direct_simplex` first resolves the direct measured curves.
For a prefill query those curves cannot bracket, it uses barycentric weights
inside a same-batch Delaunay triangle of genuine measurements. The mesh scales
each coordinate axis to its measured range and sorts sites for deterministic
construction. Every weight must lie in `[0, 1]`. Queries outside the measured
convex hull fail. Decode uses the direct capture-regime curves. The executed
evidence reports `simplex_interpolation` and its original measured vertices,
latencies and weights. This method requires a profile and `SILICON` data mode,
and supports strict query coverage with fallback denied. Its interpolation
accuracy requires independent measurements for the intended shapes.

The returned provenance pins both the selected estimation mode and interpolation
method, alongside the complete normalized profile. Reusing its `config` keeps
that selection across serialization and replay. Later timing coverage errors
never switch estimator or interpolation method. A registered model's graph
construction failure does not change SOL to direct; top-level fallback still
follows the configured estimator ordering and policy.

- `fpm_interpolation`: `text_only` (false) permits text prefill/decode profiles
  for multimodal architectures while retaining encoder weights. It does not
  supply encoder timing. `unrecorded_quant_modes` (empty) may contain `fmha`
  and/or `comm` to match an explicitly unrecorded precision field in a profile.
  The corresponding top-level quant mode must remain unset. This selects null
  profile values exactly; it does not make precision matching a wildcard.
  `fpm_parquet_path` selects an external FPM parquet with its same-stem metadata
  sidecar. Unknown fields are rejected.

Sampling defaults to `bins_per_axis: [4, 4]` and `max_observations: 64` per
logical store. Rectangular grids were already supported. Regression's
`sampling.axes` now selects one to six distinct coordinates, defaulting to
`[attention, moe]`; `bins_per_axis` must have the same length, contain positive
integers, and have a representable product. The dimension is the number of
selected axes, independent of how many features the fit uses. Regression uses
dynamic `log1p` retention coordinates and fits standardized feature values. Correction
uses fixed raw workload coordinates; its one-dimensional prefill grid uses
the product of the two axis counts. Retention evicts the oldest sample from
the most populated cell when the store exceeds its budget.

Correction explicitly reports `feature_space: legacy_workload`. Its existing
prefill/decode/mixed stores and median-ratio calculation remain unchanged at
default settings. A shared role-based correction space requires separate
accuracy validation and is not accepted as a configuration value in this release.

Use `regression_store_diagnostics()` for per-store counts/readiness. Summary
readiness means at least one store has a usable linear fit, including when
spline fitting is selected; another cold store can still return `None`.
Readiness does not guarantee coverage for every query: spline predictions require
an available linear prediction for the same query. `tune_with_fpms()` preserves
the established FPM observation contract. Native construction still uses Python
model compilation; estimator selection, regression, correction, and latency
computation are owned by Rust.

For fit configuration see [Online regression](methods/online-regression.md).
For direct measurement coverage see [Whole-forward FPM](methods/whole-forward.md).

## Unified CLI timing fields

The `engine` block of `predict` and `recommend` YAML is documented in
[Replay engine](../replay/engine/README.md). The per-role `timing` fields below
select the forward-pass timing provider. SDK constructor names differ (for
example, `parallelism.tensor` becomes `tp`). A Default Range of `x` is
concrete-only.

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `engine.workers.<role>.timing.type` | `default` | `x` | `-` | `default`, `fixed`, or `polynomial`. |
| `engine.workers.<role>.timing.prefill_ms` | `null` | `x` | `-` | Nonnegative and required for `fixed` timing. |
| `engine.workers.<role>.timing.decode_ms` | `null` | `x` | `-` | Nonnegative and required for `fixed` timing. |
| `engine.workers.<role>.timing.forward_model` | `op_level` | `x` | `-` | `op_level` or `fpm`; `default` timing only. `fpm` replays whole-forward (FPM) latency measured for the role's exact model, hardware, backend version, parallel shape and quantization, and fails closed when no such cell exists. |
| `engine.workers.<role>.timing.fpm_parquet_path` | `null` | `x` | `-` | External FPM parquet for `forward_model: fpm`; the adjacent same-stem `.metadata.json` sidecar is required. Relative paths are anchored to the working directory when the engine is constructed. Preserved per role in recommendations, candidate YAML, and regular prefill/decode companions in AFD+PD. |

The canonical `estimation_mode`, `fallback_policy`, `estimator_config`, and
`fpm_profile` controls pass through the CLI into the same native constructor.
A role's `timing.estimator_config` replaces the global dictionary for that role;
include all required controls in that override. Legacy `timing.forward_model: fpm`
and `timing.fpm_parquet_path` normalize to FPM interpolation and its external pair.
The mode does not fall back to op-level on a missing measured cell.

## Flat engine precision and execution controls

These additional public YAML fields are concrete controls:

| Field | Meaning |
| --- | --- |
| `engine.gemm_quant_mode`, `moe_quant_mode`, `fmha_quant_mode`, `comm_quant_mode` | Explicit precision overrides; omitted values follow model/profile normalization. |
| `engine.kvcache_quant_mode` | KV storage precision: `bfloat16`, `fp8`, or `int8`. Omission follows model/profile normalization and can select FP8 for quantized checkpoints. |
| `engine.attention_backend`, `moe_backend`, `moe_kernel_source` | Runtime graph/backend and exact collected source selectors, subject to canonical validation. |
| `engine.enable_eplb`, `wideep_num_slots` | Expert load balancing and redundant expert slots; require compatible MoE execution. |
| `engine.nextn`, `nextn_accepted` | Compute-side MTP identity and separate expected acceptance assumption; both are required for explicit MTP. |
| `engine.enable_chunked_prefill` | Override chunked prefill; omission preserves the backend default. |

GEMM identities `w8a8` and `w8a8_dynamic` distinguish static and per-token
dynamic INT8 activation quantization with INT8 weights. MoE
`w8a8_dynamic` selects the dynamic INT8 expert-compute identity. These use
`gpu.int8_tc_flops` for arithmetic accounting and keep separate lookup keys
from weight-only INT8, SmoothQuant and FP8. The identities do not imply that
attention, KV cache, normalization or communication are INT8. Mixed checkpoints
still require per-component model composition and matching measured data;
declaring a precision alone does not establish a supported runtime cell.

For a local `GlmMoeDsaForCausalLM` checkpoint, the model reader recognizes
`quant_model_description.json` and retains a compact per-layer precision plan.
Fused projections must agree, every routed expert declaration is checked, and
unquantized `FLOAT` entries retain their meaning without assuming a runtime
dtype. Unsupported mixed expert precisions and conflicting checkpoint metadata
fail explicitly. This metadata normalization does not by itself qualify
operator coverage or deployment accuracy.

These controls require default timing on language roles. AFD and analytical
encoder configurations reject the unsupported controls. Use `--stack engine`
and the supported model/backend; an older external adapter can reject them at
capability validation before Replay. Expected acceptance is workload behavior,
not an estimator-fitting observation. See [Replay features](../replay/features.md)
for execution and cache combinations.

FP8 tensor-core compute is rejected during estimator construction on pre-SM89
GPUs, including `a100_sxm`. Choose a BF16 checkpoint on A100, or an FP8-capable
GPU with matching backend timing data. This checks compute precision, not weight
storage or KV dtype: weight-only formats that compute in BF16 remain distinct.
FPM also checks the preserved GEMM/MoE identity before selecting a timing cell,
including profiles without an original op-level graph.
Missing hardware metadata still follows the system-spec/data diagnostics.
Serving runtimes may execute FP8 checkpoints through weight-only kernels such as
Marlin on these GPUs; AISimulate does not model that path.

### KV precision and token capacity

An explicit `engine.kvcache_quant_mode` or worker timing override takes precedence,
subject to model/backend and profile validation. An FPM profile supplies its
recorded `kv_cache_dtype`. Otherwise, the first matching rule below applies:

| Checkpoint metadata | Inferred KV precision |
|---|---|
| `kv_cache_quant_algo` is not missing/null, `fp8`, `bfloat16`, or `"none"` (for example, `int8`) | Rejected during checkpoint normalization |
| DeepSeek V4/V4.1 architecture | FP8 |
| `kv_cache_quant_algo: fp8` | FP8 |
| `kv_cache_quant_algo: "none"` | BF16 |
| KV declaration missing/null or `bfloat16`; weight `quant_algo` is `fp8`, `fp8_block`, or `nvfp4` | FP8 |
| Other supported configurations | BF16 |

These are AISimulate policies, not serving-engine defaults. `torch_dtype: bfloat16`
does not prevent FP8 KV inference. For regular language roles using default timing,
pass `--set engine.kvcache_quant_mode=bfloat16` or set it in YAML to require BF16.
The explicit engine field also accepts `int8`, subject to model/backend validation;
this does not make an `int8` checkpoint KV declaration valid.
Fixed/polynomial timing, EPD, and AFD currently reject this engine override.

Ordinary `predict` output shows each known role's effective KV dtype and source.
JSON stdout and `prediction.json` summaries expose `kv_cache_precision.<role>`
with `dtype` and `source`: `user_override`, `checkpoint` (a matching explicit KV
declaration), `fpm_profile`, or `aisimulate_policy`. `--detail memory` also includes
`kvcache_quant_mode` beside capacity. Naive fallback and roles without a capacity
estimate omit this disclosure. Fixed/polynomial timing roles currently size capacity
at compile time and do not report the resolved precision through this path.

FP8 uses one byte per KV element; BF16 uses two. At the same byte budget, linear
FP8 KV holds twice as many tokens. This does not imply twice the throughput.

## Migrating saved configuration

Use `ForwardPassPerfModelConfig.from_legacy_engine_config(old_config,
worker_type, old_options, allow_regression=False)` to convert a saved flat
EngineConfig and tuning options. It pins the old explicit native mode instead
of changing it to auto. Set `allow_regression=True` only for an old caller that
allowed direct regression fallback; the migration preserves that two-mode
order rather than adding interpolation. Legacy `forward_model: fpm` maps to
`fpm_interpolation`, and `fallback_policy: error` maps to deny. The deprecated
`regression` policy remains readable for these saved direct-fallback requests.
Legacy `extra.fpm_profile` and `extra.fpm_interpolation` migrate to the full
canonical profile and nested interpolation method; newly exported configuration
uses only the canonical fields.

Migration merges saved estimator controls with explicitly supplied legacy
options before applying ordinary defaults. It preserves disjoint settings and
accepts agreeing overlaps; contradictory explicit values report the canonical
setting's path. The same rule applies to legacy and canonical interpolation
methods, including an explicit `auto`. An omitted field does not override a
saved value. `ForwardPassPerfOptions.to_dict()` serializes only arguments
explicitly supplied to that legacy options object, including explicit defaults.

Previously saved CLI timing with `forward_model` retains explicit selection
and deny. Newly authored requests without a selection use auto. `ForwardPassPerfOptions`
is retained as a legacy migration value type; new construction has one complete
config and no separate options argument. The raw PyO3 class also exposes
`normalize_config` and migration helpers for JSON-oriented consumers.

## Decode context parallelism

Op-level DCP declarations are model- and backend-specific: DeepSeek-V3-class
MLA supports vLLM and SGLang, GPU DeepSeek-V3.2/GLM-5 DSA supports vLLM only,
and Kimi-K3 supports SGLang only. The audited GLM ModelSlim adapter on
`ascend910c` additionally supports the pinned SGLang NPU DSA DCP implementation,
with the page layout and restrictions described above. Dense/Qwen-MoE GQA on vLLM also needs enough
attention-TP ranks per KV head. These declarations do not bypass precision,
attention-kernel, communication-style or topology checks. Unsupported requests
fail validation; they are not silently priced as DCP1.

These are op-level modeling capabilities. Exact measured whole-forward DCP
profiles have a separate [coverage contract](methods/whole-forward.md#dcp-self-benchmark-profiles);
profiling one backend does not establish op-level or serving support in another.
