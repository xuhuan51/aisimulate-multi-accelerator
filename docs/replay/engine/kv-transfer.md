<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# P/D KV transfer

In `mode: disaggregated`, each request is prefilled by a `prefill` worker and
then decoded by a `decode` worker. `engine.kv_transfer` sets how long moving the
prompt's KV cache between them takes. Add it to a disaggregated configuration
such as the [P/D example](README.md#a-complete-pd-example):

```yaml
engine:
  mode: disaggregated
  kv_transfer:
    bytes_per_token: auto
    bandwidth_gb_per_second: 50
    timing_mode: destination_missing
```

Omitting `kv_transfer` is the same as `{}`: the transfer happens with no delay.
A native ReplaySpec or Sweeper search space whose prefill rank sets a positive
`kv_transfer_bandwidth` without `kv_transfer_bytes_per_token` is rejected; the
YAML above always resolves the bytes.

## Transfer fields

| Knob | Default | Recommend | Rules |
|---|---|---|---|
| `engine.kv_transfer.type` | `bandwidth` | fixed | `bandwidth` uses the scalar bandwidth below; `measured` obtains a bounded curve from the canonical prefill/decode performance models. The native engine stack currently supports measured Ascend GLM ModelSlim transfers. |
| `engine.kv_transfer.bytes_per_token` | `auto` | fixed | Positive integer or `auto`. Bytes moved per prompt token. `auto` derives it from the model and the prefill role's TP/PP/MoE shape. It can differ from either worker's `kv_cache.bytes_per_token`. With `state_cache`, `auto` uses the prefill role's resolved `kv_cache.bytes_per_token`. |
| `engine.kv_transfer.bandwidth_gb_per_second` | `null` | fixed | Positive, in decimal GB/s. `null` means no transfer delay, not a zero-bandwidth link. |
| `engine.kv_transfer.timing_mode` | `destination_missing` | fixed | `destination_missing` charges only the prompt KV the chosen decode worker does not already hold in its cache. `full_prompt` charges the whole prompt. |

All fields accept only concrete values; `recommend` does not search them.
`kv_transfer` is rejected in `aggregated` and `afd` modes.

With `type: measured`, keep `bytes_per_token: auto` and omit bandwidth. Both
language roles must use default canonical op-level timing on the same Ascend
hardware, SGLang version and local GLM ModelSlim model. The source and
destination attention TP select the actual sender/receiver mapping. The
qualified role layout uses attention TP1/2/4/8/16, either EP16/32 with MoE-TP1
or EP1 with MoE-TP16/32, PP1/CP1/DCP1 and 128-token BF16 KV pages.
Each selected attention-DP group must remain on one
node, with source and destination groups on different nodes. EP32 adds other
attention-DP groups; it does not change this request's replicated KV layout or
sender mapping. The same isolated-group curve therefore applies, while shared
link contention between groups remains uncalibrated.
The shared `estimator_config.op_level.ascend.kv_transfer_page_layout` selects
`contiguous` (default) or `fragmented`; both role configurations must agree.
The source model prepares the measured curve once and Replay shares it across
handoffs. For SGLang, Replay queries each completed prefill chunk separately;
each query preserves the curve's launch cost and token bounds. Destination
reservation and source KV retention keep their existing ordering.
Missing-token counts round up to 128-token pages, zero transfer is
free. With the default Ascend `shape_policy: measured`, an out-of-bounds query
fails without a bandwidth fallback. With `shape_policy: anchored` on both
roles, page-work ratios extend the same measured mapping, preserve a measured
latency floor and record the policy and original token bounds in metadata.
These out-of-domain values are estimates, not additional measured rows.
The runtime records the selected mapping and measurement identity in transfer
provenance. Bootstrap, per-layer streaming and contention between requests
remain outside this calibration.

Each curve retains one collection identity and timing protocol. Publications
may select different complete curves from compatible campaigns to preserve
coverage. The selection uses measured ranges and point counts, never the
smallest latency; samples from different protocols are not spliced into a curve.

For `type: bandwidth`, transfer time in seconds is

```text
(transferred_tokens × bytes_per_token + state_bytes) / (bandwidth_gb_per_second × 10^9)
```

where `transferred_tokens` follows `timing_mode`. With
`destination_missing`, a decode worker that already caches a shared prefix
receives the request sooner, so prefix reuse on the decode side can lower TTFT
even when compute time does not change. `state_bytes` is the prefill role's
`state_cache.bytes_per_request` for
[hybrid-KV models](kv-cache.md#state-cache), and 0 otherwise.

<a id="handoff"></a>

## Request flow

1. The decode worker reserves destination KV capacity according to the
   backend's handoff order below.
2. The prefill worker computes the prompt and produces the first token.
   SGLang can start sending completed chunks before the entire prompt finishes.
3. After all prompt KV has arrived, the decode worker admits the request and
   generates the remaining tokens. Source KV remains reserved until transfer
   success is processed by the prefill scheduler.

The reported TTFT includes prefill queueing, prefill compute and the transfer.
Transfer events share one virtual clock with compute and arrivals.

`engine.pd_response` selects the client response boundary. `direct` (the
default) reports tokens as the decode worker emits them. `sglang_gateway`
models the HTTP PD gateway waiting for the prefill response before exposing
the decode stream to the client. In this mode, source KV release must also
finish at a prefill scheduler boundary. A busy prefill worker can delay the
first visible token even while decode continues generating tokens. Buffered
tokens and completion become visible together when the prefill response is
ready; compute and KV lifetimes retain their normal scheduling.

This fixed policy is preserved by recommendation, saved candidate YAML and
prediction. `sglang_gateway` requires disaggregated SGLang workers in the
offline engine stack without adapters. It covers the response ordering of
the HTTP PD gateway, not additional HTTP, tokenizer or detokenizer latency.
Choose `direct` when measuring decode-worker output without that gateway.

Backends order the handoff differently:

| Backend | Order |
| --- | --- |
| vLLM, TensorRT-LLM | Source-first: the prefill side finishes and then the decode side reserves space. |
| SGLang | Destination-first: the decode side reserves space before the transfer. |

For SGLang with multiple attention-DP ranks on prefill, Replay first routes the
request to its prefill scheduler and registers the source rank at that
scheduler's next available boundary. Destination reservation waits for this
registration; prefill computation still waits for destination capacity. This
models the source-rank lookup used by round-robin DP routing without allocating
prefill KV during registration. Configured tokenization occurs once, before
registration. A canceled registration releases its route without allocating
destination KV. Single-rank prefill requires no rank lookup.

SGLang publishes each completed prefill chunk at its forward completion
boundary, even with overlap scheduling disabled. Non-final sends end at a full
KV page; the last send includes the remaining partial page. Transfers for one
request run in FIFO order and overlap later prefill computation. A slow
transfer preserves queued chunk boundaries instead of combining their timing
queries. With `destination_missing`, chunks intersect the uncached destination
suffix. State bytes in the bandwidth model are charged only on the final send.
If only a fixed native handoff delay is supplied, it remains a single final
delay; that configuration cannot predict the overlap from individual copies.

The chunk model covers the standard non-staging transport path. DCP packing,
staging grids, optional early cached-prefix sends, metadata/bootstrap latency
and competition between different requests require separate transport models.
The measured GLM Ascend transfer contract uses DCP1 and includes the NPU MLA
K/V/index buffers together in each page transfer.

After a SGLang transfer completes, source KV release and destination activation
proceed independently at each worker's next scheduler boundary. A busy decode
worker can delay activation without keeping KV pinned on an idle prefill worker.
The handoff completes after both actions finish. For vLLM and TensorRT-LLM,
source release follows destination activation.

SGLang returns the token already sampled by prefill when the destination admits
the received request into its prebuilt batch. This output is visible before the
next decode forward, including when another attention-DP rank runs a longer
batch. A one-token output completes at that admission boundary; an N-token
output needs N-1 ordinary decode steps. Prebuilt output does not increment decode
acceptance counters or materialize an additional KV token. Its first decode
forward reads the prompt plus the sampled token.

TensorRT-LLM decode workers use `GUARANTEED_NO_EVICT` scheduling, which reserves
room for the whole output while transferred KV is held.

The two roles may use different `attention_data` sizes. A request moves from one
concrete prefill rank to one concrete decode rank. A trace request's
`prefill_dp_rank` chooses the prefill rank; if omitted, its `dp_rank` is used
for both roles.

## Limitations

- The transfer is a single request-level event. Per-layer streaming, KV
  layout conversion between different TP shapes, and network contention
  between concurrent transfers are not modeled.
- Bandwidth applies to each transfer independently.
- vLLM P/D can be combined with G2 host offload; see
  [KV cache](kv-cache.md#host-offload-g2). G3 offload is aggregated-only.
