# Paired component diagnosis and independent context holdouts

Collected on npu1-20 on 2026-11-03/04 with the pinned SGLang image in
`launch.json`, global TP16 / ADP4 / EP16, aggregate serving, graph/overlap/speculation
disabled. `holdout-server-info.json` records the resolved runtime. No model or
operator performance table was updated from these observations.

`component-campaign-verified.json` contains three completed diagnostic cases.
Each retains all sixteen ranks, eight structural layers and twelve invocations
per control-before, host, events and control-after population. The first three
invocations are warmup. Numeric checks run outside timing. Nested spans overlap;
event instrumentation perturbs invocation time and is not performance-table data.
`component-summary.json` preserves source SHA256 values for decompressed records.
The original diagnostic control labelled every `verified-` case `training=false`
because of the prefix; no diagnostic data were used for fitting in any case.
The source of that exact run is `run_component_campaign-verified.py` and matches
the SHA256 in its manifest. The reusable controller's label is corrected.

The initial `component-campaign.json` failed in the controller because
`str(ForwardMode.DECODE)` in this pinned Python/runtime is `"2"`, not `"DECODE"`.
Its records and failure log remain intact; the verified campaign repeated all
three cases in new directories with the correct enum contract. This was not a
model execution failure or a hidden timing fallback.

`context-holdouts.json` contains twenty formal completed requests in three
independent validation workloads (8192/128 C4 N8, 12288/4096 C4 N4, 16384/128
C4 N8), each with two warmups and exact input/output token checks. Replay control
was removed before these cases, so component/cProfile/event instrumentation was
inactive; existing forward telemetry remains enabled. The final timed decode
iteration 5393 has records for three DP ranks, as one deferred telemetry record
had not flushed at shutdown. Whole-forward comparisons require all four ranks
and exclude that incomplete iteration; client results include all requests.

Large component records and the Docker log use lossless gzip. Their uncompressed
bytes are the original observations, including whitespace. To rerun the component
analysis, its reader accepts either `.json` or `.json.gz`; use a fresh output file
or preserve the existing summary first. Holdout/paired telemetry JSONL remains
uncompressed for the comparison tools. `container-final.json` confirms shutdown
of this diagnostic service after collection.

See the adjacent `context-transfer-validation-20261103/` for predictions, test
results, source hashes, and the external-arrival-only diagnostic. This is local
aggregate evidence, not acceptance of arbitrary PD ratios or topologies. The
Chinese report is `md/910c-glm52-context-transfer-validation-20261104.md` at the
repository root. SGLang API attribution, pinned revision and Apache-2.0 notice
are in the root and packaged `THIRD_PARTY_NOTICES.md`.
