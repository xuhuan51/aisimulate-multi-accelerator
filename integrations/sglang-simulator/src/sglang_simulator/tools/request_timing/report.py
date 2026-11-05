# SPDX-License-Identifier: Apache-2.0
# Modified for AISimulate multi-accelerator workload replay.
# Upstream attribution is retained in LICENSE and source-manifest.json.
"""Fit nonnegative service costs; keep unidentified waits out of the model."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from .trace import sha256


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0}

    def percentile(p):
        position = (len(values) - 1) * p
        low = int(position)
        return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (
            position - low
        )

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p99": percentile(0.99),
        "min": values[0],
        "max": values[-1],
    }


def fit_service(samples):
    """Least squares on a + b*x, constrained to a,b >= 0 (including axes)."""
    if not samples:
        return None
    if any(not math.isfinite(v) or v < 0 for sample in samples for v in sample):
        raise ValueError("Service samples must be finite and nonnegative")
    xs, ys = zip(*samples)
    x_mean, y_mean = statistics.mean(xs), statistics.mean(ys)
    variance = sum((x - x_mean) ** 2 for x in xs)
    candidates = [(y_mean, 0.0)]
    identifiable = len(set(xs)) >= 2
    if identifiable:
        slope = sum((x - x_mean) * (y - y_mean) for x, y in samples) / variance
        intercept = y_mean - slope * x_mean
        if slope >= 0 and intercept >= 0:
            candidates.append((intercept, slope))
        candidates.append(
            (0.0, sum(x * y for x, y in samples) / sum(x * x for x in xs))
        )
    a, b = min(
        candidates,
        key=lambda pair: sum((y - pair[0] - pair[1] * x) ** 2 for x, y in samples),
    )
    return {
        "fixed_ms": a,
        "per_unit_ms": b,
        "fit": "nonnegative_affine"
        if identifiable
        else "constant_only_no_size_variation",
        "samples": len(samples),
        "size_min": min(xs),
        "size_max": max(xs),
        "service_ms": distribution(ys),
        "mae_ms": statistics.mean(abs(y - a - b * x) for x, y in samples),
    }


def load_traces(directory):
    rows, sources, clocks = [], [], []
    for path in sorted(Path(directory).glob("*.jsonl")):
        items = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        if not items or items[0].get("kind") != "metadata":
            raise ValueError(f"Not a request timing trace: {path}")
        metadata = items[0]
        clocks.append(metadata["clock"])
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "metadata": metadata,
                "instrumented_sources": [
                    row for row in items if row["kind"] == "source"
                ],
                "ranks": [row for row in items if row["kind"] == "rank"],
            }
        )
        for row in items[1:]:
            if row["kind"] == "span":
                for key in ("start", "end", "cpu_ms", "wall_ms", "active_ms"):
                    if not math.isfinite(row[key]) or row[key] < 0:
                        raise ValueError(f"Invalid {key} in {path}")
                if row["end"] < row["start"]:
                    raise ValueError(f"Reversed span in {path}")
            rows.append({**row, "source": path.name})
    if not sources:
        raise ValueError("No server trace files; start the instrumented server first")
    return rows, sources, clocks


def summarize(
    rows, client, clocks, *, include_delivery_residuals=False, stream_interval=1
):
    requests = client["requests"]
    if not requests or any(not req.get("success") for req in requests):
        raise ValueError("Calibration requires a nonempty, fully successful client run")
    rids = {req["rid"] for req in requests}
    if len(rids) != len(requests):
        raise ValueError("Duplicate client request IDs")
    same_clock = bool(client["clock"].get("boot_id")) and all(
        c == client["clock"] for c in clocks
    )
    if include_delivery_residuals and (not same_clock or client["concurrency"] != 1):
        raise ValueError(
            "Delivery residual export requires matching host/boot clocks and concurrency 1"
        )
    stages, events = defaultdict(list), defaultdict(dict)
    mixed = 0
    for row in rows:
        if row["kind"] == "api_received" and row["rid"] in rids:
            events["api_received"][row["rid"]] = row["time"]
        elif row["kind"] == "scheduler_received":
            for rid in rids.intersection(row["rids"]):
                events["scheduler_received"][rid] = min(
                    row["time"], events["scheduler_received"].get(rid, math.inf)
                )
        elif row["kind"] == "span" and rids.intersection(row["rids"]):
            if not set(row["rids"]).issubset(rids):
                mixed += 1
                continue
            stages[row["name"]].append(row)

    for name in (
        "tokenize",
        "dispatch",
        "scheduler_pre",
        "scheduler_post",
        "detokenize",
        "api_output",
        "serialize",
    ):
        if not stages[name]:
            raise ValueError(
                f"Missing {name} samples; incomplete trace or unsupported SGLang API"
            )
    iterations = []
    for phase in ("scheduler_pre", "scheduler_post"):
        keys = [(row["source"], row["iteration"]) for row in stages[phase]]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate {phase} iterations")
        iterations.append(set(keys))
    if iterations[0] != iterations[1]:
        raise ValueError(
            "Incomplete scheduler iterations; wait for all postprocessing to finish before reporting"
        )
    tokenizations = {row["rids"][0]: row for row in stages["tokenize"]}
    if set(tokenizations) != rids or len(stages["tokenize"]) != len(rids):
        raise ValueError("Every client request must have exactly one tokenization span")
    if len({row["input_mode"] for row in tokenizations.values()}) != 1:
        raise ValueError("Collect text and input_ids workloads separately")
    if set(events["api_received"]) != rids or set(events["scheduler_received"]) != rids:
        raise ValueError("Missing API/scheduler request IDs; incomplete trace")

    fits = {
        "tokenize": fit_service(
            [(r["input_tokens"], r["active_ms"]) for r in stages["tokenize"]]
        )
    }
    for name in ("scheduler_pre", "scheduler_post"):
        fits[name] = fit_service([(len(r["rids"]), r["cpu_ms"]) for r in stages[name]])
        fits[name]["measurement"] = "thread_cpu_ms"
    fits["tokenize"]["measurement"] = "active_wall_ms"
    output_fits = {}
    for name in ("detokenize", "api_output", "serialize"):
        previous, samples = {}, []
        for row in sorted(stages[name], key=lambda r: r["start"]):
            emitted = 0
            for rid, count in zip(row["rids"], row["counts"], strict=True):
                if count < previous.get(rid, 0):
                    raise ValueError(f"Non-monotonic token count for {name}: {rid}")
                emitted += count - previous.get(rid, 0)
                previous[rid] = count
            size = len(row["rids"])
            # Tokenizer libraries can use native worker threads; the active
            # wall duration includes their work while the caller is blocked.
            duration = row["active_ms"] if name == "detokenize" else row["cpu_ms"]
            samples.append((emitted / size, duration / size))
        if set(previous) != rids or any(
            previous[req["rid"]] != req["output_tokens"] for req in requests
        ):
            raise ValueError(f"Incomplete final output counts for {name}")
        output_fits[name] = fit_service(samples)
        output_fits[name]["measurement"] = (
            "active_wall_ms" if name == "detokenize" else "thread_cpu_ms"
        )

    # The API can coalesce several detokenizer messages into one HTTP chunk.
    # Amortize fixed costs over native emissions rather than charging a full
    # serialization cost for every token when fewer HTTP chunks were sent.
    native_emissions = sum(len(row["rids"]) for row in stages["detokenize"])
    for name, fit in output_fits.items():
        fit["calls_per_native_emission"] = (
            sum(len(row["rids"]) for row in stages[name]) / native_emissions
        )

    config = {
        "tokenize_ms": fits["tokenize"]["fixed_ms"],
        "tokenize_per_token_ms": fits["tokenize"]["per_unit_ms"],
        "tokenizer_workers": 1,
        "dispatch_ms": statistics.mean(r["cpu_ms"] for r in stages["dispatch"]),
        "output_ms": sum(
            f["fixed_ms"] * f["calls_per_native_emission"] for f in output_fits.values()
        ),
        "output_per_token_ms": sum(f["per_unit_ms"] for f in output_fits.values()),
        "output_workers": 1,
        "stream_interval": stream_interval,
    }
    for name in ("scheduler_pre", "scheduler_post"):
        config[f"{name}_ms"] = fits[name]["fixed_ms"]
        config[f"{name}_per_request_ms"] = fits[name]["per_unit_ms"]

    waits = defaultdict(list)
    serialize = {}
    for row in stages["serialize"]:
        # A final marker can repeat the last token count. Match its FIRST
        # serialization to the token arrival, not the later completion marker.
        key = (row["rids"][0], row["counts"][0])
        serialize[key] = min(row["end"], serialize.get(key, math.inf))
    sends = {row["rids"][0]: row for row in stages["dispatch"]}
    by_rid = {req["rid"]: req for req in requests}
    unmatched_chunks = 0
    for req in requests:
        rid = req["rid"]
        waits["completion_ms"].append(
            1000 * (req["finished"] - req["chunks"][-1]["time"])
        )
        if req.get("previous_rid"):
            waits["client_think_ms"].append(
                1000 * (req["start"] - by_rid[req["previous_rid"]]["finished"])
            )
        if same_clock:
            waits["input_delivery_residual_ms"].append(
                1000 * (events["api_received"][rid] - req["start"])
            )
            waits["api_to_tokenize_wait_ms"].append(
                1000 * (tokenizations[rid]["start"] - events["api_received"][rid])
            )
            # IPC delivery and waiting for the next scheduler receive boundary
            # cannot be separated. Never insert this into dispatch_ms.
            waits["dispatch_to_scheduler_residual_ms"].append(
                1000 * (events["scheduler_received"][rid] - sends[rid]["end"])
            )
            for chunk in req["chunks"]:
                sent = serialize.get((rid, chunk["count"]))
                if sent is None:
                    unmatched_chunks += 1
                    continue
                waits["output_delivery_residual_ms"].append(
                    1000 * (chunk["time"] - sent)
                )

    # Some receiving processes can observe IPC before the sender returns from
    # send(). Preserve that diagnostic sign; do not clamp it into a fake sample.
    for name in (
        "completion_ms",
        "client_think_ms",
        "input_delivery_residual_ms",
        "output_delivery_residual_ms",
    ):
        if any(value < -0.001 for value in waits[name]):
            raise ValueError(
                f"Negative {name}; clock or request/chunk alignment mismatch"
            )
    config["completion_ms"] = statistics.mean(waits["completion_ms"])
    config["client_think_ms"] = (
        statistics.mean(waits["client_think_ms"])
        if waits["client_think_ms"]
        else client["think_ms"]
    )
    omitted = {
        "input_transport_ms": "Includes API/event-loop waiting; not pure transport",
        "output_transport_ms": "Includes HTTP/client-event-loop waiting; not pure transport",
    }
    if include_delivery_residuals:
        if unmatched_chunks or not waits["output_delivery_residual_ms"]:
            raise ValueError(
                "Cannot export delivery residuals with unmatched output chunks"
            )
        config["input_transport_ms"] = statistics.mean(
            waits["input_delivery_residual_ms"]
        )
        config["output_transport_ms"] = statistics.mean(
            waits["output_delivery_residual_ms"]
        )
        omitted = {}
    from sglang_simulator.simulation.parallel.timing import RequestTimingConfig

    RequestTimingConfig.parse(config)
    per_process = {}
    for name in ("scheduler_pre", "scheduler_post"):
        grouped = defaultdict(list)
        for row in stages[name]:
            grouped[row["source"]].append(row["cpu_ms"])
        per_process[name] = {
            key: distribution(values) for key, values in grouped.items()
        }
    return {
        "request_timing": config,
        "omitted_fields": omitted,
        "request_count": len(requests),
        "same_monotonic_clock": same_clock,
        "delivery_residuals_exported": include_delivery_residuals,
        "fits": fits,
        "output_component_fits": output_fits,
        "stage_wall_ms": {
            name: distribution(r["wall_ms"] for r in items)
            for name, items in stages.items()
        },
        "scheduler_cpu_ms_by_process": per_process,
        "diagnostics_ms": {
            name: distribution(values) for name, values in waits.items()
        },
        "excluded_mixed_batches": mixed,
        "unmatched_client_chunks": unmatched_chunks,
        "limitations": [
            "Scheduler/dispatch/API costs use thread CPU time. Tokenize/detokenize use active wall time to include native tokenizer worker threads. Coroutine suspension is excluded. GPU spin waits may still consume scheduler CPU.",
            "Native run_batch host preparation and kernel launch overhead is not isolated here; forward remains the predictor's responsibility.",
            "Output service adds amortized detokenizer, API handling and serialization costs; real processes can pipeline, while the simulator uses one output pool.",
            "Transport residuals include unmodeled event-loop waits. Optional C1 export is a workload-specific approximation, not isolated network latency.",
            "Affine fits apply to observed lengths, batch sizes and deployment only; fixed-size samples cannot identify a per-token slope.",
            "Instrumentation adds timestamp, JSON and file IO overhead. No GPU synchronization is added; validate with a separate uninstrumented run.",
        ],
    }


def write_report(args):
    client = json.loads(args.client.read_text())
    rows, sources, clocks = load_traces(args.traces)
    manifest = json.loads((args.traces / "server.json").read_text())
    report = summarize(
        rows,
        client,
        clocks,
        include_delivery_residuals=args.include_delivery_residuals,
        stream_interval=manifest["stream_interval"],
    )
    report.update(
        server=manifest,
        sources=sources,
        client_source={
            "path": str(args.client.resolve()),
            "sha256": sha256(args.client),
        },
    )
    outputs = {"report.json": report, "request_timing.json": report["request_timing"]}
    if args.sim_config:
        config = json.loads(args.sim_config.read_text())
        # Replace the timing object: stale manually supplied transport costs
        # must not silently survive a calibration that explicitly omitted them.
        config["request_timing"] = report["request_timing"]
        outputs["sim_config.calibrated.json"] = config
        report["sim_config_source"] = {
            "path": str(args.sim_config.resolve()),
            "sha256": sha256(args.sim_config),
        }
        # Relative predictor paths resolve against the config's directory.
        if args.output_dir.resolve() != args.sim_config.resolve().parent:
            raise ValueError(
                "With --sim-config, use its directory as --output-dir to preserve relative predictor paths"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in outputs:
        if (args.output_dir / name).exists():
            raise FileExistsError(args.output_dir / name)
    for name, obj in outputs.items():
        (args.output_dir / name).write_text(
            json.dumps(obj, indent=2, allow_nan=False) + "\n"
        )
    print(
        json.dumps(
            {
                "request_timing": report["request_timing"],
                "omitted_fields": report["omitted_fields"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
