"""Qualify real eager decoder-layer invocation observations; never fit serving residuals."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

IMAGE = "quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443"
MODEL_SOURCE = "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py"
MODEL_SHA256 = "1edf877e54ca6acf37e2d7dccb1704054175e2b78b8737bb9b617ffc97a2134e"
LAYERS = (0, 1, 2, 3, 4, 5, 6, 77)
CONTRACT = "glm52_eager_decoder_layer/v1"
SOURCE = "sglang.srt.models.deepseek_v2.DeepseekV2DecoderLayer.forward"


def qualified_boundary(records, layer, control):
    if len(records) != 16:
        raise ValueError("All sixteen rank observations are required")
    boundaries = []
    for rank, record in enumerate(records):
        if (record["rank"] != rank or record["status"] != "completed"
                or record["control"] != control or record["source_sha256"].get(MODEL_SOURCE) != MODEL_SHA256):
            raise ValueError("Layer replay rank, source or control identity mismatch")
        boundary = next(value for value in record["boundaries"] if value["layer"] == layer)
        if boundary["module"] != "DeepseekV2DecoderLayer" or boundary["function_profile"] is not None:
            raise ValueError("Expected unprofiled complete decoder-layer invocation")
        padded = (control["global_requests"] // 4 + 3) // 4 * 4
        if (boundary["forward_mode"] != "2" or boundary["batch_size"] != padded
                or boundary["global_num_tokens_cpu"] != [padded] * 4):
            raise ValueError("Captured execution mode or DP padding differs from the profile key")
        positive = [value for value in boundary["sequence_lengths"] if value > 0]
        if len(positive) != control["global_requests"] // 4:
            raise ValueError("Asymmetric attention-DP batches require separate profiles")
        if not all(control["input_length"] <= value < control["input_length"] + 192 for value in positive):
            raise ValueError("Captured layer is not a steady decode sample")
        samples = boundary["samples"]
        if len(samples) != 12 or [sample["warmup"] for sample in samples] != [True] * 3 + [False] * 9:
            raise ValueError("Expected three warmups and nine verified samples")
        if [sample["repeat"] for sample in samples] != list(range(12)):
            raise ValueError("Missing or duplicated replay sample")
        boundaries.append(boundary)
    maxima = [max(boundary["samples"][index]["device_ms"] for boundary in boundaries) for index in range(3, 12)]
    if not all(math.isfinite(value) and value > 0 for value in maxima):
        raise ValueError("Nonpositive or nonfinite layer timing")
    cv = statistics.pstdev(maxima) / statistics.mean(maxima)
    return {
        "layer": layer, "attention_tp": 4, "attention_dp": 4, "moe_ep": 16,
        "requests_per_dp": control["global_requests"] // 4,
        "kv_tokens": max(max(boundary["sequence_lengths"]) for boundary in boundaries),
        "latency_ms": statistics.median(maxima), "cv": cv, "sample_count": 9,
        "numerical_checks": 16 * 12, "boundary_contract": CONTRACT, "kernel_source": SOURCE,
    }, maxima


def export(campaign, manifests, model_config, output):
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    import yaml

    if output.exists():
        raise FileExistsError(output)
    launch = json.loads((campaign / "launch.json").read_text())
    command = launch["command"]
    if IMAGE not in command:
        raise ValueError("Unqualified runtime image")
    environment = dict(command[index + 1].split("=", 1) for index, value in enumerate(command) if value == "-e")
    if (environment.get("AISIM_CAPTURE_CPROFILE") != "0" or environment.get("TASK_QUEUE_ENABLE") != "0"
            or environment.get("SGLANG_NPU_USE_MULTI_STREAM") != "1"):
        raise ValueError("Runtime mode differs from qualified eager invocation")
    info = json.loads((campaign / "server-info.json").read_text())
    if (info["tp_size"], info["attn_dp_size"], info["ep_size"]) != (16, 4, 16):
        raise ValueError("Unqualified runtime topology")
    if not info["disable_overlap_schedule"] or info["speculative_algorithm"] is not None:
        raise ValueError("Layer observations require no overlap scheduler or speculation")
    if any(info["cuda_graph_config"][phase]["backend"] != "disabled" for phase in ("prefill", "decode")):
        raise ValueError("Graph observations cannot populate eager layer curves")
    identity = {
        "image": IMAGE, "framework_commit": "baa9befa381511cd4d56f83980951e5a45fffc6d",
        "model_config_sha256": hashlib.sha256(model_config.read_bytes()).hexdigest(),
        "quantization_sha256": hashlib.sha256((model_config.parent / "quant_model_description.json").read_bytes()).hexdigest(),
        "environment": environment, "tp": 16, "attention_dp": 4, "moe_ep": 16,
        "boundary_contract": CONTRACT, "timing_method": "median_of_per_sample_rank_max_outer_events",
        "model_source_sha256": MODEL_SHA256,
        "launcher_sha256": hashlib.sha256((campaign / "launch_layer_replay_probe.py").read_bytes()).hexdigest(),
    }
    first_manifest = json.loads(manifests[0].read_text())
    first_case = first_manifest["cases"][0]["id"]
    identity["cpu_affinity_by_rank"] = [
        json.loads((campaign / first_case / f"layer-replay-rank{rank}.json").read_text())["cpu_affinity"]
        for rank in range(16)
    ]
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    accepted, rejected, held_out, hashes = {}, [], [], {}
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "completed":
            raise ValueError("Only completed campaigns may be exported")
        for case in manifest["cases"]:
            if not case["training"]:
                held_out.append(case["id"])
                continue
            control = {name: case[name] for name in ("id", "forward_call", "training", "input_length", "global_requests")}
            records = []
            for rank in range(16):
                path = campaign / case["id"] / f"layer-replay-rank{rank}.json"
                hashes[str(path.relative_to(campaign))] = hashlib.sha256(path.read_bytes()).hexdigest()
                records.append(json.loads(path.read_text()))
                if sorted(boundary["layer"] for boundary in records[-1]["boundaries"]) != list(LAYERS):
                    raise ValueError("Incomplete or duplicate structural layer classes")
                if records[-1]["cpu_affinity"] != identity["cpu_affinity_by_rank"][rank]:
                    raise ValueError("Runtime-layer rows mix worker CPU placement")
            for layer in LAYERS:
                row, samples = qualified_boundary(records, layer, control)
                key = (layer, row["requests_per_dp"], case["input_length"])
                if row["cv"] > 0.1:
                    rejected.append({"case": case["id"], "layer": layer, "reason": "cv_exceeds_0.1", "cv": row["cv"]})
                    continue
                if key not in accepted:
                    row["runtime_identity_sha256"] = identity_hash
                    row["quantization_sha256"] = identity["quantization_sha256"]
                    accepted[key] = {"row": row, "case": case["id"], "samples_ms": samples}
    required = {(layer, 1, length) for layer in LAYERS for length in (512, 8192, 16384)}
    missing = sorted(required - accepted.keys())
    if missing:
        raise ValueError(json.dumps({"missing_qualified_cells": missing, "rejected": rejected}))
    rows = [value["row"] for _, value in sorted(accepted.items())]
    output.mkdir(parents=True)
    parquet.write_table(arrow.Table.from_pylist(rows), output / "ascend_layer_perf.parquet")
    (output / "collection_meta.yaml").write_text(yaml.safe_dump({
        "framework": "sglang", "version": "0.5.22", "device": "ascend910c",
        "scope": "Actual isolated eager decoder-layer invocations, including host/runtime boundaries; no fitted residual",
        "runtime_identity_sha256": identity_hash, "boundary_contract": CONTRACT,
        "selection": "first qualifying observation per structural type and planned request/KV cell",
        "rows": len(rows), "extrapolation": "forbidden", "serving_accuracy_acceptance": False,
    }, sort_keys=False))
    (output / "provenance.json").write_text(json.dumps({
        "runtime_identity": identity, "source_sha256": hashes,
        "accepted": list(accepted.values()), "rejected": rejected, "holdouts_excluded": held_out,
        "collector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    return {"rows": len(rows), "rejected_observations": len(rejected), "holdouts_excluded": held_out}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.campaign, args.manifest, args.model_config, args.output)))


if __name__ == "__main__":
    main()
