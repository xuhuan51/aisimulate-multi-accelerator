"""Qualify captured eager layer populations, including idle attention-DP ranks."""

import argparse
import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path

from collector.sglang.ascend_layer_export import IMAGE, LAYERS, MODEL_SHA256, MODEL_SOURCE, SOURCE

CONTRACT = "glm52_eager_decoder_layer_valid_rows/v2"
NUMERICAL_CONTRACTS = {
    "all_hidden_values_and_all_topk_columns_for_positive_kv_rows/v2",
    "all_logical_hidden_values_and_all_topk_columns_for_positive_kv_rows/v3",
}


def qualified_population(records, layer, control):
    if len(records) != 16:
        raise ValueError("All sixteen rank observations are required")
    boundaries = []
    counts = []
    for rank, record in enumerate(records):
        if (
            record["rank"] != rank
            or record["status"] != "completed"
            or record["control"] != control
            or record["source_sha256"].get(MODEL_SOURCE) != MODEL_SHA256
            or record.get("numerical_contract") not in NUMERICAL_CONTRACTS
        ):
            raise ValueError("Unqualified population source or numerical contract")
        boundary = next(value for value in record["boundaries"] if value["layer"] == layer)
        if boundary["module"] != "DeepseekV2DecoderLayer" or boundary["function_profile"] is not None:
            raise ValueError("Expected an unprofiled complete decoder-layer invocation")
        lengths = boundary["sequence_lengths"]
        if any(length < 0 for length in lengths):
            raise ValueError("Negative sequence length")
        positive = [length for length in lengths if length > 0]
        if not all(control["input_length"] <= length < control["input_length"] + 192 for length in positive):
            raise ValueError("Captured population differs from the requested workload")
        counts.append(len(positive))
        boundaries.append(boundary)
    if any(counts[rank] != counts[rank // 4 * 4] for rank in range(16)):
        raise ValueError("Tensor-parallel ranks disagree on request population")
    population = counts[::4]
    positive_counts = {count for count in population if count > 0}
    if len(positive_counts) != 1 or sum(population) != control["global_requests"]:
        raise ValueError("Requires uniform active-rank requests matching the requested concurrency")
    requests = positive_counts.pop()
    active = sum(count > 0 for count in population)
    aligned = [(count + 3) // 4 * 4 for count in population]
    if sum(aligned) * 2 >= max(aligned) * 4:
        aligned = [max(aligned)] * 4
    for rank, boundary in enumerate(boundaries):
        if (
            boundary["forward_mode"] != ("2" if counts[rank] else "4")
            or boundary["batch_size"] != aligned[rank // 4]
            or len(boundary["sequence_lengths"]) != aligned[rank // 4]
            or boundary["global_num_tokens_cpu"] != aligned
        ):
            raise ValueError("Captured idle/decode mode or DP padding differs from the source contract")
        samples = boundary["samples"]
        if (
            len(samples) != 12
            or [sample["warmup"] for sample in samples] != [True] * 3 + [False] * 9
            or [sample["repeat"] for sample in samples] != list(range(12))
        ):
            raise ValueError("Expected three warmups and nine numerically verified samples")
        for sample in samples:
            checks = sample["topk_validation"]
            if records[rank]["numerical_contract"].endswith("/v3"):
                if not isinstance(checks["hidden"], list):
                    raise ValueError("Missing hidden validity observations")
                checks = checks["topk"]
            expected_rows = counts[rank] if layer in (2, 3, 4, 6) else 0
            if checks["valid_rows"] != expected_rows:
                raise ValueError("Valid top-k rows were not numerically checked")
    maxima = [max(boundary["samples"][index]["device_ms"] for boundary in boundaries) for index in range(3, 12)]
    if not all(math.isfinite(value) and value > 0 for value in maxima):
        raise ValueError("Nonpositive or nonfinite layer timing")
    return {
        "layer": layer,
        "attention_tp": 4,
        "attention_dp": 4,
        "active_attention_dp": active,
        "moe_ep": 16,
        "requests_per_dp": requests,
        "kv_tokens": max(length for boundary in boundaries for length in boundary["sequence_lengths"]),
        "latency_ms": statistics.median(maxima),
        "cv": statistics.pstdev(maxima) / statistics.mean(maxima),
        "sample_count": 9,
        "numerical_checks": 16 * 12,
        "boundary_contract": CONTRACT,
        "kernel_source": SOURCE,
    }, maxima


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_population(campaign_set, model_config, output, collector_ref):
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    import yaml

    if output.exists():
        raise FileExistsError(output)
    accepted, rejected, required, holdouts, sources, collections = {}, [], set(), [], {}, []
    runtime_identity = None
    for item in json.loads(campaign_set.read_text()):
        campaign = Path(item["campaign"])
        command = json.loads((campaign / "launch.json").read_text())["command"]
        environment = dict(command[index + 1].split("=", 1) for index, value in enumerate(command) if value == "-e")
        info = json.loads((campaign / "server-info.json").read_text())
        if (
            IMAGE not in command
            or (info["tp_size"], info["attn_dp_size"], info["ep_size"]) != (16, 4, 16)
            or not info["disable_overlap_schedule"]
            or info["speculative_algorithm"] is not None
            or any(info["cuda_graph_config"][phase]["backend"] != "disabled" for phase in ("prefill", "decode"))
            or environment.get("TASK_QUEUE_ENABLE") != "0"
            or environment.get("SGLANG_NPU_USE_MULTI_STREAM") != "1"
            or environment.get("AISIM_CAPTURE_CPROFILE") != "0"
        ):
            raise ValueError("Unqualified runtime configuration")
        host = json.loads((campaign / "host-identity.json").read_text())
        model_identity = json.loads((campaign / "model-identity.json").read_text())
        if model_identity != {
            "config.json": digest(model_config),
            "quant_model_description.json": digest(model_config.parent / "quant_model_description.json"),
        }:
            raise ValueError("The observed campaign model metadata differs from the publication model")
        for filename in (
            "launch.json",
            "server-info.json",
            "host-identity.json",
            "model-identity.json",
            "launch_layer_replay_probe.py",
            "layer_output_validation.py",
            "run_layer_chain_campaign.py",
        ):
            sources[str(campaign / filename)] = digest(campaign / filename)
        for script in sorted(campaign.glob("*.py")):
            sources[str(script)] = digest(script)
        identity = {
            "image": IMAGE,
            "framework_commit": "baa9befa381511cd4d56f83980951e5a45fffc6d",
            "model_config_sha256": digest(model_config),
            "quantization_sha256": digest(model_config.parent / "quant_model_description.json"),
            "model_source_sha256": MODEL_SHA256,
            "boundary_contract": CONTRACT,
            "environment": {
                key: value
                for key, value in environment.items()
                if not key.startswith("AISIM_") and key != "HCCL_IF_BASE_PORT"
            },
            "host": host,
        }
        for manifest_name in item["manifests"]:
            manifest_path = campaign / manifest_name
            sources[str(manifest_path)] = digest(manifest_path)
            manifest = json.loads(manifest_path.read_text())
            if manifest["status"] != "completed":
                raise ValueError("Only completed campaigns may be published")
            for case in manifest["cases"]:
                if not case["training"]:
                    holdouts.append({"campaign": str(campaign), "case": case["id"]})
                    continue
                control = {
                    name: case[name] for name in ("id", "forward_call", "training", "input_length", "global_requests")
                }
                records = []
                for rank in range(16):
                    path = campaign / case["id"] / f"layer-replay-rank{rank}.json"
                    sources[str(path)] = digest(path)
                    records.append(json.loads(path.read_text()))
                    if records[-1]["source_sha256"].get("/campaign/layer_output_validation.py") != digest(
                        campaign / "layer_output_validation.py"
                    ):
                        raise ValueError("Numerical validator differs from the frozen source")
                    if sorted(boundary["layer"] for boundary in records[-1]["boundaries"]) != list(LAYERS):
                        raise ValueError("Incomplete structural layer classes")
                current_identity = {**identity, "cpu_affinity_by_rank": [record["cpu_affinity"] for record in records]}
                if runtime_identity is None:
                    runtime_identity = current_identity
                if current_identity != runtime_identity:
                    raise ValueError("Campaigns have different runtime/host/model identities")
                identity_hash = hashlib.sha256(json.dumps(current_identity, sort_keys=True).encode()).hexdigest()
                for layer in LAYERS:
                    row, samples = qualified_population(records, layer, control)
                    key = (layer, row["active_attention_dp"], row["requests_per_dp"], case["input_length"])
                    required.add(key)
                    if row["cv"] > 0.1:
                        rejected.append(
                            {
                                "campaign": str(campaign),
                                "case": case["id"],
                                "layer": layer,
                                "reason": "cv_exceeds_0.1",
                                "cv": row["cv"],
                            }
                        )
                        continue
                    row.update(
                        runtime_identity_sha256=identity_hash,
                        quantization_sha256=current_identity["quantization_sha256"],
                    )
                    if key not in accepted:
                        accepted[key] = {
                            "row": row,
                            "campaign": str(campaign),
                            "case": case["id"],
                            "samples_ms": samples,
                        }
                collections.append(case["end_unix"])
    missing = sorted(required - accepted.keys())
    if missing or not accepted:
        raise ValueError(json.dumps({"missing_qualified_cells": missing, "rejected": rejected}))
    rows = [value["row"] for _, value in sorted(accepted.items())]
    output.mkdir(parents=True)
    parquet.write_table(arrow.Table.from_pylist(rows), output / "ascend_layer_perf.parquet")
    runtime = {
        "framework": "sglang",
        "version": "0.5.22",
        "image_digest": IMAGE,
        "source_commit": runtime_identity["framework_commit"],
    }
    collection = {
        "collector_ref": collector_ref + "+worktree-source-sha256-in-provenance",
        "collector_hash": "sha256:" + digest(Path(__file__)),
        "case_plan_hash": "sha256:" + digest(campaign_set),
        "collected_at": datetime.fromtimestamp(max(collections), UTC).isoformat(),
        "rows": len(rows),
        "classified_failures": len(rejected),
        "status": "complete",
        "runtime": runtime,
    }
    metadata = {
        "schema_version": 2,
        "provenance": "collected",
        "runtime": runtime,
        "tables": {"ascend_layer_perf": {"rows": len(rows), "status": "complete", "collections": [collection]}},
        "boundary_contract": CONTRACT,
        "extrapolation": "forbidden",
        "serving_accuracy_acceptance": False,
        "scope": "Measured real layer invocations across request populations; no fitted serving residual",
    }
    (output / "collection_meta.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
    (output / "metadata").mkdir()
    (output / "metadata/provenance.json").write_text(
        json.dumps(
            {
                "runtime_identity": runtime_identity,
                "source_sha256": sources,
                "accepted": list(accepted.values()),
                "rejected": rejected,
                "holdouts_excluded": holdouts,
                "collector_source_sha256": digest(Path(__file__)),
                "campaign_set_sha256": digest(campaign_set),
            },
            indent=2,
        )
        + "\n"
    )
    return {"rows": len(rows), "rejected_observations": len(rejected), "holdouts_excluded": holdouts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-set", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collector-ref", required=True)
    args = parser.parse_args()
    print(json.dumps(export_population(args.campaign_set, args.model_config, args.output, args.collector_ref)))


if __name__ == "__main__":
    main()
