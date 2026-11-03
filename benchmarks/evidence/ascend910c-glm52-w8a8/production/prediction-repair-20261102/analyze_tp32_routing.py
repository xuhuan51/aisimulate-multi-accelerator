import argparse
import hashlib
import json
import statistics
from pathlib import Path


def peak_count(counts, expert_parallel):
    local_experts = 256 // expert_parallel
    return max(sum(counts[start:start + local_experts]) for start in range(0, 256, local_experts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload-output", type=Path, required=True)
    parser.add_argument("--quantization-descriptor", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.workload_output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.campaign / "routing-campaign.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "completed" or not manifest["probes_disarmed"]:
        raise ValueError("Analysis requires a completed, disarmed count-only campaign")
    counts = {layer: [0] * 256 for layer in range(3, 78)}
    training_tokens = 0
    cases = []
    excluded = []
    hashes = {str(manifest_path): hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    for case in manifest["cases"]:
        if case["status"] != "completed":
            raise ValueError("Incomplete declared routing case")
        path = args.campaign / case["id"] / "observations.json"
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        groups = json.loads(path.read_text())
        if groups["training"] != case["training"]:
            raise ValueError("Training/holdout mismatch")
        modes = {}
        for rank in range(32):
            raw = path.parent / f"routing-profile-rank{rank}.jsonl"
            hashes[str(raw)] = hashlib.sha256(raw.read_bytes()).hexdigest()
            for line in raw.read_text().splitlines():
                row = json.loads(line)
                modes.setdefault(row["model_call"], []).append(row["forward_mode"])
        pure_groups = []
        for group in groups["groups"]:
            observed_modes = modes[group["iteration"]]
            if len(observed_modes) != 32 or not set(observed_modes) <= {"EXTEND", "DECODE", "IDLE"}:
                raise ValueError("Missing or unsupported runtime forward modes")
            if "DECODE" in observed_modes:
                excluded.append({"case": case["id"], "iteration": group["iteration"], "reason": "mixed prefill/decode; retain raw observation, exclude from prefill-only workload statistics"})
                continue
            pure_groups.append(group)
            if [row["layer"] for row in group["layers"]] != list(range(3, 78)):
                raise ValueError("All75 layers are required")
            for row in group["layers"]:
                if len(row["expert_counts"]) != 256 or sum(row["expert_counts"]) != 8 * group["global_tokens"]:
                    raise ValueError("Expert assignments do not conserve tokens")
                if case["training"]:
                    counts[row["layer"]] = [old + new for old, new in zip(counts[row["layer"]], row["expert_counts"])]
            if case["training"]:
                training_tokens += group["global_tokens"]
        if not pure_groups:
            raise ValueError("No pure prefill observations for a declared case")
        cases.append((case, pure_groups))
    if training_tokens == 0:
        raise ValueError("No training tokens")
    report = []
    for case, groups in cases:
        for expert_parallel in (16, 32):
            errors, balanced_errors, ratios = [], [], []
            for group in groups:
                for row in group["layers"]:
                    actual = peak_count(row["expert_counts"], expert_parallel)
                    predicted = peak_count(counts[row["layer"]], expert_parallel) * group["global_tokens"] / training_tokens
                    balanced = 8 * group["global_tokens"] / expert_parallel
                    if actual <= 0:
                        raise ValueError("Empty critical rank")
                    errors.append(abs(predicted / actual - 1) * 100)
                    balanced_errors.append(abs(balanced / actual - 1) * 100)
                    ratios.append(actual / balanced)
            report.append({"id": case["id"], "training": case["training"], "expert_parallel": expert_parallel,
                           "scope": "ObservedEP32 placements" if expert_parallel == 32 else "Contiguous16-way repartition of observed counts only; no independentEP16 serving validation",
                           "groups": len(groups), "peak_count_mape_pct": statistics.mean(errors),
                           "balanced_peak_count_mape_pct": statistics.mean(balanced_errors),
                           "actual_peak_to_mean": {"mean": statistics.mean(ratios), "maximum": max(ratios)}})
    result = {"scope": "Shape/workload observations only, no timing calibration. Heldout inputs and seeds excluded from the routing probabilities. Sampling can change for another prompt distribution; this is not a universal error bound.",
              "training_tokens": training_tokens,
              "training_cases": [case["id"] for case, _ in cases if case["training"]],
              "expert_assignment_counts_by_layer": counts, "evaluation": report,
              "excluded_mixed_groups": excluded,
              "source_sha256": hashes, "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    workload = {"observation_id": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "quantization_sha256": hashlib.sha256(args.quantization_descriptor.read_bytes()).hexdigest(),
                "input_tokens": training_tokens, "layer_expert_counts": [counts[layer] for layer in range(3, 78)]}
    args.workload_output.write_text(json.dumps(workload, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
