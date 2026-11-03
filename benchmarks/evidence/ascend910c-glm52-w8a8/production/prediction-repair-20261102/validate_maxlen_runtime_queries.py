import argparse
import copy
import json
import math
from pathlib import Path

from aisimulate_core.sdk.errors import PerfDataNotAvailableError
from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

from compare_runtime_layer_prediction import configuration


def metrics(counts):
    return [{"scheduled_requests": {"num_decode_requests": count, "sum_decode_kv_tokens": count * 8193}}
            for count in counts]


def layer_timing(operations):
    found = [row for row in operations if row[0] == "generation_decoder_layers"]
    if len(found) != 1:
        raise ValueError("Expected exactly one replaced decoder-layer timing")
    return found[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = configuration("runtime_layers", 1, 32, 32, True)
    model = RustForwardPassPerfModel.best_available(config)
    cases = []
    for name, counts in (("uniform_active4", [2] * 4 + [0] * 28),
                         ("uniform_active31", [2] * 31 + [0]),
                         ("max_len_48requests", [2] * 16 + [1] * 16),
                         ("max_len_32requests", [2] + [1] * 30 + [0])):
        ranks = metrics(counts)
        before = copy.deepcopy(ranks)
        predicted = model.estimate_forward_pass_time_ms(ranks)
        operations = model.estimate_ascend_dp_operations(ranks)
        if (not math.isfinite(predicted) or predicted <= 0 or ranks != before
                or not math.isclose(sum(row[1] for row in operations), predicted, rel_tol=1e-12)
                or not math.isclose(model.estimate_forward_pass_time_ms(list(reversed(ranks))), predicted, rel_tol=1e-12)):
            raise ValueError("Public FPM consistency or input preservation failed")
        layer = layer_timing(operations)
        if name.startswith("max_len"):
            if layer[3] != "estimated":
                raise ValueError("Population transfer must not claim measured timing")
            homogeneous = [max(counts) if count else 0 for count in counts]
            anchor = layer_timing(model.estimate_ascend_dp_operations(metrics(homogeneous)))
            if not math.isclose(layer[1], anchor[1], rel_tol=1e-12):
                raise ValueError("Physical MAX_LEN layer query differs from its explicit homogeneous anchor")
        cases.append({"case": name, "counts": counts, "predicted_ms": predicted, "operations": operations})
    rejected = []
    strict_config = copy.deepcopy(config)
    strict_config["estimator_config"]["op_level"]["ascend"]["shape_policy"] = "measured"
    strict = RustForwardPassPerfModel.best_available(strict_config)
    for name, candidate, counts in (
        ("sum_len_cross_bucket", model, [2, 1, 1, 1] + [0] * 28),
        ("measured_policy_population_transfer", strict, [2] * 16 + [1] * 16),
        ("unmeasured_request_brackets", model, [9] * 32),
    ):
        try:
            candidate.estimate_forward_pass_time_ms(metrics(counts))
        except (ValueError, PerfDataNotAvailableError) as error:
            rejected.append({"case": name, "error": str(error)})
        else:
            raise ValueError("Unsupported query unexpectedly succeeded: " + name)
    args.output.write_text(json.dumps({"status": "passed", "configuration": config,
                                      "cases": cases, "explicit_rejections": rejected,
                                      "scope": "Canonical FPM API, provenance, geometry and coverage checks; not a measured accuracy claim."}, indent=2) + "\n")
    print(json.dumps({"passed_queries": len(cases), "explicit_rejections": len(rejected)}))


if __name__ == "__main__":
    main()
