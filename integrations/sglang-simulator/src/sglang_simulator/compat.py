"""Early compatibility checks for the SGLang surfaces used by the simulator."""

import inspect
import json
import os
from importlib import metadata

from sglang_simulator.simulation.sglang.runtime_adapter import (
    runtime_flavor,
    server_arg_values,
)

SIMULATOR_SERVER_ARG_OVERRIDES = {
    "disable_overlap_schedule": True,
    "disable_cuda_graph": True,
    "attention_backend": "torch_native",
    "prefill_attention_backend": "torch_native",
    "decode_attention_backend": "torch_native",
}


def simulator_parallel_args():
    path = os.environ.get("SGLANG_SIMULATOR_CONFIG_PATH")
    if not path:
        return {}
    with open(path) as f:
        config = json.load(f).get("scheduler", {})
    tp, dp, ep = (config.get(k, 1) for k in ("tp_size", "dp_size", "ep_size"))
    if any(type(n) is not int or n < 1 for n in (tp, dp, ep)):
        raise SGLangCompatibilityError("Parallel sizes must be positive integers")
    if tp % dp or tp % ep:
        raise SGLangCompatibilityError(
            "tp_size must be divisible by dp_size and ep_size"
        )
    if config.get("pp_size", 1) != 1 or config.get("cp_size", 1) != 1:
        raise SGLangCompatibilityError("Parallel simulation currently requires PP=CP=1")
    if (
        tp > 1
        and os.environ.get("SGLANG_SIMULATOR_OUTPUT_MODE", "OFFLINE").upper()
        != "OFFLINE"
    ):
        raise SGLangCompatibilityError("Parallel simulation requires OFFLINE mode")
    if runtime_flavor() == "ascend" or os.environ.get("SGLANG_SIMULATOR_LEGACY_ATTENTION_DP") == "1":
        return {
            "tp_size": tp,
            "dp_size": dp,
            "ep_size": ep,
            "enable_dp_attention": dp > 1,
            "pp_size": 1,
            "attn_cp_size": 1,
        }
    return {
        "tp_size": tp,
        "attn_dp_size": dp,
        "ep_size": ep,
        "dp_size": 1,
        "pp_size": 1,
        "attn_cp_size": 1,
        "dcp_size": 1,
    }


class SGLangCompatibilityError(RuntimeError):
    pass


def _sglang_version() -> str:
    try:
        return metadata.version("sglang")
    except metadata.PackageNotFoundError:
        return "source-checkout"


def _require_parameters(function, required: set[str], surface: str) -> None:
    parameters = set(inspect.signature(function).parameters)
    missing = required - parameters
    if missing:
        raise SGLangCompatibilityError(
            f"SGLang {_sglang_version()} is missing {surface} parameters: "
            f"{', '.join(sorted(missing))}. The simulator must be adapted to "
            "this SGLang revision before it can run."
        )


def apply_simulator_server_args(target) -> None:
    """Apply simulator-owned values before constructing the final ServerArgs."""
    values = {**SIMULATOR_SERVER_ARG_OVERRIDES, **simulator_parallel_args()}
    if isinstance(target, dict):
        target.update(values)
        return

    for name, value in values.items():
        setattr(target, name, value)


def validate_parallel_cli(parser, raw_args, argv):
    """Reject explicit CLI topology conflicting with the simulation JSON."""
    expected = simulator_parallel_args()
    supplied = {arg.split("=", 1)[0] for arg in argv if arg.startswith("--")}
    for action in parser._actions:
        if action.dest in expected and supplied.intersection(action.option_strings):
            actual = getattr(raw_args, action.dest)
            if actual != expected[action.dest]:
                raise SGLangCompatibilityError(
                    f"Explicit {action.option_strings[0]}={actual} conflicts with "
                    f"simulation topology ({action.dest}={expected[action.dest]}). "
                    "scheduler.dp_size in the JSON means attention DP."
                )


def validate_parallel_features(server_args):
    values = server_arg_values(server_args)
    if values["tp_size"] == 1 and runtime_flavor() != "ascend":
        return
    required = {
        "device": "cpu",
        "nnodes": 1,
        "dp_size": 1,
        "pp_size": 1,
        "attn_cp_size": 1,
        "dcp_size": 1,
        "disaggregation_mode": "null",
    }
    if runtime_flavor() == "ascend":
        required.pop("dcp_size")
        required.pop("dp_size")
    elif os.environ.get("SGLANG_SIMULATOR_LEGACY_ATTENTION_DP") == "1":
        required.pop("dp_size")
    for name, expected in required.items():
        if values.get(name) != expected:
            raise SGLangCompatibilityError(
                f"Parallel simulation requires {name}={expected!r}"
            )
    if values.get("moe_a2a_backend") not in {"none", "deepep"}:
        raise SGLangCompatibilityError(
            "Parallel simulation supports only native none/deepep metadata paths"
        )
    for name in (
        "speculative_algorithm",
        "enable_hierarchical_cache",
        "enable_two_batch_overlap",
    ):
        if values.get(name):
            raise SGLangCompatibilityError(
                f"Parallel simulation does not yet support {name}"
            )


def validate_simulator_server_args(server_args) -> None:
    """Fail early if a process bypassed a simulator-owned launch entry point."""
    expected_values = {**SIMULATOR_SERVER_ARG_OVERRIDES, **simulator_parallel_args()}
    values = server_arg_values(server_args)
    mismatches = [
        f"{name}={getattr(server_args, name, None)!r} (expected {expected!r})"
        for name, expected in expected_values.items()
        if values.get(name) != expected
    ]
    if mismatches:
        raise SGLangCompatibilityError(
            "SGLang Simulator server arguments were not prepared by a supported "
            f"entry point: {', '.join(mismatches)}"
        )
    if runtime_flavor() == "ascend":
        if values.get("device") != "cpu" or not values.get("max_total_tokens"):
            raise SGLangCompatibilityError(
                "Ascend simulation requires device=cpu and explicit --max-total-tokens"
            )
        for name in ("enable_hierarchical_cache", "speculative_algorithm"):
            if values.get(name):
                raise SGLangCompatibilityError(f"Ascend simulation does not support {name}")
        validate_parallel_features(server_args)


def validate_launch_runtime() -> None:
    from sglang.srt.entrypoints.http_server import launch_server

    _require_parameters(
        launch_server,
        {"server_args", "run_scheduler_process_func", "run_detokenizer_process_func"},
        "launch_server",
    )


def validate_benchmark_runtime() -> None:
    from sglang.benchmark import serving

    missing = [
        name
        for name in (
            "BenchmarkMetrics",
            "calculate_metrics",
            "cli_main",
            "get_request",
            "run_benchmark",
        )
        if not hasattr(serving, name)
    ]
    if missing:
        raise SGLangCompatibilityError(
            f"SGLang {_sglang_version()} is missing benchmark surfaces: "
            f"{', '.join(missing)}. The simulator must be adapted to this "
            "SGLang revision before it can run."
        )
