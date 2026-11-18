"""The simulator's contract with the current SGLang source runtime."""

import inspect
import os
import sys
from pathlib import Path


def configure_cpu_multiprocessing():
    method = os.environ.get("SGLANG_SIMULATOR_CPU_START_METHOD", "spawn")
    if method == "spawn":
        return
    if method != "forkserver":
        raise ValueError("SGLANG_SIMULATOR_CPU_START_METHOD must be spawn or forkserver")
    if os.environ.get("SGLANG_USE_CPU_ENGINE") != "1":
        raise ValueError("Simulator forkserver requires SGLANG_USE_CPU_ENGINE=1")
    import multiprocessing

    if method not in multiprocessing.get_all_start_methods():
        raise ValueError("Simulator forkserver is unavailable on this platform")
    multiprocessing.set_forkserver_preload([
        "sglang_simulator.simulation.sglang.cpu_worker_preload"
    ])
    multiprocessing.set_start_method(method, force=True)


def runtime_flavor():
    flavor = os.environ.get("SGLANG_SIMULATOR_RUNTIME", "upstream")
    if flavor not in {"upstream", "ascend"}:
        raise ValueError("SGLANG_SIMULATOR_RUNTIME must be upstream or ascend")
    return flavor


def canonical_worker_type(disaggregation_mode):
    roles = {None: "aggregated", "null": "aggregated", "prefill": "prefill", "decode": "decode"}
    if disaggregation_mode not in roles:
        raise ValueError(f"Unsupported native disaggregation mode: {disaggregation_mode}")
    return roles[disaggregation_mode]


def server_arg_values(server_args):
    resolved = getattr(server_args, "resolved_dict", None)
    if callable(resolved):
        return resolved()
    from dataclasses import asdict, is_dataclass

    if is_dataclass(server_args):
        return asdict(server_args)
    raise TypeError("Unsupported SGLang ServerArgs representation")


def server_arg_fields(server_args_type):
    declared = getattr(server_args_type, "__struct_fields__", None)
    if declared is not None:
        return set(declared)
    from dataclasses import fields, is_dataclass

    if is_dataclass(server_args_type):
        return {field.name for field in fields(server_args_type)}
    raise TypeError("Unsupported SGLang ServerArgs representation")


def request_prefix_length(req):
    if hasattr(req, "prefix_len"):
        return req.prefix_len
    return len(req.prefix_indices)


def get_parallel(server_args=None):
    if runtime_flavor() != "ascend":
        from sglang.srt.runtime_context import get_parallel as native

        return native()

    from types import SimpleNamespace

    if server_args is not None:
        return SimpleNamespace(
            tp_size=server_args.tp_size,
            attn_dp_size=server_args.dp_size if server_args.enable_dp_attention else 1,
        )

    from sglang.srt.distributed import get_attn_tp_group, get_moe_ep_group, get_tp_group
    from sglang.srt.layers.dp_attention import get_attention_dp_rank, get_attention_dp_size

    tp_group, attn_tp_group = get_tp_group(), get_attn_tp_group()
    return SimpleNamespace(
        tp_size=tp_group.world_size,
        tp_rank=tp_group.rank_in_group,
        tp_group=tp_group,
        attn_dp_size=get_attention_dp_size(),
        attn_dp_rank=get_attention_dp_rank(),
        attn_tp_size=attn_tp_group.world_size,
        attn_tp_rank=attn_tp_group.rank_in_group,
        attn_tp_group=attn_tp_group,
        moe_ep_group=get_moe_ep_group(),
    )


def select_source_runtime():
    """Use the accompanying checkout in editable installs, including spawn."""
    explicit = os.environ.get("SGLANG_SIMULATOR_SGLANG_ROOT")
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if (root / "python/sglang/srt/server_args.py").is_file():
            root = root / "python"
        if not (root / "sglang/srt/server_args.py").is_file():
            raise ValueError(f"SGLang source not found in {root}")
    else:
        root = Path(__file__).resolve().parents[6] / "python"
        if not (root / "sglang/srt/runtime_context.py").is_file():
            return
    loaded = sys.modules.get("sglang")
    if loaded is not None and Path(loaded.__file__).resolve().parent != root / "sglang":
        raise RuntimeError(
            f"Import the simulator before SGLang, or set PYTHONPATH={root}. "
            f"Already imported SGLang from {loaded.__file__}."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if str(root) not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(root), *filter(None, paths)])


class SGLangRuntimeAdapter:
    @staticmethod
    def validate():
        from sglang.srt.server_args import ServerArgs
        from sglang.version import __version__

        if runtime_flavor() == "ascend":
            import hashlib
            from dataclasses import is_dataclass

            from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
                ModelRunnerKVCacheMixin,
            )

            if not is_dataclass(ServerArgs) or not all(
                hasattr(ModelRunnerKVCacheMixin, name)
                for name in ("profile_max_num_token", "_init_pools")
            ):
                raise RuntimeError("Unsupported Ascend scheduler/KV interface")
            source_root = Path(inspect.getfile(ServerArgs)).parent
            source_hashes = {
                name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                for name in (
                    "server_args.py",
                    "managers/scheduler.py",
                    "managers/schedule_policy.py",
                    "mem_cache/allocator.py",
                    "model_executor/model_runner_kv_cache_mixin.py",
                )
            }
            return {
                "source": inspect.getfile(ServerArgs),
                "sglang_version": __version__,
                "adapter": "ascend_legacy",
                "source_hashes": source_hashes,
                "pid": os.getpid(),
            }

        from sglang.srt.distributed import bootstrap

        fields = server_arg_fields(ServerArgs)
        legacy_context = (
            os.environ.get("SGLANG_SIMULATOR_LEGACY_ATTENTION_DP") == "1"
            and {"dp_size", "enable_dp_attention"}.issubset(fields)
            and hasattr(bootstrap, "init_torch_distributed")
        )
        current_context = "attn_dp_size" in fields and hasattr(bootstrap, "init_parallel_runtime")
        if not (current_context or legacy_context):
            raise RuntimeError(
                "This simulator requires the accompanying SGLang runtime "
                "(attn_dp_size and bootstrap.init_parallel_runtime). "
                "Set PYTHONPATH to the checkout's python directory."
            )
        return {
            "source": inspect.getfile(ServerArgs),
            "sglang_version": __version__,
            "adapter": "runtime_context_legacy_attention_dp" if legacy_context else "runtime_context",
            "pid": os.getpid(),
        }

    @staticmethod
    def schedule_batch(batch):
        from sglang_simulator.time_predictor.base import ScheduleBatch, ScheduleRequest

        mode = batch.forward_mode.name
        if not batch.reqs:
            return ScheduleBatch(reqs=[], forward_mode=mode)
        if batch.forward_mode.is_extend():
            pairs = zip(batch.extend_lens, batch.prefix_lens, strict=True)
        elif batch.forward_mode.is_decode():
            pairs = ((1, int(n) - 1) for n in batch.seq_lens_cpu.tolist())
        else:
            raise ValueError(f"Unsupported simulator forward mode: {mode}")
        decoding = {id(r) for r in (getattr(batch, "decoding_reqs", None) or [])}
        reqs = [
            ScheduleRequest(
                int(n),
                int(p),
                is_decode=batch.forward_mode.is_decode() or id(req) in decoding,
                prompt_length=(
                    len(req.origin_input_ids)
                    if hasattr(req, "origin_input_ids")
                    else None
                ),
            )
            for req, (n, p) in zip(batch.reqs, pairs, strict=True)
        ]
        if len(reqs) != len(batch.reqs) or any(r.past_kv_length < 0 for r in reqs):
            raise ValueError("SGLang batch geometry does not match its requests")
        return ScheduleBatch(reqs=reqs, forward_mode=mode)


def recurrent_state_pool_info(request_pool):
    """Read native slot ownership and physical payload geometry without data access."""
    pool = getattr(request_pool, "mamba_pool", None)
    allocator = getattr(request_pool, "mamba_allocator", None)
    if pool is None or allocator is None:
        return None
    state = pool.mamba_cache
    tensors = [*state.conv, state.temporal]
    slots = pool.size
    payload_slots = slots + 1
    if any(tensor.shape[1] != payload_slots for tensor in tensors):
        raise ValueError("Native recurrent-state payload does not match its slot pool")
    payload_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    if payload_bytes % payload_slots:
        raise ValueError("Native recurrent-state payload has uneven slot geometry")
    available = allocator.available_size()
    used = slots - available
    if not 0 <= used <= slots:
        raise ValueError("Native recurrent-state allocator exceeds its slot capacity")
    peak = max(getattr(request_pool, "_simulator_peak_state_slots", 0), used)
    request_pool._simulator_peak_state_slots = peak
    return {
        "slots": slots, "available_slots": available, "peak_used_slots": peak,
        "bytes_per_request": payload_bytes // payload_slots,
        "padding_slots": 1, "payload_storage": str(state.temporal.device),
        "conv_shapes": [list(tensor.shape) for tensor in state.conv],
        "temporal_shape": list(state.temporal.shape),
        "conv_dtypes": [str(tensor.dtype) for tensor in state.conv],
        "temporal_dtype": str(state.temporal.dtype),
    }


def recurrent_state_transfer_info(request_pool):
    """Check the native PD buffer descriptors against one request's state size."""
    state = recurrent_state_pool_info(request_pool)
    if state is None:
        return None
    pool = request_pool.mamba_pool
    methods = {
        "buffers": "get_contiguous_buf_infos",
        "slice_dimensions": "get_state_dim_per_tensor",
        "layer_ids": "get_state_layer_ids",
        "slice_outer_counts": "get_state_slice_outer_counts",
        "conv_shard_groups": "get_state_conv_shard_groups",
    }
    missing = [
        name for name in methods.values() if not callable(getattr(pool, name, None))
    ]
    if missing:
        return {"available": False, "missing_native_methods": missing}
    pointers, lengths, item_lengths = pool.get_contiguous_buf_infos()
    descriptors = {
        name: getattr(pool, method)()
        for name, method in methods.items()
        if name != "buffers"
    }
    count = len(pointers)
    arrays = (lengths, item_lengths, *descriptors.values())
    if not count or any(len(values) != count for values in arrays):
        raise ValueError("Native recurrent-state PD descriptors have inconsistent lengths")
    payload_slots = state["slots"] + state["padding_slots"]
    if any(
        item <= 0 or span != item * payload_slots
        for span, item in zip(lengths, item_lengths, strict=True)
    ):
        raise ValueError("Native recurrent-state PD span differs from its state slots")
    if sum(item_lengths) != state["bytes_per_request"]:
        raise ValueError("Native recurrent-state PD payload differs from one request's state size")
    return {
        "available": True,
        "buffer_count": count,
        "bytes_per_request": sum(item_lengths),
        "item_bytes": list(item_lengths),
        **descriptors,
    }
