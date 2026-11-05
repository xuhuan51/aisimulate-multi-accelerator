"""The simulator's contract with the current SGLang source runtime."""

import inspect
import os
import sys
from pathlib import Path


def runtime_flavor():
    flavor = os.environ.get("SGLANG_SIMULATOR_RUNTIME", "upstream")
    if flavor not in {"upstream", "ascend"}:
        raise ValueError("SGLANG_SIMULATOR_RUNTIME must be upstream or ascend")
    return flavor


def server_arg_values(server_args):
    resolved = getattr(server_args, "resolved_dict", None)
    if callable(resolved):
        return resolved()
    from dataclasses import asdict, is_dataclass

    if is_dataclass(server_args):
        return asdict(server_args)
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

        fields = ServerArgs.__struct_fields__
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
