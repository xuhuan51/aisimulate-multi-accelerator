import os
import sys
import types


def install_load_utils_stub() -> None:
    """Install the kernel loader stub before importing the sgl_kernel package."""
    module_name = "sgl_kernel.load_utils"
    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        module.__package__ = "sgl_kernel"
        sys.modules[module_name] = module

    module._load_architecture_specific_ops = lambda *args, **kwargs: None
    module._preload_cuda_library = lambda *args, **kwargs: None

    if os.environ.get("SGLANG_USE_CPU_ENGINE") == "1":
        # NPU wheels omit CUDA sgl-kernel. A supplied Python interface can be
        # older than the serving source and lack this optional CPU HiCache
        # transfer export. HiCache is disabled in this simulation scope.
        # Fail if invoked rather than silently pretending to copy KV payloads.
        try:
            import sgl_kernel
        except ModuleNotFoundError:
            return
        if not hasattr(sgl_kernel, "copy_all_layer_kv_cache_cpu"):
            def unavailable_cpu_kv_copy(*args, **kwargs):
                raise RuntimeError("CPU HiCache payload transfer is outside simulation scope")

            sgl_kernel.copy_all_layer_kv_cache_cpu = unavailable_cpu_kv_copy
        if not hasattr(sgl_kernel, "assign_req_to_token_pool_cpu"):
            def assign_req_to_token_pool_cpu(
                req_indices, req_to_token, starts, ends, locations, row_stride
            ):
                # Preserve the native ragged request-to-token mapping. No KV
                # payload is touched; this is scheduler-visible bookkeeping.
                cursor = 0
                for request, start, end in zip(
                    req_indices.tolist(), starts.tolist(), ends.tolist(), strict=True
                ):
                    if end < start:
                        raise ValueError("Negative request-to-token slice")
                    count = end - start
                    req_to_token[request, start:end] = locations[cursor:cursor + count]
                    cursor += count
                if cursor != locations.numel():
                    raise ValueError("Request-to-token locations do not match lengths")

            sgl_kernel.assign_req_to_token_pool_cpu = assign_req_to_token_pool_cpu
        for name in (
            "build_tree_kernel_efficient_cpu", "verify_tree_greedy_cpu",
            "assign_draft_cache_locs_contiguous_cpu", "assign_extend_cache_locs_cpu",
        ):
            if not hasattr(sgl_kernel, name):
                def unavailable_speculative_kernel(*args, _name=name, **kwargs):
                    raise RuntimeError(f"Speculative kernel {_name} is outside simulation scope")

                setattr(sgl_kernel, name, unavailable_speculative_kernel)
        if not getattr(sgl_kernel, "_simulator_optional_cpu_exports", False):
            previous_getattr = getattr(sgl_kernel, "__getattr__", None)

            def optional_cpu_export(name):
                # Source-only imports can reference CPU kernels omitted from
                # an NPU wheel. Preserve imports while forbidding execution.
                if name.endswith("_cpu") and not name.startswith("_"):
                    def unavailable(*args, **kwargs):
                        raise RuntimeError(f"CPU kernel {name} has no simulation implementation")
                    return unavailable
                if previous_getattr is not None:
                    return previous_getattr(name)
                raise AttributeError(name)

            sgl_kernel.__getattr__ = optional_cpu_export
            sgl_kernel._simulator_optional_cpu_exports = True
