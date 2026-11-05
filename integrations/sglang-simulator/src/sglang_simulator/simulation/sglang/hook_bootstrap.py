# SPDX-License-Identifier: Apache-2.0
# Modified for AISimulate multi-accelerator workload replay.
# Upstream attribution is retained in LICENSE and source-manifest.json.
"""Install SGLang Simulator hooks in the parent and spawned SGLang worker processes."""

import os

from sglang_simulator.simulation.sglang.runtime_adapter import (
    runtime_flavor,
    select_source_runtime,
)

select_source_runtime()

# Spawned interpreters inherit this marker before usercustomize runs.
os.environ["SGLANG_SIMULATOR_BOOTSTRAP"] = "1"

import sglang_simulator.hook as sglang_simulator_hook
from sglang_simulator.simulation.sglang import (
    cache_controller,
    hicache_storage,
    mem_cache_allocator,
    mem_pool_host,
    model_runner,
    parallel_hooks,
    scheduler,
    server_args,
    sgl_kernel_hook,
    unified_radix_cache,
)

# A spawned worker imports this module while unpickling its target. ModelConfig
# can import GPU kernels while later arguments are still being unpickled, before
# the target wrapper executes, so the loader stub must already be present here.
sgl_kernel_hook.install_load_utils_stub()

_HOOKS_INSTALLED = False


def install_simulator_hooks() -> None:
    """Install hooks once in the current Python interpreter."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return

    # The package __init__ loads GPU ops before a child-module import hook can
    # run reliably under spawn. Seed the loader module before importing SGLang.
    sgl_kernel_hook.install_load_utils_stub()

    if runtime_flavor() == "ascend":
        from sglang_simulator.simulation.sglang.ascend import C_AscendKVCacheHook

        sglang_simulator_hook.install_class_hooks(
            [
                parallel_hooks.C_EngineParallelHook,
                parallel_hooks.C_DPControllerHook,
                scheduler.C_SchedulerHook,
                scheduler.C_SglangPrefillAdderHook,
                model_runner.C_ModelRunnerHook,
                C_AscendKVCacheHook,
                mem_cache_allocator.C_PagedTokenToKVPoolAllocatorHook,
            ]
        )
        _HOOKS_INSTALLED = True
        return

    sglang_simulator_hook.install_class_hooks(
        [
            parallel_hooks.C_EngineParallelHook,
            parallel_hooks.C_DPControllerHook,
            scheduler.C_SchedulerHook,
            scheduler.C_SglangPrefillAdderHook,
            scheduler.C_SchedulerRequestReceiver,
            model_runner.C_ModelRunnerHook,
            model_runner.C_KVCacheConfiguratorHook,
            hicache_storage.C_StorageBackendFactory,
            cache_controller.C_HiCacheController,
            cache_controller.C_HybridCacheController,
            unified_radix_cache.C_UnifiedRadixCacheHook,
            mem_cache_allocator.C_PagedTokenToKVPoolAllocatorHook,
            mem_pool_host.C_MHATokenToKVPoolHostHook,
            mem_pool_host.C_HostKVCacheHook,
            mem_pool_host.C_PackedSingleKVPoolHook,
            mem_pool_host.C_GenericHostKVCacheSubclassHook,
        ]
    )
    server_args.ServerArgsAdapter.install()
    _HOOKS_INSTALLED = True


def run_simulator_scheduler_process(*args, **kwargs):
    """Spawn-safe scheduler entry point which installs SGLang Simulator before SGLang imports."""
    install_simulator_hooks()
    from sglang_simulator.simulation.sglang.distributed import (
        install_distributed_adapter,
    )
    from sglang_simulator.simulation.sglang.runtime_adapter import SGLangRuntimeAdapter

    SGLangRuntimeAdapter.validate()
    install_distributed_adapter()
    coordinator = kwargs.pop("simulator_coordinator", None)
    if coordinator is not None:
        from sglang_simulator.simulation.parallel.client import bind_client

        bind_client(coordinator, kwargs.get("tp_rank", args[3] if len(args) > 3 else 0))

    # Spawned workers do not inherit parent-process monkey patches, so import
    # the scheduler only after installing hooks in this process.
    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)


def run_simulator_detokenizer_process(*args, **kwargs):
    """Spawn-safe detokenizer entry point which installs SGLang Simulator before imports."""
    install_simulator_hooks()

    # Install hooks before transitive schedule_batch and memory_pool imports
    # so this CPU-only process does not load real GPU kernels.
    from sglang.srt.managers.detokenizer_manager import run_detokenizer_process

    return run_detokenizer_process(*args, **kwargs)
