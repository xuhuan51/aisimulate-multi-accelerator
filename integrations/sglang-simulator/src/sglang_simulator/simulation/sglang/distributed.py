"""CPU transport for real SGLang metadata groups; no device kernels."""


def install_distributed_adapter():
    import inspect

    import torch
    from sglang_simulator.simulation.sglang.runtime_adapter import runtime_flavor

    if runtime_flavor() == "ascend":
        from sglang.srt.distributed import parallel_state
        from sglang.srt.model_executor import model_runner

        # Native Gloo groups are retained. CPU inference kernels and thread
        # binding are unnecessary when only scheduler metadata is executed.
        model_runner._is_cpu_amx_available = False
        model_runner._is_cpu_arm64 = False
        model_runner.ModelRunner.init_threads_binding = lambda self: None
    else:
        from sglang.srt.distributed import bootstrap, parallel_state
        from sglang.srt.utils import numa_utils

        # A simulated GPU rank is not a host NUMA-node index. Neither AMX kernels
        # nor their shared-memory allreduce are part of the simulated execution.
        bootstrap._resolve_backend = lambda **_: "gloo"
        bootstrap._bind_threads_if_cpu = lambda **_: None
        bootstrap._init_cpu_threads_env = lambda **_: None
        numa_utils.init_threads_binding = lambda **_: None
    torch.set_num_threads(1)
    group_type = parallel_state.GroupCoordinator
    if not getattr(group_type.__init__, "_simulator_cpu", False):
        original_init = group_type.__init__
        signature = inspect.signature(original_init)

        def init(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            for name in signature.parameters:
                if name.startswith("use_"):
                    bound.arguments[name] = False
            return original_init(*bound.args, **bound.kwargs)

        init._simulator_cpu = True
        group_type.__init__ = init
