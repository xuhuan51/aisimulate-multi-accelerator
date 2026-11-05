"""SGLang engine entry point with simulator-aware worker processes."""

from sglang_simulator.simulation.sglang.hook_bootstrap import (
    install_simulator_hooks,
    run_simulator_detokenizer_process,
    run_simulator_scheduler_process,
)

install_simulator_hooks()

# Install hooks before importing Engine so its worker entry points are patched.
from sglang.srt.entrypoints.engine import Engine


class SGLangSimulationEngine(Engine):
    """Engine whose spawned workers install SGLang Simulator hooks explicitly."""

    run_scheduler_process_func = staticmethod(run_simulator_scheduler_process)
    run_detokenizer_process_func = staticmethod(run_simulator_detokenizer_process)

    def __init__(self, **kwargs):
        from sglang_simulator.compat import apply_simulator_server_args

        if "server_args" in kwargs:
            from sglang_simulator.simulation.sglang.runtime_adapter import (
                server_arg_values,
            )

            kwargs = server_arg_values(kwargs["server_args"])
        apply_simulator_server_args(kwargs)
        super().__init__(**kwargs)
