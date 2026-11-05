"""Integrate the coordinator at native launcher, routing and batch boundaries."""

import atexit
import functools
import multiprocessing
import time

from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.parallel.client import get_client

_MANAGERS = []


def shutdown_coordinators():
    while _MANAGERS:
        manager = _MANAGERS.pop()
        try:
            manager._simulator_coordinator.close()
        except (EOFError, BrokenPipeError, ConnectionError):
            pass
        finally:
            manager.shutdown()


class C_EngineParallelHook(BaseHook):
    HOOK_CLASS_NAME = "Engine"
    HOOK_MODULE_NAME = "sglang.srt.entrypoints.engine"
    REQUIRED = False

    @classmethod
    def hook(cls, target):
        original_launch = target._launch_scheduler_processes.__func__
        original_shutdown = target.shutdown

        def launch(
            engine_cls, server_args, port_args, run_scheduler_process_func, **kwargs
        ):
            from sglang_simulator.simulation.sglang.runtime_adapter import get_parallel

            from sglang_simulator.simulation.manager import ConfigManager, Envs
            from sglang_simulator.simulation.parallel.coordinator import (
                CoordinatorManager,
            )

            parallel = get_parallel(server_args)
            timing_config = ConfigManager.get_request_timing_config()
            if parallel.tp_size > 1:
                from sglang_simulator.compat import validate_parallel_features

                validate_parallel_features(server_args)
                manager = CoordinatorManager(ctx=multiprocessing.get_context("spawn"))
                manager.start()
                _MANAGERS.append(manager)
                coordinator = manager.Coordinator(
                    parallel.tp_size,
                    parallel.attn_dp_size,
                    Envs.output_dir(),
                    timing_config=timing_config,
                )
                # Keep the creator's proxy alive until every spawned child has
                # unpickled its callback (the native launcher returns early).
                manager._simulator_coordinator = coordinator
                run_scheduler_process_func = functools.partial(
                    run_scheduler_process_func, simulator_coordinator=coordinator
                )
            elif ConfigManager._get_raw_config().get("request_timing"):
                raise ValueError(
                    "request_timing currently requires the multiprocess simulator (TP > 1)"
                )
            return original_launch(
                engine_cls, server_args, port_args, run_scheduler_process_func, **kwargs
            )

        def shutdown(self, *args, **kwargs):
            try:
                return original_shutdown(self, *args, **kwargs)
            finally:
                shutdown_coordinators()

        target._launch_scheduler_processes = classmethod(launch)
        target.shutdown = shutdown
        atexit.register(shutdown_coordinators)


class _DeliverySocket:
    """Keep native routing policies; deliver atomically at a logical boundary."""

    def __init__(self, coordinator, dp_rank, socket):
        self.coordinator, self.dp_rank, self.socket = coordinator, dp_rank, socket

    def send(self, data, flags=0):
        from sglang.srt.managers.io_struct import msgpack_decode

        self.coordinator.deliver(self.dp_rank, msgpack_decode(data))

    def send_pyobj(self, obj, **kwargs):
        self.coordinator.deliver(self.dp_rank, obj)

    def __getattr__(self, name):
        return getattr(self.socket, name)


class C_DPControllerHook(BaseHook):
    HOOK_CLASS_NAME = "DataParallelController"
    HOOK_MODULE_NAME = "sglang.srt.managers.data_parallel_controller"
    REQUIRED = False

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_loop = target.event_loop

        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            callback = self.run_scheduler_process_func
            self.simulator_coordinator = getattr(callback, "keywords", {}).get(
                "simulator_coordinator"
            )
            if self.simulator_coordinator is not None:
                self.workers = [
                    _DeliverySocket(self.simulator_coordinator, i, sock)
                    if sock is not None
                    else None
                    for i, sock in enumerate(self.workers)
                ]
                # Refresh budgets from the completed logical iteration, rather
                # than host-time shared-memory heartbeat timing.
                self.refresh_load_budget_on_dispatch = False

        def event_loop(self):
            if self.simulator_coordinator is None:
                return original_loop(self)
            import zmq
            from sglang_simulator.simulation.sglang.runtime_adapter import runtime_flavor

            if runtime_flavor() == "ascend":
                sock_recv = lambda socket, flags: socket.recv_pyobj(flags)
            else:
                from sglang.srt.managers.io_struct import sock_recv

            coordinator = self.simulator_coordinator
            try:
                while True:
                    self.soft_watchdog.feed()
                    while True:
                        try:
                            req = sock_recv(self.recv_from_tokenizer, flags=zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        coordinator.enqueue(req)
                    routes = coordinator.take_routes()
                    if routes is not None:
                        requests, loads = routes
                        self.dp_budget.total_requests = [v[0] for v in loads]
                        self.dp_budget.total_tokens = [v[1] for v in loads]
                        for req in requests:
                            self._request_dispatcher(req)
                        coordinator.routes_done()
                    time.sleep(0.001)
            except Exception as error:
                coordinator.fail(str(error))
                raise

        target.__init__ = init
        target.event_loop = event_loop


def install_scheduler_boundaries(target):
    original_next = target.get_next_batch_to_run

    def get_next(self, *args, **kwargs):
        plan = original_next(self, *args, **kwargs)
        client = get_client()
        if client is not None:
            client.step(getattr(plan, "batch_to_run", plan))
        return plan

    target.get_next_batch_to_run = get_next


def install_receiver_boundary(target):
    original_pull = target._pull_raw_reqs

    def pull(self):
        requests = original_pull(self)
        client = get_client()
        if client is not None:
            return client.receive(requests)
        return requests

    target._pull_raw_reqs = pull
