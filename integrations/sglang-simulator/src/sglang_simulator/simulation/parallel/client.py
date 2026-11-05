"""Rank-side adapters. All collective phases include idle ranks."""

from dataclasses import asdict

CLIENT = None


class RankClient:
    def __init__(self, coordinator, rank):
        self.coordinator = coordinator
        self.rank = rank
        self.steps = {}
        self.scheduler = None
        self.outputs = []

    def exchange(self, phase, payload):
        step = self.steps.get(phase, 0)
        self.steps[phase] = step + 1
        return self.coordinator.exchange(phase, step, self.rank, payload)

    def receive(self, incoming):
        from sglang_simulator.simulation.manager import StateManager

        scheduler = self.scheduler
        running = list(scheduler.running_batch.reqs)
        if scheduler.last_batch is not None:
            running.extend(scheduler.last_batch.reqs)
        waiting = list(scheduler.waiting_queue)
        chunked = scheduler.chunked_req
        requests = {req.rid: req for req in [*running, *waiting] if not req.finished()}
        if chunked is not None:
            requests[chunked.rid] = chunked
        pending = getattr(scheduler, "_simulator_pending_requests", [])
        reply = self.exchange(
            "receive",
            {
                "incoming": incoming or [],
                "outputs": self.outputs,
                "idle": not requests and not pending,
                "load": (
                    len(requests) + len(pending),
                    sum(
                        len(r.origin_input_ids) + len(r.output_ids)
                        for r in requests.values()
                    ) + sum(len(getattr(r, "input_ids", [])) for r in pending),
                ),
            },
        )
        self.outputs = []
        StateManager.set_global_clock(reply["clock"])
        return reply["requests"]

    def step(self, batch):
        from sglang_simulator.simulation.sglang.runtime_adapter import get_parallel

        from sglang_simulator.simulation.manager import StateManager
        from sglang_simulator.simulation.sglang.runtime_adapter import (
            SGLangRuntimeAdapter,
        )
        from sglang_simulator.time_predictor.base import ScheduleBatch

        work = (
            SGLangRuntimeAdapter.schedule_batch(batch)
            if batch
            else ScheduleBatch([], "IDLE")
        )
        padded_tokens = None
        padding_mode = None
        if batch is not None and getattr(batch, "global_num_tokens", None):
            from sglang.srt.layers.dp_attention import DpPaddingMode
            from sglang.srt.utils import ceil_align

            padded_tokens = [
                ceil_align(n, get_parallel().attn_tp_size)
                for n in batch.global_num_tokens
            ]
            mode = DpPaddingMode.get_dp_padding_mode(
                batch.is_extend_in_batch, padded_tokens
            )
            padding_mode = mode.name
            if mode.is_max_len():
                padded_tokens = [max(padded_tokens)] * len(padded_tokens)
        reply = self.exchange(
            "step",
            {
                "batch": work,
                "request_ids": [r.rid for r in batch.reqs] if batch else [],
                "tp_rank": self.rank,
                "dp_rank": get_parallel().attn_dp_rank,
                "global_num_tokens": getattr(batch, "global_num_tokens", None),
                "padded_tokens": padded_tokens,
                "padding_mode": padding_mode,
            },
        )
        StateManager.set_global_clock(reply["clock"])
        StateManager.set_current_inference_dur(reply["duration"])
        if work.reqs:
            StateManager.inc_iteration()

    def profile(self, stats):
        result = self.exchange(
            "profile", {"stats": [asdict(s) for s in stats], "outputs": self.outputs}
        )
        self.outputs = []
        return result

    def record_output(self, rid, previous_count, output_count, finished):
        from sglang_simulator.simulation.manager import StateManager

        count = output_count - previous_count
        if count < 0:
            raise RuntimeError(
                "Generated token count decreased during result processing"
            )
        if count or finished:
            self.outputs.append(
                {
                    "rid": rid,
                    "count": count,
                    "finished": finished,
                    "time": StateManager.get_global_clock(),
                }
            )


def bind_client(coordinator, rank):
    global CLIENT
    CLIENT = RankClient(coordinator, rank)


def get_client():
    return CLIENT
