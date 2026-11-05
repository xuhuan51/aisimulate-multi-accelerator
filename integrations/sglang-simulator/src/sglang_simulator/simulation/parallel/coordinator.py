# SPDX-License-Identifier: Apache-2.0
# Modified for AISimulate multi-accelerator workload replay.
# Upstream attribution is retained in LICENSE and source-manifest.json.
"""A separate CPU process owns arrivals, logical clocks and profile output.

The manager's RPC threads only exchange metadata. Native Gloo groups still
broadcast requests and perform SGLang's DP batch negotiation in the workers.
"""

import heapq
import json
import math
import threading
import time
from dataclasses import asdict
from multiprocessing.managers import BaseManager
from pathlib import Path

from sglang_simulator.simulation.parallel.timing import (
    RequestTimeline,
    RequestTimingConfig,
)


def simulation_params(req):
    sampling = getattr(req, "sampling_params", None)
    custom = getattr(sampling, "custom_params", None) or {}
    return custom.get("simulation")


class Coordinator:
    def __init__(
        self, world_size, dp_size, output_dir, timeout=120, timing_config=None
    ):
        self.world_size = world_size
        self.dp_size = dp_size
        self.attn_tp = world_size // dp_size
        self.output_dir = Path(output_dir)
        self.timeout = timeout
        self.condition = threading.Condition()
        self.exchanges = {}
        self.failure = None
        self.clock = [0.0] * world_size
        self.future = []
        self.immediate = []
        self.received = set()
        self.expected = None
        self.max_concurrency = None
        self.in_flight = set()
        self.peak_in_flight = 0
        self.sequence = 0
        self.deliveries = [[] for _ in range(dp_size)]
        self.route_work = None
        self.route_pending = False
        self.loads = [(0, 0)] * dp_size
        self.predictor = None
        self.iterations = []
        self.rank_info = {}
        self.started = time.monotonic()
        self.timing_config = RequestTimingConfig.parse(timing_config)
        self.timeline = RequestTimeline(self.timing_config)
        self.workload_started = False
        self.requests = {}

    def register(self, rank, info, model, hardware, config):
        with self.condition:
            if (
                len(info["tp_group"]) != self.world_size
                or len(info["attn_tp_group"]) != self.attn_tp
            ):
                raise ValueError("Native process groups differ from simulator topology")
            if len(info["ep_group"]) != config.ep_size:
                raise ValueError("Native EP group differs from simulator topology")
            self.rank_info[rank] = info
            if self.predictor is None:
                from sglang_simulator.simulation.manager import ConfigManager

                self.predictor = ConfigManager.get_inference_time_predictor(
                    model, hardware, config
                )
                self.config = config
            return True

    def close(self):
        if self.predictor is not None:
            self.predictor.close()

    def _add(self, req):
        # A batch envelope is unpacked before routing individual requests.
        if req.__class__.__name__ == "BatchTokenizedGenerateReqInput":
            for item in req.batch:
                self._add(item)
            return
        params = simulation_params(req)
        if params is None:
            self.immediate.append(req)
            return
        rid = req.rid
        if rid in self.received:
            raise ValueError(f"Duplicate simulator request id: {rid}")
        expected = int(params["total_request"])
        if self.expected is not None and self.expected != expected:
            raise ValueError("Inconsistent total_request in offline workload")
        if expected <= 0 or len(self.received) >= expected:
            raise ValueError("Offline request count exceeds total_request")
        concurrency = params.get("max_concurrency")
        if concurrency is not None and (
            type(concurrency) is not int or concurrency <= 0
        ):
            raise ValueError("max_concurrency must be a positive integer")
        if self.received and concurrency != self.max_concurrency:
            raise ValueError("Inconsistent max_concurrency in offline workload")
        self.max_concurrency = concurrency
        self.expected = expected
        self.timing_config.for_request(params.get("timing", {}))
        if concurrency is not None and (
            params.get("scheduler_ready_time") is not None
            or params.get("queue_start") is not None
        ):
            raise ValueError(
                "Recorded scheduler-ready timestamps require open-loop arrivals"
            )
        self.received.add(rid)
        self.requests[rid] = req
        arrival = (
            params["created_time_ms"] / 1000.0
            if "created_time_ms" in params
            else params["created_time"]
        )
        arrival = float(arrival)
        if not math.isfinite(arrival) or arrival < 0:
            raise ValueError("Request arrival time must be finite and nonnegative")
        order = params.get("request_index", self.sequence)
        heapq.heappush(self.future, (arrival, order, req))
        self.sequence += 1

    def _launch_request(self, timestamp, req):
        if req is None:
            if not self.future:
                return
            arrival, _, req = heapq.heappop(self.future)
            if arrival > timestamp:
                self.timeline.schedule(arrival, "launch", req)
                return
        params = simulation_params(req)
        params.pop("created_time_ms", None)
        params["created_time"] = timestamp
        self.in_flight.add(req.rid)
        self.peak_in_flight = max(self.peak_in_flight, len(self.in_flight))
        self.timeline.start(
            req.rid, timestamp, len(getattr(req, "input_ids", [])), params
        )

    def _client_complete(self, timestamp, rid):
        self.in_flight.remove(rid)
        if self.max_concurrency is not None:
            self.timeline.schedule(
                timestamp + self.timing_config.client_think_ms / 1000, "launch", None
            )

    def _collect_outputs(self, inputs):
        for rank in range(0, self.world_size, self.attn_tp):
            for event in inputs[rank].get("outputs", []):
                self.timeline.server_output(event)

    def _advance_timeline(self, until):
        self.timeline.advance(until, self._launch_request, self._client_complete)

    def enqueue(self, req):
        with self.condition:
            self._add(req)
            self.condition.notify_all()

    def take_routes(self):
        with self.condition:
            work, self.route_work = self.route_work, None
            return work

    def deliver(self, dp_rank, req):
        with self.condition:
            self.deliveries[dp_rank].append(req)

    def routes_done(self):
        with self.condition:
            self.route_pending = False
            self.condition.notify_all()

    def fail(self, message):
        with self.condition:
            self.failure = str(message)
            self.condition.notify_all()

    def _wait(self, predicate):
        deadline = time.monotonic() + self.timeout
        while not predicate():
            if self.failure:
                raise RuntimeError(self.failure)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.failure = "Simulator rank/route rendezvous timed out"
                self.condition.notify_all()
                raise TimeoutError(self.failure)
            self.condition.wait(min(remaining, 1))

    def exchange(self, phase, step, rank, payload):
        with self.condition:
            if self.failure:
                raise RuntimeError(self.failure)
            key = (phase, step)
            entry = self.exchanges.setdefault(key, {"inputs": {}, "readers": 0})
            if rank in entry["inputs"]:
                raise RuntimeError(f"Duplicate rank {rank} at {key}")
            entry["inputs"][rank] = payload
            if len(entry["inputs"]) == self.world_size:
                try:
                    values = [entry["inputs"][r] for r in range(self.world_size)]
                    entry["result"] = getattr(self, "_" + phase)(values)
                except Exception as error:
                    self.failure = f"Simulator {phase} failed: {error}"
                    self.condition.notify_all()
                    raise
                self.condition.notify_all()
            self._wait(lambda: "result" in entry)
            result = entry["result"][rank]
            entry["readers"] += 1
            if entry["readers"] == self.world_size:
                del self.exchanges[key]
            return result

    def _receive(self, inputs):
        self._collect_outputs(inputs)
        for rank in range(0, self.world_size, self.attn_tp):
            for req in inputs[rank]["incoming"]:
                self._add(req)
            self.loads[rank // self.attn_tp] = inputs[rank]["load"]
        now = max(self.clock)
        ready = self.expected is not None and len(self.received) == self.expected
        due, self.immediate = self.immediate, []
        if ready and not self.workload_started:
            self.workload_started = True
            if self.max_concurrency is None:
                while self.future:
                    arrival, _, req = heapq.heappop(self.future)
                    self.timeline.schedule(arrival, "launch", req)
            else:
                for _ in range(min(self.max_concurrency, len(self.future))):
                    self.timeline.schedule(self.future[0][0], "launch", None)
        if self.workload_started:
            self._advance_timeline(now)
            # Idle workers can jump to input/output/client events. In particular,
            # a delayed final response must not deadlock a full concurrency cap.
            while (
                not due
                and not self.timeline.ready
                and self.timeline.events
                and all(item["idle"] for item in inputs)
            ):
                now = max(now, self.timeline.events[0][0])
                self._advance_timeline(now)
            due.extend(self.requests[rid] for rid in self.timeline.ready)
            self.timeline.ready.clear()
        self.clock = [max(t, now) for t in self.clock]
        for req in due:
            self.timeline.mark_received(getattr(req, "rid", None), now)
        if self.dp_size == 1:
            self.deliveries[0].extend(due)
        elif due:
            self.route_work = (due, list(self.loads))
            self.route_pending = True
            self.condition.notify_all()
            self._wait(lambda: not self.route_pending)
        if not due and all(item["idle"] for item in inputs):
            self.condition.wait(0.002)
        result = [
            {
                "clock": self.clock[r],
                "requests": self.deliveries[r // self.attn_tp]
                if r % self.attn_tp == 0
                else [],
            }
            for r in range(self.world_size)
        ]
        self.deliveries = [[] for _ in range(self.dp_size)]
        return result

    def _step(self, inputs):
        from sglang_simulator.time_predictor.parallel import predict_parallel_step

        batches = [inputs[r]["batch"] for r in range(0, self.world_size, self.attn_tp)]
        if not any(b.reqs for b in batches):
            return [{"clock": t, "duration": 0.0} for t in self.clock]
        for leader in range(0, self.world_size, self.attn_tp):
            for rank in range(leader + 1, leader + self.attn_tp):
                if asdict(inputs[rank]["batch"]) != asdict(inputs[leader]["batch"]):
                    raise ValueError("Attention TP peers selected different batches")
        geometry = next((item for item in inputs if item.get("padded_tokens")), {})
        estimate = predict_parallel_step(
            self.predictor,
            batches,
            self.config,
            geometry.get("padded_tokens"),
            geometry.get("padding_mode"),
        )
        # All ranks negotiate this batch before entering the model. Preserve
        # local completion times; the next negotiation waits for all peers.
        iteration_start = max(self.clock)
        pre = [
            self.timing_config.scheduler_seconds("pre", len(b.reqs)) for b in batches
        ]
        post = [
            self.timing_config.scheduler_seconds("post", len(b.reqs)) for b in batches
        ]
        start = max(
            self.clock[r] + pre[r // self.attn_tp] for r in range(self.world_size)
        )
        for rank in range(0, self.world_size, self.attn_tp):
            self.timeline.mark_forward(inputs[rank].get("request_ids", []), start)
        for rank in range(self.world_size):
            dp = rank // self.attn_tp
            self.clock[rank] = start + estimate["durations"][dp] + post[dp]
        self.iterations.append(
            dict(
                step=len(self.iterations),
                start=start,
                end=start + max(estimate["durations"]),
                iteration_start=iteration_start,
                postprocess_end=max(self.clock),
                host_pre_seconds=pre,
                host_post_seconds=post,
                rank_batches=inputs,
                **estimate,
            )
        )
        return [
            {
                "clock": self.clock[r],
                "duration": estimate["durations"][r // self.attn_tp],
            }
            for r in range(self.world_size)
        ]

    def _profile(self, inputs):
        from sglang_simulator.simulation.types import RequestStats
        from sglang_simulator.simulation.utils import calc_metrics
        from sglang_simulator.utils.json import CustomJsonEncoder

        self._collect_outputs(inputs)
        if self.workload_started:
            # Physical Engine completion may precede logical client delivery.
            # Drain the output pipeline before reporting or resetting a run.
            while self.timeline.events:
                self._advance_timeline(
                    max(self.timeline.time, self.timeline.events[0][0])
                )
            if self.in_flight or self.future or self.timeline.ready:
                raise RuntimeError(
                    "Cannot finalize a simulation with unfinished requests"
                )

        stats = [
            RequestStats(**s)
            for r in range(0, self.world_size, self.attn_tp)
            for s in inputs[r]["stats"]
        ]
        if len({s.rid for s in stats}) != len(stats):
            raise ValueError("A logical request was reported by multiple DP groups")
        stats.sort(key=lambda s: (s.created_time, s.rid))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if stats:
            origin = min(s.created_time for s in stats)
            for item in stats:
                for field in (
                    "created_time",
                    "queue_start",
                    "queue_end",
                    "last_event_time",
                ):
                    setattr(item, field, getattr(item, field) - origin)
                if item.rid in self.timeline.records:
                    item.timeline = self.timeline.export(item.rid, origin)
                    tokens = item.timeline["client_token_times"]
                    if len(tokens) != item.output_length or not tokens:
                        raise RuntimeError(
                            f"Incomplete logical output for {item.rid}: "
                            f"received {len(tokens)} tokens, expected {item.output_length}"
                        )
                    item.gen_token_latencies = [
                        b - a for a, b in zip([item.created_time, *tokens[:-1]], tokens)
                    ]
                    item.last_event_time = item.timeline["client_finished"]
            metrics = calc_metrics(stats)
            metrics.update(
                time_cost=time.monotonic() - self.started,
                simulation_world_size=self.world_size,
                attention_dp_size=self.dp_size,
                cpu_overhead_model=(
                    "configured"
                    if any(
                        self.timing_config.scheduler_seconds(phase, 1)
                        for phase in ("pre", "post")
                    )
                    else "excluded"
                ),
                request_timing=self.timeline.describe(),
                latency_boundary="client",
                rank_info=self.rank_info,
                load_max_concurrency=self.max_concurrency,
                peak_in_flight=self.peak_in_flight,
            )
            if self.predictor is not None:
                metrics.update(self.predictor.get_metrics())
            for name, rows in (
                ("request", [asdict(s) for s in stats]),
                ("iteration", self.iterations),
            ):
                with (self.output_dir / f"{name}.jsonl").open("w") as f:
                    for row in rows:
                        f.write(json.dumps(row, cls=CustomJsonEncoder) + "\n")
            (self.output_dir / "metrics.json").write_text(
                json.dumps(metrics, cls=CustomJsonEncoder) + "\n"
            )
        result = {"total_request": len(stats), "output_directory": str(self.output_dir)}
        self.clock = [0.0] * self.world_size
        self.future.clear()
        self.received.clear()
        self.expected = None
        self.max_concurrency = None
        self.in_flight.clear()
        self.peak_in_flight = 0
        self.iterations.clear()
        self.requests.clear()
        self.sequence = 0
        self.workload_started = False
        self.timeline = RequestTimeline(self.timing_config)
        self.started = time.monotonic()
        if self.predictor is not None:
            self.predictor.reset_metrics()
        return [result] * self.world_size


class CoordinatorManager(BaseManager):
    pass


CoordinatorManager.register("Coordinator", Coordinator)
