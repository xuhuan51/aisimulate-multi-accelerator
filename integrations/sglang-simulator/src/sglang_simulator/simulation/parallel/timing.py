"""Logical host stages around the native scheduler and forward predictor.

Service times occupy a resource; transport delays do not. All reservations
are made in event-time order, including events during an outstanding forward.
No duration is inferred from the wall time of the CPU simulator itself.
"""

import heapq
import math
from dataclasses import asdict, dataclass, fields, replace


@dataclass(frozen=True)
class RequestTimingConfig:
    input_transport_ms: float = 0.0
    tokenize_ms: float = 0.0
    tokenize_per_token_ms: float = 0.0
    tokenizer_workers: int = 1
    dispatch_ms: float = 0.0
    scheduler_pre_ms: float = 0.0
    scheduler_pre_per_request_ms: float = 0.0
    scheduler_post_ms: float = 0.0
    scheduler_post_per_request_ms: float = 0.0
    output_ms: float = 0.0
    output_per_token_ms: float = 0.0
    output_workers: int = 1
    output_transport_ms: float = 0.0
    completion_ms: float = 0.0
    client_think_ms: float = 0.0
    stream_interval: int = 1

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in {"tokenizer_workers", "output_workers", "stream_interval"}:
                if type(value) is not int or value < 1:
                    raise ValueError(
                        f"request_timing.{field.name} must be a positive integer"
                    )
            elif (
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
            ):
                raise ValueError(
                    f"request_timing.{field.name} must be finite and nonnegative"
                )

    @classmethod
    def parse(cls, values=None):
        values = {} if values is None else values
        if not isinstance(values, dict):
            raise TypeError("request_timing must be an object")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown request_timing fields: {sorted(unknown)}")
        return cls(**values)

    def for_request(self, overrides):
        allowed = {
            "input_transport_ms",
            "tokenize_ms",
            "tokenize_per_token_ms",
            "dispatch_ms",
            "output_ms",
            "output_per_token_ms",
            "output_transport_ms",
            "completion_ms",
        }
        if not isinstance(overrides, dict) or set(overrides) - allowed:
            raise ValueError(
                "Request timing overrides may only change input/output durations"
            )
        return replace(self, **overrides)

    def scheduler_seconds(self, phase, num_requests):
        return (
            getattr(self, f"scheduler_{phase}_ms")
            + getattr(self, f"scheduler_{phase}_per_request_ms") * num_requests
        ) / 1000


class ServicePool:
    def __init__(self, workers):
        self.available = [0.0] * workers

    def reserve(self, arrival, duration):
        start = max(arrival, heapq.heappop(self.available))
        end = start + duration
        heapq.heappush(self.available, end)
        return start, end


class RequestTimeline:
    def __init__(self, config):
        self.config = config
        self.events = []
        self.sequence = 0
        self.time = 0.0
        self.records = {}
        self.plans = {}
        self.sent_tokens = {}
        self.last_delivery = {}
        self.ready = []
        self.tokenizer = ServicePool(config.tokenizer_workers)
        self.output = ServicePool(config.output_workers)

    def schedule(self, timestamp, kind, payload):
        if not math.isfinite(timestamp) or timestamp < self.time - 1e-9:
            raise ValueError(f"Invalid or past logical event: {kind} at {timestamp}")
        heapq.heappush(self.events, (timestamp, self.sequence, kind, payload))
        self.sequence += 1

    def start(self, rid, timestamp, input_tokens, params):
        if rid in self.records:
            raise ValueError(f"Request already started: {rid}")
        plan = self.config.for_request(params.get("timing", {}))
        self.plans[rid] = plan
        self.sent_tokens[rid] = 0
        self.last_delivery[rid] = timestamp
        self.records[rid] = {
            "request_index": params.get("request_index"),
            "input_tokens": input_tokens,
            "client_send": timestamp,
            "api_received": None,
            "tokenize_start": None,
            "tokenize_end": None,
            "scheduler_ready": None,
            "scheduler_received": None,
            "first_forward": None,
            "server_finished": None,
            "client_finished": None,
            "server_token_times": [],
            "client_token_times": [],
            "output_chunks": [],
            "timing_overrides": params.get("timing", {}),
            "input_source": "model",
        }
        ready = params.get("scheduler_ready_time", params.get("queue_start"))
        if ready is not None:
            if not math.isfinite(ready) or ready < timestamp:
                raise ValueError(
                    "scheduler_ready_time must be finite and >= client send"
                )
            self.records[rid]["input_source"] = "recorded_scheduler_ready"
            self.schedule(float(ready), "ready", rid)
        else:
            self.schedule(timestamp + plan.input_transport_ms / 1000, "input", rid)

    def mark_received(self, rid, timestamp):
        if rid in self.records:
            self.records[rid]["scheduler_received"] = timestamp

    def mark_forward(self, rids, timestamp):
        for rid in rids:
            if rid in self.records and self.records[rid]["first_forward"] is None:
                self.records[rid]["first_forward"] = timestamp

    def server_output(self, event):
        if event["rid"] in self.records:
            self.schedule(event["time"], "server_output", event)

    def advance(self, until, on_launch, on_complete):
        while self.events and self.events[0][0] <= until:
            timestamp, _, kind, payload = heapq.heappop(self.events)
            self.time = timestamp
            if kind == "launch":
                on_launch(timestamp, payload)
            elif kind == "complete":
                self.records[payload]["client_finished"] = timestamp
                on_complete(timestamp, payload)
            elif kind == "input":
                record, plan = self.records[payload], self.plans[payload]
                record["api_received"] = timestamp
                start, end = self.tokenizer.reserve(
                    timestamp,
                    (
                        plan.tokenize_ms
                        + plan.tokenize_per_token_ms * record["input_tokens"]
                    )
                    / 1000,
                )
                record["tokenize_start"], record["tokenize_end"] = start, end
                self.schedule(end + plan.dispatch_ms / 1000, "ready", payload)
            elif kind == "ready":
                self.records[payload]["scheduler_ready"] = timestamp
                self.ready.append(payload)
            elif kind == "server_output":
                self._server_output(timestamp, payload)
            elif kind == "delivered":
                rid, count, finished = payload
                self.records[rid]["client_token_times"].extend([timestamp] * count)
                if finished:
                    self.schedule(
                        timestamp + self.plans[rid].completion_ms / 1000,
                        "complete",
                        rid,
                    )
            else:
                raise RuntimeError(f"Unknown timeline event {kind}")
        self.time = max(self.time, until)

    def _server_output(self, timestamp, event):
        rid, count, finished = event["rid"], event["count"], event["finished"]
        record, plan = self.records[rid], self.plans[rid]
        if record["server_finished"] is not None:
            raise ValueError(f"Output after request completion: {rid}")
        record["server_token_times"].extend([timestamp] * count)
        if finished:
            record["server_finished"] = timestamp
        total = len(record["server_token_times"])
        pending = total - self.sent_tokens[rid]
        # The first token is flushed immediately, then chunks at stream_interval.
        if not finished and self.sent_tokens[rid] and pending < plan.stream_interval:
            return
        start, end = self.output.reserve(
            timestamp, (plan.output_ms + plan.output_per_token_ms * pending) / 1000
        )
        delivery = max(end + plan.output_transport_ms / 1000, self.last_delivery[rid])
        record["output_chunks"].append(
            {
                "ready": timestamp,
                "start": start,
                "end": end,
                "received": delivery,
                "tokens": pending,
                "finished": finished,
            }
        )
        self.sent_tokens[rid] = total
        self.last_delivery[rid] = delivery
        self.schedule(delivery, "delivered", (rid, pending, finished))

    def export(self, rid, origin=0.0):
        record = self.records[rid]
        result = dict(record)
        for name in (
            "client_send",
            "api_received",
            "tokenize_start",
            "tokenize_end",
            "scheduler_ready",
            "scheduler_received",
            "first_forward",
            "server_finished",
            "client_finished",
        ):
            if result[name] is not None:
                result[name] -= origin
        for name in ("server_token_times", "client_token_times"):
            result[name] = [t - origin for t in result[name]]
        result["output_chunks"] = [
            {
                k: v - origin if k in {"ready", "start", "end", "received"} else v
                for k, v in chunk.items()
            }
            for chunk in record["output_chunks"]
        ]
        return result

    def describe(self):
        return {
            "config": asdict(self.config),
            "clock": "logical_seconds",
            "input_resource": "shared_tokenizer_pool",
            "output_resource": "shared_output_pool",
            "scheduler": "normal_loop_per_rank_pre_and_post_forward",
            "completion_boundary": "client_finished",
        }
