"""Conservative event time shared by native Prefill and Decode cohorts."""

import heapq
import math
import threading
import time


class NativePDClock:
    def __init__(self, workers, timeout=120):
        if not workers or len(set(workers)) != len(workers):
            raise ValueError("Native PD clock requires distinct worker IDs")
        self.condition = threading.Condition()
        self.status = {worker: "ready" for worker in workers}
        self.time = 0.0
        self.timeout = timeout
        self.events = []
        self.sequence = 0
        self.wakeups = {}
        self.transfers = {}
        self.failure = None
        self.arrived = set()
        self.queue_times = {}
        self.admission = None
        self.admission_epoch = 0
        self.admission_seen = {worker: 0 for worker in workers}

    def configure_requests(self, request_ids, concurrency, completion_worker, request_routes=None):
        with self.condition:
            if self.admission is not None or self.arrived or self.time != 0:
                raise ValueError("Native PD request admission must be configured before intake")
            if not request_ids or any(not isinstance(rid, str) or not rid for rid in request_ids):
                raise ValueError("Native PD requests need nonempty string IDs")
            if len(set(request_ids)) != len(request_ids):
                raise ValueError("Native PD request IDs must be distinct")
            if type(concurrency) is not int or concurrency < 1:
                raise ValueError("Native PD concurrency must be a positive integer")
            if request_routes is not None:
                if set(request_routes) != set(request_ids) or any(
                    set(route) != {"prefill", "decode"}
                    or route["prefill"] == route["decode"]
                    or any(worker not in self.status for worker in route.values())
                    for route in request_routes.values()
                ):
                    raise ValueError("Native PD routes must assign each request to distinct known P/D workers")
                request_routes = {rid: dict(route) for rid, route in request_routes.items()}
            elif completion_worker not in self.status:
                raise ValueError("Unknown native PD completion worker")
            self.admission = {"request_ids": list(request_ids), "concurrency": concurrency,
                              "completion_worker": completion_worker, "request_routes": request_routes,
                              "launched": [],
                              "active": set(), "completed": [], "peak_active": 0}
            self._admit_requests(0)

    def _admit_requests(self, timestamp):
        admission = self.admission
        count = min(admission["concurrency"] - len(admission["active"]),
                    len(admission["request_ids"]) - len(admission["launched"]))
        if count == 0:
            return
        for _ in range(count):
            rid = admission["request_ids"][len(admission["launched"])]
            admission["active"].add(rid)
            admission["launched"].append({"rid": rid, "time": timestamp})
        admission["peak_active"] = max(admission["peak_active"], len(admission["active"]))
        self.admission_epoch += 1
        self.queue_times.clear()
        # A queue poll from an older admission epoch must return to receive the
        # newly released requests before joining the next queue boundary.
        for worker, status in self.status.items():
            if status == "queue_wait":
                self.status[worker] = "ready"
        self.condition.notify_all()

    def request_admissions(self, after, worker=None):
        with self.condition:
            if self.failure:
                raise RuntimeError(self.failure)
            if self.admission is None:
                raise ValueError("Native PD request admission is not configured")
            launched = self.admission["launched"]
            if type(after) is not int or not 0 <= after <= len(launched):
                raise ValueError("Invalid native PD admission cursor")
            if worker is not None and worker not in self.status:
                raise ValueError("Unknown native PD admission worker")
            requests = launched[after:]
            routes = self.admission["request_routes"]
            if routes is not None and worker is not None:
                requests = [item for item in requests if worker in routes[item["rid"]].values()]
            return {"epoch": self.admission_epoch, "requests": requests, "next_index": len(launched)}

    def admissions_received(self, worker, epoch):
        with self.condition:
            if worker not in self.status or epoch > self.admission_epoch or epoch < self.admission_seen[worker]:
                raise ValueError("Invalid native PD admission acknowledgement")
            self.admission_seen[worker] = epoch
            self._release_queues()

    def request_completed(self, worker, rid, timestamp):
        timestamp = self._timestamp(timestamp)
        with self.condition:
            if self.failure:
                raise RuntimeError(self.failure)
            if self.admission is None:
                raise ValueError("Only the native Decode client completion releases admission")
            completion_worker = self.admission["completion_worker"]
            routes = self.admission["request_routes"]
            if routes is not None:
                completion_worker = routes.get(rid, {}).get("decode")
            if worker not in self.status or worker != completion_worker:
                raise ValueError("Only the native Decode client completion releases admission")
            if rid not in self.admission["active"] or timestamp > self.time + 1e-9:
                raise ValueError("Native PD completion must name an active request at the current clock")
            self.admission["active"].remove(rid)
            self.admission["completed"].append({"rid": rid, "time": timestamp})
            self._admit_requests(timestamp)

    def request_report(self):
        with self.condition:
            if self.admission is None:
                return None
            return {**self.admission, "active": sorted(self.admission["active"]),
                    "epoch": self.admission_epoch, "seen": dict(self.admission_seen)}

    def _needs_admissions(self, worker):
        return self.admission is not None and self.admission_seen[worker] != self.admission_epoch

    def describe(self):
        with self.condition:
            return {"time": self.time, "status": dict(self.status),
                    "arrived": sorted(self.arrived), "queue_times": dict(self.queue_times),
                    "pending_events": len(self.events), "failure": self.failure,
                    "admission_epoch": self.admission_epoch, "admission_seen": dict(self.admission_seen)}

    def arrivals_ready(self, worker):
        with self.condition:
            self.arrived.add(worker)
            self.condition.notify_all()
            deadline = time.monotonic() + self.timeout
            while len(self.arrived) != len(self.status):
                if self.failure:
                    raise RuntimeError(self.failure)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.failure = "Native PD workload intake timed out"
                    self.condition.notify_all()
                    raise TimeoutError(self.failure)
                self.condition.wait(remaining)

    def workload_started(self):
        with self.condition:
            return len(self.arrived) == len(self.status)

    @staticmethod
    def _timestamp(value):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Native PD event times must be finite and nonnegative")
        return float(value)

    def current_time(self):
        with self.condition:
            return self.time

    def _push(self, timestamp, kind, key):
        heapq.heappush(self.events, (timestamp, self.sequence, kind, key))
        self.sequence += 1

    def _advance(self):
        # A ready cohort may still publish a send, receive or next forward at
        # this time. Every cohort must yield before any future event can run.
        while not any(status in {"ready", "queue_wait"} or (status == "idle" and self._needs_admissions(worker))
                      for worker, status in self.status.items()):
            candidates = list(self.wakeups.values())
            if self.events:
                candidates.append(self.events[0][0])
            if not candidates:
                break
            self.time = max(self.time, min(candidates))
            while self.events and self.events[0][0] <= self.time:
                _, _, kind, key = heapq.heappop(self.events)
                if kind == "forward":
                    if self.status[key] == "running":
                        self.status[key] = "ready"
                else:
                    self.transfers[key] = True
            # Idle native loops poll bootstrap/transfer queues at each event,
            # including while another cohort's model is still running.
            for worker, status in self.status.items():
                if status == "idle":
                    self.status[worker] = "ready"
            self.wakeups.clear()
        self.condition.notify_all()

    def _release_queues(self):
        epoch = (self.time, self.admission_epoch)
        pending = any((status == "ready" and self.queue_times.get(worker) != epoch)
                      or (status == "idle" and self._needs_admissions(worker))
                      for worker, status in self.status.items())
        if not pending:
            for worker, status in self.status.items():
                if status == "queue_wait":
                    self.status[worker] = "ready"
            self.condition.notify_all()

    def queues_ready(self, worker, admission_epoch=None):
        with self.condition:
            if self.failure:
                raise RuntimeError(self.failure)
            if self.status.get(worker) != "ready":
                raise ValueError(f"Native PD queue boundary is unavailable: {worker}")
            if self.admission is not None and admission_epoch != self.admission_epoch:
                return self.time
            epoch = (self.time, self.admission_epoch)
            if self.queue_times.get(worker) == epoch:
                return self.time
            self.queue_times[worker] = epoch
            self.status[worker] = "queue_wait"
            self._release_queues()
            deadline = time.monotonic() + self.timeout
            while self.status[worker] == "queue_wait":
                if self.failure:
                    raise RuntimeError(self.failure)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.failure = f"Native PD queue boundary timed out for {worker}"
                    self.condition.notify_all()
                    raise TimeoutError(self.failure)
                self.condition.wait(remaining)
            if self.failure:
                raise RuntimeError(self.failure)
            return self.time

    def step(self, worker, ready_time, duration=None, wake_at=None):
        ready_time = self._timestamp(ready_time)
        duration = None if duration is None else self._timestamp(duration)
        wake_at = None if wake_at is None else self._timestamp(wake_at)
        with self.condition:
            if self.failure:
                raise RuntimeError(self.failure)
            if self.status.get(worker) != "ready":
                raise ValueError(f"Native PD worker is unavailable or already yielding: {worker}")
            start = max(self.time, ready_time)
            if duration is None:
                self.status[worker] = "idle"
                if wake_at is not None:
                    self.wakeups[worker] = max(start, wake_at)
            else:
                self.status[worker] = "running"
                self._push(start + duration, "forward", worker)
            self._release_queues()
            self._advance()
            deadline = time.monotonic() + self.timeout
            while self.status[worker] not in {"ready", "stopped"}:
                if self.failure:
                    raise RuntimeError(self.failure)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.failure = f"Native PD clock timed out waiting for {worker}"
                    self.condition.notify_all()
                    raise TimeoutError(self.failure)
                poll_intake = duration is None and not self.events and not self.wakeups
                self.condition.wait(min(remaining, 0.005) if poll_intake else remaining)
                if duration is None and self.status[worker] == "idle" and (
                    self._needs_admissions(worker) or not self.events and not self.wakeups
                ):
                    # No logical event is ready. Allow intake of requests and
                    # control RPCs arriving on the physical native sockets.
                    self.status[worker] = "ready"
                    self.wakeups.pop(worker, None)
            if self.status[worker] == "stopped":
                raise RuntimeError(f"Native PD worker stopped: {worker}")
            if self.failure:
                raise RuntimeError(self.failure)
            return {"start": start, "end": self.time}

    def schedule_transfer(self, transfer_id, ready_time, duration):
        ready_time = self._timestamp(ready_time)
        duration = self._timestamp(duration)
        with self.condition:
            if self.failure:
                raise RuntimeError(self.failure)
            if transfer_id in self.transfers:
                raise ValueError(f"Duplicate native PD transfer: {transfer_id}")
            finish = max(self.time, ready_time) + duration
            self.transfers[transfer_id] = False
            self._push(finish, "transfer", transfer_id)
            self._advance()
            return finish

    def transfer_complete(self, transfer_id):
        with self.condition:
            if self.failure:
                raise RuntimeError(self.failure)
            return self.transfers[transfer_id]

    def stop(self, worker):
        with self.condition:
            if worker not in self.status:
                raise ValueError(f"Unknown native PD worker: {worker}")
            self.status[worker] = "stopped"
            self.wakeups.pop(worker, None)
            self._release_queues()
            self._advance()
            self.condition.notify_all()

    def fail(self, message):
        with self.condition:
            self.failure = str(message)
            self.condition.notify_all()
