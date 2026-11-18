from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from sglang_simulator.simulation.parallel.client import RankClient, disaggregation_requests
from sglang_simulator.simulation.parallel.coordinator import Coordinator
from sglang_simulator.simulation.parallel.pd_clock import NativePDClock
from sglang_simulator.simulation.sglang.parallel_hooks import install_scheduler_boundaries
from sglang_simulator.time_predictor.base import ScheduleBatch, ScheduleRequest


class NativePDClockTests(TestCase):
    def test_idle_cohorts_keep_yielding_while_a_logical_event_waits_for_a_slow_peer(self):
        clock = NativePDClock(['p', 'd0', 'd1', 'd2'], timeout=1)
        clock.schedule_transfer('copy', 0, 0.01)
        with ThreadPoolExecutor(max_workers=4) as executor:
            first = executor.submit(clock.step, 'd0', 0)
            time.sleep(0.02)
            self.assertFalse(first.done())
            second = executor.submit(clock.step, 'd1', 0)
            time.sleep(0.02)
            self.assertFalse(second.done())
            prefill = executor.submit(clock.step, 'p', 0, 0.02)
            self.assertAlmostEqual(clock.step('d2', 0)['end'], 0.01)
            self.assertAlmostEqual(first.result(timeout=1)['end'], 0.01)
            self.assertAlmostEqual(second.result(timeout=1)['end'], 0.01)
            self.assertTrue(clock.transfer_complete('copy'))
            for worker in ('d0', 'd1', 'd2'):
                clock.stop(worker)
            self.assertAlmostEqual(prefill.result(timeout=1)['end'], 0.02)

    def test_repeated_queue_polls_at_one_timestamp_do_not_start_another_barrier(self):
        clock = NativePDClock(["p", "d"], timeout=1)
        with ThreadPoolExecutor() as executor:
            prefill = executor.submit(clock.queues_ready, "p")
            clock.queues_ready("d")
            prefill.result(timeout=1)
            self.assertEqual(clock.queues_ready("d"), 0)
            forward = executor.submit(clock.step, "p", 0, 0.005)
            self.assertAlmostEqual(clock.step("d", 0)["end"], 0.005)
            forward.result(timeout=1)

    def test_queue_boundary_settles_destination_metadata_before_prefill_selection(self):
        clock = NativePDClock(["p", "d"], timeout=1)
        with ThreadPoolExecutor() as executor:
            prefill = executor.submit(clock.queues_ready, "p")
            self.assertFalse(prefill.done())
            clock.schedule_transfer("copy", 0, 0.001)
            self.assertFalse(clock.transfer_complete("copy"))
            self.assertEqual(clock.queues_ready("d"), 0)
            self.assertEqual(prefill.result(timeout=1), 0)
            self.assertFalse(clock.transfer_complete("copy"))

    def test_stopped_cohort_releases_queue_boundary(self):
        clock = NativePDClock(["p", "d"], timeout=1)
        with ThreadPoolExecutor() as executor:
            prefill = executor.submit(clock.queues_ready, "p")
            clock.stop("d")
            self.assertEqual(prefill.result(timeout=1), 0)

    def test_queue_boundary_excludes_a_busy_peer(self):
        clock = NativePDClock(["p", "d"], timeout=1)
        with ThreadPoolExecutor() as executor:
            prefill = executor.submit(clock.step, "p", 0, 0.005)
            with clock.condition:
                self.assertTrue(clock.condition.wait_for(lambda: clock.status["p"] == "running", 1))
            self.assertEqual(clock.queues_ready("d"), 0)
            self.assertAlmostEqual(clock.step("d", 0)["end"], 0.005)
            self.assertAlmostEqual(prefill.result(timeout=1)["end"], 0.005)

    def test_decode_steps_advance_inside_a_longer_prefill(self):
        clock = NativePDClock(["p", "d"], timeout=1)
        with ThreadPoolExecutor() as executor:
            prefill = executor.submit(clock.step, "p", 0, 0.005)
            self.assertAlmostEqual(clock.step("d", 0, 0.002)["end"], 0.002)
            self.assertFalse(prefill.done())
            self.assertAlmostEqual(clock.step("d", 0.002, 0.002)["end"], 0.004)
            self.assertAlmostEqual(clock.step("d", 0.004)["end"], 0.005)
            self.assertAlmostEqual(prefill.result(timeout=1)["end"], 0.005)

    def test_transfer_completion_is_ordered_before_next_forward_completion(self):
        clock = NativePDClock(["p", "d"], timeout=1)
        clock.schedule_transfer("chunk", 0, 0.001)
        self.assertFalse(clock.transfer_complete("chunk"))
        with ThreadPoolExecutor() as executor:
            prefill = executor.submit(clock.step, "p", 0, 0.005)
            self.assertAlmostEqual(clock.step("d", 0, 0.002)["end"], 0.002)
            self.assertTrue(clock.transfer_complete("chunk"))
            clock.step("d", 0.002)
            self.assertAlmostEqual(prefill.result(timeout=1)["end"], 0.005)

    def test_idle_decode_wakes_at_handoff_before_another_forward_finishes(self):
        clock = NativePDClock(["p", "d"], timeout=1)
        clock.schedule_transfer("final-state", 0, 0.003)
        with ThreadPoolExecutor() as executor:
            prefill = executor.submit(clock.step, "p", 0, 0.005)
            self.assertAlmostEqual(clock.step("d", 0)["end"], 0.003)
            self.assertTrue(clock.transfer_complete("final-state"))
            self.assertFalse(prefill.done())
            decode = executor.submit(clock.step, "d", 0.003, 0.004)
            # Prefill's 5ms completion releases it to publish its next boundary.
            # Retiring that cohort permits Decode to finish its 7ms forward.
            self.assertAlmostEqual(prefill.result(timeout=1)["end"], 0.005)
            clock.stop("p")
            self.assertAlmostEqual(decode.result(timeout=1)["end"], 0.007)

    def test_idle_only_workers_can_receive_physical_requests(self):
        clock = NativePDClock(["p", "d"], timeout=1)
        with ThreadPoolExecutor() as executor:
            first = executor.submit(clock.step, "p", 0)
            second = executor.submit(clock.step, "d", 0)
            self.assertEqual(first.result(timeout=1)["end"], 0)
            self.assertEqual(second.result(timeout=1)["end"], 0)

    def test_future_arrival_is_a_fleet_event(self):
        clock = NativePDClock(["p", "d"], timeout=1)

        def await_arrival(worker):
            while clock.current_time() < 0.01:
                clock.step(worker, clock.current_time(), wake_at=0.01)
            return clock.current_time()

        with ThreadPoolExecutor() as executor:
            first = executor.submit(await_arrival, "p")
            second = executor.submit(await_arrival, "d")
            self.assertAlmostEqual(first.result(timeout=1), 0.01)
            self.assertAlmostEqual(second.result(timeout=1), 0.01)

    def test_duplicate_and_invalid_events_fail(self):
        clock = NativePDClock(["p", "d"])
        clock.schedule_transfer("chunk", 0, 0.01)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            clock.schedule_transfer("chunk", 0, 0.01)
        for duration in (-1, float("nan"), True):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                clock.step("p", 0, duration)

    def test_failed_cohort_wakes_the_other_role(self):
        clock = NativePDClock(["p", "d"], timeout=0.01)
        with self.assertRaises(TimeoutError):
            clock.step("p", 0, 0.005)
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            clock.step("d", 0)

    def test_local_future_arrivals_wait_for_the_shared_clock(self):
        with TemporaryDirectory() as directory, ThreadPoolExecutor() as executor:
            clock = NativePDClock(["p", "d"], timeout=1)
            prefill = Coordinator(2, 1, directory, fleet_clock=clock, worker_id="p")
            req = SimpleNamespace(
                rid="future", input_ids=[1],
                sampling_params=SimpleNamespace(custom_params={"simulation": {
                    "created_time": 0.02, "total_request": 1,
                }}),
            )
            payload = [{"incoming": [req], "outputs": [], "idle": True, "load": (0, 0)},
                       {"incoming": [], "outputs": [], "idle": True, "load": (0, 0)}]
            intake = executor.submit(clock.arrivals_ready, "d")
            with prefill.condition:
                received = prefill._receive(payload)
            intake.result(timeout=1)
            self.assertEqual(received[0]["clock"], 0)
            self.assertAlmostEqual(prefill.next_wakeup, 0.02)
            forward = executor.submit(clock.step, "d", 0, 0.005)
            idle = [{"batch": ScheduleBatch([], "IDLE")} for _ in range(2)]
            self.assertAlmostEqual(prefill._step(idle)[0]["clock"], 0.005)
            self.assertAlmostEqual(forward.result(timeout=1)["end"], 0.005)
            prefill.close()
            clock.stop("d")

    def test_existing_coordinators_share_one_pd_clock(self):
        with TemporaryDirectory() as directory, ThreadPoolExecutor() as executor:
            clock = NativePDClock(["p", "d"], timeout=1)
            prefill = Coordinator(2, 1, Path(directory, "p"), fleet_clock=clock, worker_id="p")
            decode = Coordinator(2, 1, Path(directory, "d"), fleet_clock=clock, worker_id="d")
            prefill.config, decode.config = "p", "d"
            active = [{"batch": ScheduleBatch([ScheduleRequest(2048, 0)], "EXTEND")} for _ in range(2)]
            idle = [{"batch": ScheduleBatch([], "IDLE")} for _ in range(2)]

            def estimate(predictor, batches, config, *args):
                return {"durations": [0.005 if config == "p" else 0.002]}

            with patch("sglang_simulator.time_predictor.parallel.predict_parallel_step", side_effect=estimate):
                forward = executor.submit(prefill._step, active)
                self.assertAlmostEqual(decode._step(active)[0]["clock"], 0.002)
                self.assertAlmostEqual(decode._step(idle)[0]["clock"], 0.005)
                self.assertAlmostEqual(forward.result(timeout=1)[0]["clock"], 0.005)
            prefill.close()
            decode.close()


class NativePDAdmissionTests(TestCase):
    def test_fleet_filters_intake_without_losing_the_global_cursor(self):
        clock = NativePDClock(['p0', 'p1', 'd0', 'd1'], timeout=1)
        routes = {
            'a': {'prefill': 'p0', 'decode': 'd0'},
            'b': {'prefill': 'p1', 'decode': 'd1'},
            'c': {'prefill': 'p0', 'decode': 'd1'},
        }
        clock.configure_requests(['a', 'b', 'c'], 1, None, routes)
        empty = clock.request_admissions(0, 'p1')
        self.assertEqual(empty['requests'], [])
        self.assertEqual(empty['next_index'], 1)
        with self.assertRaisesRegex(ValueError, 'Only.*Decode'):
            clock.request_completed('d1', 'a', 0)
        self.assertEqual(clock.request_report()['active'], ['a'])
        clock.request_completed('d0', 'a', 0)
        second = clock.request_admissions(empty['next_index'], 'p1')
        self.assertEqual(second['requests'], [{'rid': 'b', 'time': 0}])
        self.assertEqual(second['next_index'], 2)
        clock.request_completed('d1', 'b', 0)
        self.assertEqual(clock.request_admissions(2, 'p0')['requests'], [{'rid': 'c', 'time': 0}])
        clock.request_completed('d1', 'c', 0)
        self.assertEqual(clock.request_report()['active'], [])
        self.assertEqual(clock.request_report()['peak_active'], 1)

    def test_fleet_rejects_missing_and_foreign_request_owners_before_admission(self):
        for routes in ({}, {'a': {'prefill': 'p', 'decode': 'missing'}},
                       {'a': {'prefill': 'p', 'decode': 'p'}}):
            clock = NativePDClock(['p', 'd'], timeout=1)
            with self.subTest(routes=routes), self.assertRaisesRegex(ValueError, 'routes'):
                clock.configure_requests(['a'], 1, None, routes)
            self.assertIsNone(clock.request_report())

    def test_only_decode_completion_releases_the_next_request(self):
        clock = NativePDClock(["p", "d"], timeout=1)
        clock.configure_requests(["a", "b", "c"], 2, "d")
        self.assertEqual(clock.request_admissions(0)["requests"], [{"rid": "a", "time": 0}, {"rid": "b", "time": 0}])
        with self.assertRaisesRegex(ValueError, "Only.*Decode"):
            clock.request_completed("p", "a", 0)
        clock.request_completed("d", "a", 0)
        self.assertEqual(clock.request_admissions(2)["requests"], [{"rid": "c", "time": 0}])
        report = clock.request_report()
        self.assertEqual(report["active"], ["b", "c"])
        self.assertEqual(report["peak_active"], 2)
        with self.assertRaisesRegex(ValueError, "active request"):
            clock.request_completed("d", "a", 0)

    def test_new_admission_wakes_an_old_queue_boundary(self):
        clock = NativePDClock(["p", "d"], timeout=1)
        clock.configure_requests(["a", "b"], 1, "d")
        clock.admissions_received("p", 1)
        clock.admissions_received("d", 1)
        with ThreadPoolExecutor() as executor:
            waiting = executor.submit(clock.queues_ready, "p", 1)
            with clock.condition:
                self.assertTrue(clock.condition.wait_for(lambda: clock.status["p"] == "queue_wait", 1))
            clock.request_completed("d", "a", 0)
            self.assertEqual(waiting.result(timeout=1), 0)
            self.assertEqual(clock.request_admissions(1)["epoch"], 2)
            clock.admissions_received("d", 2)
            waiting = executor.submit(clock.queues_ready, "d", 2)
            self.assertFalse(waiting.done())
            clock.admissions_received("p", 2)
            self.assertEqual(clock.queues_ready("p", 2), 0)
            self.assertEqual(waiting.result(timeout=1), 0)

    def test_unseen_admission_prevents_a_future_event_from_skipping_intake(self):
        clock = NativePDClock(["p", "d"], timeout=1)
        clock.configure_requests(["a", "b"], 1, "d")
        clock.admissions_received("p", 1)
        clock.admissions_received("d", 1)
        clock.request_completed("d", "a", 0)
        clock.schedule_transfer("future", 0, 0.01)
        with ThreadPoolExecutor() as executor:
            waiting = executor.submit(clock.step, "p", 0)
            clock.step("d", 0)
            waiting.result(timeout=1)
            self.assertEqual(clock.current_time(), 0)
            self.assertFalse(clock.transfer_complete("future"))
            clock.admissions_received("p", 2)
            clock.admissions_received("d", 2)

            def advance(worker):
                while clock.current_time() < 0.01:
                    clock.step(worker, clock.current_time())

            waiting = executor.submit(advance, "p")
            advance("d")
            waiting.result(timeout=1)
            self.assertTrue(clock.transfer_complete("future"))

    def test_coordinators_share_client_start_after_decode_finishes(self):
        clock = NativePDClock(["p", "d"], timeout=1)
        clock.configure_requests(["a", "b"], 1, "d")

        def request(rid, index):
            return SimpleNamespace(rid=rid, input_ids=[1], sampling_params=SimpleNamespace(custom_params={"simulation": {
                "created_time": 0, "total_request": 2, "max_concurrency": 1, "request_index": index,
            }}))

        def payload(incoming):
            return [{"incoming": incoming, "outputs": [], "idle": True, "load": (0, 0)}]

        def receive(coordinator, incoming):
            with coordinator.condition:
                return coordinator._receive(payload(incoming))

        with TemporaryDirectory() as directory, ThreadPoolExecutor() as executor:
            p = Coordinator(1, 1, Path(directory, "p"), fleet_clock=clock, worker_id="p",
                            pd_context={"shared_admission": True, "role": "prefill"})
            d = Coordinator(1, 1, Path(directory, "d"), fleet_clock=clock, worker_id="d",
                            pd_context={"shared_admission": True, "role": "decode"})
            pending = executor.submit(receive, p, [request("a", 0), request("b", 1)])
            self.assertEqual([r.rid for r in receive(d, [request("a", 0), request("b", 1)])[0]["requests"]], ["a"])
            self.assertEqual([r.rid for r in pending.result(timeout=1)[0]["requests"]], ["a"])
            forward = executor.submit(clock.step, "d", 0, 0.01)
            clock.step("p", 0)
            forward.result(timeout=1)
            p._client_complete(0.01, "a")
            self.assertEqual(receive(p, [])[0]["requests"], [])
            d._client_complete(0.01, "a")
            self.assertEqual([r.rid for r in receive(d, [])[0]["requests"]], ["b"])
            self.assertEqual([r.rid for r in receive(p, [])[0]["requests"]], ["b"])
            self.assertEqual(p.timeline.records["b"]["client_send"], 0.01)
            self.assertEqual(d.timeline.records["b"]["client_send"], 0.01)
            self.assertEqual(clock.request_report()["peak_active"], 1)
            p.close()
            d.close()


class NativePDQueueTests(TestCase):
    def test_finished_prefill_holding_a_transfer_still_counts_as_pending(self):
        req = SimpleNamespace(rid="copying", origin_input_ids=[1] * 8,
                              output_ids=[2], finished=lambda: True)
        scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(reqs=[]), last_batch=None,
            waiting_queue=[], chunked_req=None,
            disagg_prefill_inflight_queue=[req],
        )
        payloads = []
        coordinator = SimpleNamespace(exchange=lambda *args: payloads.append(args[-1]) or {"clock": 0, "requests": []})
        client = RankClient(coordinator, 0)
        client.scheduler = scheduler
        self.assertEqual(client.receive([]), [])
        self.assertFalse(payloads[0]["idle"])
        self.assertEqual(payloads[0]["load"], (1, 9))

    def test_bootstrap_preallocation_and_inflight_queues_are_deduplicated(self):
        first, second = SimpleNamespace(rid="first"), SimpleNamespace(rid="second")
        scheduler = SimpleNamespace(
            disagg_prefill_bootstrap_queue=SimpleNamespace(queue=[first]),
            disagg_prefill_inflight_queue=[first],
            disagg_decode_prealloc_queue=SimpleNamespace(
                pending_reqs=[SimpleNamespace(req=second)],
                queue=[SimpleNamespace(req=second)],
                retracted_queue=[second],
            ),
            disagg_decode_transfer_queue=SimpleNamespace(queue=[SimpleNamespace(req=second)]),
        )
        self.assertEqual(disaggregation_requests(scheduler), {"first": first, "second": second})
        self.assertEqual(disaggregation_requests(SimpleNamespace(disagg_prefill_inflight_queue=None)), {})

    def test_all_native_role_batch_boundaries_are_timed_once(self):
        calls = []

        class Scheduler:
            def get_next_batch_to_run(self, value):
                return value

            def get_next_disagg_prefill_batch_to_run(self, value):
                return SimpleNamespace(batch_to_run=value)

            def get_next_disagg_decode_batch_to_run(self, value):
                return SimpleNamespace(batch_to_run=value)

        install_scheduler_boundaries(Scheduler)
        with patch("sglang_simulator.simulation.sglang.parallel_hooks.get_client", return_value=SimpleNamespace(step=calls.append)):
            scheduler = Scheduler()
            scheduler.get_next_batch_to_run("agg")
            scheduler.get_next_disagg_prefill_batch_to_run("prefill")
            scheduler.get_next_disagg_decode_batch_to_run("decode")
        self.assertEqual(calls, ["agg", "prefill", "decode"])
