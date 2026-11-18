import ctypes
from concurrent.futures import ThreadPoolExecutor
import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from sglang_simulator.simulation.parallel.pd_clock import NativePDClock
from sglang_simulator.simulation.parallel.pd_transfer import NativePDTransfer
from sglang_simulator.simulation.parallel.client import RankClient
from sglang_simulator.simulation.sglang.pd_transfer import metadata_rows, observe_native_pd_outputs


class NativePDTransferTests(TestCase):
    def transport(self):
        clock = NativePDClock(["p", "d"], timeout=1)
        transfer = NativePDTransfer(clock, 1000)
        transfer.open_sender(10, 0, "p", 0, 100, 200)
        transfer.register_receiver(10, 0, 4, 2, 200)
        transfer.init_sender(10, 0, 4)
        return clock, transfer

    def drain(self, clock):
        def wait(worker):
            while not clock.transfer_complete((10, 0, 1)):
                clock.step(worker, clock.current_time())
        with ThreadPoolExecutor() as executor:
            left = executor.submit(wait, "p")
            right = executor.submit(wait, "d")
            left.result(timeout=2)
            right.result(timeout=2)

    def test_final_state_and_auxiliary_bytes_share_the_serial_link(self):
        clock, transfer = self.transport()
        self.assertAlmostEqual(transfer.send(10, 0, 2, None, None), 0.2)
        self.assertAlmostEqual(transfer.send(10, 0, 2, 1, [b"12345678"]), 0.608)
        self.assertEqual(transfer.poll(10, 0, "receiver")["status"], "transferring")
        self.drain(clock)
        result = transfer.poll(10, 0, "receiver")
        self.assertEqual(result, {"status": "success", "metadata": [b"12345678"]})
        self.assertEqual([row["state_bytes"] for row in transfer.report()["transfers"]], [0, 200])

    def test_cancellation_keeps_copies_inflight_until_drained(self):
        clock, transfer = self.transport()
        transfer.send(10, 0, 2, None, None)
        transfer.send(10, 0, 2, 1, [b"x"])
        transfer.cancel(10, 0)
        self.assertEqual(transfer.poll(10, 0, "sender")["status"], "transferring")
        self.drain(clock)
        self.assertEqual(transfer.poll(10, 0, "sender")["status"], "failed")
        self.assertEqual(transfer.poll(10, 0, "receiver")["status"], "failed")

    def test_state_cannot_be_sent_before_final_chunk(self):
        _, transfer = self.transport()
        with self.assertRaisesRegex(ValueError, "final chunk"):
            transfer.send(10, 0, 2, 1, [b"x"])
        self.assertEqual(transfer.report()["transfers"], [])

    def test_sender_waits_for_actual_destination_reservation(self):
        clock = NativePDClock(["p", "d"])
        transfer = NativePDTransfer(clock, 1000)
        transfer.open_sender(10, 0, "p", 0, 100, 200)
        self.assertEqual(transfer.poll(10, 0, "sender")["status"], "bootstrap")
        transfer.register_receiver(10, 0, 4, 1, 200)
        self.assertEqual(transfer.poll(10, 0, "sender")["status"], "ready")
        with self.assertRaisesRegex(ValueError, "page counts"):
            transfer.init_sender(10, 0, 3)


class NativePDAuxiliaryTests(TestCase):
    def test_output_observation_supports_native_slotted_streamers(self):
        class Streamer:
            __slots__ = ()

            def stream_output(self, requests, *args, **kwargs):
                return "native-result"

        client = RankClient(None, 0)
        scheduler = SimpleNamespace(output_streamer=Streamer())
        req = SimpleNamespace(rid="handoff", output_ids=[1], finished=lambda: False)
        with patch("sglang_simulator.simulation.parallel.client.get_client", return_value=client):
            observe_native_pd_outputs(scheduler, "decode")
            observe_native_pd_outputs(scheduler, "decode")
            self.assertEqual(scheduler.output_streamer.stream_output([req]), "native-result")
        self.assertEqual([event["count"] for event in client.outputs], [1])

    def test_registered_cpu_rows_copy_and_retain_bootstrap_room_bytes(self):
        source, target = ctypes.create_string_buffer(b"row-one!row-two!"), ctypes.create_string_buffer(17)
        src = SimpleNamespace(aux_data_ptrs=[ctypes.addressof(source)], aux_data_lens=[16], aux_item_lens=[8])
        dst = SimpleNamespace(aux_data_ptrs=[ctypes.addressof(target)], aux_data_lens=[16], aux_item_lens=[8])
        with patch.dict(os.environ, {"SGLANG_USE_CPU_ENGINE": "1"}):
            rows = metadata_rows(src, 1)
            self.assertEqual(rows, [b"row-two!"])
            self.assertEqual(metadata_rows(dst, 0, rows), rows)
            with self.assertRaisesRegex(ValueError, "outside"):
                metadata_rows(dst, 2)
            with self.assertRaisesRegex(ValueError, "sizes differ"):
                metadata_rows(dst, 0, [b"bad"])

    def test_prebuilt_handoff_and_batch_observers_do_not_duplicate_tokens(self):
        client = RankClient(None, 0)
        client.record_output("r", 0, 1, False)
        client.record_output("r", 0, 1, False)
        client.record_output("r", 1, 2, False)
        client.record_output("r", 2, 2, True)
        client.record_output("r", 2, 2, True)
        self.assertEqual([event["count"] for event in client.outputs], [1, 1, 0])
        self.assertEqual(sum(event["finished"] for event in client.outputs), 1)
