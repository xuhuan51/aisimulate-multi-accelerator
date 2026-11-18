"""Couple native SGLang PD interfaces to virtual KV/state and CPU metadata copies.

Interface adaptation follows the pinned SGLang revision in THIRD_PARTY_NOTICES.
Payload tensors stay on meta. Only registered CPU auxiliary rows are copied.
"""

import ctypes
import os
from types import SimpleNamespace


def metadata_rows(args, slot, payload=None):
    if os.environ.get("SGLANG_USE_CPU_ENGINE") != "1":
        raise ValueError("Native PD metadata copying requires the CPU simulator")
    buffers = list(zip(args.aux_data_ptrs, args.aux_data_lens, args.aux_item_lens, strict=True))
    if payload is not None and len(payload) != len(buffers):
        raise ValueError("Native PD auxiliary buffer counts differ")
    rows = []
    for index, (pointer, span, item_bytes) in enumerate(buffers):
        if pointer <= 0 or item_bytes <= 0 or slot < 0 or (slot + 1) * item_bytes > span:
            raise ValueError("Native PD auxiliary row is outside its registered CPU buffer")
        address = pointer + slot * item_bytes
        if payload is not None:
            if len(payload[index]) != item_bytes:
                raise ValueError("Native PD auxiliary row sizes differ")
            ctypes.memmove(address, payload[index], item_bytes)
        rows.append(ctypes.string_at(address, item_bytes))
    return rows


def state_slot(indices):
    if indices is None:
        return None
    if len(indices) != 1 or len(indices[0]) != 1:
        raise ValueError("Native virtual PD currently requires one Mamba state component")
    slots = indices[0][0].reshape(-1).tolist()
    if len(slots) != 1 or slots[0] <= 0:
        raise ValueError("Native PD state payload must name one allocated slot")
    return int(slots[0])


def observe_native_pd_outputs(scheduler, role):
    from sglang_simulator.simulation.manager import StateManager
    from sglang_simulator.simulation.parallel.client import get_client
    from sglang_simulator.simulation.sglang.req_stats_manager import request_stats_manager

    streamer_type = type(scheduler.output_streamer)
    if getattr(streamer_type, "_simulator_pd_observer", False):
        return
    original = streamer_type.stream_output

    def stream(self, requests, *args, **kwargs):
        result = original(self, requests, *args, **kwargs)
        client = get_client()
        for req in requests:
            if role == "prefill":
                request_stats_manager.get_req_stats(req.rid).output_length = 1
            stats = request_stats_manager.get_req_stats(req.rid)
            if req.output_ids and stats.queue_end < 0:
                stats.queue_end = StateManager.get_global_clock()
            client.record_output(req.rid, client.output_counts.get(req.rid, 0),
                                 len(req.output_ids), req.finished())
        return result

    streamer_type.stream_output = stream
    streamer_type._simulator_pd_observer = True


def install_native_pd_backend(context):
    from sglang.srt.disaggregation import decode, prefill
    from sglang.srt.disaggregation.base.conn import KVPoll
    from sglang.srt.disaggregation.fake.conn import FakeKVManager, FakeKVReceiver, FakeKVSender
    from sglang.srt.disaggregation.utils import KVClassType
    from sglang_simulator.simulation.parallel.client import get_client

    transfer = context["transport"]
    poll_values = {"bootstrap": KVPoll.Bootstrapping, "ready": KVPoll.WaitingForInput,
                   "transferring": KVPoll.Transferring, "success": KVPoll.Success,
                   "failed": KVPoll.Failed}

    class Manager(FakeKVManager):
        def __init__(self, args, *remaining):
            super().__init__(args, *remaining)
            self.rank = args.engine_rank % context["attention_tp"]
            self.global_rank = get_client().rank
            self.state_bytes = sum(sum(component) for component in args.state_item_lens)
            if self.state_bytes != context["state_bytes_per_request"]:
                raise ValueError("Native PD state bytes differ from canonical capacity geometry")
            transfer.register_geometry(context["worker_id"], self.global_rank,
                                       sum(args.kv_item_lens), context["kv_page_bytes"], self.state_bytes)
            self.prefill_info_table = {context["bootstrap_address"]: SimpleNamespace(
                dp_size=context.get("prefill_attention_dp", context["attention_dp"]), follow_bootstrap_room=True,
            )}
            self.is_dummy_cp_rank = False

    class Sender(FakeKVSender):
        def __init__(self, mgr, bootstrap_addr, bootstrap_room, *args, **kwargs):
            super().__init__(mgr, bootstrap_addr, bootstrap_room, *args, **kwargs)
            self.room, self.rank = bootstrap_room, mgr.rank
            transfer.open_sender(self.room, self.rank, context["worker_id"],
                                 mgr.global_rank, context["kv_page_bytes"], mgr.state_bytes)

        def init(self, num_kv_indices, aux_index=None):
            self.aux_index = aux_index
            transfer.init_sender(self.room, self.rank, num_kv_indices)

        def poll(self):
            return poll_values[transfer.poll(self.room, self.rank, "sender")["status"]]

        def send(self, kv_indices, state_indices=None, num_kv_tokens=None):
            slot = state_slot(state_indices)
            rows = metadata_rows(self.kv_mgr.kv_args, self.aux_index) if slot is not None else None
            transfer.send(self.room, self.rank, len(kv_indices), slot, rows)

        def should_send_kv_chunk(self, num_pages, last_chunk):
            return num_pages > 0 or last_chunk

        def abort(self):
            transfer.cancel(self.room, self.rank)

    class Receiver(FakeKVReceiver):
        def __init__(self, mgr, bootstrap_addr, bootstrap_room=None):
            super().__init__(mgr, bootstrap_addr, bootstrap_room)
            self.room, self.rank = bootstrap_room, mgr.rank
            self.metadata_written = False

        def send_metadata(self, kv_indices, aux_index=None, state_indices=None, decode_prefix_len=None):
            if decode_prefix_len:
                raise ValueError("Native virtual PD prefix reuse is not qualified")
            self.aux_index = aux_index
            transfer.register_receiver(self.room, self.rank, len(kv_indices),
                                       state_slot(state_indices), self.kv_mgr.state_bytes,
                                       context["worker_id"], self.kv_mgr.global_rank)
            self.has_sent_metadata = True

        def poll(self):
            if not self.bootstrap_done:
                return KVPoll.Bootstrapping
            if not self.has_sent_metadata:
                return KVPoll.WaitingForInput
            result = transfer.poll(self.room, self.rank, "receiver")
            if result["status"] == "success" and not self.metadata_written:
                if int.from_bytes(result["metadata"][-1][:8], "little") != self.room:
                    raise ValueError("Native PD received metadata for a different bootstrap room")
                copied = metadata_rows(self.kv_mgr.kv_args, self.aux_index, result["metadata"])
                if copied != result["metadata"]:
                    raise ValueError("Native PD auxiliary metadata copy verification failed")
                self.metadata_written = True
            return poll_values[result["status"]]

        def abort(self):
            self.abort_notified = True
            transfer.cancel(self.room, self.rank)

    mapping = {KVClassType.MANAGER: Manager, KVClassType.SENDER: Sender, KVClassType.RECEIVER: Receiver}
    original = prefill.get_kv_class

    def get_class(backend, kind):
        return mapping[kind] if kind in mapping else original(backend, kind)

    prefill.get_kv_class = decode.get_kv_class = get_class

    original_bootstrap = prefill.PrefillBootstrapQueue.pop_bootstrapped
    original_decode = decode.SchedulerDisaggregationDecodeMixin.process_decode_queue

    def pop_bootstrapped(self, *args, **kwargs):
        client = get_client()
        while client.pd_queues_ready():
            client.scheduler.ingest_requests()
        return original_bootstrap(self, *args, **kwargs)

    def process_decode_queue(self, *args, **kwargs):
        result = original_decode(self, *args, **kwargs)
        while get_client().pd_queues_ready():
            self.ingest_requests()
            result = original_decode(self, *args, **kwargs)
        return result

    prefill.PrefillBootstrapQueue.pop_bootstrapped = pop_bootstrapped
    decode.SchedulerDisaggregationDecodeMixin.process_decode_queue = process_decode_queue
