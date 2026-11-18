"""Rank-local KV/state copies on the shared native PD event clock."""

import math
import threading


class NativePDTransfer:
    def __init__(self, clock, bandwidth_bytes_per_second):
        if not math.isfinite(bandwidth_bytes_per_second) or bandwidth_bytes_per_second <= 0:
            raise ValueError("Native PD transfer bandwidth must be positive and finite")
        self.clock = clock
        self.bandwidth = bandwidth_bytes_per_second
        self.lock = threading.RLock()
        self.rooms = {}
        self.links = {}
        self.audit = []
        self.geometries = {}

    def register_geometry(self, worker, rank, compact_page_bytes, logical_page_bytes, state_bytes):
        with self.lock:
            if min(compact_page_bytes, logical_page_bytes, state_bytes) <= 0:
                raise ValueError("Native PD buffer geometry must be positive")
            self.geometries[f"{worker}:{rank}"] = {
                "native_compact_kv_page_bytes": compact_page_bytes,
                "modeled_logical_kv_page_bytes": logical_page_bytes,
                "state_bytes_per_request": state_bytes,
            }

    def _room(self, room, rank):
        return self.rooms.setdefault((room, rank), {
            "sender": None, "receiver": None, "expected_pages": None,
            "sent_pages": 0, "chunks": [], "final": False,
            "metadata": None, "canceled": False,
        })

    def open_sender(self, room, rank, worker, global_rank, page_bytes, state_bytes):
        with self.lock:
            record = self._room(room, rank)
            if record["sender"] is not None:
                raise ValueError("Duplicate native PD sender")
            if page_bytes <= 0 or state_bytes <= 0:
                raise ValueError("Native Qwen PD needs positive KV page and state bytes")
            record["sender"] = (worker, global_rank, page_bytes, state_bytes)

    def init_sender(self, room, rank, pages):
        with self.lock:
            record = self._room(room, rank)
            if pages <= 0 or record["expected_pages"] is not None:
                raise ValueError("Invalid or duplicate native PD sender initialization")
            record["expected_pages"] = pages
            if record["receiver"] is not None and record["receiver"][0] != pages:
                raise ValueError("Native PD source and destination page counts differ")

    def register_receiver(self, room, rank, pages, state_slot, state_bytes, worker=None, global_rank=None):
        with self.lock:
            record = self._room(room, rank)
            if record["receiver"] is not None or pages <= 0 or state_slot is None or state_slot <= 0:
                raise ValueError("Invalid or duplicate native PD destination")
            if record["expected_pages"] not in (None, pages):
                raise ValueError("Native PD source and destination page counts differ")
            record["receiver"] = (pages, state_slot, state_bytes, worker, global_rank)

    def send(self, room, rank, pages, state_slot, metadata):
        with self.lock:
            record = self._room(room, rank)
            if record["canceled"] or record["final"]:
                raise ValueError("Native PD send after cancellation or final chunk")
            if record["sender"] is None or record["receiver"] is None or record["expected_pages"] is None:
                raise ValueError("Native PD send before destination reservation")
            sent = record["sent_pages"] + pages
            if pages < 0 or sent > record["expected_pages"]:
                raise ValueError("Native PD sent more pages than reserved")
            final = state_slot is not None
            if final != (sent == record["expected_pages"]):
                raise ValueError("Native PD recurrent state must accompany the final chunk")
            worker, global_rank, page_bytes, state_bytes = record["sender"]
            if final and (state_slot <= 0 or state_bytes != record["receiver"][2] or metadata is None):
                raise ValueError("Native PD final state geometry or metadata mismatch")
            auxiliary_bytes = sum(len(row) for row in metadata) if final else 0
            payload_bytes = pages * page_bytes + (state_bytes if final else 0) + auxiliary_bytes
            link = (worker, global_rank)
            start = max(self.clock.current_time(), self.links.get(link, 0))
            transfer_id = (room, rank, len(record["chunks"]))
            finish = self.clock.schedule_transfer(transfer_id, start, payload_bytes / self.bandwidth)
            self.links[link] = finish
            record["sent_pages"] = sent
            record["final"] = final
            record["chunks"].append(transfer_id)
            if final:
                record["metadata"] = metadata
            self.audit.append({
                "room": room, "rank": rank, "chunk": transfer_id[2],
                "source_worker": worker, "source_global_rank": global_rank,
                "destination_worker": record["receiver"][3], "destination_global_rank": record["receiver"][4],
                "pages": pages, "kv_bytes": pages * page_bytes,
                "state_bytes": state_bytes if final else 0,
                "auxiliary_bytes": auxiliary_bytes,
                "source_state_slot": state_slot,
                "destination_state_slot": record["receiver"][1],
                "start": start, "finish": finish,
            })
            return finish

    def poll(self, room, rank, side):
        with self.lock:
            record = self._room(room, rank)
            drained = all(self.clock.transfer_complete(key) for key in record["chunks"])
            if record["canceled"]:
                return {"status": "failed" if drained else "transferring"}
            if record["final"] and drained:
                return {"status": "success", "metadata": record["metadata"] if side == "receiver" else None}
            if side == "sender" and record["receiver"] is None:
                return {"status": "bootstrap"}
            return {"status": "transferring" if record["chunks"] else "ready"}

    def cancel(self, room, rank):
        with self.lock:
            self._room(room, rank)["canceled"] = True

    def report(self):
        with self.lock:
            return {"bandwidth_bytes_per_second": self.bandwidth,
                    "contention_scope": "source_rank", "transfers": list(self.audit),
                    "rooms": len(self.rooms), "rank_geometries": dict(self.geometries)}
