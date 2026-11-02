import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch

from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator


def state(allocator):
    return {
        "ready_page_ids": (allocator.free_pages.cpu() - 1).tolist(),
        "ready_cursor_elements": allocator.free_pages.storage_offset(),
        "staged_pages": allocator.num_staged_pages,
        "staged_page_ids": [(pages.cpu() - 1).tolist() for pages in allocator.staged_pages],
        "available_pages": allocator.available_size() // allocator.page_size,
    }


def execute(total_pages, steps):
    allocator = NPUPagedTokenToKVPoolAllocator(total_pages * 128, 128, torch.bfloat16, "cpu", None, True)
    allocations, observations = {}, []
    for operation in steps:
        before = state(allocator)
        pages = None
        if operation["operation"] == "allocate":
            result = allocator.alloc(operation["pages"] * 128)
            if result is not None:
                allocations[operation["name"]] = result
                pages = (result.reshape(-1, 128)[:, 0] // 128 - 1).tolist()
        else:
            allocated = allocations[operation["name"]].reshape(-1, 128)
            selected = allocated[operation.get("start", 0):operation.get("end", len(allocated))].flatten()
            allocator.free(selected)
        observations.append({"operation": operation, "before": before, "returned_pages": pages, "after": state(allocator)})
    return observations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recycle = execute(6, [
        {"operation": "allocate", "name": "first", "pages": 2},
        {"operation": "free", "name": "first"},
        {"operation": "allocate", "name": "second", "pages": 3},
        {"operation": "free", "name": "second", "start": 1, "end": 2},
        {"operation": "allocate", "name": "after_merge", "pages": 2},
        {"operation": "allocate", "name": "tail", "pages": 2},
    ])
    failed = execute(4, [
        {"operation": "allocate", "name": "first", "pages": 3},
        {"operation": "free", "name": "first", "end": 1},
        {"operation": "allocate", "name": "failed", "pages": 3},
        {"operation": "allocate", "name": "after_failed_merge", "pages": 1},
        {"operation": "allocate", "name": "tail", "pages": 1},
    ])
    sources = {inspect.getfile(owner) for owner in NPUPagedTokenToKVPoolAllocator.__mro__ if owner is not object}
    result = {"scope": "CPU execution of the pinned Ascend paged allocator alloc/free state, not hardware timing. Source page IDs begin at1; output subtracts1 to match simulator abstract IDs.",
              "page_size": 128, "need_sort": True,
              "source_sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sorted(sources)},
              "recycle": recycle, "failed_allocation": failed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"recycle_allocations": [row["returned_pages"] for row in recycle if row["operation"]["operation"] == "allocate"],
                      "failed_attempt": failed[2], "allocation_after_failure": failed[3]["returned_pages"]}))


if __name__ == "__main__":
    main()
