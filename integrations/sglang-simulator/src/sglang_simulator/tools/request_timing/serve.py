# SPDX-License-Identifier: Apache-2.0
# Modified for AISimulate multi-accelerator workload replay.
# Upstream attribution is retained in LICENSE and source-manifest.json.
"""Launch the user's real server, installing only stage measurement wrappers."""

from __future__ import annotations

import argparse
import json
import os
import platform
from functools import partial
from pathlib import Path

from .hooks import (
    install_serializer,
    install_tokenizer,
    provenance,
    run_detokenizer,
    run_scheduler,
)
from .trace import Recorder, sha256


def validate_server_args(args):
    expected = {
        "disable_overlap_schedule": True,
        "tokenizer_worker_num": 1,
        "detokenizer_worker_num": 1,
        "nnodes": 1,
        "pp_size": 1,
        "enable_dynamic_batch_tokenizer": False,
        "enable_tokenizer_batch_encode": False,
        "skip_tokenizer_init": False,
    }
    for name, value in expected.items():
        if getattr(args, name, value) != value:
            raise ValueError(f"Request timing collection requires {name}={value}")
    if getattr(args, "speculative_algorithm", None) not in (None, "NONE"):
        raise ValueError(
            "Request timing collection requires speculative decoding disabled"
        )
    if getattr(args, "disaggregation_mode", "null") not in (None, "null"):
        raise ValueError("Request timing collection requires an aggregated server")


def serve(argv):
    if (
        os.getenv("SGLANG_SIMULATOR_BOOTSTRAP") == "1"
        or os.getenv("SGLANG_USE_CPU_ENGINE") == "1"
    ):
        raise ValueError(
            "Launch this collector in the real serving environment, without simulator/CPU-engine flags"
        )
    from sglang.srt.entrypoints import http_server
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.server_args import ServerArgs
    from sglang.version import __version__

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    ServerArgs.add_cli_args(parser)
    options = parser.parse_args(argv)
    args = ServerArgs.from_cli_args(options)
    validate_server_args(args)
    directory = options.trace_dir.expanduser().resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Trace directory must be new or empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "sglang_version": __version__,
        "python": platform.python_version(),
        "collector_source_sha256": {
            path.name: sha256(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        **{
            name: getattr(args, name, None)
            for name in (
                "model_path",
                "tokenizer_path",
                "tp_size",
                "dp_size",
                "attn_dp_size",
                "ep_size",
                "moe_ep_size",
                "enable_dp_attention",
                "stream_interval",
                "disable_overlap_schedule",
                "chunked_prefill_size",
                "max_running_requests",
                "quantization",
                "attention_backend",
                "disable_radix_cache",
            )
        },
    }
    (directory / "server.json").write_text(json.dumps(manifest, indent=2) + "\n")
    recorder = Recorder(directory, "tokenizer")
    provenance(recorder, TokenizerManager)
    provenance(recorder, http_server.generate_request)
    install_tokenizer(TokenizerManager, recorder)
    install_serializer(http_server, recorder)
    print(f"Request timing traces: {directory}")
    try:
        http_server.launch_server(
            args,
            run_scheduler_process_func=partial(run_scheduler, trace_dir=str(directory)),
            run_detokenizer_process_func=partial(
                run_detokenizer, trace_dir=str(directory)
            ),
        )
    finally:
        recorder.close()
