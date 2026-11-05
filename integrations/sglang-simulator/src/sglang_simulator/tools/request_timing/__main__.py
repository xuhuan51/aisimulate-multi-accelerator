# SPDX-License-Identifier: Apache-2.0
# Modified for AISimulate multi-accelerator workload replay.
# Upstream attribution is retained in LICENSE and source-manifest.json.
"""Measure request stages: serve (real server), bench (client), report (CPU)."""

import argparse
import asyncio
import math
import sys
from pathlib import Path


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    # Let the native ServerArgs parser own all serving flags.
    if argv and argv[0] == "serve":
        from .serve import serve

        serve(argv[1:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="Launch the real SGLang server with stage hooks")
    bench = commands.add_parser(
        "bench", help="Send streaming requests to an instrumented server"
    )
    bench.add_argument("--base-url", default="http://127.0.0.1:30000")
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument(
        "--requests-jsonl",
        type=Path,
        help="One text/input_ids + optional sampling_params object per line; consumes every row once",
    )
    bench.add_argument(
        "--num-requests", type=int, default=64, help="Generated workload only"
    )
    bench.add_argument(
        "--prompt-words",
        default="128,1024,4096",
        help="Generated text sizes; actual input token counts are measured server-side",
    )
    bench.add_argument("--output-tokens", type=int, default=32)
    bench.add_argument("--concurrency", type=int, default=1)
    bench.add_argument("--warmup-requests", type=int, default=4)
    bench.add_argument("--think-ms", type=float, default=0)
    bench.add_argument("--timeout", type=float, default=300)
    report = commands.add_parser(
        "report", help="Export request_timing.json and measurement diagnostics"
    )
    report.add_argument("--traces", type=Path, required=True)
    report.add_argument("--client", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, required=True)
    report.add_argument(
        "--include-delivery-residuals",
        action="store_true",
        help="Opt into C1/same-host delivery delay estimates, including event-loop waits",
    )
    report.add_argument(
        "--sim-config",
        type=Path,
        help="Also write sim_config.calibrated.json; output-dir must be this config's directory",
    )
    args = parser.parse_args(argv)
    if args.command == "bench":
        for name in ("num_requests", "output_tokens", "concurrency"):
            if getattr(args, name) < 1:
                parser.error(f"--{name.replace('_', '-')} must be positive")
        if (
            args.warmup_requests < 0
            or not math.isfinite(args.think_ms)
            or args.think_ms < 0
        ):
            parser.error("warmup-requests and think-ms must be nonnegative")
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            parser.error("timeout must be finite and positive")
        from .client import collect

        asyncio.run(collect(args))
    else:
        from .report import write_report

        write_report(args)


if __name__ == "__main__":
    main()
