import gzip
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]


def main():
    source = REPO / "build/runtime-audit-20261029/gateway/pd_router.rs"
    tests = {}
    for name, marker in (("replay", "473 passed; 0 failed"), ("python-tests", "1407 passed")):
        original = Path(f"/tmp/aisim-gateway-{name}-20261102.log") if name != "replay" else Path("/tmp/aisim-gateway-replay-tests-20261102.log")
        raw = original.read_bytes()
        if marker.encode() not in raw:
            raise ValueError(f"Verification incomplete: {original}")
        target = ROOT / f"gateway-{name}-20261102.log.gz"
        target.write_bytes(gzip.compress(raw, mtime=0))
        tests[name] = {"result": marker, "path": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    rows = json.loads((REPO / "aisimulate-output/paired-pd-gateway-repair-20261102/comparison.json").read_text())["rows"]
    record = {
        "source_audit": {
            "runtime_image_digest": "sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443",
            "sglang_commit": "baa9befa381511cd4d56f83980951e5a45fffc6d",
            "gateway_package": "sglang-router==0.3.2",
            "path": "sgl-model-gateway/src/routers/http/pd_router.rs",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "finding": "execute_dual_dispatch_internal sends P/D concurrently but awaits process_prefill_response, including prefill_response.bytes(), before create_streaming_response exposes the Decode stream.",
            "source_locations": [653, 833, 846, 869, 1267, 1352],
            "implementation": "Independent causal replay implementation; no upstream source code copied into the package."
        },
        "policy": "Explicit engine.pd_response=sglang_gateway; direct remains the default. Client output and completion wait for P response; D computation continues while source KV awaits scheduler-boundary release.",
        "identity": "Policy survives recommendation, saved candidate YAML, prediction, native ReplaySpec and replay. Unsupported backends/adapters/aggregated modes are rejected.",
        "tests": tests,
        "image_id": subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", "aisimulate-910c:gateway-repair-20261102"], text=True).strip(),
        "same_campaign_gateway_only_rows": rows,
        "limits": "Gateway ordering alone does not remove all low-concurrency TTFT error. Tokenizer/arrival timing and cold costs are independently measured in the ongoing request-linked campaign; no serving-accuracy acceptance is asserted."
    }
    (ROOT / "gateway-response-validation.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
