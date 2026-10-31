#!/usr/bin/env python3

import argparse
import gzip
import hashlib
import json
import os
import pathlib
import time
import urllib.request


FIELDS = (
    "model_path", "version", "status", "tp_size", "attn_dp_size", "ep_size",
    "nnodes", "node_rank", "pp_size", "attn_cp_size", "mem_fraction_static",
    "max_running_requests", "max_total_num_tokens", "chunked_prefill_size",
    "max_prefill_tokens", "context_length", "page_size", "num_reserved_decode_tokens",
    "disable_overlap_schedule", "disable_radix_cache", "speculative_algorithm",
    "cuda_graph_config", "disaggregation_mode", "disaggregation_transfer_backend",
    "moe_a2a_backend", "deepep_mode", "disable_shared_experts_fusion",
    "load_balance_method", "schedule_policy", "quantization", "kv_cache_dtype",
    "launch_command", "startup_time",
    "dtype", "enable_dp_lm_head", "enable_tp_lm_head_all_to_all", "enable_fp32_lm_head",
    "moe_dense_tp_size", "max_req_input_len",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", action="append", required=True)
    args = parser.parse_args()
    output = pathlib.Path(os.environ["SNAPSHOT_OUTPUT"])
    nodes = []
    captured_replicas = set()
    for endpoint in args.endpoint:
        replica, url = endpoint.split("=", 1)
        if replica in captured_replicas:
            nodes.append({"replica": replica, "endpoint": url, "metrics_only_node": True, "configuration_source": "same replica leader /get_server_info"})
            continue
        with urllib.request.urlopen(url + "/get_server_info", timeout=15) as response:
            content = response.read()
        info = json.loads(content)
        if info.get("status") != "ready":
            raise RuntimeError(f"{endpoint} 尚未 ready")
        if not info.get("disable_overlap_schedule") or info.get("speculative_algorithm") is not None:
            raise RuntimeError(f"{endpoint} overlap/speculative 参数不符合对照条件")
        if any(info["cuda_graph_config"][phase]["backend"] != "disabled" for phase in ("prefill", "decode")):
            raise RuntimeError(f"{endpoint} graph 未关闭")
        if not info.get("disable_radix_cache"):
            raise RuntimeError(f"{endpoint} prefix cache 未关闭")
        target = output / (url.split("//", 1)[1].replace(":", "-") + "-server-info.json.gz")
        with gzip.open(target, "wb") as compressed:
            compressed.write(content)
        nodes.append({"replica": replica, "endpoint": url, "sha256": hashlib.sha256(content).hexdigest(), "effective": {key: info.get(key) for key in FIELDS}})
        captured_replicas.add(replica)
    (output / "runtime-config.json").write_text(json.dumps({"captured_at": time.time(), "nodes": nodes}, indent=2) + "\n")
    print(f"已保存 {len(nodes)} 个节点的实际启动配置")


if __name__ == "__main__":
    main()
