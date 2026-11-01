import csv
import gzip
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]


def main():
    campaign = ROOT / "pd-cold-1p1d"
    replay = REPO / "aisimulate-output/pd-cold-closed-loop-20261102"
    comparison = json.loads((replay / "comparison.json").read_text())
    analysis = json.loads((ROOT / "pd-cold-timeline-analysis.json").read_text())
    if len(comparison["rows"]) != 2 or sum(row["requests"] for row in comparison["rows"]) != 24:
        raise ValueError("Expected both completed cold/warm controls")
    rows = comparison["rows"]
    cold = next(row for row in rows if row["concurrency"] == 1)
    cold_trace = next(case for case in analysis["cases"] if case["case"] == cold["case"])
    repeated = [row for row in cold_trace["rows"] if all(row["previous_client_requests_by_rank"].values())]
    repeated_error = 100 * (cold["predicted_ttft_ms"] / statistics.mean(row["ttft_ms"] for row in repeated) - 1)
    exclusions = []
    progress = json.loads((campaign / "progress.json").read_text())
    for case in progress["cases"]:
        if case["status"] == "completed":
            continue
        limits = {}
        for path in (campaign / case["name"]).glob("*server-info.json.gz"):
            info = json.loads(gzip.decompress(path.read_bytes()))
            limits[path.name] = {key: info[key] for key in ("context_length", "max_req_input_len", "max_total_num_tokens", "mem_fraction_static")}
        exclusions.append({"case": case["name"], "status": case["status"], "runtime_limits": limits, "reason": "32768 input tokens exceed both actual per-rank admission limits; HTTP200 does not make the requested token workload successful."})
    result = {"scope": "Independent 1P1D same pinned software, no new fitted parameter; all24 successful formal requests retained. Repeated-rank subset is a diagnostic, not a replacement for all-request accuracy.", "rows": rows, "repeated_rank_c1_ttft_error_percent": repeated_error, "exclusions": exclusions}
    (ROOT / "cold-warm-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    with (ROOT / "cold-warm-validation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# 1P1D 冷启动与重复请求对照（2026-11-02）", "",
        "node20/25、GLM5.2 W8A8、P/D均TP16/ADP4/EP16；固定0.5.22/CANN9.1镜像，graph、调度overlap、投机解码关闭。每组仍为两个warmup；本轮未重新拟合P/D执行残差或tokenizer参数。",
        "", "8K/128用于定位首字与初始化开销；完整8K/1K独立验证见[请求时序总表](request-timing-repair-comparison.md)。所有24条成功正式请求都计入下面两行，没有删除慢请求。", "",
        "| 组别 | C/N | TTFT 实测/预测 ms | TTFT误差 | TPOT 实测/预测 ms | TPOT误差 | 输入吞吐 实测/预测 tok/s | 输出吞吐 实测/预测 tok/s | 吞吐误差 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['case']} | {row['concurrency']}/{row['requests']} | {row['measured_ttft_ms']:.2f}/{row['predicted_ttft_ms']:.2f} | {row['error_percent_ttft_ms']:+.2f}% | {row['measured_tpot_ms']:.2f}/{row['predicted_tpot_ms']:.2f} | {row['error_percent_tpot_ms']:+.2f}% | {row['measured_input_tok_s']:.3f}/{row['predicted_input_tok_s']:.3f} | {row['measured_output_tok_s']:.3f}/{row['predicted_output_tok_s']:.3f} | {row['error_percent_output_tok_s']:+.2f}% |")
    lines.extend([
        "", "## 首次访问仍有额外延迟", "",
        "C1的两个warmup只覆盖两个P/D attention-DP rank。正式请求首次访问另两个rank时，TTFT分别为5553.82和6801.18ms；之后重复访问六条的均值为3169.17ms。",
        f"同一冻结模型预测3052.42ms，与重复访问六条均值的差为{repeated_error:+.2f}%。这是帮助定位的子集对照；完整C1八条仍低估22.16%，不能用子集掩盖。",
        "", "| C1顺序 | P/D rank | TTFT ms | P forward外间隙 ms | D bootstrap后分配/metadata ms | 首次可见文本对应token序号 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for index, row in enumerate(cold_trace["rows"], 1):
        lines.append(f"| {index} | {row['prefill_dp_rank']}/{row['decode_dp_rank']} | {row['ttft_ms']:.2f} | {row['prefill_outside_run_batch_ms']:.2f} | {row['decode_allocation_metadata_ms']:.2f} | {row['first_visible_text_token']} |")
    lines.extend([
        "", "第一条慢请求的D分配/metadata阶段约2116ms；第二条的P计算区间约5284ms，但两次Scheduler.run_batch主机调用之和约3133ms，剩余约2151ms发生在forward外。这里只记录区间，未把它伪装成算子耗时，也未加入随意的冷启动常数。新采集将继续细分_pre_alloc、KV元数据、send_kv_chunk等步骤。",
        "", "host_run_batch为主机墙钟，不额外同步NPU；异步kernel与主机时间不能简单相加。P/client可直接使用node20同一单调时钟，D只使用node25本地阶段差值。一次访问包含正常warmup与正式client记录，服务内部初始化和health请求未被假定为相同输入shape预热。",
        "", "首次可见文本有时是第3或第4个token。客户端TTFT按非空文本计时，与第一个生成token并非总相等；这也是短请求首字对照的剩余边界。",
        "", "## 32K失败记录", "",
        "同一服务的context_length虽然为180000，实际P max_total_num_tokens=17408、max_req_input_len=17402；D分别为32384和32378。因此32768-token输入被拒绝，不能作为32K性能数据。启动容量、每rank显存分配也必须对齐，不能只看context_length。",
        "32K客户端收到HTTP200，但输出token数不符合请求；采集校验已拒绝该点，原始结果与日志仍保留。失败warmup也不生成tokenizer校准配置。本轮不修改已验证8K配置来冒充32K结果。",
        "", "原始本地/远端记录与哈希在pd-cold-1p1d和cold-warm-validation-manifest.json；该轮三个诊断容器已停止。原四节点23点总表保持不变。",
    ])
    (ROOT / "cold-warm-validation.md").write_text("\n".join(lines) + "\n")
    artifacts = {}
    for path in sorted(campaign.rglob("*")):
        if not path.is_file() or "decode-live" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix in (".jsonl", ".log") or path.name.startswith(("host-forward-", "paired-forward-", "request-stages-", "tokenizer-ingress-")) and path.suffix != ".gz":
            archive = path.with_suffix(path.suffix + ".gz")
            archive.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            path = archive
        artifacts[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("cold-warm-validation.json", "cold-warm-validation.md", "cold-warm-validation.csv", "pd-cold-timeline-analysis.json"):
        path = ROOT / name
        if name.endswith("analysis.json"):
            archive = path.with_suffix(".json.gz")
            archive.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            path = archive
        artifacts[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for case in rows:
        for relative in ("execution.json", "result/prediction.json", "result/requests.jsonl"):
            source = replay / case["case"] / relative
            destination = ROOT / "cold-warm-replays" / case["case"] / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            artifacts[str(destination.relative_to(ROOT))] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (ROOT / "cold-warm-validation-manifest.json").write_text(json.dumps({"artifacts": artifacts, "scope": result["scope"]}, indent=2) + "\n")
    print(json.dumps({"validated_cases": len(rows), "formal_requests": 24, "manifest_files": len(artifacts), "repeated_rank_c1_ttft_error_percent": repeated_error}))


if __name__ == "__main__":
    main()
