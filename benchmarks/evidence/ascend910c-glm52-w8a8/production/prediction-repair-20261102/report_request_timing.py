import csv
import gzip
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]


def read_json(path):
    return json.loads(path.read_text())


def decode_occupancy(path):
    changes = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        changes.extend(((row["destination_activated_ms"], 1), (row["last_token_ms"], -1)))
    active = peak = 0
    area = duration = 0.0
    previous = 0.0
    for timestamp, delta in sorted(changes):
        if active:
            area += active * (timestamp - previous)
            duration += timestamp - previous
        active += delta
        peak = max(peak, active)
        previous = timestamp
    return {"peak": peak, "mean_when_nonzero": area / duration, "scope": "Destination activation-to-last-token occupancy; includes prebuilt waiting, not the exact committed forward batch."}


def main():
    original = [row for row in read_json(ROOT / "original-23-point-comparison.json")["rows"] if row["mode"] == "stock"]
    baseline = {(row["configuration"], row["concurrency"]): row for row in original}
    latest = [row for row in read_json(REPO / "aisimulate-output/four-node-request-timing-20261102/comparison.json")["rows"] if row["mode"] == "stock"]
    paired = read_json(REPO / "aisimulate-output/pd-ingress-closed-loop-20261102/comparison.json")["rows"]
    conditional = read_json(REPO / "aisimulate-output/pd-ingress-conditional-v2-20261102/comparison.json")["rows"]
    if len(latest) != 23 or len(paired) != 5 or len(conditional) != 5:
        raise ValueError("Finish all original and independent replays")
    for row in latest:
        old = baseline[(row["configuration"], row["concurrency"])]
        for metric in ("ttft_ms", "tpot_ms", "input_tok_s", "output_tok_s"):
            row[f"baseline_predicted_{metric}"] = old[f"predicted_{metric}"]
            row[f"baseline_error_percent_{metric}"] = old[f"error_percent_{metric}"]
    for row in paired:
        metrics = read_json(ROOT / "pd-ingress-1p1d" / row["case"] / "metrics-summary.json")
        row["measured_running_peak"] = metrics["decode_running_peak"]
        row["measured_running_mean_when_nonzero"] = metrics["decode_running_mean_when_nonzero"]
        row["measured_rank_running_peak"] = metrics["decode_rank_running_peak"]
        row["predicted_decode_occupancy"] = decode_occupancy(REPO / "aisimulate-output/pd-ingress-closed-loop-20261102" / row["case"] / "result/requests.jsonl")
    metrics = ("ttft_ms", "tpot_ms", "input_tok_s", "output_tok_s")
    acceptance = {metric: {"within_15_percent": sum(abs(row[f"error_percent_{metric}"]) <= 15 for row in latest), "total": len(latest), "mape": statistics.mean(abs(row[f"error_percent_{metric}"]) for row in latest), "maximum_absolute_error_percent": max(abs(row[f"error_percent_{metric}"]) for row in latest)} for metric in metrics}
    rankings = []
    for concurrency in sorted({row["concurrency"] for row in latest}):
        group = [row for row in latest if row["concurrency"] == concurrency]
        actual = max(group, key=lambda row: row["measured_output_tok_s"])
        predicted = max(group, key=lambda row: row["predicted_output_tok_s"])
        rankings.append({"concurrency": concurrency, "measured_best": actual["configuration"], "predicted_best": predicted["configuration"], "tested_configurations": len(group), "measured_best_output_tok_s": actual["measured_output_tok_s"], "predicted_best_output_tok_s": predicted["predicted_output_tok_s"]})
    payload = {"scope": "Original23 independent serving points plus five request-linked1P1D controls; exact pinned runtime and workload conditions. Pure C1 tokenizer timing and previous short-C4 prefill forwards calibrated separately; original23 serving points are never fitted.", "error_formula": "100*(prediction/measurement-1)", "original23_acceptance": acceptance, "same_concurrency_rankings": rankings, "original23_rows": latest, "paired_pd_rows": paired, "conditional_measured_arrivals": conditional}
    (ROOT / "request-timing-repair-comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (ROOT / "request-timing-repair-original23.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(latest[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(latest)
    lines = ["# 请求时序修复：同配置预测与实测", "", "2026-11-02；原四节点23个已完成点、1760条正式请求保留。新增node20+25的1P1D、五组42条请求，均逐条核对8192输入、128或1024输出、错误数组及实际启动参数。warmup固定2条，输出最多32token，排除在正式统计外。", "", "## 当前误差", "", "| 指标 | 原23点中绝对误差≤15% | MAPE | 最大绝对误差 |", "| --- | ---: | ---: | ---: |"]
    for metric, summary in acceptance.items():
        lines.append(f"| {metric} | {summary['within_15_percent']}/{summary['total']} | {summary['mape']:.2f}% | {summary['maximum_absolute_error_percent']:.2f}% |")
    lines.extend(["", "初步达到多数验证点10%～15%以内，但仍有7个低并发TTFT点超出15%，最大约54%；不能宣称全部场景准确。C4、C8的首次分配、未预热rank及实际异步到达是继续检查的对象。", "", "## 同配置8k/1k原始四节点结果", "", "N=2×并发。TTFT、TPOT单位ms；输出吞吐单位token/s；输入吞吐等于输出吞吐×8，完整原始数值在CSV。", "", "| 配置 | C | 实测TTFT | 修复后TTFT | TTFT误差 | 实测TPOT | 修复后TPOT | TPOT误差 | 实测/预测输出吞吐 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in latest:
        lines.append(f"| {row['configuration']} | {row['concurrency']} | {row['measured_ttft_ms']:.2f} | {row['predicted_ttft_ms']:.2f} | {row['error_percent_ttft_ms']:+.2f}% | {row['measured_tpot_ms']:.2f} | {row['predicted_tpot_ms']:.2f} | {row['error_percent_tpot_ms']:+.2f}% | {row['measured_output_tok_s']:.3f}/{row['predicted_output_tok_s']:.3f} |")
    lines.extend(["", "## 新增独立1P1D对照", "", "P/D均globalTP16、attentionDP4、EP16。以下是由模拟器自己生成完成后补发的闭环请求，未使用实测到达时间替代预测。C1的API处理间隔用于tokenization校准；其整体TTFT不用于拟合。", "", "| case | C/N | TTFT实测/预测ms | TTFT误差 | TPOT实测/预测ms | TPOT误差 | 输入吞吐实测/预测 | 输出吞吐实测/预测 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in paired:
        lines.append(f"| {row['case']} | {row['concurrency']}/{row['requests']} | {row['measured_ttft_ms']:.2f}/{row['predicted_ttft_ms']:.2f} | {row['error_percent_ttft_ms']:+.2f}% | {row['measured_tpot_ms']:.2f}/{row['predicted_tpot_ms']:.2f} | {row['error_percent_tpot_ms']:+.2f}% | {row['measured_input_tok_s']:.3f}/{row['predicted_input_tok_s']:.3f} | {row['measured_output_tok_s']:.3f}/{row['predicted_output_tok_s']:.3f} |")
    lines.extend(["", "| case | 实测全D running peak/活跃期均值 | 实测单rank峰值 | 预测D占用峰值/活跃期均值 |", "| --- | ---: | ---: | ---: |"])
    for row in paired:
        occupied = row["predicted_decode_occupancy"]
        lines.append(f"| {row['case']} | {row['measured_running_peak']:.1f}/{row['measured_running_mean_when_nonzero']:.3f} | {row['measured_rank_running_peak']:.1f} | {occupied['peak']}/{occupied['mean_when_nonzero']:.3f} |")
    lines.extend(["", "实测running来自每秒Prometheus采样，已按DP rank去重；预测占用来自destination activation到last token区间，包含prebuilt等待，不等于逐轮forward batch。两列保留口径说明，不能据此宣称running batch精确一致。", "", "## 修复内容与校准边界", "", "1. 修复D的全局TP LM-head及eager实际行数。", "2. TP16和TP32每轮eager decode残差分别为197.609ms与132.119ms，仅用各自C1稳态forward拟合；其他并发/长度独立核验。", "3. 真实HTTP PD网关会等Prefill响应体返回后才转发Decode流。新增engine.pd_response=sglang_gateway，D继续计算，客户端输出在P响应完成后可见。", "4. 补齐P源rank登记先于D预留、在P调度边界处理的顺序。", "5. Prefill单轮91.625ms残差只用先前1P1D短输出C4的8个完整正式forward拟合；其他点和原四节点23点未参与拟合。", "6. 从新C1请求ID对齐的API时间测得P tokenization/派发约0.8435ms，worker数为1；固定8192个已分词token，不推导任意长度的线性斜率。该配置现在可穿过recommend、候选缓存、保存YAML和predict。", "", "三份forward残差和tokenizer参数均显式提供，不修改算子实测表，不默认套用到graph开启、其他软件或其他拓扑。新原生网关策略默认仍为direct，需在对应HTTP PD部署显式启用sglang_gateway。", "", "运行时get_server_info确认cuda_graph_config的P/D backend均为disabled，disable_overlap_schedule=true，speculative_algorithm=null。这个版本旧兼容字段disable_cuda_graph可以为false，应看归一化graph backend和实际日志；本轮日志npu graph为False。", "", "## 排序与使用", "", "在已测的4种候选中，C4/8/16/32/64/128的预测和实测输出吞吐第一名均为recommended-2p1d-dtp32：2台独立P，各TP16/ADP4/EP16；另2台联合组成1个D，TP32/ADP32/EP32。C128只有3种已完成候选；这不是所有并行组合的全局最优证明，也未附加用户未提供的TTFT/TPOT SLO。", "", "使用本机aisimulate-910c:request-timing-repair-20261102镜像；原aisimulate-910c:glm52-w8a8保持为冻结基线。每个预测命令、精确image ID、模型/配置及实测SHA位于aisimulate-output/four-node-request-timing-20261102/*/c*/stock/execution.json；完整23点重跑入口是同目录replay_completed_measurements.py。", "", "另保留基于实测到达trace的条件诊断，见JSON的conditional_measured_arrivals。它用于定位调度与到达差异，不能用其吞吐误差作为独立搜索准确性证据。完整RID阶段分解见pd-ingress-timeline-analysis.json，后续冷启动轮次对照在pd-cold-1p1d继续采集。"])
    (ROOT / "request-timing-repair-comparison.md").write_text("\n".join(lines) + "\n")
    artifacts = {}
    campaign = ROOT / "pd-ingress-1p1d"
    for path in sorted(campaign.rglob("*")):
        if not path.is_file() or "decode-live" in path.parts or "__pycache__" in path.parts or path.suffix == ".gz":
            continue
        if path.suffix == ".log" or path.name.startswith(("paired-forward-", "request-stages-", "tokenizer-ingress-")):
            archive = path.with_suffix(path.suffix + ".gz")
            archive.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            path = archive
        artifacts[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in campaign.rglob("*.log.gz"):
        artifacts[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("request-timing-repair-comparison.json", "request-timing-repair-original23.csv", "request-timing-repair-comparison.md", "pd-ingress-timeline-analysis.json", "prefill-tokenizer-8k-profile.json", "eager-overhead-prefill-tp16-adp4.json", "pd-prefill-calibration-analysis.json"):
        path = ROOT / name
        if name.endswith("analysis.json"):
            archive = path.with_suffix(path.suffix + ".gz")
            archive.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            path = archive
        artifacts[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / "request-timing-repair-manifest.json").write_text(json.dumps({"status": "completed_bounded_validation", "accuracy_goal": "Majority initial15percent target met for original23; low-concurrency TTFT still open. No global-search or arbitrary-topology accuracy claim.", "original23_acceptance": acceptance, "artifact_sha256": artifacts}, indent=2) + "\n")


if __name__ == "__main__":
    main()
