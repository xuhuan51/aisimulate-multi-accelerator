import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT / "component-boundaries-node20-20261103"
VALIDATION = ROOT / "context-transfer-validation-20261103"
METRICS = (("mean_ttft_ms", "mean_ttft_ms", "TTFT ms"),
           ("mean_tpot_ms", "mean_tpot_ms", "TPOT ms"),
           ("input_throughput", "input_throughput_tok_s", "输入 tok/s"),
           ("output_throughput", "output_throughput_tok_s", "输出 tok/s"))


def main():
    campaign = json.loads((CAMPAIGN / "context-holdouts.json").read_text())
    if campaign["status"] != "completed":
        raise ValueError("Incomplete serving validation")
    rows = []
    for case in campaign["cases"]:
        actual_path = CAMPAIGN / case["id"] / "requests.jsonl"
        actual = json.loads(actual_path.read_text().splitlines()[-1])
        for scope in ("isolated_kernels", "runtime_layers"):
            predicted_path = VALIDATION / "serving-predictions" / f"{case['id']}-{scope}" / "prediction/prediction.json"
            predicted = json.loads(predicted_path.read_text())
            if (predicted["completed_requests"] != case["prompts"]
                    or predicted["total_input_tokens"] != case["input_length"] * case["prompts"]
                    or predicted["total_output_tokens"] != case["output_length"] * case["prompts"]
                    or any(predicted[key] for key in ("failed_requests", "canceled_requests", "rejected_requests"))):
                raise ValueError("Prediction request/token conservation failed")
            row = {"case": case["id"], "scope": scope, "input_tokens": case["input_length"],
                   "output_tokens": case["output_length"], "concurrency": case["concurrency"], "requests": case["prompts"],
                   "nodes": 1, "deployment": "aggregated", "tp": 16, "adp": 4, "ep": 16,
                   "kv_pages_per_rank": 254, "graph": False, "overlap_schedule": False, "speculation": False,
                   "measured_sha256": hashlib.sha256(actual_path.read_bytes()).hexdigest(),
                   "prediction_sha256": hashlib.sha256(predicted_path.read_bytes()).hexdigest()}
            for measured_name, predicted_name, _ in METRICS:
                row[measured_name + "_measured"] = actual[measured_name]
                row[measured_name + "_predicted"] = predicted[predicted_name]
                row[measured_name + "_error_pct"] = 100 * (predicted[predicted_name] / actual[measured_name] - 1)
            rows.append(row)
    averages = {scope: {name: statistics.mean(abs(row[name + "_error_pct"]) for row in rows if row["scope"] == scope)
                        for name, _, _ in METRICS} for scope in ("isolated_kernels", "runtime_layers")}
    paired = json.loads((VALIDATION / "paired-forward-comparison.json").read_text())
    ingress = json.loads((VALIDATION / "ingress-prefill-diagnosis.json").read_text())
    external = json.loads((VALIDATION / "external-ingress/prediction/prediction.json").read_text())
    long_row = next(row for row in rows if row["case"] == "holdout-12288-4096-c4" and row["scope"] == "runtime_layers")
    external_metrics = {name: {"measured": long_row[name + "_measured"], "predicted": external[predicted_name],
                               "error_pct": 100 * (external[predicted_name] / long_row[name + "_measured"] - 1)}
                        for name, predicted_name, _ in METRICS}
    summary = {"scope": "Independent local aggregated validation; not arbitrary PD/topology accuracy acceptance",
               "rows": rows, "case_equal_weight_mape_pct": averages,
               "conditional_external_arrivals_12k4k": external_metrics,
               "tests": {"rust_passed": 2135, "rust_ignored": 1, "python_public_and_both_parity_passed": 528},
               "operator_tables_changed": False, "new_forward_residual": False,
               "acceptance": {"all_metrics_below_20_pct": False, "ttft_requires_further_work": True}}
    (VALIDATION / "serving-comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (VALIDATION / "serving-comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 910C GLM-5.2 W8A8：长上下文外推、独立实测与剩余误差（2026-11-04）", "",
             "本轮完成了受约束的长上下文外推，并在本机 npu1-20 实测验证。12K/4K 已能使用 runtime_layers 完整预测；没有用本轮实测结果修改性能表，也没有增加按 workload 拟合的残差。**任意 PD 配比、任意并行策略且各指标平均误差低于 20% 的总目标尚未完成。**", "",
             "## 配置和验证资格", "",
             "所有主表数据都是单机合并部署，16 个可见 NPU，global TP16 / attention DP4 / EP16；AISimulate 的 attention-local tensor=4。关闭 prefill/decode graph、overlap 调度和投机解码；保留实际启用的 shared-expert multistream，它和 overlap 调度是不同开关。SGLang 0.5.22，固定镜像 digest `5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443`；模型 `/data_lib/data/models/glm-q`。", "",
             "对齐 page=128、每 rank 254 页、全局 chunk=16384（每 attention DP rank=4096）、每 rank max_sequences=32（服务全局 128）、max_prefill_tokens=32768、关闭 radix cache、global LM head。每组 warmup=2，正式请求共 20 个，全部完成且 I/O token 数与目标相符。输入 random-ids，客户端 `--tokenize-prompt`，seed=974。完整启动参数、实际 server-info、请求结果和逐轮记录均保留在 evidence。采集结束后已停止本轮诊断容器。", "",
             "K=isolated_kernels；L=runtime_layers，P 仍是独立算子组成。两种模式均显式运行，主表固定使用 L，K 单独列出；不按误差挑结果。外推规则在这批验证结果产生前确定。本轮数据没有写入算子或层性能表。误差定义 `(预测/实测-1)×100%`；MAPE 对三组 workload 等权平均，并对各指标分别验收。", "",
             "## 同条件完整请求对照：L", "",
             "| I/O | 并发/请求数 | TTFT ms：实测 / 预测（误差） | TPOT ms：实测 / 预测（误差） | 输入 tok/s：实测 / 预测（误差） | 输出 tok/s：实测 / 预测（误差） |",
             "|---|---|---|---|---|---|"]

    def format_row(row):
        cells = [f"{row['input_tokens']}/{row['output_tokens']}", f"{row['concurrency']}/{row['requests']}"]
        for name, _, _ in METRICS:
            cells.append(f"{row[name + '_measured']:.3f} / {row[name + '_predicted']:.3f}（{row[name + '_error_pct']:+.2f}%）")
        return "| " + " | ".join(cells) + " |"

    lines.extend(format_row(row) for row in rows if row["scope"] == "runtime_layers")
    means = averages["runtime_layers"]
    lines.extend(["", f"三组 MAPE：TTFT **{means['mean_ttft_ms']:.2f}%**，TPOT **{means['mean_tpot_ms']:.2f}%**，输入/输出吞吐 **{means['output_throughput']:.2f}%**。TPOT 和吞吐达到了这三组的 20% 门槛，TTFT 没有。不能把指标混在一起平均来掩盖 TTFT 的误差。", "",
                  "## 同条件完整请求对照：K", "",
                  "| I/O | 并发/请求数 | TTFT ms：实测 / 预测（误差） | TPOT ms：实测 / 预测（误差） | 输入 tok/s：实测 / 预测（误差） | 输出 tok/s：实测 / 预测（误差） |",
                  "|---|---|---|---|---|---|"])
    lines.extend(format_row(row) for row in rows if row["scope"] == "isolated_kernels")
    lines.extend(["", "K 的解码低估仍约 50%，说明纯 kernel 组成尚未完整覆盖 eager 调用成本。本轮的长上下文修复没有把 K 变准确，也没有声称已经支持所有并行拓扑的准确外推。", "",
                  "## 真实 batch 的单轮验证", "",
                  "此表输入实测逐轮 batch，仅用于隔离算子/层预测误差，不等同于完整请求的自主调度预测。只比较完整的四个 DP rank 记录、固定并发、至少生成 16 token 的稳定 decode 轮。最后一轮日志存在一个 rank 未刷出的记录，未用于完整轮统计。", "",
                  "| workload | 模式 | 完整轮数 | 实测平均 ms | 预测平均 ms | 逐轮 MAPE |", "|---|---|---:|---:|---:|---:|"])
    for row in paired["results"]:
        lines.append(f"| {row['case']} | {row['timing_scope']} | {row['forward_groups']} | {row['measured_mean_ms']:.3f} | {row['predicted_mean_ms']:.3f} | {row['mape_pct']:.2f}% |")
    lines.extend(["", "## 本轮修改了什么", "",
                  "以前 L 在 active DP4 等场景下的 KV 覆盖止于约 10K；即便底层注意力表能够计算 12K，整层表仍拒绝查询。现在 `shape_policy: anchored` 在相同请求数范围、活跃 DP、TP padding、并行拓扑和量化身份下，取八类层共同覆盖的最长 KV anchor，用 Rust 算子模型计算 `实测层(anchor) + 注意力(new KV) - 注意力(anchor)`。增量严格限定为 78 次 sparse attention 与 21 次 indexer；投影、MoE、通信和 host 调用成本不随 KV 长度整体放大，不重复计入层内部 kernel。所有这种外推标为 `estimated`。", "",
                  "只允许向更长 KV 外推，anchor 必须不小于 DSA top-k=2048；没有 request-count bracket、跨 TP padding bucket、跨 topology、缺少对应 attention 数据时仍报错。`shape_policy: measured` 保留原有严格检查，覆盖范围内的数值不变。单节点并不能验证 TP32 跨节点通信，L 的 TP16/ADP4/EP16/DCP1 限制保留。45 个标准 API 查询中 38 个成功，7 个 strict measured 查询按预期拒绝；anchored 的 30 个查询全部完成，包含 12K/16K/32K。32K 这里只证明查询可用，不是实测精度验收。", "",
                  "新增分段诊断在 8K-C4、12K-C4、8K-C8 上检查全部 16 rank、八种层类型，分别运行 control/host/events/control。结果显示 event 插桩可使组合层耗时增加约 23–25%；因此没有把这些被扰动的分段时间导入性能数据库。纯算子外的主机调用、流管理和层边界仍要继续拆分，不能将 host CPU 时间直接加到已经含主机间隙的 NPU event 耗时上。", "",
                  "## TTFT 为何仍偏低", "",
                  "本轮证据把剩余差距拆成两个部分。首先，合成 concurrency 负载将第一批请求放在同一虚拟时刻；实际客户端和 API/IPC 处理使请求陆续到达。在 12K/4K 的四个正式请求中，外部发送间隔约为 0、12.63、14.32、15.71 ms，而后三个请求直到约 1469 ms 才被调度器接收：无 overlap 的 scheduler 正在运行第一轮 prefill。于是实际出现一轮仅一个 DP 活跃，随后其他 DP 加入；同一时刻全部进入的离线负载少了一段排队。其次，在实际相同 prefill/mixed batch 上，预测仍有约 12–14% 的 MAPE，多活跃 DP 的 normal MoE 路由负载与运行时边界仍不完整。", "",
                  "| workload | P/mixed 单轮 MAPE | 外部发送到 scheduler 接收的平均间隔 ms |", "|---|---:|---:|"])
    for case in ingress["cases"]:
        delay = statistics.mean(row["send_to_receive_ms"] for row in case["requests"])
        lines.append(f"| {case['case']} | {case['prefill_and_mixed_mape_pct']:.2f}% | {delay:.3f} |")
    lines.extend(["", "为验证请求到达过程的贡献，另外只把 12K/4K 的四个 tokenizer 外部发送时间作为 trace 输入，仍由模拟器自行决定调度、batch 和耗时；没有注入 scheduler 接收时间、DP 路由、实测 forward 耗时或整轮残差。结果如下，**这是有条件的到达轨迹回放，不是仅凭并发=4 的通用预测保证**：", "",
                  "| 指标 | 实测 | 到达轨迹回放预测 | 误差 |", "|---|---:|---:|---:|"])
    for name, _, label in METRICS:
        row = external_metrics[name]
        lines.append(f"| {label} | {row['measured']:.3f} | {row['predicted']:.3f} | {row['error_pct']:+.2f}% |")
    lines.extend(["", "## 后续缺口与可复现材料", "",
                  "下一步优先做客户端/API/IPC 入队成本的独立测量与显式模型，再补齐 normal MoE 的实际 expert-count 分布和 host/stream 边界；继续用未参与建表的 workload 验证。不同 PD 配比、TP/ADP/EP 的搜索能力与精度资格必须分别记录：本轮单机结果不能替代两节点/四节点 PD 验证。原先 39 组历史对照保持不变，本次也未用新观测重拟合它们。", "",
                  "代码检查：Rust 2135 passed、1 ignored；Python canonical API/model/单一算子真值契约及两套 parity 共 528 passed；原有 golden 未刷新。性能表未修改，无新增残差参数。运行镜像 `aisimulate-910c:context-transfer-20261104`，必须显式选择该镜像；默认旧镜像标签没有偷偷改指向。", "",
                  "材料位于 `benchmarks/evidence/ascend910c-glm52-w8a8/production/prediction-repair-20261102/`：`component-boundaries-node20-20261103/` 保存实测与诊断；`context-transfer-validation-20261103/` 保存构建、测试、标准 API 覆盖、两模式完整预测、CSV/JSON 对照、逐轮比较和外部到达回放。原始大日志使用无损 gzip，SHA256 与来源清单一并保存。", "",
                  "本阶段没有宣告总 goal 完成。当前可验证结论是：相同 TP16/ADP4/EP16 的 12K/4K 长上下文预测已打通，三组 L 的 TPOT/吞吐误差较小，TTFT 的主要额外缺口已定位到请求进入调度的时序和多 DP prefill 成本。", ""])
    report = ROOT.parents[4] / "md/910c-glm52-context-transfer-validation-20261104.md"
    report.write_text("\n".join(lines))
    print(json.dumps({"report": str(report), "case_equal_weight_mape_pct": averages}, ensure_ascii=False))


if __name__ == "__main__":
    main()
