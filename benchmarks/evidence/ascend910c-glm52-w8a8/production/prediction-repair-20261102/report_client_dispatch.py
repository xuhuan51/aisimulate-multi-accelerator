import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


METRICS = (("ttft_ms", "mean_ttft_ms", "mean_ttft_ms"),
           ("tpot_ms", "mean_tpot_ms", "mean_tpot_ms"),
           ("input_tok_s", "input_throughput", "input_throughput_tok_s"),
           ("output_tok_s", "output_throughput", "output_throughput_tok_s"))


def compare(measured, prediction, count, input_length, output_length):
    expected = {"completed_requests": count, "total_input_tokens": count * input_length,
                "total_output_tokens": count * output_length,
                "failed_requests": 0, "canceled_requests": 0, "rejected_requests": 0}
    if any(prediction[key] != value for key, value in expected.items()):
        raise ValueError("Prediction request/token conservation failed")
    result = {}
    for name, actual_key, predicted_key in METRICS:
        result[name + "_measured"] = measured[actual_key]
        result[name + "_predicted"] = prediction[predicted_key]
        result[name + "_error_pct"] = 100 * (prediction[predicted_key] / measured[actual_key] - 1)
    return result


def averages(rows):
    return {name: statistics.mean(abs(row[name + "_error_pct"]) for row in rows)
            for name, _, _ in METRICS}


def cell(row, metric):
    return f"{row[metric + '_measured']:.2f} / {row[metric + '_predicted']:.2f} ({row[metric + '_error_pct']:+.2f}%)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--pd", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    local_manifest = json.loads((args.local / "manifest.json").read_text())
    pd_manifest = json.loads((args.pd / "manifest.json").read_text())
    profile = json.loads(args.profile.read_text())
    rows = []
    for job in local_manifest["cases"]:
        if job["returncode"]:
            raise ValueError(f"Local replay failed: {job['folder']}")
        campaign = Path(job["campaign"])
        manifest = json.loads((campaign / "context-holdouts.json").read_text())
        case = next(case for case in manifest["cases"] if case["id"] == job["case"])
        source = campaign / case["id"] / "requests.jsonl"
        actual = json.loads(source.read_text().splitlines()[-1])
        path = Path(job["folder"]) / "prediction/prediction.json"
        predicted = json.loads(path.read_text())
        fresh = campaign.name == "ingress-validation-node20-20261104"
        rows.append({"group": "本轮单机" if fresh else "前轮单机", "case": case["id"],
                     "mode": job["mode"], "scope": "L", "input_length": case["input_length"],
                     "output_length": case["output_length"], "concurrency": case["concurrency"],
                     "requests": case["prompts"], "nodes": 1, "parallelism": "16/4/16",
                     "qualification": "独立 CPU 成本模型下的完整并发预测；无实测到达注入",
                     "prediction_path": str(path), "measurement_path": str(source),
                     "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     **compare(actual, predicted, case["prompts"], case["input_length"], case["output_length"])})
    for record in pd_manifest["records"]:
        history = record["historical"]
        number = record["number"]
        group = "A·4节点2P1D" if number <= 6 else "B·4节点1P3D" if number <= 12 else (
            "C·4节点2P2D" if number <= 18 else "D·4节点3P1D" if number <= 23 else "E·2节点1P1D")
        measured = {actual_key: history["measured_" + name] for name, actual_key, _ in METRICS}
        spec = history["configuration_spec"]
        parallel = "; ".join("/".join(str(spec[role + "_" + key + "_SIZE"]) for key in ("TP", "DP", "EP"))
                             for role in ("PREFILL", "DECODE"))
        for mode, key in record["jobs"].items():
            job = pd_manifest["jobs"][key]
            if job["status"] != "completed":
                raise ValueError(f"PD replay failed: {job['folder']}")
            path = Path(job["folder"]) / "prediction/prediction.json"
            predicted = json.loads(path.read_text())
            rows.append({"group": group, "case": str(number).zfill(2), "mode": mode,
                         "scope": "K" if record["scope"] == "isolated_kernels" else "L",
                         "input_length": history["input_length"], "output_length": history["output_length"],
                         "concurrency": history["concurrency"], "requests": history["requests"],
                         "nodes": 4 if number <= 23 else 2, "parallelism": parallel,
                         "kv_pages": history["observed_blocks_per_rank"],
                         "qualification": "历史回归；非新盲测；保留固定拓扑选用 K/L 的既有规则",
                         "prediction_path": str(path), "measurement_path": history["measurement_path"],
                         "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         **compare(measured, predicted, history["requests"], history["input_length"], history["output_length"])})
    groups = list(dict.fromkeys(row["group"] for row in rows))
    grouped = {group: {mode: averages([row for row in rows if row["group"] == group and row["mode"] == mode])
                      for mode in ("instantaneous", "profiled_client")} for group in groups}
    result = {"rows": rows, "group_mape_pct": grouped,
              "acceptance_by_group": {group: all(value < 20 for value in metrics["profiled_client"].values())
                                      for group, metrics in grouped.items()},
              "universal_accuracy_claim": False,
              "scope": "Per-case MAPE evaluated separately for TTFT, TPOT, input and output throughput; groups are not pooled to hide a failing topology"}
    (args.output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    cost = profile["client_dispatch"]
    lines = ["# 910C / GLM-5.2 W8A8：客户端发送节奏修复与同配置验证", "", "日期：2026-11-04。",
             "", "本轮增加的是闭环负载中的客户端请求准备队列：并发上限仍然由请求完成后补位控制，请求之间的发送间隔由独立 CPU 微测确定。没有加入整轮 forward 残差，也没有把真实请求到达时间、路由、batch 或 forward 耗时输入预测。所有预测统一显式使用当前修复版本；K/L 选择沿用事先按拓扑确定的规则，不按结果误差挑选。",
             "", "## 实现与来源", "",
             "实际 SGLang benchmark 在单个 asyncio 客户端逐个构造和序列化请求；关闭 overlap 后，后来到达的请求可能错过正在启动的一轮 prefill，毫秒级发送差异会变成一轮 forward 的等待。原来的理想并发负载在零时刻同时发出首批请求，容易把更多 DP rank 合并到首轮。因此本次在 Rust WorkloadDriver 增加一个跨所有 serving replica 共享的客户端发送通道，按照上一条请求的输入长度占用准备时间；空闲时不额外等待，完成补位、多轮会话思考时间和并发上限仍保持原规则。该参数贯穿 predict、recommend、候选配置保存和重新加载。不开启时保留原来的理想同时到达语义。算子性能表、KV 容量和 eager/graph/overlap 设置不随误差调整。",
             "", f"CPU profile 使用相同固定 SGLang 源码的客户端调用，连接单独的模拟 HTTP 进程，无模型 forward。用 512/2048/8192/32768 输入长度的准备间隔拟合后固定：`base_ms={cost['base_ms']:.12f}`，`per_input_token_us={cost['per_input_token_us']:.12f}`。4096/12288/16384 是 CPU 曲线验证点；其误差依次为 " + ", ".join(f"{row['error_pct']:+.3f}%" for row in profile["heldout_cpu_lengths"]) + "。这组参数只代表本机和本次客户端实现，不能直接承诺其他客户端、HTTP 格式或繁忙程度相同。",
             "", "## 分组误差", "", "MAPE 对每条配置等权，四项指标分别计算。箭头为理想同时发送 → 加入独立 CPU 准备成本。", "",
             "| 组别 | TTFT MAPE | TPOT MAPE | 输入吞吐 MAPE | 输出吞吐 MAPE | 四项均低于20% |",
             "|---|---:|---:|---:|---:|---|"]
    for group in groups:
        metrics = grouped[group]
        cells = [f"{metrics['instantaneous'][name]:.2f}% → {metrics['profiled_client'][name]:.2f}%" for name, _, _ in METRICS]
        lines.append("| " + " | ".join([group, *cells, "是，仅限本组数据" if result["acceptance_by_group"][group] else "否，继续修复"]) + " |")
    lines += ["", "## 单机完整请求预测", "", "全局 TP16/ADP4/EP16、attention local TP4、PP1/DCP1，KV 每 rank 254 页、页大小128。graph/overlap/speculation/radix cache 关闭；保留真实 shared-expert 多 stream。每个 workload 两条 warmup，表中仅正式请求。本轮 48 条、前轮 20 条均成功；前轮含12K/4K长输出。", "",
              "| 组别/I/O | C/N | TTFT ms：实/预（误差） | TPOT ms：实/预（误差） | 输入 tok/s：实/预（误差） | 输出 tok/s：实/预（误差） |",
              "|---|---|---|---|---|---|"]
    for row in rows:
        if "单机" in row["group"] and row["mode"] == "profiled_client":
            lines.append("| " + " | ".join([f"{row['group']} {row['input_length']}/{row['output_length']}",
                                             f"{row['concurrency']}/{row['requests']}", *[cell(row, metric) for metric, _, _ in METRICS]]) + " |")
    lines += ["", "## 历史两节点/四节点39组同配置回归", "", "C/N 为并发/请求数；P、D 的并行参数均按全局 TP/ADP/EP 表示。A 是两个独立 P 实例和一个跨两节点 D 实例。K 为 isolated_kernels，L 为 runtime_layers；P 仍由独立算子组成。实测容量逐配置保留，预测使用原报告每角色的保守最小每-rank页数。原始实测、配置和压缩证据均先验证 SHA256，再重跑。", "",
              "| 编号/组别 | I/O | C/N | P；D 并行 | 页数 P/D | 模式 | TTFT ms：实/预（误差） | TPOT ms：实/预（误差） | 输入 tok/s：实/预（误差） | 输出 tok/s：实/预（误差） |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        if "单机" not in row["group"] and row["mode"] == "profiled_client":
            pages = row["kv_pages"]
            lines.append("| " + " | ".join([row['case'] + "·" + row['group'], f"{row['input_length']}/{row['output_length']}",
                                             f"{row['concurrency']}/{row['requests']}", row['parallelism'],
                                             f"{pages['prefill']} / {pages['decode']}", row['scope'],
                                             *[cell(row, metric) for metric, _, _ in METRICS]]) + " |")
    lines += ["", "## 边界与下一步", "",
              "不能把这些结果理解成任意 workload、任意并行配置均已达标。客户端模型目前没有 HTTP 多进程重排、网络抖动和 IPC 轮询延迟；prefill 的 normal MoE 仍有真实 expert 分布与均衡 profile 的差异；TP32 eager 的细粒度运行时覆盖与独立验证仍需补齐。某个分组未达20%时，不能用其他分组的低误差把它平均掉后宣称通过。历史39组是回归材料，不是新盲测；新采集的5组用于检验此次独立 CPU 成本模型的完整闭环效果。",
              "", "本轮未启动 PD 服务重测历史39组；只在本机启动了独立 aggregated 诊断实例，完成后已停止。独立微测及正式实测期间有固定 CPU 范围内的构建任务，执行信息保留在证据中；如需严格隔离的置信区间，还需要重复采样。",
              "", "使用新的完整 YAML，在 `traffic.load` 中增加下列配置，`recommend` 使用同一字段并保留到候选配置；执行镜像须显式选择 `aisimulate-910c:client-dispatch-20261104`。", "", "```yaml",
              "load:", "  type: concurrency", "  concurrency: 4", "  client_dispatch:",
              f"    base_ms: {cost['base_ms']}", f"    per_input_token_us: {cost['per_input_token_us']}", "```", "",
              "详细 CSV、JSON、每组预测配置、逐请求输出和固定 CPU profile 位于本报告对应的 `client-dispatch-validation-20261104` 证据目录。"]
    if args.validation:
        validation = json.loads(args.validation.read_text())
        lines += ["", "## 实现验证与可复现入口", "",
                  f"执行镜像 `{validation['runtime_image']}`，镜像 ID `{validation['runtime_image_id']}`；wheel SHA256 `{validation['wheel_sha256']}`。",
                  "", "Rust 全库测试 2138 通过、1 忽略；新增客户端闭环功能测试6项通过。Python 扩展回归1181项通过、4项失败，4项在修改前的 context-transfer 镜像上均复现，保留双方日志：两项 idle-tail KV admission 断言、一项 detail 输出字段、一项 recommend 旧 fixture 缺少字段。两个 Rust/Python parity 测试集包含在通过项中，未刷新 golden 来消除差异。",
                  "", "已实际运行单机 predict 示例，以及最多4节点预算的 PD 搜索示例：30个可行候选全部完成，客户端参数在30份保存配置中均保留。首选1P3D、每实例 TP16/ADP4/EP16、C32/N64 的 TTFT、TPOT、输入输出吞吐和完成请求数与重新加载 YAML 的 predict 结果一致（浮点容差1e-12）。这只是给定候选范围内的搜索与保存验证，尚不能证明跨拓扑精度或全局最优。",
                  "", "完整入口见 `examples/cli/ascend910c-glm52-eager-client.yaml`、`examples/cli/ascend910c-glm52-pd-client-search.yaml` 和使用说明。CPU容器临时目录改为独立 tmpfs，避免系统盘满时非 root 用户无法创建配置；首次失败日志也保留，没有覆盖。"]
    args.document.write_text("\n".join(lines) + "\n")
    print(json.dumps({"group_mape_pct": grouped, "acceptance_by_group": result["acceptance_by_group"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
