import argparse
import csv
import hashlib
import json
import pathlib
import statistics
import tarfile

from analyze_serving_diagnostics import analyze_matrix


HERE = pathlib.Path(__file__).resolve().parent
LABELS = {
    "recommended-2p1d-dtp32": "2P1D：P TP16/ADP4/EP16，D 跨两节点 TP32/ADP32/EP32",
    "control-1p3d-tp16": "1P3D：P/D TP16/ADP4/EP16",
    "control-2p2d-tp16": "2P2D：P/D TP16/ADP4/EP16",
    "control-3p1d-tp16": "3P1D：P/D TP16/ADP4/EP16",
}
TABLE_LABELS = {
    "recommended-2p1d-dtp32": "2P1D（D TP32）",
    "control-1p3d-tp16": "1P3D（D TP16）",
    "control-2p2d-tp16": "2P2D（D TP16）",
    "control-3p1d-tp16": "3P1D（D TP16）",
}


def comparison_table(plan, indexed, measurements):
    lines = [
        "每个性能单元格依次为 **实测 / 原预测 / 相对误差**；TTFT 使用秒，TPOT 使用毫秒，吞吐使用 token/s。正误差表示预测偏大，负误差表示预测偏小。running batch 只列服务端实测采样值，不能与客户端设置的并发混为一谈。", "",
        "| 配置 | 并发 C / 请求数 N | TTFT 秒：实测 / 预测 / 误差 | TPOT ms：实测 / 预测 / 误差 | 输入 tok/s：实测 / 预测 / 误差 | 输出 tok/s：实测 / 预测 / 误差 | 实测活跃 D running 峰值 / 非零均值 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    metrics = (("ttft_ms", 1000, 3), ("tpot_ms", 1, 2), ("input_tok_s", 1, 2), ("output_tok_s", 1, 2))
    for configuration in plan["configurations"]:
        label = configuration["label"]
        for concurrency in plan["workload"]["concurrencies"]:
            row = indexed[(label, concurrency, "stock")]
            measured = measurements[(label, concurrency)]
            cells = [TABLE_LABELS[label], f"{concurrency} / {row['requests']}"]
            for metric, scale, precision in metrics:
                cells.append(
                    f"{row[f'measured_{metric}'] / scale:.{precision}f} / "
                    f"{row[f'predicted_{metric}'] / scale:.{precision}f} / "
                    f"{row[f'error_percent_{metric}']:+.1f}%"
                )
            cells.append(f"{float(measured['decode_running_peak']):.0f} / {float(measured['decode_running_mean_when_nonzero']):.2f}")
            lines.append("| " + " | ".join(cells) + " |")
    return lines + [""]


def archive_measurement(configuration, source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / f"{configuration}.tar.gz"
    manifest_path = destination / f"{configuration}-manifest.json"
    if archive_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if hashlib.sha256(archive_path.read_bytes()).hexdigest() != manifest["archive_sha256"]:
            raise ValueError(f"Archive digest mismatch: {archive_path}")
        return
    files = []
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            include = path.name != "metrics.jsonl.gz"
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            files.append({"path": str(relative), "bytes": path.stat().st_size, "sha256": digest.hexdigest(), "in_archive": include})
            if include:
                archive.add(path, arcname=str(relative), recursive=False)
    manifest_path.write_text(json.dumps({
        "source_directory": str(source), "configuration": configuration,
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "raw_metrics_policy": "Unmodified per-second metrics.jsonl.gz retained at source_directory; digest and size recorded here. All benchmark request traces, summaries and archived worker logs included.",
        "files": files,
    }, indent=2) + "\n")


def write_report(measurement_root, prediction_root, output):
    plan = json.loads((measurement_root / "comparison-plan.json").read_text())
    comparisons = json.loads((prediction_root / "comparison.json").read_text())["rows"]
    with (measurement_root / "matrix-summary.csv").open() as stream:
        measurements = {(row["configuration"].removesuffix("-20261101"), int(row["point"].split("-")[0][1:])): row for row in csv.DictReader(stream)}
    indexed = {(row["configuration"], row["concurrency"], row["mode"]): row for row in comparisons}
    expected = len(plan["configurations"]) * len(plan["workload"]["concurrencies"])
    if len(measurements) != expected or len(indexed) != expected * 2:
        raise ValueError("Incomplete measurement/prediction matrix")
    lines = [
        "# 四节点 GLM5.2 W8A8：8K/1K 实测与算子级预测对照", "",
        "本次比较固定的四组候选配置，不构成所有并行方案的全局最优性证明。", "",
        "每档输入 8192、输出 1024 token；并发 C=4/8/16/32/64/128；正式请求数 N=2C；预热 2 条，预热输出由镜像基准脚本限制为 32 token且不计入正式结果。随机种子 910、token-ID 请求、ignore-EOS、流式响应。每个点只测一轮，包含启动填充与末尾排空，不能当成长时间稳态吞吐。", "",
        "服务使用固定 digest 的 SGLang 0.5.22/CANN 9.1.0 A3；关闭 graph、调度 overlap、投机解码和前缀缓存；保留共享专家多流执行。P/D 的 HCCL_BUFFSIZE 均为 650。", "",
        "预测走 AISimulate 原有 predict --stack engine --capture-per-request，使用同一候选、并发和 N=2C。stock 保留原推荐的内存推算；observed-kv 仅替换为实际 KV pool 容量，是定位内存/准入影响的诊断，不是拟合后的新推荐。误差定义为 100×(预测/实测−1)。正值表示预测值偏大。", "",
        "已知执行路径差异：当前服务 P 开启 enable_dp_lm_head，D 关闭；D 实际 LM head 使用全局 TP，而冻结的 AISimulate Ascend 模型按 attention TP 计算 LM head/Logits gather。推荐组 D 为实际 TP32 对模型 TP1，并开启有条件的 TP LM-head all-to-all；对照组 D 为实际 TP16 对模型 TP4，关闭此 all-to-all 选项。开启选项不代表所有 batch 都满足该分支条件。下面是原推荐的误差基线，虽然 PD/TP/attention-DP/EP 和请求口径一致，尚未做到所有算子路径完全一致。原始 server-info 与 comparison-jobs.json 中的 logits_layout_audit 可核验；本次不在采集中途改服务路径或拟合计时。", "",
        "## 实测与原推荐预测", "",
    ]
    rankings = []
    recovery_path = measurement_root / "bootstrap-timeout-recovery.json"
    if recovery_path.exists():
        recovery = json.loads(recovery_path.read_text())
        previous, current = recovery["previous_limits"], recovery["new_limits"]
        note = f"等待时限变更：1P3D C128 首次尝试触发 PD bootstrap {previous['PD_BOOTSTRAP_TIMEOUT']} 秒超时，已中止并保留于 failed-attempts，不计入下表。重启后的 C128 重测及后续配置使用 bootstrap {current['PD_BOOTSTRAP_TIMEOUT']} 秒、Router request {current['ROUTER_REQUEST_TIMEOUT']} 秒；此前成功点使用原 bootstrap {previous['PD_BOOTSTRAP_TIMEOUT']} 秒、Router request {previous['ROUTER_REQUEST_TIMEOUT']} 秒。并行参数和执行开关保持各候选定义，延长等待不代表提高计算吞吐；这些结果不能被解释为原超时策略下全部成功。"
        lines[-2:-2] = [note, ""]
    lines.extend([f"- {LABELS[configuration['label']]}。" for configuration in plan["configurations"]])
    lines.extend(["", "2P1D 的 D 是跨两台物理节点的一个实例；四种配置都占用四台节点。attention DP 简写为 ADP。", ""])
    lines.extend(comparison_table(plan, indexed, measurements))
    lines.extend([
        "输入吞吐与输出吞吐都使用各自 token 总量除以正式测量时长；由于固定 8192/1024，本次输入吞吐恰为输出吞吐的 8 倍。完整 TTFT/TPOT 百分位、输入吞吐、请求明细与指标摘要见归档和 comparison.csv。", "",
        "running batch 是每秒采样近似值：按 D 实例和 attention-DP rank 去掉 TP 重复指标，再仅统计该采样间隔 decode token counter 有增长的 rank，避免空闲 rank 残留 gauge 被误当成并发；不是逐轮 forward 的精确 trace。不同 rank 的刷新时刻不一致，且可能包含系统探活，采样峰值可能略高于客户端并发。原始 gauge 与未修改的逐秒数据均保留，不按客户端上限强行截断。", "",
        "## 候选集合内的吞吐排序", "",
        "这里以同一并发下的输出吞吐最大为选择标准；没有隐含 TTFT/TPOT SLA。若有延迟限制，必须重新过滤候选。", "",
        "| 并发 | 实测最佳 | 原预测选择 | 原预测所选配置的实测吞吐 | 实测最佳吞吐 | 选择损失 |",
        "|---|---|---|---:|---:|---:|",
    ])
    for concurrency in plan["workload"]["concurrencies"]:
        rows = [row for row in comparisons if row["concurrency"] == concurrency and row["mode"] == "stock"]
        actual_best = max(rows, key=lambda row: row["measured_output_tok_s"])
        selected = max(rows, key=lambda row: row["predicted_output_tok_s"])
        regret = 100 * (1 - selected["measured_output_tok_s"] / actual_best["measured_output_tok_s"])
        rankings.append({"concurrency": concurrency, "measured_best": actual_best["configuration"], "predicted_choice": selected["configuration"], "selected_measured_tok_s": selected["measured_output_tok_s"], "best_measured_tok_s": actual_best["measured_output_tok_s"], "selection_loss_percent": regret})
        lines.append(f"| {concurrency} | {actual_best['configuration']} | {selected['configuration']} | {selected['measured_output_tok_s']:.2f} | {actual_best['measured_output_tok_s']:.2f} | {regret:.1f}% |")
    lines.extend(["", "## 实际 KV 容量诊断", "", "下表为所有 24 个点的未加权平均绝对百分比误差。实际容量取同一角色各副本容量的最小值；副本容量不完全一致时，这是保守近似。", "", "源码核对：EP 模式下共享专家权重已经完整计入每个 rank；运行时/通信开销仍采用固定系统常数，未按 HCCL 缓冲区与 DeepEP 布局分别建模。已知 LM-head 分片差异还会影响驻留权重。不能把全部容量偏差归因于漏计某个缓冲区，或在未核对分配阶段和 mem_fraction_static 预留空间时重复扣减。证据见 memory-capacity-source-audit.json。", "", "| 口径 | TTFT MAPE | TPOT MAPE | 输出吞吐 MAPE |", "|---|---:|---:|---:|"])
    for mode in ("stock", "observed-kv"):
        selected = [row for row in comparisons if row["mode"] == mode]
        errors = [statistics.mean(abs(row[f"error_percent_{metric}"]) for row in selected) for metric in ("ttft_ms", "tpot_ms", "output_tok_s")]
        lines.append(f"| {mode} | {errors[0]:.1f}% | {errors[1]:.1f}% | {errors[2]:.1f}% |")
    diagnostics = analyze_matrix(measurement_root, prediction_root / "serving-stage-diagnostics.json")
    lines.extend([
        "", "## 排队与调度阶段诊断", "",
        "下表来自每秒采样的近似正式窗口。队列按实例/attention-DP rank 对 TP 重复项取最大值后求和；KV列取所有 D rank 上报比例的最大值。run_batch 占比是稳定计数器窗口内的调度墙钟时间占比，包含 CPU 提交、设备执行和同步，不能解释成纯 NPU kernel 时间。汇总包含该角色的空闲实例，不等于单个请求关键路径占比。缺失指标显示 —，不当作零。", "",
        "| 配置 | 并发 | D 预分配队列峰值 | D 传输队列峰值 | D 单 rank KV 使用率峰值 | D run_batch 占比 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for diagnostic in diagnostics:
        gauges = {row["metric"]: row for row in diagnostic["gauges"] if row["role"] == "decode"}
        cells = []
        for name in ("sglang:num_decode_prealloc_queue_reqs", "sglang:num_decode_transfer_queue_reqs"):
            value = gauges.get(name)
            cells.append(f"{value['peak']:.0f}" if value is not None else "—")
        usage = gauges.get("sglang:token_usage")
        cells.append(f"{100 * usage['rank_peak']:.1f}%" if usage is not None else "—")
        run_percent = diagnostic["scheduler"].get("decode", {}).get("run_batch_percent")
        cells.append(f"{run_percent:.2f}%" if run_percent is not None else "—")
        lines.append(f"| {diagnostic['configuration']} | {diagnostic['concurrency']} | " + " | ".join(cells) + " |")
    lines.extend([
        "", "完整阶段增量、KV 传输/分配/引导与请求阶段直方图均值、有效 rank 数、缺失边界和计数器重置信息保存在 predictions/serving-stage-diagnostics.json。这些是服务级指标，未按本轮压测 request ID 过滤，可能包含探活等流量。直方图按 rank 观测加权，不是唯一正式请求数；该窗口也可能未覆盖最早或最后的指标刷新，不能将各均值相加当成客户端 TTFT。推荐组 C4–C64 使用早期采集字段，缺少新增阶段与 PD 队列指标；C128 和对照组使用扩展字段。", "",
    ])
    diagnostic_path = measurement_root / "recommended-2p1d-dtp32-20261101/mid-c64-scheduler-window.json"
    if diagnostic_path.exists():
        diagnostic = json.loads(diagnostic_path.read_text())
        stage_rows = [row for row in diagnostic["counter_deltas"] if row["name"] == "sglang:scheduler_stage_seconds_total"]
        totals = {category: statistics.mean(row["delta"] for row in stage_rows if row["category"] == category) for category in {row["category"] for row in stage_rows}}
        interval = diagnostic["last"][0]["time"] - diagnostic["first"][0]["time"]
        lines.extend([
            "", "## 单窗口调度诊断", "",
            f"推荐组 C64 的独立 {interval:.2f} 秒窗口中，32 个 Decode rank 的平均 run_batch 增量为 {totals['run_batch']:.3f} 秒，占已归类调度墙钟时间的 {100*totals['run_batch']/sum(totals.values()):.2f}%。原始首尾计数器和增量保留在 mid-c64-scheduler-window.json。",
            "这只描述该窗口：主要时间在 run_batch 内，应进一步核对 forward 内部的算子、通信、CPU 提交和同步开销，不能把全部 TPOT 偏差都归因于外层请求排队。run_batch 墙钟时间不等于 NPU kernel 总时间，也不能由这个窗口推断其他并发或配置的占比。",
        ])
    lines.extend(["", "## 保留的数据与解释范围", "", "- 每个点保留客户端原始 JSONL（含逐请求长度、TTFT、ITL、错误）、基准日志、服务运行参数、指标摘要；每组结束后归档各 Worker/Router 日志。", "- Git 中的 measurements/*.tar.gz 包含上述证据；体积较大的原始逐秒 metrics.jsonl.gz 保留在测量目录，manifest 记录绝对来源、大小和 SHA256。", "- predictions 目录保存预测原始结果、逐请求 trace、执行命令、冻结镜像 ID 和输入哈希；比较失败不会被算成成功采集点。", "- 随机 token workload、两波请求、RR 路由与一次重复的结论不能直接代表真实业务、长期稳态或统计置信区间。warmup 后服务路由游标不一定与空模拟器完全一致。", "- 对照只揭示误差，不自动证明误差来自调度或某个算子；需要结合队列、KV、forward/通信时间进一步区分。此次不改动计时系数来迎合实测。", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    with (output.parent / "ranking.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rankings[0]))
        writer.writeheader()
        writer.writerows(rankings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-root", type=pathlib.Path, required=True)
    parser.add_argument("--prediction-root", type=pathlib.Path)
    parser.add_argument("--archive-configuration", choices=list(LABELS))
    parser.add_argument("--output", type=pathlib.Path, default=HERE / "comparison-report.md")
    args = parser.parse_args()
    if args.archive_configuration:
        source = args.measurement_root / (args.archive_configuration + "-20261101")
        if sum(1 for path in source.glob("c*-n*/.complete")) != 6:
            raise ValueError("Archive only a completed six-point sweep")
        if not (source / "worker-logs").is_dir():
            raise ValueError("Worker logs have not been archived yet")
        archive_measurement(args.archive_configuration, source, HERE / "measurements")
    else:
        if args.prediction_root is None:
            parser.error("--prediction-root is required when writing the comparison report")
        write_report(args.measurement_root, args.prediction_root, args.output)


if __name__ == "__main__":
    main()
