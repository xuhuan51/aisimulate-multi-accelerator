import csv
import gzip
import hashlib
import json
import statistics
from pathlib import Path

from analyze_ingress_campaign import records
from report_request_timing import decode_occupancy


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]


def load(path):
    return json.loads(path.read_text())


def main():
    original = load(ROOT / "original-23-point-comparison.json")["rows"]
    previous = load(REPO / "aisimulate-output/four-node-request-timing-20261102/comparison.json")["rows"]
    current = load(REPO / "aisimulate-output/four-node-active-rank-20261102/comparison.json")["rows"]
    groups = {label: [row for row in rows if row["mode"] == "stock"] for label, rows in (("original", original), ("previous", previous), ("current", current))}
    if any(len(rows) != 23 for rows in groups.values()) or len(current) != 46:
        raise ValueError("Complete all matched original serving replays first")
    prior = {(row["configuration"], row["concurrency"]): row for row in groups["previous"]}
    for row in groups["current"]:
        old = prior[(row["configuration"], row["concurrency"])]
        for key in ("ttft_ms", "tpot_ms", "input_tok_s", "output_tok_s"):
            if row[f"measured_{key}"] != old[f"measured_{key}"]:
                raise ValueError("Original measurement changed")
    accuracy = {}
    for label, rows in groups.items():
        accuracy[label] = {}
        for metric in ("ttft_ms", "tpot_ms", "input_tok_s", "output_tok_s"):
            errors = [abs(row[f"error_percent_{metric}"]) for row in rows]
            accuracy[label][metric] = {"mape_percent": statistics.mean(errors), "max_absolute_error_percent": max(errors), "within15": sum(value <= 15 for value in errors), "points": len(errors)}
    independent = []
    for campaign, output in (("pd-ingress-1p1d", "pd-ingress-active-rank-20261102"), ("pd-cold-1p1d", "pd-cold-active-rank-20261102"), ("pd-phases-1p1d", "pd-phase-active-rank-20261102")):
        for row in load(REPO / "aisimulate-output" / output / "comparison.json")["rows"]:
            metrics = load(ROOT / campaign / row["case"] / "metrics-summary.json")
            row["campaign"] = campaign
            row["measured_running_peak"] = metrics["decode_running_peak"]
            row["measured_running_mean_when_nonzero"] = metrics["decode_running_mean_when_nonzero"]
            row["predicted_occupancy_proxy"] = decode_occupancy(REPO / "aisimulate-output" / output / row["case"] / "result/requests.jsonl")
            independent.append(row)
    if len(independent) != 9 or sum(row["requests"] for row in independent) != 82:
        raise ValueError("Incomplete independent PD controls")
    ranking = []
    for concurrency in (4, 8, 16, 32, 64, 128):
        candidates = [row for row in groups["current"] if row["concurrency"] == concurrency]
        measured = max(candidates, key=lambda row: row["measured_output_tok_s"])
        predicted = max(candidates, key=lambda row: row["predicted_output_tok_s"])
        ranking.append({"concurrency": concurrency, "measured_winner": measured["configuration"], "predicted_winner": predicted["configuration"], "completed_candidates": len(candidates), "agree": measured["configuration"] == predicted["configuration"]})
    result = {"scope": "Same pinned eager deployment; original23 serving points and9new PD controls were not used to fit the added active-rank term. Two previously held-out forward cases are explicitly reclassified as training. First-use KV allocation compilation and first nonempty text remain outside the steady model.", "accuracy": accuracy, "original23": groups["current"], "independent": independent, "same_concurrency_ranking": ranking}
    (ROOT / "active-rank-serving-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    with (ROOT / "active-rank-serving-comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(groups["current"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(groups["current"])
    lines = ["# 活跃 attention-DP 数量修复与同配置实测对照", "", "2026-11-02，固定GLM5.2 W8A8、SGLang0.5.22/CANN9.1镜像、8192/1024原始工作负载，graph/调度overlap/投机解码关闭。原23点1760条正式请求没有重测、修改或参与本次参数拟合；46次新回放包含默认内存和实测KV容量两种口径。汇总采用默认内存口径，与前一版一致。", "", "## 原23点汇总", "", "| 指标 | 原始MAPE | 上一版MAPE | 本版MAPE | 本版最大绝对误差 | 本版误差≤15% |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for metric, title in (("ttft_ms", "TTFT"), ("tpot_ms", "TPOT"), ("input_tok_s", "输入吞吐"), ("output_tok_s", "输出吞吐")):
        new = accuracy["current"][metric]
        lines.append(f"| {title} | {accuracy['original'][metric]['mape_percent']:.2f}% | {accuracy['previous'][metric]['mape_percent']:.2f}% | {new['mape_percent']:.2f}% | {new['max_absolute_error_percent']:.2f}% | {new['within15']}/23 |")
    lines.extend(["", "本轮主要改善TPOT和吞吐，TTFT没有整体达标：15/23点在15%内，仍有8点超标、最差约53%。上一版是16/23；更准确的Decode时长改变闭环请求到达及Prefill分批，个别TTFT反而变差。保留两版，不把吞吐改善说成所有指标都改善。", "", "| 配置 | 并发 | TTFT 实测/预测 ms | TTFT误差 | TPOT 实测/预测 ms | TPOT误差 | 输入吞吐 实测/预测 tok/s | 输出吞吐 实测/预测 tok/s | 吞吐误差 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in groups["current"]:
        lines.append(f"| {row['configuration']} | {row['concurrency']} | {row['measured_ttft_ms']:.2f}/{row['predicted_ttft_ms']:.2f} | {row['error_percent_ttft_ms']:+.2f}% | {row['measured_tpot_ms']:.2f}/{row['predicted_tpot_ms']:.2f} | {row['error_percent_tpot_ms']:+.2f}% | {row['measured_input_tok_s']:.3f}/{row['predicted_input_tok_s']:.3f} | {row['measured_output_tok_s']:.3f}/{row['predicted_output_tok_s']:.3f} | {row['error_percent_output_tok_s']:+.2f}% |")
    lines.extend(["", "## 独立1P1D验证", "", "P/D均TP16/ADP4/EP16，82条正式请求，仍为每组两个warmup。8K/128短输出只用于定位；validation-c4-8k1k为完整8K/1K。所有成功正式请求保留，C1冷请求不剔除。", "", "| case | C/N | TTFT 实测/预测 ms | TTFT误差 | TPOT 实测/预测 ms | TPOT误差 | 输出吞吐 实测/预测 tok/s | 吞吐误差 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in independent:
        lines.append(f"| {row['case']} | {row['concurrency']}/{row['requests']} | {row['measured_ttft_ms']:.2f}/{row['predicted_ttft_ms']:.2f} | {row['error_percent_ttft_ms']:+.2f}% | {row['measured_tpot_ms']:.2f}/{row['predicted_tpot_ms']:.2f} | {row['error_percent_tpot_ms']:+.2f}% | {row['measured_output_tok_s']:.3f}/{row['predicted_output_tok_s']:.3f} | {row['error_percent_output_tok_s']:+.2f}% |")
    lines.extend(["", "JSON保留输入吞吐、实测running peak/均值及预测占用代理值。实测为每秒Prometheus采样，预测为destination activation到last token的占用区间，包含prebuilt等待，仍不能当作完全相同的逐轮running batch指标。", "", "## 本次模型修改与校准边界", "", "原每轮eager残差不变，新增每个额外活跃attention-DP rank的残差：TP16/ADP4为6.373190ms，TP32/ADP32为1.497163ms。Rust根据真实rank请求向量计数，空rank即使因通信padding获得token也不计入；Prefill沿用原91.625ms，空batch仍为零。该项是显式estimated执行残差，不改算子实测数据。", "", "仅用先前逐轮配对TP16的validation-c4和TP32的validation-c8拟合新增项；这两组现在明确属于训练，原文件名不变。原C1基值保持冻结。其他逐轮case、原四节点23点和新9组PD端到端数据用于验证。TP32 calibration-c1曾用于基值，不能再计为完全独立验证；详见active-rank-forward-validation.json。", "", "控制字段为estimator_config.op_level.ascend.eager_runtime_overhead.decode_per_extra_active_rank_ms，默认0。所有值通过同一Rust canonical FPM接口；保存YAML和搜索保留校准身份及完整拓扑，不在Python结果表里乘修正系数。EngineSpec升级为32，GPU默认路径保持不变。", "", "本机镜像为aisimulate-910c:active-rank-repair-20261102，可直接执行的五份YAML在active-rank-configs/。旧request-timing-repair镜像和配置继续保留，run-ascend910c.sh的原默认镜像仍是冻结基线，使用本版需显式设置AISIMULATE_910C_IMAGE。", "", "相同并发的已测候选中，输出吞吐第一名仍6/6一致：2个独立P各TP16/ADP4/EP16，另两节点组成1个TP32/ADP32/EP32的D。C128仅三种完成候选；这不是任意并行组合的全局最优证明，也不保证未建模的冷启动TTFT SLO。", "", "## 已定位的剩余首字误差", "", "在新1P1D中，DecodePreallocQueue._pre_alloc首次实际8K调用约2.1秒，后续仅数毫秒。独立单NPU、无模型权重、无HCCL的metadata allocator复现了该差异；进一步直接计时Triton compiler函数：首次4K/8K编译调用约2063/2069ms，重复调用没有compile事件、allocator约0.5ms。首次1-token路径还包含约9秒的其他JIT/模块初始化，通常已被服务启动预热吸收。", "", "因此单次forward稳态算子表不会自动包含首次shape编译。两个benchmark warmup不能覆盖所有P/D副本、attention-DP rank及首次shape。Prefill也观察到约2.17秒发生在run_batch之外；当前细分的send_kv_chunk/结果处理没有解释这段时间，结合allocator探针推测与首次分配路径有关，尚未把该推测作为已验证的固定延迟写入预测。", "", "该冷编译时间仍未进入请求模拟；没有用TTFT实测差额硬加一个常数。客户端首次非空文本有时晚于第一个生成token，也是尚未完全对齐的部分。32K尝试因实际P/D KV容量不足被拒绝，保留失败记录，不纳入精度表。", "", "该阶段模型服务及allocator探针均已退出，20/25上的其他用户进程未操作。原始配对记录、独立allocator测量、失败尝试、实测配置、预测命令及哈希见active-rank-repair-manifest.json。"])
    lines.extend(["", "## 验证记录", "", "1031项Rust性能模型测试通过，1项既有R17外部数据测试忽略；433项Python/模型/OpSpec/单一计算入口及两套GPU parity测试通过，未修改GPU golden。固定1P1D候选经过recommend→保存YAML→predict，以及直接保存predict配置再运行，两条路径的请求数、token总量、TTFT、TPOT、输入/输出吞吐均与直接预测一致（浮点容差内）。详情见active-rank-code-validation.json.gz。", "", "独立性说明：pd-ingress的diagnostic-c1曾用于0.8435ms tokenizer阶段的测量，不能把这一组对整个组合模型称为完全独立留出。完整8K/1K、cold与phases新组均未用于参数拟合；本次新增rank系数没有使用任何PD端到端点。"])
    (ROOT / "active-rank-serving-comparison.md").write_text("\n".join(lines) + "\n")
    phase_analysis = load(ROOT / "pd-phase-timeline-analysis.json")
    stage_rows = records(ROOT / "pd-phases-1p1d/decode", "host-phases-*.jsonl*")
    client_ids = {row["rid"] for case in phase_analysis["cases"] for row in case["rows"]}
    phases = [row for row in stage_rows if any(request["rid"] in client_ids for request in row["requests"])]
    (ROOT / "pd-phase-allocation-summary.json").write_text(json.dumps({"scope": "Nested host phase timings, do not sum parent and child intervals", "phases": phases}, indent=2) + "\n")
    artifacts = {}
    for directory in ("pd-phases-1p1d", "allocator-probe", "allocator-compile-probe", "allocator-compile-probe-v2", "active-rank-configs"):
        for path in sorted((ROOT / directory).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix in (".jsonl", ".log"):
                archive = path.with_suffix(path.suffix + ".gz")
                archive.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
                path = archive
            if path.suffix == ".gz":
                gzip.decompress(path.read_bytes())
            artifacts[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("active-rank-serving-comparison.json", "active-rank-serving-comparison.md", "active-rank-serving-comparison.csv", "eager-overhead-active-tp16-adp4.json", "eager-overhead-active-tp32-adp32.json", "pd-phase-allocation-summary.json", "active-rank-forward-validation.json", "pd-phase-timeline-analysis.json", "active-rank-code-validation.json", "active-rank-native-tests.log.gz", "active-rank-package-tests.log.gz", "active-rank-recommend-roundtrip.log.gz"):
        path = ROOT / name
        if name.endswith("validation.json") or name.endswith("analysis.json"):
            archive = path.with_suffix(path.suffix + ".gz")
            archive.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            path = archive
        artifacts[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for directory in ("four-node-active-rank-20261102", "pd-ingress-active-rank-20261102", "pd-cold-active-rank-20261102", "pd-phase-active-rank-20261102", "active-rank-saved-c4-20261102", "active-rank-recommend-roundtrip-20261102", "active-rank-recommend-saved-predict-20261102"):
        source = REPO / "aisimulate-output" / directory
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.name not in ("comparison.json", "comparison-jobs.json", "execution.json", "prediction.json", "requests.jsonl", "recommendation.json", "0001.yaml"):
                continue
            destination = ROOT / "active-rank-replays" / directory / path.relative_to(source)
            archive_json = path.suffix == ".jsonl" or path.name in ("execution.json", "comparison-jobs.json", "recommendation.json")
            if archive_json:
                destination = destination.with_suffix(destination.suffix + ".gz")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(gzip.compress(path.read_bytes(), mtime=0) if archive_json else path.read_bytes())
            artifacts[str(destination.relative_to(ROOT))] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (ROOT / "active-rank-repair-manifest.json").write_text(json.dumps({"scope": result["scope"], "artifacts": artifacts}, indent=2) + "\n")
    print(json.dumps({"accuracy": accuracy["current"], "independent_cases": len(independent), "manifest_files": len(artifacts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
