import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from report_client_dispatch import METRICS, compare
from report_joint_comparison import batch_summary


def averages(rows):
    return {name: statistics.mean(abs(row[name + "_error_pct"]) for row in rows)
            for name, _, _ in METRICS} if rows else None


def value_cell(row, name):
    actual = row.get(name + "_measured")
    predicted = row.get(name + "_predicted")
    if predicted is None:
        return f"{actual:.2f} / 未覆盖" if actual is not None else "待完成"
    return f"{actual:.2f} / {predicted:.2f}（{row[name + '_error_pct']:+.2f}%）"


def batch_cell(row, role):
    actual = row.get("measured_" + role + "_batch")
    predicted = row.get("predicted_" + role + "_batch")
    return (f"{actual['mean']:.3f}" if actual else "无逐轮记录") + " / " + (
        f"{predicted['mean']:.3f}" if predicted else "未覆盖")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pd", type=Path, action="append", required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--layer-comparison", type=Path, action="append", default=[])
    parser.add_argument("--layer-provenance", type=Path)
    parser.add_argument("--serving", type=Path)
    parser.add_argument("--serving-predictions", type=Path)
    parser.add_argument("--prefill-before", type=Path)
    parser.add_argument("--prefill-after", type=Path)
    parser.add_argument("--previous-serving-predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--document", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    previous = json.loads(args.previous.read_text())
    baseline = {int(row["case"]): row for row in previous["rows"]
                if row["group"].startswith(("A·", "B·", "C·", "D·", "E·")) and row["mode"] == "profiled_client"}
    manifests = [path / "manifest.json" for path in args.pd]
    sources = [args.previous, *manifests]
    manifest = {"records": [], "jobs": {}}
    identities = set()
    numbers = set()
    for path in manifests:
        part = json.loads(path.read_text())
        identities.add(tuple(part[key] for key in ("source_manifest_sha256", "source_archives_sha256", "cpu_profile_sha256", "source_sha256")))
        if numbers.intersection(record["number"] for record in part["records"]):
            raise ValueError("Historical rows must be disjoint across replay parts")
        if manifest["jobs"].keys() & part["jobs"].keys():
            raise ValueError("Repeated prediction jobs must be resolved before report composition")
        numbers.update(record["number"] for record in part["records"])
        manifest["records"].extend(part["records"])
        manifest["jobs"].update(part["jobs"])
    if len(identities) != 1:
        raise ValueError("Replay parts differ in historical source, client profile or controller")
    manifest["records"].sort(key=lambda record: record["number"])
    rows = []
    for record in manifest["records"]:
        number, history = record["number"], record["historical"]
        old = baseline[number]
        job = manifest["jobs"][record["job"]]
        row = {key: old[key] for key in ("group", "case", "input_length", "output_length", "concurrency", "requests", "nodes", "parallelism", "kv_pages")}
        row.update(status=job["status"], scope="L", configuration=job["config"],
                   previous={name: old[name + "_error_pct"] for name, _, _ in METRICS})
        for role in ("prefill", "decode"):
            row["measured_" + role + "_batch"] = history.get("measured_" + role + "_batch")
        actual = {actual_key: history["measured_" + name] for name, actual_key, _ in METRICS}
        row.update({name + "_measured": actual[actual_key] for name, actual_key, _ in METRICS})
        if job["status"] == "completed":
            path = Path(job["folder"]) / "prediction/prediction.json"
            predicted = json.loads(path.read_text())
            row.update(compare(actual, predicted, history["requests"], history["input_length"], history["output_length"]))
            row.update({"predicted_" + role + "_batch": batch_summary(predicted["forward_pass_batches"], role)
                        for role in ("prefill", "decode")})
            sources.append(path)
        else:
            path = Path(job["folder"]) / "prediction.log"
            row["failure_log"] = str(path)
            if path.exists():
                row["failure_tail"] = path.read_text(errors="replace").splitlines()[-8:]
        rows.append(row)
    grouped = {}
    for group in dict.fromkeys(row["group"] for row in rows):
        selected = [row for row in rows if row["group"] == group]
        completed = [row for row in selected if row["status"] == "completed"]
        current = averages(completed)
        prior_all = {name: statistics.mean(abs(row["previous"][name]) for row in selected) for name, _, _ in METRICS}
        prior = {name: statistics.mean(abs(row["previous"][name]) for row in completed) for name, _, _ in METRICS} if completed else None
        grouped[group] = {"planned": len(selected), "completed": len(completed), "before_mape_pct": prior,
                          "before_all_rows_mape_pct": prior_all,
                          "after_mape_pct": current, "accepted": len(selected) == len(completed) and current is not None and all(value < 20 for value in current.values())}
    published_cells = {}
    if args.layer_provenance:
        sources.append(args.layer_provenance)
        for item in json.loads(args.layer_provenance.read_text())["accepted"]:
            if (item["row"]["attention_tp"], item["row"]["attention_dp"], item["row"]["moe_ep"]) == (1, 32, 32):
                published_cells[item["case"]] = published_cells.get(item["case"], 0) + 1
    forwards = []
    for path in args.layer_comparison:
        sources.append(path)
        for case in json.loads(path.read_text())["cases"]:
            forwards.append({"case": case["case"]["id"], "training": case["case"]["training"],
                             "published_layer_cells": published_cells.get(case["case"]["id"], 0) if args.layer_provenance else None,
                             **case["summary"],
                             "coverage_gaps": case["coverage_gaps"], "source": str(path)})
    serving = []
    previous_serving = {}
    if args.previous_serving_predictions:
        prior_path = args.previous_serving_predictions / "manifest.json"
        sources.append(prior_path)
        previous_serving = {case["id"]: case for case in json.loads(prior_path.read_text())["cases"]}
    if args.serving and args.serving_predictions:
        actual_path, predicted_path = args.serving / "campaign.json", args.serving_predictions / "manifest.json"
        sources += [actual_path, predicted_path]
        actual = json.loads(actual_path.read_text())
        predictions = {case["id"]: case for case in json.loads(predicted_path.read_text())["cases"]}
        measured = {case["id"]: case for case in actual["cases"]}
        for name, input_length, output_length, concurrency, prompts in actual["plan"]:
            case, prediction = measured.get(name), predictions.get(name)
            row = {"case": name, "input_length": input_length, "output_length": output_length,
                   "concurrency": concurrency, "requests": prompts, "status": "pending"}
            if case and case["status"] == "completed":
                path = args.serving / name / "requests.jsonl"
                observation = json.loads(path.read_text().splitlines()[-1])
                sources.append(path)
                row.update(status="prediction_unavailable", **{metric + "_measured": observation[actual_key] for metric, actual_key, _ in METRICS})
                if prediction and prediction["status"] == "completed":
                    path = Path(prediction["folder"]) / "prediction/prediction.json"
                    row.update(status="completed", **compare(observation, json.loads(path.read_text()), prompts, input_length, output_length))
                    sources.append(path)
                    prior = previous_serving.get(name)
                    if prior and prior["status"] == "completed":
                        if prior["config_sha256"] != prediction["config_sha256"]:
                            raise ValueError("Before/after serving configurations differ")
                        path = Path(prior["folder"]) / "prediction/prediction.json"
                        row["previous"] = compare(observation, json.loads(path.read_text()), prompts, input_length, output_length)
                        sources.append(path)
            serving.append(row)
    prefill = []
    if args.prefill_before or args.prefill_after:
        if not args.prefill_before or not args.prefill_after:
            raise ValueError("Both prefill attribution reports are required")
        sources.extend([args.prefill_before, args.prefill_after])
        before, after = [json.loads(path.read_text()) for path in (args.prefill_before, args.prefill_after)]
        if before["configuration"] != after["configuration"]:
            raise ValueError("Before/after attribution configurations differ")
        old_cases = {case["case"]: case for case in before["cases"]}
        if set(old_cases) != {case["case"] for case in after["cases"]}:
            raise ValueError("Before/after attribution case sets differ")
        for case in after["cases"]:
            old = old_cases[case["case"]]
            identities = lambda value: [(row["iteration"], row["metrics"], row["measured_ms"]) for row in value["forwards"]]
            if identities(old) != identities(case):
                raise ValueError("Observed forward inputs or measurements changed")
            for summary in case["summaries"]:
                prior = next(item for item in old["summaries"] if item["mixed"] == summary["mixed"])
                prefill.append({"case": case["case"], "before": prior, "after": summary})
    report = {"historical_rows": rows, "grouped": grouped, "forward_cases": forwards, "serving_cases": serving,
              "prefill_attribution": prefill,
              "universal_accuracy_claim": False,
              "scope": "Historical same-condition regression and independent serving holdouts are separate; missing predictions are retained and block acceptance; no topology choice by error",
              "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with (args.output / "historical.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 910C / GLM-5.2 W8A8：TP32 数据接入与同条件对照", "", "日期：2026-11-04。", "",
             "历史数据保持相同输入输出、C/N、PD副本数、P/D并行设置和KV容量；关闭graph、调度overlap、投机解码。L表示真实decoder层调用数据，Prefill仍用独立算子。两种已审计拓扑事先统一选择L，未覆盖的组合直接记录失败，不按误差挑模式，也不回退到旧结果。", "",
             "## 分组误差", "", "四个指标分别计算逐配置MAPE，不能用吞吐改善代替TTFT验收。部分覆盖时，前后MAPE均只比较同一组成功预测的行，整组不算通过；旧版全部行MAPE也保存在JSON中。", "",
             "| 组别 | 已预测/总数 | TTFT MAPE：前→后 | TPOT MAPE：前→后 | 输入吞吐 MAPE：前→后 | 输出吞吐 MAPE：前→后 | 完整覆盖且四项<20% |", "|---|---:|---:|---:|---:|---:|---|"]
    for group, result in grouped.items():
        cells = [f"{result['before_mape_pct'][name]:.2f}% → {result['after_mape_pct'][name]:.2f}%" if result["after_mape_pct"] else "未覆盖" for name, _, _ in METRICS]
        lines.append(f"| {group} | {result['completed']}/{result['planned']} | " + " | ".join(cells) + f" | {'是' if result['accepted'] else '否'} |")
    lines += ["", "## 历史同条件明细", "", "每格：实测 / 当前预测（有符号相对误差）。TP/ADP/EP以实例的global TP列出。C是并发，N是请求数。Batch按每个worker的完整forward轮次取算术平均，缺失实测不补造。", "",
              "| 编号/配置 | I/O | C/N | P；D：TP/ADP/EP | KV页/rank | TTFT ms | TPOT ms | 输入tok/s | 输出tok/s | P平均batch：实/预 | D平均batch：实/预 | 状态 |", "|---|---|---|---|---|---:|---:|---:|---:|---|---|---|"]
    for row in rows:
        cells = [value_cell(row, name) for name, _, _ in METRICS]
        lines.append(f"| {row['case']}·{row['group']} | {row['input_length']}/{row['output_length']} | {row['concurrency']}/{row['requests']} | {row['parallelism']} | {json.dumps(row['kv_pages'])} | " + " | ".join(cells) + f" | {batch_cell(row, 'prefill')} | {batch_cell(row, 'decode')} | {row['status']} |")
    if forwards:
        lines += ["", "## 单轮Decode验证", "", "来自正常forward的设备计时，排除层重复调用及相邻轮次。12K/16K层验证点没有写入性能表；这部分不能当作完整PD请求TTFT验收。", "",
                  "| case | 原计划采集 / 实际入库层数 | 实测ms | 独立kernel预测ms | 层调用预测ms | 层调用MAPE | 覆盖轮数/总数 |", "|---|---|---:|---:|---:|---:|---:|"]
        for row in forwards:
            estimate = f"{row['runtime_layer_prediction_ms']:.2f}" if "runtime_layer_prediction_ms" in row else "未覆盖"
            error = f"{row['runtime_layer_mape_percent']:.2f}%" if "runtime_layer_mape_percent" in row else "未覆盖"
            cells = str(row['published_layer_cells']) if row['published_layer_cells'] is not None else '未标注'
            lines.append(f"| {row['case']} | {'是' if row['training'] else '预先保留验证'} / {cells} | {row['measured_ms']:.2f} | {row['isolated_kernel_prediction_ms']:.2f} | {estimate} | {error} | {row['runtime_layer_supported_groups']}/{row['unprofiled_groups']} |")
    if serving:
        lines += ["", "## 独立完整请求验证", "", "这是两节点TP32合并部署，2个warmup、独立输入随机种子、关闭所有层重复采样；与历史PD部署分开评价。", "",
                  "| case | I/O | C/N | TTFT ms | TPOT ms | 输入tok/s | 输出tok/s | 状态 |", "|---|---|---|---:|---:|---:|---:|---|"]
        for row in serving:
            lines.append(f"| {row['case']} | {row['input_length']}/{row['output_length']} | {row['concurrency']}/{row['requests']} | " + " | ".join(value_cell(row, name) for name, _, _ in METRICS) + f" | {row['status']} |")
    if prefill:
        lines += ["", "## 注意力外推缺口与修复", "",
                  "已确认一个算子级算法问题：prefix=4096时，最小已采集query为4096tokens、约12.02ms。旧算法把这12.02ms全部当作固定启动开销，向1024tokens外推时仍收取接近完整4096tokens的计算成本，随后按prefix插值形成不存在的耗时峰值。修复改为用相邻算子实测点及其理论工作量推导非负固定部分，再缩放可变计算部分；单点无法识别固定成本时明确按工作量估算，不伪造启动耗时。此修改不使用TTFT/forward残差，也不改变原始算子记录、命中的实测点或严格measured策略。", "",
                  "下表使用完全相同的实测forward、batch和配置作算子归因，不能替代独立调度验证。修复消除了低并发Prefill的错误高估，也暴露了高并发此前被抵消的另一处低估；误差变大的行同样保留。", "",
                  "| workload | 纯P / 混合 | 实测forward ms | 旧预测ms | 新预测ms | 逐轮MAPE：前→后 |", "|---|---|---:|---:|---:|---:|"]
        for row in prefill:
            old, current = row["before"], row["after"]
            lines.append(f"| {row['case']} | {'混合' if current['mixed'] else '纯P'} | {current['measured_ms']:.2f} | {old['predicted_ms']:.2f} | {current['predicted_ms']:.2f} | {old['mape_percent']:.2f}% → {current['mape_percent']:.2f}% |")
        lines += ["", "同一输入与启动配置下的完整请求TTFT变化：", "",
                  "| workload | 实测TTFT ms | 旧预测ms（误差） | 新预测ms（误差） |", "|---|---:|---:|---:|"]
        for row in serving:
            if "previous" in row:
                old = row["previous"]
                lines.append(f"| {row['case']} | {row['ttft_ms_measured']:.2f} | {old['ttft_ms_predicted']:.2f}（{old['ttft_ms_error_pct']:+.2f}%） | {row['ttft_ms_predicted']:.2f}（{row['ttft_ms_error_pct']:+.2f}%） |")
        lines += ["", "32个活跃DP的Prefill仍有明显缺口，需继续用真实算子时间线区分MoE、通信和host边界；当前不声称任意并发、PD比例与并行策略已全部达到平均20%的目标。", ""]
    lines += ["", "## 修复说明与边界", "",
              "本轮发布320条TP32合格层记录，与原有240条TP16记录合并为560条；原值逐行保持。2K、32个活跃DP、每DP4请求的layer4在三次采集中均超过CV=0.1，因此这一人口组合的8条层记录均未入库，完整失败证据保留；同人口组合已有合格8K数据，2K预测由逐层attention/indexer上下文差值给出，并单列为待验证的估计覆盖，不能宣称原采集计划全部通过。", "",
              "本次为双节点global TP32/ADP32/EP32增加真实decoder层调用的采集和读取路径：每个采集点检查全部32个rank的有效hidden输出与top-k，按8类真实层结构及78层的数量组合，并替换其内部kernel耗时以避免重复相加；TP16与TP32保留各自的运行时身份，数据合并逐行保留原值。较长或略短于采样点的KV上下文，仅在同属DSA稀疏注意力区间时通过原始attention/indexer算子的差值估算，密集投影、MoE与host边界不按KV长度缩放，缺失请求数或并行布局数据继续报错。这修复了暖机后捕获的KV长度比请求初始Decode稍长而导致早期步骤无覆盖的问题。采集128并发时曾只形成96条同时运行请求，失败原样保留，扩大采集服务请求上限后重新得到每DP4条请求的有效数据；没有将失败样本改名为目标shape。上述改动没有加入整轮forward残差，也没有把实测到达时间、路由或batch输入离线预测。历史冷编译、首个非空文本的流式口径、Prefill路由不均衡及跨padding bucket的异构batch仍需分别验证，不能据此宣称任意workload和并行策略都已达到20%误差目标。", "",
              f"完整机器可读记录：`{args.output / 'comparison.json'}`；历史表CSV：`{args.output / 'historical.csv'}`。", ""]
    args.document.write_text("\n".join(lines))
    print(json.dumps(grouped, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
