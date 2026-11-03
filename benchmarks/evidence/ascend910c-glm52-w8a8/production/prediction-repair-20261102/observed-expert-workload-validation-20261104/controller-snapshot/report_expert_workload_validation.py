import argparse
import copy
import hashlib
import json
import statistics
from pathlib import Path

import yaml

from report_client_dispatch import METRICS, compare


def validate_change(before, after, workload):
    normalized = copy.deepcopy(after)
    found = []

    def remove(value):
        if isinstance(value, dict):
            if "prefill_expert_workload" in value:
                found.append(value.pop("prefill_expert_workload"))
            for child in value.values():
                remove(child)
        elif isinstance(value, list):
            for child in value:
                remove(child)

    remove(normalized)
    if found != [workload] or normalized != before:
        raise ValueError("Configurations differ beyond the declared independent expert workload")


def read_config(job):
    path = Path(job["config"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != job["config_sha256"]:
        raise ValueError("Prediction configuration changed")
    return yaml.safe_load(path.read_text())


def errors(rows, name):
    complete = [row[name] for row in rows if row.get(name)]
    return {metric: statistics.mean(abs(row[metric + "_error_pct"]) for row in complete)
            for metric, _, _ in METRICS} if complete else None


def cell(row, metric):
    actual = row["actual"][metric]
    before = row.get("before")
    after = row.get("after")
    previous = f"{before[metric + '_predicted']:.2f}（{before[metric + '_error_pct']:+.2f}%）" if before else "未覆盖"
    current = f"{after[metric + '_predicted']:.2f}（{after[metric + '_error_pct']:+.2f}%）" if after else "未覆盖"
    return f"{actual:.2f} / {previous} → {current}"


def main():
    parser = argparse.ArgumentParser()
    for name in ("old-pd", "new-pd", "baseline-report", "old-serving", "new-serving", "actual-serving",
                 "expert-workload", "routing-analysis", "prefill-before", "prefill-after", "output", "document"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.document.exists():
        raise FileExistsError("Report outputs must be fresh")
    workload = json.loads(args.expert_workload.read_text())
    metadata = {int(row["case"]): row for row in json.loads(args.baseline_report.read_text())["historical_rows"]}
    old_pd, new_pd = [json.loads((path / "manifest.json").read_text()) for path in (args.old_pd, args.new_pd)]
    old_records = {row["number"]: row for row in old_pd["records"]}
    if set(old_records) != {row["number"] for row in new_pd["records"]}:
        raise ValueError("Historical comparison must retain every declared row")
    rows = []
    for record in new_pd["records"]:
        before_record = old_records[record["number"]]
        if record["historical"] != before_record["historical"]:
            raise ValueError("Historical observations changed")
        history = record["historical"]
        before_job, after_job = old_pd["jobs"][before_record["job"]], new_pd["jobs"][record["job"]]
        validate_change(read_config(before_job), read_config(after_job), workload)
        row = {key: metadata[record["number"]][key] for key in ("group", "case", "input_length", "output_length", "concurrency", "requests", "nodes", "parallelism", "kv_pages")}
        observation = {actual_key: history["measured_" + name] for name, actual_key, _ in METRICS}
        row["actual"] = {name: observation[actual_key] for name, actual_key, _ in METRICS}
        for label, job in (("before", before_job), ("after", after_job)):
            row[label + "_status"] = job["status"]
            if job["status"] == "completed":
                predicted = json.loads((Path(job["folder"]) / "prediction/prediction.json").read_text())
                row[label] = compare(observation, predicted, history["requests"], history["input_length"], history["output_length"])
        rows.append(row)
    old_serving, new_serving = [{row["id"]: row for row in json.loads((path / "manifest.json").read_text())["cases"]}
                               for path in (args.old_serving, args.new_serving)]
    actual = json.loads((args.actual_serving / "campaign.json").read_text())
    serving = []
    for name, length, output_length, concurrency, prompts in actual["plan"]:
        before_job, after_job = old_serving[name], new_serving[name]
        validate_change(read_config(before_job), read_config(after_job), workload)
        observation = json.loads((args.actual_serving / name / "requests.jsonl").read_text().splitlines()[-1])
        row = {"case": name, "input_length": length, "output_length": output_length, "concurrency": concurrency,
               "requests": prompts, "actual": {name: observation[actual_key] for name, actual_key, _ in METRICS}}
        for label, job in (("before", before_job), ("after", after_job)):
            row[label + "_status"] = job["status"]
            if job["status"] == "completed":
                prediction = json.loads((Path(job["folder"]) / "prediction/prediction.json").read_text())
                row[label] = compare(observation, prediction, prompts, length, output_length)
        serving.append(row)
    before, after = [json.loads(path.read_text()) for path in (args.prefill_before, args.prefill_after)]
    validate_change(before["configuration"], after["configuration"], workload)
    if [row["case"] for row in before["cases"]] != [row["case"] for row in after["cases"]]:
        raise ValueError("Prefill attribution case set changed")
    prefill = []
    for old, new in zip(before["cases"], after["cases"]):
        identities = lambda case: [(row["iteration"], row["metrics"], row["measured_ms"]) for row in case["forwards"]]
        if identities(old) != identities(new):
            raise ValueError("Observed batch or timing changed")
        for summary in new["summaries"]:
            prior = next(row for row in old["summaries"] if row["mixed"] == summary["mixed"])
            prefill.append({"case": old["case"], "before": prior, "after": summary})
    grouped = {}
    for group in dict.fromkeys(row["group"] for row in rows):
        selected = [row for row in rows if row["group"] == group]
        paired = [row for row in selected if row.get("before") and row.get("after")]
        grouped[group] = {"planned": len(selected), "paired": len(paired),
                          "before": errors(paired, "before"), "after": errors(paired, "after")}
    report = {"historical_rows": rows, "serving_rows": serving, "prefill_attribution": prefill, "grouped": grouped,
              "routing": json.loads(args.routing_analysis.read_text()), "serving_accuracy_accepted": False,
              "scope": "Same deployments and observations; only independent expert-count workload added. Default balanced transport is unchanged. No whole-forward residual or heldout serving fit.",
              "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (args.baseline_report, args.expert_workload, args.routing_analysis,
                                             args.prefill_before, args.prefill_after, Path(__file__))}}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = ["# 910C GLM-5.2：专家工作量修正的独立验证", "", "日期：2026-11-04。", "",
             "仅增加独立采集的专家分配次数，部署配置、原始实测、KV容量、请求数、CPU准备模型保持一致；graph、调度overlap、投机解码均关闭。Decode仍用原来的560条层数据。所有表项按相同配置检查，失败项保留。", "",
             "8K、两个预先声明的随机种子用于统计75层各256个专家的工作量；12K、16K留作验证。按候选EP连续分配专家，计算每层最忙rank的工作量，再由原生算子表预测GMM/激活耗时，并按层与shared expert组合。没有新增固定毫秒残差。通信仍使用原有最大sender的balanced profile；此修正尚未解决非均匀路由的通信关键路径。", "",
             "## 历史39条同条件对照", "", "每格：实测 / 修正前预测（误差）→修正后预测（误差）。C/N为并发/请求数；并行列为P；D的TP/ADP/EP。", "",
             "| 编号/组别 | I/O | C/N | 并行 | KV页/rank | TTFT ms | TPOT ms | 输入tok/s | 输出tok/s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['case']}·{row['group']} | {row['input_length']}/{row['output_length']} | {row['concurrency']}/{row['requests']} | {row['parallelism']} | {json.dumps(row['kv_pages'])} | " + " | ".join(cell(row, name) for name, _, _ in METRICS) + " |")
    lines += ["", "## 独立请求验证", "", "以下是两节点合并TP32/ADP32/EP32服务，用于验证新输入输出长度；不等同于PD性能验收。", "",
              "| I/O | C/N | TTFT ms | TPOT ms | 输入tok/s | 输出tok/s |", "|---|---|---|---|---|---|"]
    for row in serving:
        lines.append(f"| {row['input_length']}/{row['output_length']} | {row['concurrency']}/{row['requests']} | " + " | ".join(cell(row, name) for name, _, _ in METRICS) + " |")
    lines += ["", "## 相同Prefill batch的误差", "", "逐轮输入相同；此表只用于归因，不代替端到端验证。", "",
              "| workload | 混合轮 | 实测ms | 预测ms：前→后 | MAPE：前→后 |", "|---|---|---|---|---|"]
    for row in prefill:
        old, new = row["before"], row["after"]
        lines.append(f"| {row['case']} | {new['mixed']} | {new['measured_ms']:.2f} | {old['predicted_ms']:.2f} → {new['predicted_ms']:.2f} | {old['mape_percent']:.2f}% → {new['mape_percent']:.2f}% |")
    lines += ["", "仍须分别验证TTFT、TPOT和吞吐，不能把专家数量的预测误差当作时延误差。输入内容分布变化、未覆盖的DP batch、通信接收端拥塞以及host同步仍可能产生误差；当前不宣布任意workload/并行/PD的平均误差低于20%。", ""]
    args.document.write_text("\n".join(lines))
    print(json.dumps(grouped, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
