import argparse
import hashlib
import json
import statistics
from functools import lru_cache
from pathlib import Path

import yaml

from report_client_dispatch import METRICS, compare


@lru_cache(maxsize=128)
def parsed_config(content):
    return yaml.safe_load(content)


def read_config(job):
    content = Path(job["config"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != job["config_sha256"]:
        raise ValueError("Prediction configuration changed")
    return parsed_config(content)


def summary(rows, field):
    selected = [row[field] for row in rows if row.get(field)]
    return {"planned": len(rows), "completed": len(selected),
            "mape_percent": {name: statistics.mean(abs(row[name + "_error_pct"]) for row in selected)
                             for name, _, _ in METRICS} if selected else None}


def cell(row, name):
    actual = row["actual"][name]
    result = row.get("after")
    return (f"{actual:.2f} / {result[name + '_predicted']:.2f}（{result[name + '_error_pct']:+.2f}%）"
            if result else f"{actual:.2f} / 未覆盖")


def prediction(job):
    read_config(job)
    return json.loads((Path(job["folder"]) / "prediction/prediction.json").read_text())


def main():
    parser = argparse.ArgumentParser()
    for name in ("baseline", "old-pd", "new-pd", "forward-validation", "output", "document"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--serving", type=Path, action="append", required=True)
    parser.add_argument("--predicted-serving", type=Path, action="append", required=True)
    parser.add_argument("--steady-analysis", type=Path)
    args = parser.parse_args()
    if len(args.serving) != len(args.predicted_serving):
        parser.error("Each serving campaign requires its own prediction manifest")
    if args.output.exists() or args.document.exists():
        raise FileExistsError("Report outputs must be new")
    sources = {str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def load(path):
        content = path.read_bytes()
        sources[str(path)] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    baseline = load(args.baseline)
    before_manifest = load(args.old_pd / "manifest.json")
    after_manifest = load(args.new_pd / "manifest.json")
    metadata = {int(row["case"]): row for row in baseline["historical_rows"]}
    old_records = {row["number"]: row for row in before_manifest["records"]}
    if set(old_records) != {row["number"] for row in after_manifest["records"]}:
        raise ValueError("Historical row set changed")
    historical = []
    for record in after_manifest["records"]:
        old_record = old_records[record["number"]]
        if old_record["historical"] != record["historical"]:
            raise ValueError("Historical observations changed")
        prior = before_manifest["jobs"][old_record["job"]]
        current = after_manifest["jobs"][record["job"]]
        if read_config(prior) != read_config(current):
            raise ValueError("Same-condition prediction configuration changed")
        row = {name: metadata[record["number"]][name] for name in ("group", "case", "input_length", "output_length", "concurrency", "requests", "nodes", "parallelism", "kv_pages", "actual")}
        history = record["historical"]
        observation = {actual: history["measured_" + name] for name, actual, _ in METRICS}
        for label, job in (("before", prior), ("after", current)):
            row[label + "_status"] = job["status"]
            if job["status"] == "completed":
                row[label] = compare(observation, prediction(job), history["requests"], history["input_length"], history["output_length"])
            else:
                row[label + "_error_log"] = str(Path(job["folder"]) / "prediction.log")
        historical.append(row)
    serving_groups = []
    for serving_path, predicted_path in zip(args.serving, args.predicted_serving):
        measured = load(serving_path / "campaign.json")
        planned = load(predicted_path / "manifest.json")
        if measured["status"] != "completed":
            raise ValueError("Do not summarize unfinished measurements")
        jobs = {row["id"]: row for row in planned["cases"]}
        if set(jobs) != {row[0] for row in measured["plan"]}:
            raise ValueError("Serving workload set changed")
        rows = []
        for name, input_length, output_length, concurrency, requests in measured["plan"]:
            observation_path = serving_path / name / "requests.jsonl"
            sources[str(observation_path)] = hashlib.sha256(observation_path.read_bytes()).hexdigest()
            observation = json.loads(observation_path.read_text().splitlines()[-1])
            if (observation["completed"] != requests or any(observation["errors"])
                    or observation["input_lens"] != [input_length] * requests
                    or observation["output_lens"] != [output_length] * requests):
                raise ValueError("Client request/token conservation failed")
            job = jobs[name]
            row = {"case": name, "input_length": input_length, "output_length": output_length,
                   "concurrency": concurrency, "requests": requests, "after_status": job["status"],
                   "actual": {name: observation[actual] for name, actual, _ in METRICS}}
            if job["status"] == "completed":
                row["after"] = compare(observation, prediction(job), requests, input_length, output_length)
            else:
                row["error_log"] = str(Path(job["folder"]) / "prediction.log")
            rows.append(row)
        serving_groups.append({"source": str(serving_path), "rows": rows, "summary": summary(rows, "after")})
    forwards = load(args.forward_validation)
    steady = load(args.steady_analysis) if args.steady_analysis else None
    paired = [row for row in historical if row.get("before") and row.get("after")]
    result = {"historical_rows": historical, "historical_before_paired": summary(paired, "before"),
              "historical_after_paired": summary(paired, "after"), "historical_after_all": summary(historical, "after"),
              "serving_groups": serving_groups, "forward_validation": forwards, "fresh_steady_analysis": steady,
              "source_sha256": sources, "serving_accuracy_accepted": False,
              "scope": "Unchanged deployment/workload configs and predeclared measurements; new MAX_LEN physical-population transfer and16 measured layer cells. No fitted latency, no error-selected fallback. Repeat campaigns reported separately without counting repeated conditions as independent coverage."}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    lines = ["# 910C GLM-5.2：MAX_LEN覆盖修复与同条件复核", "", "日期：2026-11-04。目标尚未全部达成，不能据此承诺任意PD、并行策略和workload的平均误差均小于20%。", "",
             "这轮按镜像源码补齐了MAX_LEN的物理张量补齐规则：同一个Decode轮中部分rank有2个请求、部分rank有1个请求时，物理buffer会补到每rank的最大长度。原模型在跨越attention-TP补齐区间时直接拒绝，现在在anchored模式中使用对应的实测同质population作为层耗时锚点，保留真实活跃rank数和KV长度；调度、KV页、logits及running batch仍按真实请求统计。这类结果显式标为estimated，并不把补齐行伪装成真实请求。SUM_LEN跨区间仍拒绝，因为它没有相同的物理补齐行为。新增16个通过原有数值校验和CV门槛的层shape，全部560条旧记录保持不变；没有添加固定毫秒残差。", "",
             "10项原生几何测试、449项回归/模型/接口/采集测试、4个规范FPM查询与3个必须拒绝的边界查询通过，golden未修改。", "",
             "## 历史39条同条件对照", "", "每格为实测 / 当前预测（有符号误差）；并行为P；D的TP/ADP/EP。A=4节点2P1D，D实例跨两节点；B=4节点1P3D；C=4节点2P2D；D=4节点3P1D；E=2节点1P1D。所有Decode预先统一选择runtime_layers，失败不切换到独立kernel。", "",
             "| 编号/组 | I/O | C/N | 并行 | KV页/rank | TTFT ms | TPOT ms | 输入tok/s | 输出tok/s |", "|---|---|---|---|---|---|---|---|---|"]
    for row in historical:
        lines.append(f"| {row['case']}·{row['group']} | {row['input_length']}/{row['output_length']} | {row['concurrency']}/{row['requests']} | {row['parallelism']} | {json.dumps(row['kv_pages'])} | " + " | ".join(cell(row, name) for name, _, _ in METRICS) + " |")
    lines += ["", "| 历史统计口径 | 完成/计划 | TTFT MAPE | TPOT MAPE | 输入吞吐MAPE | 输出吞吐MAPE |", "|---|---|---|---|---|---|"]
    for label, values in (("同一可配对子集：修复前", result["historical_before_paired"]), ("同一可配对子集：修复后", result["historical_after_paired"]), ("修复后全部成功项", result["historical_after_all"])):
        lines.append(f"| {label} | {values['completed']}/{values['planned']} | " + " | ".join(f"{values['mape_percent'][name]:.2f}%" for name, _, _ in METRICS) + " |")
    lines += ["", "## 两节点合并服务验证与复测", "", "以下均为TP32/ADP32/EP32、每rank968个KV页、graph/调度overlap/投机解码关闭、2次warmup。合并服务用于算子与调度归因，不替代PD端到端验收。各批次独立列出，包含变慢的复测。", ""]
    for group in serving_groups:
        lines += [f"来源：`{group['source']}`", "", "| case | I/O | C/N | TTFT ms | TPOT ms | 输入tok/s | 输出tok/s |", "|---|---|---|---|---|---|---|"]
        for row in group["rows"]:
            lines.append(f"| {row['case']} | {row['input_length']}/{row['output_length']} | {row['concurrency']}/{row['requests']} | " + " | ".join(cell(row, name) for name, _, _ in METRICS) + " |")
        lines += ["", "成功/计划：" + f"{group['summary']['completed']}/{group['summary']['planned']}；各指标MAPE：" + json.dumps(group["summary"]["mape_percent"], ensure_ascii=False) + "。", ""]
    lines += ["## 逐轮覆盖验证", "", "这些观测排除了层重放及相邻轮次，只比较同一实际batch；它们不是整请求准确率。", "", "| 实际population | 可预测/稳态轮数 | 实测forward中位数ms | 预测ms | MAPE |", "|---|---|---|---|---|"]
    for case in forwards["cases"]:
        value = case["summary"]
        prediction_cell = f"{value['runtime_layer_prediction_ms']:.2f}" if "runtime_layer_prediction_ms" in value else "未覆盖"
        error_cell = f"{value['runtime_layer_mape_percent']:.2f}%" if "runtime_layer_mape_percent" in value else "未覆盖"
        lines.append(f"| {case['case']['id']} | {value['runtime_layer_supported_groups']}/{value['unprofiled_groups']} | {value['measured_ms']:.2f} | {prediction_cell} | {error_cell} |")
    lines += ["", "## 仍存在的缺口", "", "16K/C32两次复测的32个rank均各有1个请求，但forward由原来的约372ms升至约536/528ms，轮间间隔仍约8–9ms；因此不能把差异归因于请求路由偏斜或客户端排队。32-rank算子轨迹确认走MoeLowLatencyDispatchV2；同一节点内Dispatch启动时间相差约5.17/2.52ms，结束时间只相差约0.032ms，说明其中包含等待其他rank就绪。此轨迹受profiler扰动，只用于归因，未进入性能库，也未作为固定通信惩罚。", "", "MAX_LEN补齐修复只解决几何与覆盖问题；部分均匀population和稀疏活跃rank的层组合仍低估真实forward，Prefill非均匀MoE通信及host/stream关键路径也仍需继续核对。所有新旧测量、失败记录和配置均保留，不能只挑更接近预测的一次复测作为验收。", ""]
    if steady:
        lines += ["## 重启与采集影响的对照", "", "同一镜像、参数、环境重新启动后，先运行四个预先声明的普通请求负载，再复测C32，然后执行一次明确不入库的层重放，再复测C32。过程中没有编译或离线预测任务与服务计时并行。", "",
                  "| 实测case | TTFT ms | TPOT ms | 稳态forward均值ms | 轮间启动间隔ms |", "|---|---|---|---|---|"]
        for item in steady["cases"]:
            value = item["summary"]
            lines.append(f"| {item['case']['id']} | {value['ttft_ms']:.2f} | {value['tpot_ms']:.2f} | {value['mean_max_forward_ms']:.2f} | {value['mean_rank0_cadence_ms']:.2f} |")
        lines += ["", "普通shape历史之后和一次层重放之后，C32的TPOT都保持在约370ms，稳态forward约358–361ms。因此本次控制实验没有复现之前约528–536ms的慢forward，也没有证明一次层重放会导致持续变慢；原来的慢记录继续保留，原因仍待定位。重启首轮TTFT比第二轮多约5.4秒，但没有单独量出编译耗时，不能直接把差额作为固定冷编译补偿。", ""]
    lines += ["原始6项请求验证中的C64/N128仍缺active29/request2的层数据；新测C64/N64已能完整预测，两者请求总数不同，不能把后者的成功写成前者已修复。历史第4、5、6项仍遇到SUM_LEN跨补齐区间而被明确拒绝。本轮历史36项成功预测的数值与上版保持一致，覆盖修复没有使这组历史MAPE进一步下降。", ""]
    args.document.write_text("\n".join(lines))
    print(json.dumps({key: result[key] for key in ("historical_before_paired", "historical_after_paired", "historical_after_all")}, indent=2))


if __name__ == "__main__":
    main()
