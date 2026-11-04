import argparse
import copy
import csv
import datetime
import gzip
import hashlib
import json
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from report_client_dispatch import METRICS, compare


MODES = {"K": "isolated_kernels", "L": "runtime_layers"}
GROUPS = {"A": "4节点2P1D，D跨2节点", "B": "4节点1P3D", "C": "4节点2P2D",
          "D": "4节点3P1D", "E": "2节点1P1D"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def group_for(number):
    return "A" if number <= 6 else "B" if number <= 12 else "C" if number <= 18 else "D" if number <= 23 else "E"


def batch_summary(prediction, role):
    rows = [row for row in prediction["forward_pass_batches"]
            if row["dp_rank"] is None and row["stage"] == role and row["phase"] == role]
    passes = sum(row["passes"] for row in rows)
    if passes == 0:
        raise ValueError("Missing completed logical-worker batch counters: " + role)
    return {"mean": sum(row["request_instances"] for row in rows) / passes,
            "peak": max(row["max_batch_size"] for row in rows), "passes": passes}


def run_job(job):
    folder = Path(job["folder"])
    command = ["aisimulate", "predict", "--stack", "engine", "--config", job["config"],
               "--capture-per-request", "--detail", "source", "--output-dir", str(folder / "prediction")]
    started = time.time()
    update = {"command": command, "started_unix": started}
    try:
        with (folder / "prediction.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
        update.update(returncode=result.returncode, status="completed" if result.returncode == 0 else "failed")
    except subprocess.TimeoutExpired:
        update.update(returncode=None, status="timeout")
    update["wall_seconds"] = time.time() - started
    if update["status"] != "completed":
        text = (folder / "prediction.log").read_text()
        update["error_tail"] = text[-6000:]
        if "SUM_LEN skewed active-rank profiles" in text:
            update["failure_reason"] = "SUM_LEN跨补齐区间，缺少对应层覆盖"
        else:
            update["failure_reason"] = update["status"] + "，见prediction.log"
    return update


def summarize(rows, mode):
    selected = [row["results"][mode] for row in rows if row["results"][mode]["status"] == "completed"]
    return {"planned": len(rows), "completed": len(selected),
            "mape_percent": {name: statistics.mean(abs(row[name + "_error_pct"]) for row in selected)
                             for name, _, _ in METRICS} if selected else None}


def metric_cell(row, name):
    values = [f"{row['actual'][name]:.2f}"]
    for mode in MODES:
        result = row["results"][mode]
        values.append(f"{result[name + '_predicted']:.2f}（{result[name + '_error_pct']:+.2f}%）"
                      if result["status"] == "completed" else "未覆盖")
    return " / ".join(values)


def write_report(args, manifest):
    previous = json.loads(args.baseline.read_text())
    old_rows = {int(row["case"]): row for row in previous["historical_rows"]}
    rows = []
    unchanged = []
    for record in manifest["records"]:
        history = record["historical"]
        row = {"number": record["number"], "group": group_for(record["number"]),
               "nodes": 4 if record["number"] <= 23 else 2,
               "case": history["case"], "input_length": history["input_length"],
               "output_length": history["output_length"], "concurrency": history["concurrency"],
               "requests": history["requests"], "actual": record["actual"],
               "kv_pages_measured": history["observed_blocks_per_rank"],
               "measurement_path": history["measurement_path"],
               "measurement_sha256": record["measurement_sha256"],
               "measured_prefill_batch": history["measured_prefill_batch"],
               "measured_decode_batch": history["measured_decode_batch"], "results": {}}
        spec = history["configuration_spec"]
        row["parallelism"] = "; ".join("/".join(str(spec[role + "_" + name + "_SIZE"])
                                                for name in ("TP", "DP", "EP"))
                                        for role in ("PREFILL", "DECODE"))
        for mode, job_id in record["jobs"].items():
            job = manifest["jobs"][job_id]
            if digest(Path(job["config"])) != job["config_sha256"]:
                raise ValueError("Configuration changed during replay")
            result = {"status": job["status"], "job": job_id, "config": job["config"],
                      "config_sha256": job["config_sha256"], "kv_pages_predicted": job["kv_pages"]}
            if job["status"] == "completed":
                path = Path(job["folder"]) / "prediction/prediction.json"
                prediction = json.loads(path.read_text())
                measured = {actual_key: row["actual"][name] for name, actual_key, _ in METRICS}
                result.update(compare(measured, prediction, row["requests"], row["input_length"], row["output_length"]))
                result.update(prediction_path=str(path), prediction_sha256=digest(path),
                              prefill_batch=batch_summary(prediction, "prefill"),
                              decode_batch=batch_summary(prediction, "decode"))
            else:
                result["failure_reason"] = job["failure_reason"]
            row["results"][mode] = result
        prior = old_rows[row["number"]]
        if row["actual"] != prior["actual"]:
            raise ValueError("Historical measurements changed")
        result = row["results"]["L"]
        if prior.get("after") and result["status"] == "completed":
            unchanged.append(all(abs(result[name + "_predicted"] - prior["after"][name + "_predicted"]) < 1e-8
                                 for name, _, _ in METRICS))
        rows.append(row)
    if len(rows) != 39:
        raise ValueError("Expected all39 historical observations")
    cohorts = {"全部39条": rows, "四节点": [row for row in rows if row["nodes"] == 4],
               "两节点": [row for row in rows if row["nodes"] == 2]}
    cohorts.update({GROUPS[group]: [row for row in rows if row["group"] == group] for group in GROUPS if group != "E"})
    summaries = {name: {mode: summarize(selected, mode) for mode in MODES} for name, selected in cohorts.items()}
    paired = [row for row in rows if all(result["status"] == "completed" for result in row["results"].values())]
    report = {"created": datetime.datetime.now(datetime.timezone.utc).isoformat(), "rows": rows,
              "summary": summaries, "paired_summary": {mode: summarize(paired, mode) for mode in MODES},
              "layer_replay_prior_successes": len(unchanged), "layer_replay_unchanged": all(unchanged),
              "scope": manifest["scope"], "source_manifest_sha256": digest(args.source),
              "baseline_sha256": digest(args.baseline), "manifest_sha256": digest(args.output / "manifest.json"),
              "runtime": manifest["runtime"], "accuracy_accepted": False}
    save(args.output / "comparison.json", report)
    flat = []
    for row in rows:
        for mode, result in row["results"].items():
            item = {key: row[key] for key in ("number", "group", "nodes", "case", "input_length", "output_length", "concurrency", "requests", "parallelism")}
            item.update(mode=mode, status=result["status"], kv_pages_measured=json.dumps(row["kv_pages_measured"]),
                        kv_pages_predicted=json.dumps(result["kv_pages_predicted"]))
            for name, _, _ in METRICS:
                item[name + "_measured"] = row["actual"][name]
                item[name + "_predicted"] = result.get(name + "_predicted")
                item[name + "_error_pct"] = result.get(name + "_error_pct")
            for role in ("prefill", "decode"):
                measured = row["measured_" + role + "_batch"]
                predicted = result.get(role + "_batch")
                item[role + "_batch_measured"] = measured["mean"] if measured else None
                item[role + "_batch_predicted"] = predicted["mean"] if predicted else None
            item.update(failure_reason=result.get("failure_reason", ""), config=result["config"],
                        measurement_path=row["measurement_path"], prediction_path=result.get("prediction_path", ""))
            flat.append(item)
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = ["# 两节点、四节点历史实测：当前版本重新预测", "", "日期：2026-11-04。39条历史实测原值保留，使用当前镜像重新执行预测；并非新启动模型实测。", "",
             "预先声明同时运行K与L，不按误差挑选模式，失败不切换。K=Prefill与Decode使用独立算子；L=Prefill使用独立算子，纯Decode内部使用实测decoder层组合。两套配置仅修改Decode的decode_timing_scope。", "",
             "固定GLM-5.2 W8A8、SGLang0.5.22、关闭graph/调度overlap/投机解码；保留原PD实例数、并行、请求数、KV容量和调度参数。shared-expert内部多流与调度overlap是不同设置，此处保留真实部署的shared_expert_overlap=true。", "",
             "使用独立采集的CPU客户端准备参数与Prefill专家计数分布，没有引入目标请求到达轨迹、实测forward耗时或固定残差。KV容量采用历史观测值；这是历史回放对照，不是新workload盲测。39条包含重复实测，实际31种不同配置、每种运行K/L共62个预测任务。", "",
             f"预测镜像：`{manifest['runtime']['tag']}`；ID：`{manifest['runtime']['id']}`；源码提交：`{manifest['runtime']['source_commit']}`。", "",
             "## 平均误差与覆盖", "", "MAPE=成功预测项的平均绝对百分比误差；失败项不进入数值平均，但保留分母。K/L成功子集不同，另列共同成功子集。输入输出长度固定时，两种吞吐的相对误差相同。", "",
             "| 数据组 | 模式 | 完成/总数 | TTFT MAPE | TPOT MAPE | 输入吞吐MAPE | 输出吞吐MAPE |",
             "|---|---|---|---|---|---|---|"]
    for name, summary in summaries.items():
        for mode, values in summary.items():
            errors = values["mape_percent"]
            lines.append(f"| {name} | {mode} | {values['completed']}/{values['planned']} | " + " | ".join(f"{errors[metric]:.2f}%" for metric, _, _ in METRICS) + " |")
    for mode, values in report["paired_summary"].items():
        lines.append(f"| K/L共同成功子集 | {mode} | {values['completed']}/{values['planned']} | " + " | ".join(f"{values['mape_percent'][metric]:.2f}%" for metric, _, _ in METRICS) + " |")
    lines += ["", f"与上次L报告逐项核对：{len(unchanged)}条可配对成功结果，数值全部一致={all(unchanged)}。本次没有修改模型公式或性能表，重新运行本身不意味着精度改善。", "",
              "## 配置与逐条性能", "", "每个性能单元格依次为：**实测 / K预测（有符号误差） / L预测（有符号误差）**。误差=(预测/实测−1)×100%；负值表示预测偏低。TTFT/TPOT单位ms，吞吐单位token/s。C/N=并发/请求总数。", "",
              "A=4节点2P1D（D跨两节点）；B=4节点1P3D；C=4节点2P2D；D=4节点3P1D；E=2节点1P1D。并行为SGLang全局TP/Attention DP/EP，P；D分别列出。KV列为实测每实例页数→预测每rank页数，页大小128token；多实例容量不同时沿用历史配置的最小值。", "",
              "| 编号/组 | I/O | C/N | P；D：TP/ADP/EP | KV页：实→预 | TTFT：实/K/L | TPOT：实/K/L | 输入吞吐：实/K/L | 输出吞吐：实/K/L |",
              "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        capacity = row["results"]["K"]["kv_pages_predicted"]
        if capacity != row["results"]["L"]["kv_pages_predicted"]:
            raise ValueError("Mode change altered KV capacity")
        pages = "; ".join(f"{role[0].upper()}{row['kv_pages_measured'][role]}→{capacity[role]}" for role in ("prefill", "decode"))
        lines.append(f"| {row['number']:02d}·{row['group']} | {row['input_length']}/{row['output_length']} | {row['concurrency']}/{row['requests']} | {row['parallelism']} | {pages} | " + " | ".join(metric_cell(row, name) for name, _, _ in METRICS) + " |")
    lines += ["", "## Running batch", "", "实测/K/L均按完整logical worker的逐轮batch算术平均；不把客户端并发、padding或等待队列算作running batch。缺逐轮实测的项目保留缺失，不反推补齐。", "",
              "| 编号/组 | P平均batch：实/K/L | D平均batch：实/K/L |", "|---|---|---|"]
    for row in rows:
        columns = []
        for role in ("prefill", "decode"):
            actual = row["measured_" + role + "_batch"]
            values = [f"{actual['mean']:.3f}" if actual else "缺逐轮实测"]
            for mode in MODES:
                batch = row["results"][mode].get(role + "_batch")
                values.append(f"{batch['mean']:.3f}" if batch else "未覆盖")
            columns.append(" / ".join(values))
        lines.append(f"| {row['number']:02d}·{row['group']} | " + " | ".join(columns) + " |")
    lines += ["", "## 未覆盖项与证据", ""]
    for row in rows:
        for mode, result in row["results"].items():
            if result["status"] != "completed":
                lines.append(f"- {row['number']:02d}·{row['group']}，{mode}：{result['failure_reason']}；配置和完整错误日志保留在`{Path(result['config']).parent}`。")
    lines += ["", "L精度不能当作纯独立kernel的精度；K与L分别验收。历史回放、已知KV容量及少数拓扑覆盖均不能证明任意workload/PD/并行策略的误差低于20%。", "",
              f"机器可读表：`{args.output}/comparison.csv`；原始预测与请求记录：`{args.output}/jobs/`；配置、来源哈希和逐任务命令：`{args.output}/manifest.json`。", ""]
    args.document.write_text("\n".join(lines))
    print(json.dumps({"summary": summaries, "layer_replay_unchanged": all(unchanged)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "baseline", "runtime", "output", "document"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    if args.document.exists() or args.jobs < 1:
        raise ValueError("Use a fresh report and positive concurrency")
    source = json.loads(args.source.read_text())
    root = Path(__file__).resolve().parent
    repo = root.parents[4]
    archives = json.loads((root / "joint-serving-comparison-manifest.json").read_text())
    artifacts = {item["source"]: item for item in archives["artifacts"]}
    manifest = {"status": "preparing", "scope": "Fresh39-row same-condition historical replay; predeclared K/L comparison, no fitting or fallback",
                "runtime": json.loads(args.runtime.read_text()), "source_sha256": digest(args.source),
                "runner_sha256": digest(Path(__file__)), "records": [], "jobs": {},
                "started": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    for original in source["records"]:
        history = original["historical"]
        measurement = Path(history["measurement_path"])
        artifact = artifacts[str(measurement)]
        archived = repo / artifact["archive"]
        if digest(measurement) != artifact["source_sha256"] or digest(archived) != artifact["archive_sha256"]:
            raise ValueError("Historical evidence hash mismatch")
        if hashlib.sha256(gzip.decompress(archived.read_bytes())).hexdigest() != digest(measurement):
            raise ValueError("Archive payload differs from original measurement")
        measured = json.loads(measurement.read_text().splitlines()[-1])
        if (measured["completed"], measured["total_input_tokens"], measured["total_output_tokens"]) != (
                history["requests"], history["requests"] * history["input_length"], history["requests"] * history["output_length"]):
            raise ValueError("Measured request/token conservation failed")
        actual = {name: measured[actual_key] for name, actual_key, _ in METRICS}
        if any(actual[name] != history["measured_" + name] for name in actual):
            raise ValueError("Stored historical summary differs from raw benchmark")
        origin = source["jobs"][original["job"]]
        config_path = Path(origin["config"])
        if digest(config_path) != origin["config_sha256"]:
            raise ValueError("Original config changed")
        config = yaml.load(config_path.read_text(), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
        if "eager_runtime_overhead" in json.dumps(config) or config["engine"]["nextn"] != 0:
            raise ValueError("Residual or speculation found")
        for role in ("prefill", "decode"):
            timing = config["engine"]["workers"][role]["timing"]
            if timing["fallback_policy"] != "deny" or timing["estimator_config"]["correction"]["enabled"]:
                raise ValueError("Unexpected fallback or correction")
            if timing["estimator_config"]["op_level"]["ascend"]["decode_execution_mode"] != "eager":
                raise ValueError("Unexpected graph mode")
        record = {"number": original["number"], "historical": history, "actual": actual,
                  "measurement_sha256": digest(measurement), "jobs": {}}
        for mode, scope in MODES.items():
            candidate = copy.deepcopy(config)
            candidate["engine"]["workers"]["decode"]["timing"]["estimator_config"]["op_level"]["ascend"]["decode_timing_scope"] = scope
            control = copy.deepcopy(candidate)
            control["engine"]["workers"]["decode"]["timing"]["estimator_config"]["op_level"]["ascend"]["decode_timing_scope"] = "runtime_layers"
            if control != config:
                raise ValueError("More than decode timing scope changed")
            job_id = mode + "-" + hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest()
            record["jobs"][mode] = job_id
            if job_id not in manifest["jobs"]:
                folder = args.output / "jobs" / job_id
                folder.mkdir(parents=True, exist_ok=False)
                path = folder / "config.yaml"
                path.write_text(yaml.dump(candidate, Dumper=getattr(yaml, "CSafeDumper", yaml.SafeDumper), sort_keys=False))
                manifest["jobs"][job_id] = {"mode": mode, "config": str(path), "config_sha256": digest(path),
                                            "folder": str(folder), "row_numbers": [], "status": "pending",
                                            "kv_pages": {role: candidate["engine"]["workers"][role]["kv_cache"]["capacity"]["blocks"]
                                                         for role in ("prefill", "decode")}}
            manifest["jobs"][job_id]["row_numbers"].append(record["number"])
        manifest["records"].append(record)
    destination = args.output / "manifest.json"
    manifest["status"] = "running"
    save(destination, manifest)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_job, job): job_id for job_id, job in manifest["jobs"].items()}
        for future in as_completed(futures):
            job = manifest["jobs"][futures[future]]
            job.update(future.result())
            save(destination, manifest)
            print(json.dumps({key: job[key] for key in ("row_numbers", "mode", "status", "wall_seconds")}), flush=True)
    manifest["status"] = "completed" if all(job["status"] == "completed" for job in manifest["jobs"].values()) else "completed_with_coverage_or_execution_failures"
    manifest["completed"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(destination, manifest)
    write_report(args, manifest)


if __name__ == "__main__":
    main()
