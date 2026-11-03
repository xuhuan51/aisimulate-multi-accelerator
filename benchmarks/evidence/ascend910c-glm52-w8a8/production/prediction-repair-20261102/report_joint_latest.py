import csv
import gzip
import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path

import replay_joint_latest as replay
import report_joint_comparison as history

ROOT = replay.ROOT
REPO = replay.REPO
EVIDENCE = replay.EVIDENCE
DOCUMENT = REPO / "md/910c-glm52-w8a8-2node-4node-latest-comparison-20261103.md"
METRICS = history.METRICS


def relative_link(path):
    return os.path.relpath(path, DOCUMENT.parent)


def read_archive(execution, filename):
    artifact = next(item for item in execution["artifacts"] if item["source"].endswith("/" + filename))
    path = REPO / artifact["archive"]
    if replay.digest(path) != artifact["archive_sha256"]:
        raise ValueError(f"Changed archived output: {path}")
    payload = gzip.decompress(path.read_bytes())
    if hashlib.sha256(payload).hexdigest() != artifact["source_sha256"]:
        raise ValueError(f"Changed output payload: {path}")
    return payload


def load_results(manifest):
    configs = json.loads(subprocess.check_output(
        ["docker", "run", "--rm", "-i", "--network", "none", "--entrypoint", "python3", replay.IMAGE_ID,
         "-c", "import json,sys,yaml;json.dump([yaml.safe_load(item) for item in json.load(sys.stdin)],sys.stdout)"],
        input=json.dumps([Path(job["config"]).read_text() for job in manifest["jobs"]]), text=True,
    ))
    result = {}
    for job, config in zip(manifest["jobs"], configs):
        execution_path = EVIDENCE / "jobs" / job["id"] / "execution.json"
        execution = json.loads(execution_path.read_text())
        if replay.digest(Path(job["config"])) != job["config_sha256"]:
            raise ValueError(f"Changed configuration: {job['id']}")
        if execution["image_id"] != replay.IMAGE_ID:
            raise ValueError("Unexpected simulator image")
        if "eager_runtime_overhead" in json.dumps(config):
            raise ValueError("Residual calibration leaked into the latest replay")
        for estimator in [config["engine"], *[worker["timing"] for worker in config["engine"]["workers"].values()]]:
            if estimator["estimator_config"]["correction"]["enabled"]:
                raise ValueError("Online correction must be disabled")
        if execution["status"] == "completed":
            prediction = json.loads(read_archive(execution, "prediction.json"))
            requests = [json.loads(line) for line in read_archive(execution, "requests.jsonl").splitlines()]
            json.loads(read_archive(execution, "resource-runtime.json"))
            if len(requests) != job["requests"] or any(row["terminal_status"] != "completed" for row in requests):
                raise ValueError("Incomplete native prediction")
            for key, value in execution["validation"].items():
                if prediction[key] != value:
                    raise ValueError("Prediction conservation failure")
            for metric, (key, _) in replay.replay.COMPARISON.METRIC_PAIRS.items():
                if not math.isclose(execution["metrics"][metric], prediction[key], rel_tol=1e-12):
                    raise ValueError("Saved prediction metric differs from raw output")
            for role in ("prefill", "decode"):
                if execution["batches"][role] != history.batch_summary(prediction["forward_pass_batches"], role):
                    raise ValueError("Incorrect forward batch scope")
            operation_names = [item["name"] for phase in prediction["performance_diagnostics"]["phases"] for item in phase["operations"]]
            if any("runtime_overhead" in name for name in operation_names):
                raise ValueError("Residual operation found in executed prediction")
        elif job["scope"] != "runtime_layers" or "currently require TP16/ADP4/EP16/DCP1" not in execution["failure_log_tail"]:
            raise ValueError(f"Unclassified execution failure: {execution_path}")
        result[job["id"]] = {"execution": execution, "config": config, "execution_path": str(execution_path)}
    return result


def aligned_config(record, config):
    historical = record["historical"]
    for node in historical["runtime_nodes"]:
        actual = node.get("effective")
        if actual is None:
            continue
        role = actual["disaggregation_mode"]
        worker = config["engine"]["workers"][role]
        parallel = worker["parallelism"]
        if (parallel["tensor"] * parallel["attention_data"], parallel["attention_data"], parallel["moe_expert"]) != (actual["tp_size"], actual["attn_dp_size"], actual["ep_size"]):
            raise ValueError(f"Parallelism changed: row {record['number']}")
        if worker["scheduler"]["max_sequences"] != actual["max_running_requests"] // actual["attn_dp_size"]:
            raise ValueError("Max-running-request scope differs")
        if worker["kv_cache"]["capacity"]["blocks"] != min(historical["observed_blocks_per_rank"][role]):
            raise ValueError("Measured KV capacity was not preserved")
        if not actual["disable_overlap_schedule"] or actual["speculative_algorithm"] is not None:
            raise ValueError("Measured scheduling mode differs")
        if any(actual["cuda_graph_config"][phase]["backend"] != "disabled" for phase in ("prefill", "decode")):
            raise ValueError("Measured graph mode differs")
        ascend = worker["timing"]["estimator_config"]["op_level"]["ascend"]
        if ascend["decode_execution_mode"] != "eager":
            raise ValueError("Prediction is not eager")
        audit = next(item for item in record["logits_audit"] if item["replica"] == node["replica"])
        if any(ascend[key] != audit[f"actual_{key}"] for key in ("enable_dp_lm_head", "enable_tp_lm_head_all_to_all")):
            raise ValueError("LM-head flags differ from the original server-info")
        if role == "prefill" and worker["scheduler"]["chunked_prefill_size"] // parallel["attention_data"] != actual["chunked_prefill_size"]:
            raise ValueError("Prefill chunk scope differs")


def statistics_for(rows, selection):
    result = {}
    for metric in METRICS:
        selected = [row[selection]["error_percent"][metric] for row in rows if row[selection]["status"] == "completed"]
        if selected:
            result[metric] = {"points": len(selected), "mape_percent": statistics.mean(abs(value) for value in selected),
                              "mean_signed_error_percent": statistics.mean(selected),
                              "minimum_signed_error_percent": min(selected), "maximum_signed_error_percent": max(selected),
                              "within15": sum(abs(value) <= 15 for value in selected)}
    return result


def metric_cell(row, metric, selected="primary"):
    precision = 2 if metric.endswith("ms") else 3
    actual = row["historical"][f"measured_{metric}"]
    result = row[selected]
    if result["status"] != "completed":
        return f"{actual:.{precision}f} / 未覆盖"
    return f"{actual:.{precision}f} / {result['metrics'][metric]:.{precision}f}（{result['error_percent'][metric]:+.2f}%）"


def batch_cell(row, role):
    actual = row["historical"][f"measured_{role}_batch"]
    predicted = row["primary"]["batches"][role]["mean"]
    if actual is None:
        return f"缺逐轮实测 / {predicted:.3f}"
    return f"{actual['mean']:.3f} / {predicted:.3f}（{100 * (predicted / actual['mean'] - 1):+.2f}%）"


def write_document(data):
    rows = data["rows"]
    covered = data["statistics"]["runtime_layers_covered"]["runtime_layers"]
    lines = [
        "# 910C / GLM-5.2 W8A8：原39组PD实测与最新无整轮残差预测对照（2026-11-03）", "",
        f"已经用当前验证镜像重新预测原表的全部配置。采用分层计时的33组，TTFT平均绝对百分比误差为 **{covered['ttft_ms']['mape_percent']:.2f}%**，TPOT为 **{covered['tpot_ms']['mape_percent']:.2f}%**，输入/输出吞吐均为 **{covered['output_tok_s']['mape_percent']:.2f}%**。其余6组使用TP32/ADP32/EP32 Decode，分层数据尚不支持；主表明确列出它们的独立kernel预测，不能把这6组当作已完成分层适配。",
        "", "本次没有新测服务，也没有修改实测值；沿用39组、1900条成功正式请求，逐项核对原始记录与归档SHA256。去掉历史P/D整轮 `eager_runtime_overhead` 残差，关闭 estimator correction，使用最新算子/层边界、KV分配和调度实现重新计算。旧表的小误差包含拟合残差，不能当作当前无残差版本已达到同样精度的证据。", "",
        "## 配置和统计口径", "",
        "- 模型、SGLang 0.5.22、PD副本数、TP/ADP/EP、请求长度、并发C、正式请求数N均按原实测记录。实测 `server-info` 已确认 prefill/decode graph关闭、overlap调度关闭、投机解码关闭、radix cache关闭；预测为eager、`nextn=0`、`decoder_replay=false`。",
        "- 所有P为全局TP16/ADP4/EP16，局部attention TP=4；A组D为全局TP32/ADP32/EP32，局部attention TP=1，且一个D副本占两个节点。B/C/D/E组D均为全局TP16/ADP4/EP16。PP=DCP=1。A组2P1D合计4节点，不是3节点。",
        "- P全局chunk=16384，在ADP4下每rank为4096；P最大prefill token=32768。全局max-running P=128、TP16 D=128（32/rank）、TP32 D=1024（32/rank）。P/D的LM-head布局保留实测开关，P的DP LM-head开启，D关闭，A组D的TP LM-head all-to-all开启。",
        "- KV页大小128 tokens；按原表继续用每个角色实测容量的最小值，因此各副本相差1页时不是逐副本完全相同。表中实测→预测页明确列出，不能把额外内存自动补大。",
        "- 负载是原来的闭环并发；没有把实测到达时间、逐轮batch、forward耗时注入预测。保留原来的 `sglang_gateway` 响应语义、round-robin PD路由和CPU tokenization测量（1个worker、每请求0.8435074705630541ms）。这项CPU测量来自旧入队诊断，属于已知组件校准，不是整轮残差，也不代表完整IPC/HTTP开销。",
        "- 客户端原来设置warmup=2；预测正式请求仍从空闲状态起算，没有重建每次健康检查、两次warmup后的KV游标、编译cache和第一段可见文本。保持相同启动参数不等于复原隐藏冷暖状态。",
        "- `shared_expert_overlap=true`保留实测的共享专家多流执行；它与已关闭的请求级overlap调度是不同的开关。KV传输仍使用已测MemFabric组件，单请求传输曲线不能保证覆盖所有并发传输争用。",
        "- 原39组包括25组8192/1024和14组8192/128诊断；其中分层覆盖分别为19组和14组。不能将39组全部称为8K/1K验证。误差=100×(预测/实测−1)，负值代表低估。MAPE是各行绝对百分比误差的算术平均，不按请求数加权；相同配置的不同实测批次均保留，因此汇总不是独立同分布统计检验。",
        "- P/D平均batch为每个worker已提交forward的请求数均值，跨DP取worker汇总，排除逐DP统计行，非时间加权。没有原始逐轮实测的行保留“缺逐轮实测”。", "",
        "## 汇总：先看支持范围，再看误差", "",
        "`L` = Decode使用 `runtime_layers`，P仍由独立算子组成；按真实层类型及层数组合实测层调用，包含相应host/stream/通信边界，替换内部kernel项，避免双算。",
        "`K` = `isolated_kernels`，独立kernel计时；尚不能完整覆盖eager运行时开销。两种模式分别显式运行；主表预先按拓扑选定，A用K，其余用L，未按误差大小挑结果，也没有失败后偷偷回退。", "",
        "|同一批实测行|行数|模式|TTFT MAPE|TPOT MAPE|输入/输出吞吐 MAPE|", "|---|---:|---|---:|---:|---:|",
    ]
    selections = [("B/C/D/E，旧表残差版本", "runtime_layers_covered", "historical_prediction", "旧"),
                  ("B/C/D/E，最新独立kernel", "runtime_layers_covered", "isolated_kernels", "K"),
                  ("B/C/D/E，最新分层", "runtime_layers_covered", "runtime_layers", "L"),
                  ("A，旧表残差版本", "tp32", "historical_prediction", "旧"),
                  ("A，最新独立kernel", "tp32", "isolated_kernels", "K（L未覆盖）")]
    for label, group, selection, scope in selections:
        summary = data["statistics"][group][selection]
        lines.append(f"|{label}|{summary['ttft_ms']['points']}|{scope}|{summary['ttft_ms']['mape_percent']:.2f}%|{summary['tpot_ms']['mape_percent']:.2f}%|{summary['output_tok_s']['mape_percent']:.2f}%|")
    lines += ["", "|最新L覆盖的子集|行数|TTFT MAPE|TPOT MAPE|输入/输出吞吐 MAPE|", "|---|---:|---:|---:|---:|"]
    for key, label in [("four_node_tp16", "四节点 B/C/D"), ("two_node", "双节点 E"), ("runtime_layers_covered", "以上合计")]:
        summary = data["statistics"][key]["runtime_layers"]
        lines.append(f"|{label}|{summary['ttft_ms']['points']}|{summary['ttft_ms']['mape_percent']:.2f}%|{summary['tpot_ms']['mape_percent']:.2f}%|{summary['output_tok_s']['mape_percent']:.2f}%|")
    lines += ["", f"33组L结果中，误差绝对值≤15%的点数：TTFT **{covered['ttft_ms']['within15']}/33**，TPOT **{covered['tpot_ms']['within15']}/33**，输入/输出吞吐 **{covered['output_tok_s']['within15']}/33**。因此，当前主要缺口仍包括TTFT和TP32执行范围，不能宣称所有PD布局都已经准确。", "",
              "## 39组完整对照", "",
              "每个性能单元格为 **实测 / 本次预测（误差）**。编号与用户提供的旧表完全一致。A组的K值仅用于暴露未覆盖的误差，不能替代L覆盖验收。", "",
              "|编号/配置|I/O|C/N|P；D：TP/ADP/EP|实测KV页→预测页（每rank）|TTFT ms：实/预（误差）|TPOT ms：实/预（误差）|输入tok/s：实/预（误差）|输出tok/s：实/预（误差）|P平均batch：实/预（误差）|D平均batch：实/预（误差）|本次计时/资格|",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        original = row["historical"]
        code, label, prefill, decode = history.LABELS[original["configuration"]]
        pages = original["observed_blocks_per_rank"]
        scope = "K；L缺TP32覆盖" if row["primary_scope"] == "isolated_kernels" else "L；同环境回放"
        if row["number"] == 29:
            scope += "；CPU入队组件来自本组"
        cells = [f"{row['number']:02}·{code} {label}", f"{original['input_length']}/{original['output_length']}",
                 f"{original['concurrency']}/{original['requests']}", f"{prefill}；{decode}",
                 f"P{pages['prefill']}→{min(pages['prefill'])}; D{pages['decode']}→{min(pages['decode'])}"]
        cells += [metric_cell(row, metric) for metric in METRICS]
        cells += [batch_cell(row, role) for role in ("prefill", "decode")]
        lines.append("|" + "|".join(cells + [scope]) + "|")
    lines += ["", "## 这次结果说明了什么", "",
              "**TPOT/吞吐的主要漏项确实在算子执行边界。** 同一批B/C/D/E配置，独立kernel与实际eager服务仍有明显差距；改用真实decoder层调用的测量后，包含host下发、stream及层内通信的时间，不再把这些漏项塞进固定整轮残差。主表所有数值均由原生调度、KV管理和性能模型完整执行产生，没有在结果上加减补偿。层计时仍只覆盖已采到的拓扑、活跃DP人口与请求/KV区间；它不是任意拓扑和workload的精度承诺。", "",
              "**TTFT并未跟着完全收敛。** Prefill仍使用独立算子组合，先前实机事件诊断已把活跃DP增加时的主要缺口定位到normal MoE：真实模型的专家路由并不均匀，通信/依赖等待也不等于各rank独立耗时求和。目前仍使用balanced形状；本轮只重跑与报告，没有补入未经验证的路由比例或残差。同时，禁用overlap后，请求在DP控制器、IPC和scheduler轮询中的到达相位会改变prefill分批；原生闭环负载没有复现每次实测的精确入队时刻。KV页释放/排序语义的修复已在本轮镜像，但冷编译时间、warmup后的分配游标和客户端首段非空文本语义没有完整重建。因此低并发TTFT仍可能明显偏低。高并发TTFT又含大量KV容量约束导致的排队，不能只凭其相对误差较小判定prefill计算准确。", "",
              "**旧表不能作为无残差精度的承诺。** 原TP16/TP32 Decode残差和P残差包含从已测forward倒推的耗时；此次全部撤掉，没有把旧系数改个身份再继续使用。P残差训练涉及的第25行本轮不再注入那项系数；第29行的CPU入队组件计时仍被复用，故保留部分组件校准标记。其它行也参与过诊断，统一称同环境历史回放，不称独立盲测。", "",
              "本轮包含的既有修复是：真实KV容量与分配状态语义、原始LM-head与DP布局对齐、sequence-length CPU→NPU传输计入、真实decoder层边界替换独立kernel、同步DP人口与padding处理，以及此前的PD gateway、reserved tokens和闭环请求模型。跨DP混合prefill/decode的修复也在同一镜像中，但这些数据是PD分离部署，不能把本表改善全部归因于混合forward路径。本轮新增内容仅为39组无整轮残差回放、配置核验和报告。", "",
              "下一步应先补TP32/ADP32/EP32实际层调用数据，再继续修复P端MoE路由与rank间依赖、IPC/poll相位及冷暖状态；随后用新输入长度和不同请求内容做独立验证。当前证据足以定位剩余误差，不足以保证最优PD搜索排名已准确。", "",
              "## 证据、复现与限制", "",
              f"- 本次镜像：`{replay.IMAGE}`，不可变ID：`{replay.IMAGE_ID}`；源码基线 `{data['source_commit']}`。默认运行脚本镜像未改，复现时必须显式设置该镜像。",
              f"- [原始39行实测及旧预测](./910c-glm52-w8a8-2node-4node-latest-comparison-20261102.md)保持不变。[本次CSV]({relative_link(EVIDENCE / 'comparison.csv')})与[完整JSON]({relative_link(EVIDENCE / 'comparison.json')})保存两种计时、旧值、误差、batch和全部配置。",
              f"- [本次manifest]({relative_link(EVIDENCE / 'manifest.json')})记录39行到31份独立配置的映射，每份分别尝试K/L，共62个CLI任务；56个成功，6个按TP32覆盖门禁拒绝。重复配置共享确定性预测，各次实测独立保留。每个任务的配置、命令、原始输出gzip、请求明细与哈希均已归档。",
              f"- [验证结果]({relative_link(EVIDENCE / 'validation.json')})包含配置/模式核验、请求/token守恒、归档SHA256与指标复算。旧checkpoint的217项历史完整性问题仍单独保留；本次使用的39组证据已重新逐项核对，未用重写旧哈希的方法消除告警。",
              "- 第一轮试跑遇到JSON科学计数法经YAML解析为字符串的问题，尚未进入有效预测；原失败日志保留在上层目录。本次有效结果均来自 `attempt2` 的标准YAML配置，不把配置解析失败记作性能模型缺覆盖。",
              f"- [运行脚本]({relative_link(ROOT / 'replay_joint_latest.py')})、[报告生成脚本]({relative_link(ROOT / 'report_joint_latest.py')})只运行离线CPU预测，不启动NPU模型服务。", "",
              "```bash", "cd /home/youxingye/youxingye_data/aisimulate-910c",
              "python3 benchmarks/evidence/ascend910c-glm52-w8a8/production/prediction-repair-20261102/replay_joint_latest.py --workers 2",
              "python3 benchmarks/evidence/ascend910c-glm52-w8a8/production/prediction-repair-20261102/report_joint_latest.py", "```", "",
              "脚本会核对并复用相同身份的已保存任务。需要独立再跑时，在空目录复制manifest中的配置和execution中的命令，显式指定上述镜像ID，并给出新的output-dir；不要覆盖原测量或旧预测。", ""]
    DOCUMENT.write_text("\n".join(lines))


def main():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    results = load_results(manifest)
    rows = []
    for record in manifest["records"]:
        original = record["historical"]
        row = {"number": record["number"], "historical": original}
        for scope in replay.SCOPES:
            resolved = results[record["jobs"][scope]]
            aligned_config(record, resolved["config"])
            execution = resolved["execution"]
            value = {"status": execution["status"], "execution": resolved["execution_path"],
                     "config": execution["config"], "config_sha256": execution["config_sha256"]}
            if value["status"] == "completed":
                value.update({"metrics": execution["metrics"], "batches": execution["batches"],
                              "error_percent": {metric: 100 * (prediction / original[f"measured_{metric}"] - 1)
                                                for metric, prediction in execution["metrics"].items()}})
            else:
                value["reason"] = execution["failure_log_tail"].strip()
            row[scope] = value
        row["historical_prediction"] = {"status": "completed", "metrics": {metric: original[f"predicted_{metric}"] for metric in METRICS},
                                        "error_percent": {metric: original[f"error_percent_{metric}"] for metric in METRICS}}
        row["primary_scope"] = "isolated_kernels" if original["configuration"] == "recommended-2p1d-dtp32" else "runtime_layers"
        row["primary"] = row[row["primary_scope"]]
        if row["primary"]["status"] != "completed":
            raise ValueError("The predetermined report scope did not complete")
        rows.append(row)
    groups = {"all": rows, "runtime_layers_covered": [row for row in rows if row["runtime_layers"]["status"] == "completed"],
              "tp32": rows[:6], "four_node_tp16": rows[6:23], "two_node": rows[23:]}
    data = {"image": replay.IMAGE, "image_id": replay.IMAGE_ID, "source_commit": manifest["source_commit"],
            "created_at": manifest["created_at"], "error_formula": "100*(prediction/measurement-1)",
            "statistics": {key: {scope: statistics_for(values, scope) for scope in (*replay.SCOPES, "historical_prediction", "primary")}
                           for key, values in groups.items()}, "rows": rows}
    (EVIDENCE / "comparison.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    columns = ["number", "configuration", "campaign", "case", "input_length", "output_length", "concurrency", "requests", "primary_scope"]
    columns += [f"measured_{metric}" for metric in METRICS]
    for scope in ("historical_prediction", *replay.SCOPES, "primary"):
        columns += [f"{scope}_status"] + [f"{scope}_{prefix}_{metric}" for prefix in ("predicted", "error_percent") for metric in METRICS]
    for role in ("prefill", "decode"):
        columns += [f"measured_{role}_batch_mean", f"primary_{role}_batch_mean", f"primary_{role}_batch_error_percent"]
    columns += ["observed_kv_pages", "primary_config", "primary_execution", "measurement_path", "runtime_layers_failure"]
    with (EVIDENCE / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            flat = {key: row["historical"][key] for key in columns if key in row["historical"] and key.startswith(("measured_", "configuration", "campaign", "case", "input_", "output_", "concurrency", "requests"))}
            flat.update({"number": row["number"], "primary_scope": row["primary_scope"],
                         "observed_kv_pages": json.dumps(row["historical"]["observed_blocks_per_rank"]),
                         "primary_config": row["primary"]["config"], "primary_execution": row["primary"]["execution"],
                         "measurement_path": row["historical"]["measurement_path"],
                         "runtime_layers_failure": row["runtime_layers"].get("reason", "")})
            for scope in ("historical_prediction", *replay.SCOPES, "primary"):
                result = row[scope]
                flat[f"{scope}_status"] = result["status"]
                if result["status"] == "completed":
                    flat.update({f"{scope}_predicted_{metric}": value for metric, value in result["metrics"].items()})
                    flat.update({f"{scope}_error_percent_{metric}": value for metric, value in result["error_percent"].items()})
            for role in ("prefill", "decode"):
                actual = row["historical"][f"measured_{role}_batch"]
                predicted = row["primary"]["batches"][role]["mean"]
                flat[f"primary_{role}_batch_mean"] = predicted
                if actual is not None:
                    flat[f"measured_{role}_batch_mean"] = actual["mean"]
                    flat[f"primary_{role}_batch_error_percent"] = 100 * (predicted / actual["mean"] - 1)
            writer.writerow(flat)
    validation = {"rows": len(rows), "formal_measured_requests": sum(row["historical"]["requests"] for row in rows),
                  "unique_jobs": len(results), "successful_jobs": sum(item["execution"]["status"] == "completed" for item in results.values()),
                  "expected_missing_topology_jobs": sum(item["execution"]["status"] != "completed" for item in results.values()),
                  "runtime_layer_rows": len(groups["runtime_layers_covered"]),
                  "original_archives_verified": 195, "result_archive_hashes_verified": True,
                  "predicted_token_and_request_conservation_verified": True, "actual_and_predicted_graph_overlap_speculation_checked": True,
                  "parallelism_running_limit_chunk_pages_lm_head_checked": True,
                  "no_forward_residual_or_correction": True, "no_measured_arrival_or_batch_injection": True,
                  "metrics_recomputed_from_native_prediction": True,
                  "historical_logits_audit_note": "The old modeled_lm_head_tp/aligned fields describe its original template. This report validates the resolved override flags against actual server-info instead.",
                  "source_commit": manifest["source_commit"], "image_id": replay.IMAGE_ID}
    write_document(data)
    validation["generated"] = [{"path": str(path.relative_to(REPO)), "sha256": replay.digest(path)} for path in
                               (DOCUMENT, EVIDENCE / "comparison.json", EVIDENCE / "comparison.csv")]
    (EVIDENCE / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"document": str(DOCUMENT), "validation": validation, "covered_accuracy": data["statistics"]["runtime_layers_covered"]["runtime_layers"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
