import csv
import datetime
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
PREDICTIONS = REPO / "aisimulate-output/joint-serving-table-20261102"
DOCUMENT = REPO / "md/910c-glm52-w8a8-2node-4node-latest-comparison-20261102.md"
METRICS = ("ttft_ms", "tpot_ms", "input_tok_s", "output_tok_s")
LABELS = {
    "recommended-2p1d-dtp32": ("A", "4节点·2P1D", "16/4/16", "32/32/32"),
    "control-1p3d-tp16": ("B", "4节点·1P3D", "16/4/16", "16/4/16"),
    "control-2p2d-tp16": ("C", "4节点·2P2D", "16/4/16", "16/4/16"),
    "control-3p1d-tp16": ("D", "4节点·3P1D", "16/4/16", "16/4/16"),
    "paired-1p1d-tp16": ("E", "2节点·1P1D", "16/4/16", "16/4/16"),
}
CAMPAIGN_ORDER = ["four-node", "pd-paired-1p1d", "pd-ingress-1p1d", "pd-cold-1p1d", "pd-phases-1p1d", "pd-allocator-1p1d"]


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(path, destination, artifacts):
    payload = path.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(gzip.compress(payload, mtime=0))
    artifacts.append({"source": str(path), "source_sha256": hashlib.sha256(payload).hexdigest(),
                      "archive": str(destination.relative_to(REPO)), "archive_sha256": sha256(destination)})


def averages(rows):
    result = {}
    for metric in METRICS:
        errors = [abs(row[f"error_percent_{metric}"]) for row in rows]
        result[metric] = {"mape_percent": statistics.mean(errors), "max_absolute_error_percent": max(errors),
                          "within15": sum(error <= 15 for error in errors), "points": len(rows)}
    return result


def batch_data():
    matched = {}
    unavailable = {}
    for campaign, filename in [
        ("pd-ingress-1p1d", "forward-batch-ingress-comparison.json"),
        ("pd-cold-1p1d", "forward-batch-cold-comparison.json"),
        ("pd-phases-1p1d", "forward-batch-phase-comparison.json"),
        ("pd-allocator-1p1d", "forward-batch-allocator-comparison.json"),
    ]:
        source = ROOT / filename
        data = read_json(source)
        for case in data["cases"]:
            matched[(campaign, case["case"])] = (case, source)
        for case in data["unavailable_batch_comparisons"]:
            unavailable[(campaign, case["case"])] = case["reason"]
    return matched, unavailable


def batch_summary(rows, role):
    relevant = [row for row in rows if row["stage"] == role and row["dp_rank"] is None]
    passes = sum(row["passes"] for row in relevant)
    if not passes:
        return None
    return {"mean": sum(row["request_instances"] for row in relevant) / passes,
            "peak": max(row["max_batch_size"] for row in relevant), "passes": passes}


def main():
    source = read_json(PREDICTIONS / "comparison.json")
    if len(source["rows"]) != 39:
        raise ValueError("The report requires all 39 completed PD cases")
    matched, unavailable = batch_data()
    artifacts = []
    rows = []
    for base in source["rows"]:
        row = dict(base)
        execution_path = Path(row["execution"])
        execution = read_json(execution_path)
        prediction_path = execution_path.parent / "result/prediction.json"
        prediction = read_json(prediction_path)
        runtime_path = Path(execution["runtime_path"])
        measured_path = Path(execution["measured_path"])
        runtime = read_json(runtime_path)
        measured = json.loads(measured_path.read_text().splitlines()[-1])
        if execution["image_id"] != source["image_id"]:
            raise ValueError(f"Mixed simulator builds: {execution_path}")
        if sha256(runtime_path) != execution["runtime_sha256"] or sha256(measured_path) != execution["measured_sha256"]:
            raise ValueError(f"Changed serving inputs: {execution_path}")
        for metric, predicted_key, measured_key in [
            ("ttft_ms", "mean_ttft_ms", "mean_ttft_ms"),
            ("tpot_ms", "mean_tpot_ms", "mean_tpot_ms"),
            ("input_tok_s", "input_throughput_tok_s", "input_throughput"),
            ("output_tok_s", "output_throughput_tok_s", "output_throughput"),
        ]:
            if row[f"predicted_{metric}"] != prediction[predicted_key] or row[f"measured_{metric}"] != measured[measured_key]:
                raise ValueError(f"Metric provenance mismatch: {execution_path}, {metric}")
            error = 100 * (prediction[predicted_key] / measured[measured_key] - 1)
            if not math.isclose(error, row[f"error_percent_{metric}"], abs_tol=1e-10):
                raise ValueError(f"Error calculation mismatch: {execution_path}, {metric}")
        if any(prediction[key] != 0 for key in ("failed_requests", "canceled_requests", "rejected_requests")):
            raise ValueError(f"Incomplete prediction: {execution_path}")
        if prediction["completed_requests"] != row["requests"] or measured["completed"] != row["requests"]:
            raise ValueError(f"Request conservation failed: {execution_path}")
        for direction in ("input", "output"):
            if prediction[f"total_{direction}_tokens"] != row["requests"] * row[f"{direction}_length"]:
                raise ValueError(f"Token conservation failed: {execution_path}")
        row.update({"runtime_nodes": runtime["nodes"], "configuration_spec": execution["configuration_spec"],
                    "observed_blocks_per_rank": execution["observed_blocks_per_rank"],
                    "capacity_policy": "minimum observed per-role pages; replicas with one extra page are represented conservatively",
                    "measurement_path": str(measured_path), "prediction_path": str(prediction_path),
                    "runtime_path": str(runtime_path), "simulator_image_id": source["image_id"]})
        identity = (row["campaign"], row["case"])
        row["batch_measurement_qualification"] = unavailable.get(identity, "No complete matched per-forward measurement in this report")
        for role in ("prefill", "decode"):
            row[f"predicted_{role}_batch"] = batch_summary(prediction["forward_pass_batches"], role)
            row[f"measured_{role}_batch"] = None
        if identity in matched:
            evidence, batch_source = matched[identity]
            if evidence["overlapping_health_checks"]:
                raise ValueError("Keep background-health batch attribution explicit")
            for role in ("prefill", "decode"):
                row[f"measured_{role}_batch"] = batch_summary(evidence["measured"], role)
            row["batch_measurement_qualification"] = "Complete matched committed forwards, arithmetic mean per worker pass"
            row["batch_measurement_source"] = str(batch_source)
        row["calibration_qualification"] = "未用于数值拟合；同一软件/硬件范围内回放验证"
        if row["campaign"] == "pd-paired-1p1d" and row["case"] == "diagnostic-c4-8k128":
            row["calibration_qualification"] = "部分校准：P执行残差来自本组forward"
        if row["campaign"] == "pd-ingress-1p1d" and row["case"] == "diagnostic-c1-8k128":
            row["calibration_qualification"] = "部分校准：入队处理间隔来自本组API记录"
        destination = ROOT / "joint-comparison-replays" / row["campaign"] / row["case"]
        for original, filename in [(execution_path, "execution.json.gz"), (prediction_path, "prediction.json.gz"),
                                   (runtime_path, "runtime-config.json.gz"), (measured_path, "measurement.jsonl.gz"),
                                   (execution_path.parent / "result/requests.jsonl", "requests.jsonl.gz")]:
            archive(original, destination / filename, artifacts)
        row["evidence_directory"] = str(destination.relative_to(REPO))
        rows.append(row)
    rows.sort(key=lambda row: (CAMPAIGN_ORDER.index(row["campaign"]), LABELS[row["configuration"]][0], row["concurrency"], row["output_length"], row["case"]))
    groups = {"four_node": [row for row in rows if row["campaign"] == "four-node"],
              "two_node": [row for row in rows if row["campaign"] != "four-node"], "all": rows}
    accuracy = {name: averages(values) for name, values in groups.items()}
    if sum(row["requests"] for row in rows) != 1900:
        raise ValueError("Expected 1900 original successful formal requests")
    original = read_json(ROOT / "active-rank-serving-comparison.json")["accuracy"]["original"]
    result = {"created_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              "image": source["image"], "image_id": source["image_id"], "error_formula": "100*(prediction/measurement-1)",
              "accuracy": accuracy, "original_four_node_accuracy": original, "rows": rows}
    (ROOT / "joint-serving-comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    columns = ["campaign", "case", "configuration", "input_length", "output_length", "concurrency", "requests"]
    columns += [f"{prefix}_{metric}" for metric in METRICS for prefix in ("measured", "predicted", "error_percent")]
    columns += ["configuration_spec", "observed_blocks_per_rank", "calibration_qualification", "batch_measurement_qualification", "evidence_directory"]
    with (ROOT / "joint-serving-comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], dict) else row[key] for key in columns})
    write_document(result)
    manifest = {"artifacts": artifacts, "generated": [{"path": str(path.relative_to(REPO)), "sha256": sha256(path)} for path in
                [DOCUMENT, ROOT / "joint-serving-comparison.json", ROOT / "joint-serving-comparison.csv"]],
                "validation": {"rows": 39, "four_node_rows": 23, "two_node_rows": 16, "formal_requests": 1900,
                               "all_inputs_hash_verified": True, "all_metrics_recomputed": True,
                               "all_predicted_request_and_token_totals_verified": True}}
    (ROOT / "joint-serving-comparison-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"document": str(DOCUMENT), "rows": len(rows), "accuracy": accuracy}, ensure_ascii=False))


def metric_cell(row, metric):
    precision = 2 if metric.endswith("ms") else 3
    return f"{row[f'measured_{metric}']:.{precision}f} / {row[f'predicted_{metric}']:.{precision}f}（{row[f'error_percent_{metric}']:+.2f}%）"


def batch_cell(row, role):
    actual, predicted = row[f"measured_{role}_batch"], row[f"predicted_{role}_batch"]
    if actual is None:
        return f"缺逐轮实测 / {predicted['mean']:.3f}"
    return f"{actual['mean']:.3f} / {predicted['mean']:.3f}（{100*(predicted['mean']/actual['mean']-1):+.2f}%）"


def write_document(result):
    evidence = "../benchmarks/evidence/ascend910c-glm52-w8a8/production/prediction-repair-20261102/"
    lines = [
        "# 910C + GLM-5.2 W8A8：双节点、四节点实测与最新预测总表", "",
        "更新日期：2026-11-02。本次**重新执行全部39组离线预测**：四节点23组、双节点16组，对应1900条已完成正式实测请求；没有重新启动模型服务，没有替换原始实测，也没有用表中的端到端TTFT/TPOT反推修正倍率。", "",
        "当前四节点23点的TTFT平均绝对百分比误差（MAPE）为**14.73%**，TPOT为**1.69%**，输入/输出吞吐均为**2.49%**；双节点16点分别为**18.41%、2.12%、3.89%**。合计39点为**16.24%、1.87%、3.06%**。TPOT和吞吐全部在15%内；TTFT只有24/39在15%内，四节点仍有8个、双节点仍有7个超标点，最大绝对TTFT误差53.42%，因此不能认为全部精度目标已经完成。MAPE按工况等权统计，不按请求数加权；双节点汇总包含短输出诊断，不能当作16组完整8K/1K。", "",
        "## 版本、配置与比较口径", "",
        "实测模型为`/data_lib/data/models/glm-q`中的GLM-5.2 W8A8，ModelSlim量化；W8A8没有被解释为所有attention、KV和通信都使用INT8。实测服务镜像固定为`quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443`，对应已核对的SGLang0.5.22/CANN9.1.0 B243。", "",
        f"本表实际运行的预测镜像为`{result['image']}`，镜像ID为`{result['image_id']}`；它包含活跃attention-DP执行残差和原生forward batch统计。本轮刚修改的KV页延迟复用代码**尚未打入该镜像**，冷编译延迟也**尚未进入预测**，两者不计入本表已有改善。不要用启动脚本原默认冻结镜像代替这里的显式镜像。", "",
        "共同服务参数：prefill/decode graph backend均为disabled；scheduler overlap关闭；speculative algorithm=None；prefix cache关闭；shared-expert多流重叠保留并与预测一致。P/D静态显存比例0.88/0.90，page size128，D reserved decode tokens512，context length180000；P启动chunk16384，经ADP4归一化为4096/rank，max-prefill-tokens32768。PP=1、attention CP=1，不启用额外DCP；HCCL_BUFFSIZE=650，P/D DeepEP分别normal/low_latency。max-running-requests为P128、普通D128、TP32 D1024，均对应每attention-DP rank32。", "",
        "配置A：4节点2P1D，P分别在node20、node25；node31和node32共同组成**一个**跨节点D。P全局TP16/ADP4/EP16，D全局TP32/ADP32/EP32。配置B：4节点1P3D，P=node20，三个D=node25/31/32。配置C：4节点2P2D，P=node20/25，D=node31/32。配置D：4节点3P1D，P=node20/25/31，D=node32。B/C/D每个实例都是TP16/ADP4/EP16。配置E：2节点1P1D，P=node20、D=node25，两端都是TP16/ADP4/EP16。这里TP为SGLang全局TP，ADP为attention DP；TP16/ADP4意味着attention TP4，TP32/ADP32意味着attention TP1。", "",
        "请求使用random-ids、seed910、固定输入输出长度、tokenize-prompt及ignore-EOS，request-rate=inf，通过固定并发闭环补发。每个实测工况有2条benchmark warmup，warmup最多32输出token且不计入正式N；C是最大并发，不是始终达到的running batch。预测从同一并发/请求数生成自己的闭环到达，不回灌实测完成时间。没有恢复实测warmup、健康检查和编译缓存的完整初始状态，这是尚未对齐的边界。", "",
        "本轮采用observed-kv：按每组runtime-config读出的实际KV容量设定预测页数，同角色多个实例取最小值。表中`P[136,135]→135`表示实测两个P分别有136/135页，预测两个P统一按135页；这是一页级保守近似，并非声称逐副本容量完全相同。四节点本轮数值与此前默认容量预测在浮点容差内一致，不能把已有巨大误差归咎于这次容量口径切换。", "",
        "所有性能格按**实测 / 最新预测（有符号误差）**列出，误差=`100×(预测/实测−1)`。TTFT、TPOT均为请求均值，单位ms；吞吐为整个正式测量窗口的input/output token/s。负延迟误差表示低估等待，正吞吐误差表示高估性能。8K/1K为8192/1024；8192/128是短诊断。输入与输出吞吐使用同一测量窗口，因此固定长度下二者百分比误差相同。", "",
        "## 全部有效PD数据：一张表", "",
        "`paired`、`ingress`、`cold`、`phases`、`allocator`表示不同采集批次；同名工况跨批次仍是不同实测，不能合并或挑选其中较好的结果。批次完整目录分别为`pd-paired-1p1d`、`pd-ingress-1p1d`、`pd-cold-1p1d`、`pd-phases-1p1d`、`pd-allocator-1p1d`。表中running batch采用完成forward的算术平均，跨attention-DP求和后按逻辑worker轮次统计；它与Prometheus每秒采样的running gauge、队列占用代理不是同一指标。", "",
        "| 编号/配置 | 批次/工况 | I/O | C/N | P；D：TP/ADP/EP | 实测KV页→预测页（每rank） | TTFT ms：实/预（误差） | TPOT ms：实/预（误差） | 输入tok/s：实/预（误差） | 输出tok/s：实/预（误差） | P平均batch：实/预（误差） | D平均batch：实/预（误差） | 数据资格 |",
        "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for index, row in enumerate(result["rows"], 1):
        label, topology, prefill, decode = LABELS[row["configuration"]]
        campaign = row["campaign"].removeprefix("pd-").removesuffix("-1p1d")
        case = "原四节点" if campaign == "four-node" else f"{campaign}/{row['case']}"
        capacity = "; ".join(f"{short}{row['observed_blocks_per_rank'][role]}→{min(row['observed_blocks_per_rank'][role])}" for role, short in [("prefill", "P"), ("decode", "D")])
        qualification = row["calibration_qualification"]
        if qualification.startswith("未用于"):
            qualification = "回放验证"
        cells = [f"{index:02d}·{label} {topology}", case, f"{row['input_length']}/{row['output_length']}", f"{row['concurrency']}/{row['requests']}",
                 f"{prefill}；{decode}", capacity] + [metric_cell(row, metric) for metric in METRICS] + [batch_cell(row, "prefill"), batch_cell(row, "decode"), qualification]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "", "缺逐轮实测表示没有完整、同口径的forward采样可比，不表示batch为0；这些行的TTFT/TPOT/吞吐仍是有效数据。四节点原数据主要为Prometheus采样，early paired批次未纳入当前严格RID逐轮配对；ingress的repeat、cold的warm-validation、phases的C4存在末轮异步计时rank记录不全，不补零。峰值、轮数和完整runtime节点配置见配套JSON。", "",
        "四节点C64/C128部分TTFT达到数百秒或上千秒，是高并发超过实际KV/Decode承载后产生排队的原始结果，不是毫秒/秒单位换算错误。推荐配置A在已完成候选中，C4/8/16/32/64/128六档的实测与预测输出吞吐第一名一致；C128的3P1D未成功完成，只有三种候选，不能据此声称任意PD、任意并行布局下的全局最优都得到证明。", "",
        "## 对齐修复总结（整段）", "",
        "这一轮精度改善首先来自把真实服务身份、请求口径和执行路径对齐：固定GLM-5.2 W8A8权重元数据、SGLang0.5.22/CANN9.1镜像、PD实例数与物理节点数、全局TP/attention-DP/EP、chunk归一化、KV页大小及容量、Decode预留、输入输出长度、并发与正式请求数，并核实graph、调度overlap、投机解码确实关闭；在算子侧修正Ascend输出LM-head的切分与通信路径，区分P端DP LM-head、D端全局TP LM-head和TP32 all-to-all，补采相应shape，避免用attention TP错误代替实际输出层TP；在执行时间侧，将真实逐轮forward中稳态算子表未覆盖的eager执行残差作为有拓扑和模式约束的显式参数，由Rust统一FPM接口使用，TP16/ADP4的Decode基值为197.609ms、TP32/ADP32为132.119ms，P端增加91.625ms，并再依据真实活跃attention-DP rank数量加入每个额外活跃rank的6.373ms/1.497ms，空rank的通信padding不计为真实活跃rank，这主要消除了TPOT偏低、吞吐偏高的问题；在调度和请求侧，补齐多attention-DP下P先登记source rank、D再预留KV的顺序及取消清理，对齐SGLang Gateway虽并发请求P/D、但等待P响应完成才向客户端暴露D流的门控，并将实测约0.8435ms的P端单worker请求处理间隔放到入队事件中，使原先被理想化为完全同时到达的请求按真实因果顺序参与chunk和batch组合，同时保存配置、搜索候选与再次预测走同一套参数；在验证侧，增加来自已完成forward的原生batch统计，按请求ID和rank核对真实运行量，避免把排队请求、KV占用或padding当作running batch，但这项遥测本身不改变预测耗时。上述组合使原四节点23点TTFT MAPE由48.47%降到14.73%、TPOT由51.01%降到1.69%、输入/输出吞吐由105.48%降到2.49%，不是每个修复单独都能让每个点单调变好，活跃rank项主要改善TPOT/吞吐，部分TTFT因闭环到达改变反而略变差。关于刚定位的KV页问题，更准确的描述是“分配状态语义不一致”而不是显存损坏：真实PD采用need_sort，释放的页先暂存，ready页不足时再合并排序，即使最终分配失败也可能已经重排空闲表，而模拟器此前立即复用；相应源码修正和CPU源分配器核验已完成、Rust测试2123通过/1项既有忽略，但尚未打入本表预测镜像，不能把本表改善归功于它。冷编译也只是已取得直接证据、尚未接入预测的剩余因素：新双节点正式请求中，D的一次KV分配耗时2134.522ms、其中compile API2132.784ms，P的另一次分配2163.331ms、其中compile API2160.130ms，正常无编译分配中位数仅约0.577ms；不同shape及free_pages指针对齐会产生不同特化，同特化跨rank可复用缓存，健康检查又会改变页游标与cache状态，因此不能简单给每rank或每请求固定加2秒，也不能把冷编译塞进稳态算子表。剩余TTFT误差还受客户端首个非空文本可能晚于第一个生成token、初始缓存/健康检查状态及真实入队时序影响，当前两条warmup不能保证全覆盖，这些未完成项与已生效修复需要明确分开。", "",
        "## 如何理解当前误差和数据边界", "",
        "双节点两个完整8192/1024、C4/N8使用同一配置的批次都保留：early paired实测TTFT6282.18ms，对应最新预测4998.83ms，误差−20.43%；ingress批次实测5424.56ms，同一预测4998.83ms，误差−7.85%。这说明配置相同也不足以重建每次服务的冷暖状态和首个可见文本时间，不能只列较小误差那一次。当前完整8K/1K证据为四节点23点加双节点2点；其余14点是8192/128诊断。", "",
        "校准与验证分开：TP16/TP32的C1纯Decode逐轮记录用于残差基值，原名validation-c4的TP16和validation-c8的TP32逐轮记录后来用于活跃rank系数，已经属于训练；early paired的diagnostic-c4使用了其P forward拟合P残差，ingress的diagnostic-c1用于入队处理间隔，因此这两行标记部分校准。其余行未用于数值系数拟合，但它们参与过问题诊断和模型迭代，属于同环境回放验证，不能全部称作完全未接触的盲测。", "",
        "本表只收录完成且输入输出长度逐条正确的PD服务工况。3P1D/C128路由失败、32K试验超出当时每rank实际KV输入上限，以及测试后用于排空异步计时的512/32请求不作为有效性能点；双节点TP32合并服务的纯Decode逐轮校准另存，不能冒充PD端到端TTFT数据。测量钩子未额外插入NPU同步，但仍存在观测开销；表中的同条件表示已核对的部署参数与负载一致，并不表示隐藏冷状态和客户端文本流语义已完全建模。", "",
        "## 数据与复现", "",
        f"- [完整39点CSV]({evidence}joint-serving-comparison.csv)：未四舍五入的实测、预测、误差、并行配置、KV容量及证据目录。",
        f"- [完整39点JSON]({evidence}joint-serving-comparison.json)：额外保留每个节点runtime配置、P/D平均与峰值batch、校准资格和数据来源。",
        f"- [证据哈希清单]({evidence}joint-serving-comparison-manifest.json)：每组实测、runtime、预测、请求明细和原命令均存为gzip快照，可追溯原文件SHA256。",
        f"- [本轮重跑脚本]({evidence}replay_joint_comparison.py)与[文档生成脚本]({evidence}report_joint_comparison.py)：使用固定镜像与已保存原始实测，39组全部成功并核对请求/token守恒；不会启动模型。",
        f"- [修复前后四节点阶段记录]({evidence}active-rank-serving-comparison.md)、[冷编译直接证据]({evidence}allocator-cold-validation.md)、[原生batch统计说明]({evidence}forward-batch-comparison.md)。", "",
        "离线回放原始输出目录为`aisimulate-output/joint-serving-table-20261102/`；每个`execution.json`含完整predict命令和配置覆盖，使用显式镜像ID。复用既有命令前必须保留全部`--set`，尤其`pd_response=sglang_gateway`、tokenization、P/D eager_runtime_overhead及实测KV容量；只选相同PD ratio而漏掉这些参数，得到的不是本表预测。", "",
    ])
    DOCUMENT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
