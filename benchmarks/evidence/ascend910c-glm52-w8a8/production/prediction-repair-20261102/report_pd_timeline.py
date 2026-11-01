import csv
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
CAMPAIGN = ROOT / "pd-paired-1p1d"


def main():
    state = json.loads((CAMPAIGN / "progress.json").read_text())
    if state["status"] != "completed":
        raise ValueError("Only archive a completed PD campaign")
    source = REPO / "aisimulate-output/paired-pd-registration-repair-20261102/comparison.json"
    comparison = json.loads(source.read_text())
    timeline = json.loads((ROOT / "pd-request-timeline-analysis.json").read_text())
    (ROOT / "paired-pd-validation-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    with (ROOT / "paired-pd-validation-comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison["rows"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparison["rows"])
    lines = [
        "# node20 + node25：独立 PD 逐请求验证",
        "",
        "2026-11-02。1P1D，P/D均为全局TP16、attention-DP4、EP16。相同GLM权重和SGLang0.5.22/CANN9.1镜像，graph、调度overlap、投机解码和prefix cache关闭。HCCL_BUFFSIZE=650，P/D静态显存比例0.88/0.90，P chunk4096/rank，warmup2，正式请求N=2C。开启内置FPM及批次/请求日志钩子用于定位；钩子没有额外插入NPU事件或同步，观测本身仍可能增加开销。",
        "",
        "197.609ms的TP16 Decode执行残差保持不变，未用这些PD请求拟合。表内按“实测 / 预测 / 误差”排列，TTFT与TPOT单位ms，吞吐token/s。8K/128是短诊断；只有标明8K/1K的一行是完整目标负载。",
        "",
        "| case | C / N | TTFT | TPOT | 输入吞吐 | 输出吞吐 | D running峰值 / 非零均值 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison["rows"]:
        summary = json.loads((CAMPAIGN / row["case"] / "metrics-summary.json").read_text())
        values = [f"{row['measured_' + metric]:.2f} / {row['predicted_' + metric]:.2f} / {row['error_percent_' + metric]:+.2f}%" for metric in ("ttft_ms", "tpot_ms", "input_tok_s", "output_tok_s")]
        lines.append(f"| {row['case']} | {row['concurrency']} / {row['requests']} | " + " | ".join(values) + f" | {summary['decode_running_peak']:.0f} / {summary['decode_running_mean_when_nonzero']:.2f} |")
    lines.extend([
        "",
        "完整8K/1K的TPOT和吞吐达到本轮10%范围，TTFT仍明显偏低。短负载repeat的实测TTFT也会变化；不能用单点固定倍率把TTFT强行拉齐。新补的源端路由登记约束通过470项Rust回放测试，但在所有请求同一时刻入队的这些合成输入上，端到端数值基本不变；不把语义修复等同于TTFT精度已经改善。",
        "",
        "## 逐请求阶段证据",
        "",
        "通过bootstrap_room关联P/D，排除最多32输出token的warmup。阶段时间在同一节点的单调时钟内相减，避免跨节点时钟偏差。P传输后持有KV的时间包含本端轮询延迟，不能直接加到客户端TTFT。",
        "",
        "| case | P bootstrap均值ms | P计算区间均值ms | D分配/元信息均值ms | D收到请求至首token均值ms | 客户端TTFT多出的ms | 首个可见文本前额外token均值 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for case in timeline["cases"]:
        stages = case["stages"]
        values = [stages[key]["mean_ms"] for key in ("prefill_bootstrap_ms", "prefill_chunks_elapsed_ms", "decode_allocation_and_metadata_ms", "decode_scheduler_to_first_token_ms")]
        lines.append(f"| {case['case']} | " + " | ".join(f"{value:.2f}" for value in values) + f" | {case['client_minus_decode_scheduler_first_token_ms']:.2f} | {case['mean_additional_tokens_before_text']:.3f} |")
    lines.extend([
        "",
        "C1两条正式请求：D分配/元信息阶段第一次约2164ms、第二次约5ms；对应P计算区间约5277ms与3145ms。这说明首次执行成本不能简单用稳态算子表解释，也不足以证明每个DP rank都会固定支付同一开销。",
        "",
        "同镜像benchmark只把首个非空text计为TTFT。原始output_lens减去ITL条数可恢复首个可见文本对应的token位置；这些请求存在第2、3、4个token才产生非空text的情况。模拟器首token时间与客户端可见文本时间存在口径差异，应显式区分，不能静默改写实测。其余差距还包含HTTP/Tokenizer/调度入队时间；当前请求记录从Scheduler接收开始，尚不能把所有客户端差值归因于某一个阶段。",
        "",
        "实际P/D接收与bootstrap完成会错峰，影响chunk组合、D首轮入队和下一波请求完成时间。仅加入1ms的串行入队敏感性实验可以改变TTFT约0.7～1.1秒，但这不是实测的Tokenizer参数，未作为默认校准。",
        "",
        "源端登记修复：在多attention-DP的SGLang round-robin路径中，P在可调度边界登记请求的DP rank，D才能完成该rank查询并预留资源，P随后计算。已补齐这个因果顺序及tokenization只执行一次、取消不遗留D端KV的测试。HTTP往返、首次分配成本及节点间争用仍需独立建模。",
        "",
        "原始记录在pd-paired-1p1d/；逐请求分解在pd-request-timeline-analysis.json；精确镜像、模型元数据、参数与文件哈希在paired-pd-validation-manifest.json。node25使用数据盘上的独立Docker daemon，原Docker未修改。两个诊断模型与Router已停止；其他占用节点的服务未停止。当前继续单独采集TP32，尚未接受整体搜索准确性。",
    ])
    (ROOT / "paired-pd-validation.md").write_text("\n".join(lines) + "\n")
    artifacts = {}
    for path in sorted(CAMPAIGN.rglob("*")):
        if not path.is_file() or "decode-live" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix == ".log" or (path.suffix == ".jsonl" and ("paired-forward-rank" in path.name or "request-stages-rank" in path.name)):
            archive = path.with_suffix(path.suffix + ".gz")
            archive.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            path = archive
        artifacts[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("paired-pd-validation.md", "paired-pd-validation-comparison.json", "paired-pd-validation-comparison.csv", "pd-request-timeline-analysis.json"):
        artifacts[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    launchers = {role: json.loads((CAMPAIGN / role / "launch.json").read_text()) for role in ("prefill", "decode")}
    if launchers["prefill"]["image_id"] != launchers["decode"]["image_id"] or launchers["prefill"]["model_metadata_sha256"] != launchers["decode"]["model_metadata_sha256"]:
        raise ValueError("PD image/model metadata mismatch")
    manifest = {"status": "completed", "nodes": [20, 25], "workload_validation": "Five points,42formalrequests,each length and client error array validated; full8192/1024 C4/N8 included", "image_id": launchers["prefill"]["image_id"],
                "model_metadata_sha256": launchers["prefill"]["model_metadata_sha256"], "decode_calibration_unchanged": json.loads((ROOT / "eager-overhead-tp16-adp4.json").read_text()),
                "prediction_image_id": json.loads((REPO / "aisimulate-output/paired-pd-registration-repair-20261102/validation-c4-8k1k/execution.json").read_text())["image_id"],
                "artifact_sha256": artifacts}
    (ROOT / "paired-pd-validation-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
