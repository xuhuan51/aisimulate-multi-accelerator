import csv
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]


def main():
    campaign = ROOT / "tp32-paired-node20-25"
    progress = json.loads((campaign / "node0/progress.json").read_text())
    analysis = json.loads((ROOT / "tp32-paired-forward-analysis.json").read_text())
    profile = json.loads((ROOT / "eager-overhead-tp32-adp32.json").read_text())
    if progress["status"] != "completed" or analysis["campaign_status"] != "completed" or profile != analysis["profile"]:
        raise ValueError("Complete the paired campaign before publishing")
    requests = 0
    for case in progress["cases"]:
        measured = json.loads((campaign / "node0" / f"{case['name']}.jsonl").read_text().splitlines()[-1])
        expected = 2 * case["concurrency"]
        if measured["completed"] != expected or measured["input_lens"] != [case["input_length"]] * expected or measured["output_lens"] != [case["output_length"]] * expected or any(measured["errors"]):
            raise ValueError(case["name"])
        requests += expected
    comparisons = json.loads((REPO / "aisimulate-output/four-node-dual-calibration-20261102/comparison.json").read_text())
    original = json.loads((ROOT / "original-23-point-comparison.json").read_text())
    before = {(row["configuration"], row["concurrency"]): row for row in original["rows"] if row["mode"] == "stock"}
    rows = []
    for result in comparisons["rows"]:
        if result["mode"] != "stock":
            continue
        row = dict(result)
        baseline = before[(row["configuration"], row["concurrency"])]
        for metric in ("ttft_ms", "tpot_ms", "input_tok_s", "output_tok_s"):
            row[f"baseline_predicted_{metric}"] = baseline[f"predicted_{metric}"]
            row[f"baseline_error_percent_{metric}"] = baseline[f"error_percent_{metric}"]
        rows.append(row)
    if len(rows) != 23:
        raise ValueError("Preserve the original 23 completed validation points")
    scope = "C1-only, topology-specific eager decode calibration; independent original four-node measurements. TTFT not accepted. No gateway response correction in this comparison."
    (ROOT / "dual-eager-serving-comparison.json").write_text(json.dumps({"scope": scope, "rows": rows}, indent=2) + "\n")
    with (ROOT / "dual-eager-serving-comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# TP32 独立补测与原四节点回放", "", "2026-11-02，node20+25，固定 v0.5.22/CANN9.1 镜像，globalTP32/attentionDP32/EP32，graph/调度overlap/投机解码关闭。只用C1稳态纯decode拟合；其他并发、长度和原四节点实测均未参与拟合。", "", f"完成6组、{requests}条正式请求；每条输入输出长度、completed及errors均校验。拟合额外每轮decode开销 {profile['decode_ms']:.6f} ms。它表示eager运行中算子表尚未解释的开销，不是伪造的单算子耗时，也不能用于graph开启或其他并行拓扑。", "", "| 验证case | 用于拟合 | 完整32rank轮数 | 实测单轮中位数ms | 原MAPE | 修复后MAPE |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for summary in analysis["summaries"]:
        lines.append(f"| {summary['case']} | {'是' if summary['used_for_fit'] else '否'} | {summary['groups']} | {summary['median_measured_ms']:.2f} | {summary['raw_mape']:.2f}% | {summary['corrected_mape']:.2f}% |")
    lines.extend(["", "诊断采用PD合并服务分离采样纯decode，启用与真实PD Decode一致的TP LM-head all-to-all。由于Prefill不是目标验证对象，chunk限定为每rank1024，避免诊断Prefill要求更大的HCCL buffer；保留原650MB设置。第一次4096/rank失败的日志单独保留，未用于拟合。32rank按实际计时forward编号关联，并筛除warmup、mixed/prefill、首尾波动及不完整组。", "", "## 同配置原四节点数据", "", "8k/1k固定8192/1024、N=2C。原实测与原预测均保留。TP16与TP32分别使用其C1校准，未拿这些23个验证点重新拟合。误差=(预测/实测-1)。", "", "| 配置 | 并发 | TTFT实测/预测ms | TTFT误差 | TPOT实测/预测ms | 原TPOT误差 | 新TPOT误差 | 输出吞吐实测/预测tok/s | 吞吐误差 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in rows:
        lines.append(f"| {row['configuration']} | {row['concurrency']} | {row['measured_ttft_ms']:.1f}/{row['predicted_ttft_ms']:.1f} | {row['error_percent_ttft_ms']:+.2f}% | {row['measured_tpot_ms']:.2f}/{row['predicted_tpot_ms']:.2f} | {row['baseline_error_percent_tpot_ms']:+.2f}% | {row['error_percent_tpot_ms']:+.2f}% | {row['measured_output_tok_s']:.2f}/{row['predicted_output_tok_s']:.2f} | {row['error_percent_output_tok_s']:+.2f}% |")
    lines.extend(["", "完整输入吞吐、请求数、预测前后数值在dual-eager-serving-comparison.csv/json。新TPOT误差明显下降，但TTFT仍未通过目标；不能据此宣称所有workload和PD比例均准确，或最优配置排序已经验收。下一步核对HTTP PD网关输出门控、真实到达时间及Tokenizer/冷启动时间。", "", "原始文件、镜像/模型元数据和SHA256在tp32-paired-validation-manifest.json。node20/25诊断TP32模型已停止，其他用户模型未触碰。"])
    (ROOT / "tp32-paired-validation.md").write_text("\n".join(lines) + "\n")
    artifacts = {}
    paths = list(campaign.rglob("*")) + [ROOT / name for name in ("tp32-paired-forward-analysis.json", "eager-overhead-tp32-adp32.json", "dual-eager-serving-comparison.csv", "dual-eager-serving-comparison.json", "tp32-paired-validation.md")]
    for path in sorted(paths):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".gz":
            continue
        if path.name == "tp32-paired-forward-analysis.json" or path.suffix == ".log" or (path.suffix == ".jsonl" and path.name.startswith(("paired-forward-", "request-stages-", "tokenizer-ingress-"))):
            archive = path.with_suffix(path.suffix + ".gz")
            archive.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            path = archive
        artifacts[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in campaign.rglob("server.log.gz"):
        artifacts[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    launches = {node: json.loads((campaign / node / "launch.json").read_text()) for node in ("node0", "node1")}
    if launches["node0"]["image_id"] != launches["node1"]["image_id"] or launches["node0"]["model_metadata_sha256"] != launches["node1"]["model_metadata_sha256"]:
        raise ValueError("Different model metadata or runtime image across ranks")
    (ROOT / "tp32-paired-validation-manifest.json").write_text(json.dumps({"status": "completed", "formal_requests": requests, "scope": scope, "profile": profile, "image_id": launches["node0"]["image_id"], "model_metadata_sha256": launches["node0"]["model_metadata_sha256"], "source_uncompressed_sha256": analysis["source_sha256"], "artifact_sha256": artifacts}, indent=2) + "\n")


if __name__ == "__main__":
    main()
