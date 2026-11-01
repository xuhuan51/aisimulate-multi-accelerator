import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    original = json.loads((ROOT / "original-23-point-comparison.json").read_text())["rows"]
    repaired = json.loads((ROOT / "eager-overhead-serving-comparison.json").read_text())["rows"]
    baseline = {(row["configuration"], row["concurrency"]): row for row in original if row["mode"] == "stock"}
    rows = [row for row in repaired if row["mode"] == "stock"]
    csv_rows = []
    lines = [
        "# 2026-11-02：TP16 eager 开销修复后的同配置对照",
        "",
        "23 个已完成原始实测点中，17 个采用 D 全局 TP16 / attention-DP4 / EP16。下表只列这 17 点；D 跨两节点 TP32/ADP32 的推荐组尚无独立 forward 开销校准，不套用此参数。",
        "",
        "部署、镜像、8192/1024、请求数 N=2C、关闭 graph/调度 overlap/投机解码均沿用原始实测。保留原始预测，并修复全局 LM-head 和 eager logits shape；本轮额外加入每轮197.609ms的独立 decode 残差。该值来自 node20 同镜像、同 TP/ADP/EP 合并部署的 C1 逐轮记录，未用这17个PD结果拟合。",
        "",
        "这属于有范围的经验校准：原始算子延迟保留，额外开销单独标记 estimated。它可能包含 CPU 提交、流等待和漏计的算子，不能解释为纯 CPU 时间。FPM 异步回收事件可能导致相邻轮次错配；C1 稳态中间段及下列请求级验证支持该残差，但 Prefill 尚未据此校准。",
        "",
        "下列 TTFT 单位为秒，TPOT 为毫秒，吞吐为token/s；每个性能格为“实测 / 修复后预测”。误差=100×(预测/实测−1)。",
        "",
        "| 配置 | 并发 | TTFT 实测/预测 s | TTFT 误差 | TPOT 实测/预测 ms | TPOT 误差 | 输入吞吐 实测/预测 | 输出吞吐 实测/预测 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        prior = baseline[row["configuration"], row["concurrency"]]
        short = row["configuration"].replace("control-", "").replace("-tp16", "").upper()
        lines.append(
            f"| {short} | {row['concurrency']} | {row['measured_ttft_ms']/1000:.3f} / {row['predicted_ttft_ms']/1000:.3f} | {row['error_percent_ttft_ms']:+.2f}% | "
            f"{row['measured_tpot_ms']:.3f} / {row['predicted_tpot_ms']:.3f} | {row['error_percent_tpot_ms']:+.2f}% | "
            f"{row['measured_input_tok_s']:.3f} / {row['predicted_input_tok_s']:.3f} | {row['measured_output_tok_s']:.3f} / {row['predicted_output_tok_s']:.3f} |"
        )
        csv_rows.append({**row, **{f"original_{key}": value for key, value in prior.items() if key.startswith(("predicted_", "error_percent_"))}})
    lines.extend(["", "## 验证范围与剩余问题", ""])
    for metric in ("ttft_ms", "tpot_ms", "output_tok_s"):
        old = statistics.mean(abs(baseline[row["configuration"], row["concurrency"]]["error_percent_" + metric]) for row in rows)
        new = statistics.mean(abs(row["error_percent_" + metric]) for row in rows)
        lines.append(f"- {metric}：17点平均绝对百分比误差 {old:.2f}% → {new:.2f}%。")
    lines.extend([
        "- 独立并发/长度的逐轮验证平均绝对误差约4.7%–10.3%；P95约15.7%–17.7%，不能声称所有轮次都在15%内。",
        "- 低并发 TTFT 仍明显低估，PD bootstrap、完成通知、冷 shape 与 Prefill 执行开销还需拆开核对；不能用最终 TTFT 反推一个全局倍率。",
        "- 推荐 TP32/ADP32 与其他并行布局尚未使用该校准，不应将本表直接当作更新后的全局最优配置搜索结果。",
        "- stock 与 observed-kv 两种预测都保存于JSON/CSV，主表用stock，未替换原测量。",
        "",
        "原始计时：`forward-metrics-node20/forward-metrics.jsonl.gz`；校准：`eager-overhead-calibration.json`；逐轮验证：`eager-overhead-forward-validation.json`；完整前后数值：`eager-overhead-serving-comparison.csv`。",
    ])
    (ROOT / "eager-overhead-serving-comparison.md").write_text("\n".join(lines) + "\n")
    with (ROOT / "eager-overhead-serving-comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    print("Reported", len(rows), "matched control measurements")


if __name__ == "__main__":
    main()
