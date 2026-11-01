import json
from pathlib import Path


root = Path(__file__).resolve().parent
report = json.loads((root / "paired-forward-analysis.json").read_text())
if report["status"] != "completed":
    raise ValueError("Only report a completed paired campaign")
lines = [
    "# 按实际 forward 轮次对齐的二次验证",
    "",
    "2026-11-02，在 node20 用相同的 SGLang0.5.22/CANN9.1 镜像、TP16/ADP4/EP16、关闭graph和调度overlap重新采集。保留既有197.609ms的Decode残差，没有使用这次数据重新拟合。模型权重、量化和全局TP LM-head与前一轮相同；这是合并部署的诊断，不替代PD整请求结果。",
    "",
    "原生FPM把已完成的异步NPU事件累计后配给当前batch，Prefill/Decode交替时可能错配。新的诊断入口在计时事件创建时保存batch快照和轮次，事件完成后再输出对应记录；没有添加额外NPU事件或设备同步。四个attention-DP rank按轮次合并，混合prefill/decode组单独排除。",
    "",
    "Decode只使用生成位置40至输出长度−8，排除最多32输出token的warmup。表中的C1是重复工况验证；4K/16K、C4/C8/C12为其他工况。原始文件沿用采集case名，其中calibration-c1和calibration-prefill4k不代表本轮参与拟合。",
    "",
    "| 采集case | 纯Decode轮数 | 实测中位数 ms | 原算子模型 MAPE | 加既有残差后的 MAPE |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for case in report["cases"]:
    value = case["decode"]
    lines.append(f"| {case['case']} | {value['forwards']} | {value['median_measured_ms']:.3f} | {value['raw_mape']:.2f}% | {value['corrected_mape']:.2f}% |")
lines.extend([
    "",
    "Prefill仍仅作诊断：C1稳态算子模型已接近实测，其他并发的偏差还涉及进入batch的时刻、冷shape以及实际expert负载。不能把前一轮错配到相邻batch的长时间直接当作Prefill固定开销。",
    "",
    "原始逐轮文件和客户端记录在`forward-paired-node20/`，分析结果为`paired-forward-analysis.json`。`paired-telemetry-source-audit.json`核对了实际镜像中的7个调度/计时源文件，其SHA256与本地审计副本一致。",
    "",
    "node19、31、32当前被其他服务占用。node25可用，因此后续在node20+25建立独立1P1D补测握手、传输和完整8K/1K；node25缺少指定镜像且系统盘只剩约10GB，诊断镜像放在数据盘上的独立Docker实例，原有Docker及其他服务保持原状。",
])
(root / "paired-forward-validation.md").write_text("\n".join(lines) + "\n")
