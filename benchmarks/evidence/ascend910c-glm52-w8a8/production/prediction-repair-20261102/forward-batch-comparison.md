# 完成 forward 的 running batch 对照（2026-11-02）

本轮补齐 Rust 原生 replay 的逐轮 batch 统计，经 canonical Python/CLI 导出。统计来自已完成 forward，按逻辑 worker 汇总 attention-DP ranks，并分别保留 rank 行；不是请求占用量代理，也不把 padding 或等待队列当作运行 batch。开启 `--detail time` 后读取 `prediction.json` 的 `forward_pass_batches`。

9组相同配置回放的 TTFT、TPOT、输入输出吞吐和请求计数与上一版逐项一致。476项Rust replay、433项已打包Python/model/契约/GPU parity测试通过。新增遥测没有改动校准参数，原23点TTFT仍有8点超过15%，准确性goal继续进行。

| 工况 | 阶段 | 实测平均batch | 预测平均batch | 误差 | 实测forward数 | 预测forward数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| diagnostic-c1-8k128 | decode | 1.0000 | 1.0000 | +0.00% | 254 | 254 |
| diagnostic-c1-8k128 | prefill | 1.0000 | 1.0000 | +0.00% | 4 | 4 |
| diagnostic-c4-8k128 | decode | 3.7910 | 3.7910 | +0.00% | 268 | 268 |
| diagnostic-c4-8k128 | prefill | 2.2857 | 2.2857 | +0.00% | 7 | 7 |
| diagnostic-c8-8k128 | decode | 7.1049 | 7.1802 | +1.06% | 286 | 283 |
| diagnostic-c8-8k128 | prefill | 2.2857 | 2.6667 | +16.67% | 14 | 12 |
| validation-c4-8k1k | decode | 3.9805 | 3.9728 | -0.19% | 2056 | 2060 |
| validation-c4-8k1k | prefill | 2.6667 | 2.2857 | -14.29% | 6 | 7 |
| cold-rank-cycle-c1 | decode | 1.0000 | 1.0000 | +0.00% | 1016 | 1016 |
| cold-rank-cycle-c1 | prefill | 1.0000 | 1.0000 | +0.00% | 16 | 16 |
| phase-c1-8k128 | decode | 1.0000 | 1.0000 | +0.00% | 1016 | 1016 |
| phase-c1-8k128 | prefill | 1.0000 | 1.0000 | +0.00% | 16 | 16 |

完整8192/1024、C4/N8的decode平均batch为3.9805/3.9728（-0.19%）；峰值均为4。Prefill平均batch为2.6667/2.2857（-14.29%），分别6/7轮，说明闭环到达与分批还有差异。该工况上一版TTFT误差-7.85%、TPOT-2.51%、吞吐+2.68%保持不变。

严格完整性检查要求每组 timed_forward_iteration 的全部4个rank记录齐全，且正式decode实例数等于N×(O−1)。9个serving工况中6个有完整逐轮数据；其余3组最后一次异步计时上报只留下部分rank，不补零、不伪造：
- `validation-repeat-c4-8k128`：Incomplete timed forward group: decode, 3348。serving延迟/吞吐仍有效，只缺完整group batch证据。
- `warm-validation-c4`：Incomplete timed forward group: decode, 1672。serving延迟/吞吐仍有效，只缺完整group batch证据。
- `phase-c4-8k128`：Incomplete timed forward group: decode, 1389。serving延迟/吞吐仍有效，只缺完整group batch证据。

计数采用真实请求身份；重叠的HEALTH_CHECK请求单独列出，并保留包含健康请求与仅正式请求两种统计。Warmup会改变正式请求起始rank；本镜像的health轮询使用独立计数器，不推进用户轮询序号。单rank对照不能忽略此区别。

源码统计默认关闭，不需要启动模型或占用NPU即可重新预测。校准仍限定现有固定镜像、模型、TP16/ADP4/EP16或TP32/ADP32/EP32以及eager模式；本次统计不扩大精度外推资格。

证据：`forward-batch-{ingress,cold,phase}-comparison.json/csv`、`forward-batch-code-validation.json`、`forward-batch-replays/`。独立的冷分配/编译补测仍在进行，将单独保留其结果。
