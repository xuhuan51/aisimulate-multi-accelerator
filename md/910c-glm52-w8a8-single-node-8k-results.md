# 单节点 GLM-5.2 W8A8：8K/1K 最优配置预测

2026-10-31，CPU 离线预测，没有启动真实推理服务。
固定 256 条请求后，推荐的最小饱和并发是 **C20**：单实例全局 TP16、Attention DP4、Attention TP4、MoE TP16、EP1；输出吞吐 **86.08 token/s**，平均 TTFT **12.730 s**、TPOT **216.85 ms/token**。
在本轮测试的 C20～C32、48、64、96、128 档位中，吞吐基本相同，而等待增加，因此选择并列吞吐第一名里平均 TTFT 最低的 C20。该配置采用张量 MoE，不能把 EP1 改成 EP16 后仍沿用此预测。

## 条件与范围

模型 `/data_lib/data/models/glm-q`，单节点16个逻辑device；合并部署、一个副本、PP1/CP1、无DCP；eager、关闭graph和overlap调度。
KV page128、memory fraction0.90、prefix cache关闭；context180000；每rank max_sequences32，chunked_prefill_size16384为DP归一化前的全局值，max_prefill_tokens32768。
精度GEMM W8A8、MoE W8A8 dynamic、attention和KV为BF16。balanced专家路由、anchored shape外推，没有延迟SLA。
先扫13档并发、N=2C，再固定N=256扫C1～32每个整数及48/64/96/128。每档6种合法并行配置，共294个候选，全部成功并耗尽声明空间。只搜索固定调度和内存设置下的并行配置，不代表所有调度参数的全局最优。

## 固定256请求的并发曲线

| 并发 | Attention TP / DP | MoE TP / EP | 输出吞吐 token/s | mean TTFT s | mean TPOT ms |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | 16 / 1 | 16 / 1 | 6.81 | 2.510 | 144.60 |
| 2 | 16 / 1 | 16 / 1 | 13.20 | 4.884 | 146.93 |
| 4 | 16 / 1 | 16 / 1 | 25.27 | 7.325 | 151.31 |
| 8 | 16 / 1 | 16 / 1 | 48.61 | 12.209 | 152.81 |
| 12 | 4 / 4 | 16 / 1 | 57.03 | 8.509 | 196.18 |
| 16 | 4 / 4 | 16 / 1 | 76.73 | 10.720 | 198.26 |
| 18 | 4 / 4 | 16 / 1 | 74.87 | 11.557 | 218.63 |
| 19 | 4 / 4 | 16 / 1 | 79.47 | 12.093 | 219.65 |
| 20 | 4 / 4 | 16 / 1 | 86.08 | 12.730 | 216.85 |
| 21 | 4 / 4 | 16 / 1 | 86.08 | 23.791 | 216.85 |
| 24 | 4 / 4 | 16 / 1 | 86.08 | 56.974 | 216.85 |
| 32 | 4 / 4 | 16 / 1 | 86.08 | 145.463 | 216.85 |
| 48 | 4 / 4 | 16 / 1 | 86.08 | 311.380 | 216.85 |
| 64 | 4 / 4 | 16 / 1 | 86.08 | 466.236 | 216.85 |
| 96 | 4 / 4 | 16 / 1 | 86.08 | 742.764 | 216.85 |
| 128 | 4 / 4 | 16 / 1 | 86.08 | 967.673 | 216.85 |

N=2C菜单的第一名也是C20，但N40下为86.79 token/s、TTFT12.864s、TPOT218.09ms。请求总数改变会影响有限请求的填充/排空统计，所以实测应使用与预测相同的N。
为控制等待，可按业务SLA比较C8/C16/C20；没有指定SLA时，不应仅凭吞吐将大并发称为更优。

## 复现与验证

```bash
cd /home/youxingye/youxingye_data/aisimulate-910c
AISIMULATE_910C_MODEL=/data_lib/data/models/glm-q bash scripts/run-ascend910c.sh predict \
  --stack engine \
  --config benchmarks/evidence/ascend910c-glm52-w8a8/production/single-node-8k-eager-20261031/best-requests_256-c20.yaml \
  --capture-per-request --output-dir aisimulate-output/single-node-best-recheck
```

294个候选都核对了请求和token总数；另将7份保存配置复算，1576条请求完成，指标与搜索一致。实测精度尚未验收。
结果与全部候选CSV：`benchmarks/evidence/ascend910c-glm52-w8a8/production/single-node-8k-eager-20261031/`。
运行包源码 `db5f2e72`，native SHA256 `5ee56d5c3b3096d63582604baf040b2cb07d68f73755752de709db80d156cbad`。
适配检查点仍处于 stage2/in_progress；后续继续四节点启动参数对齐和同条件服务验证。
