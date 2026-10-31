# 四节点 GLM5.2 W8A8：实测与 AISimulate 对照

更新日期：2026-11-01，Asia/Shanghai。目标是修复实际 PD 启动、采集相同工作负载下的
多配置实测，再与未调参的算子级预测比较。当前已经启动推荐配置并取得首档有效结果；
完整矩阵的完成情况以 `matrix-progress.json` 和各点 `.complete` 为准。

## 当前实际部署

脚本目录：`/home/youxingye/youxingye_data/910c_script/PD disaggregation`。
模型目录：`/data_lib/data/models/glm-q`。

| 实例 | 物理节点 | 全局 TP | attention DP | EP | 每个 attention DP rank 的请求上限 |
| --- | --- | ---: | ---: | ---: | ---: |
| P0 | node20 | 16 | 4 | 16 | 32 |
| P1 | node25 | 16 | 4 | 16 | 32 |
| D0 | node31、node32 | 32 | 32 | 32 | 32 |

这是 **2P1D 实例配比、2 个 P 物理节点 + 2 个 D 物理节点**。D 两台机器的
`nnodes=2`、`node-rank=0/1`，rendezvous 为 `10.120.72.31:29500`，Router 注册
node31 的 D 地址。node32 暴露独立 metrics，必须一起采样。Router/Store 在 node20。

角色参数分别为 `PREFILL_TP_SIZE/PREFILL_DP_SIZE/PREFILL_EP_SIZE` 和
`DECODE_TP_SIZE/DECODE_DP_SIZE/DECODE_EP_SIZE`。`max-running-requests` 是实例级
参数，因此 P 设置 128、D 设置 1024，对应每个 attention DP rank 32。
P 的 chunk 启动值为 16384，SGLang 根据 attention DP4 归一化后的实际值为 4096。
P/D memory fraction 为 0.88/0.90，KV page=128，D reserved decode tokens=512。

`DECODE_NODES` 列的是物理成员，每连续 `DECODE_NNODES` 台组成一个 D 实例。
检查脚本已按照该规则计算实例数，不能把两个 D 容器直接解释为两个 D 实例。

## 镜像与启动故障

镜像固定为 `quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443`，
对应已核对的 `v0.5.22-cann9.1.0-a3`：SGLang 0.5.22、CANN 9.1.0 B243、
torch 2.10.0+cpu、torch-npu 2.10.0.post6、
DeepEP `1.0.0+0234a271.cann.9.1.0.b243`。node20 驱动为 26.1.1。
实际故障容器的镜像 ID 是
`sha256:d4c350d7d4e233d3cd203d2f783467c5f5be1453b9eee073f71b758069392edb`。

执行 `sudo -E ./service/start-pd.sh` 会读取配置，并通过
`run-prefill-container.sh` / `run-decode-container.sh` 创建这些容器。
首次故障并非“镜像没下载”：两台 P 已完成模型加载，在 normal-mode DeepEP 的
`CamMoeDispatchNormal` tiling 阶段失败；尾部 py-spy/CUDA coredump 信息属于退出诊断。

当时 P 没有显式设置 HCCL buffer，SGLang 的对应默认值为 200 MB。
用户在 P 脚本加入 `HCCL_BUFFSIZE=650` 和
`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32` 后启动成功，首档 8K/1K
请求也全部成功。已将 HCCL buffer 固化为 `pd-config.sh` 中的
`PREFILL_HCCL_BUFFSIZE=650`、`DECODE_HCCL_BUFFSIZE=650`，传递到各节点和容器。
没有更换镜像、驱动或 CANN。独立单算子 A/B 诊断因与用户的服务启动重叠而主动停止，
不能把该未完成测试当作严格的单因素证明。650 MB 仅是当前配置已验证的值，不能保证
任意 chunk、隐藏维度或其他模型均够用。

graph 的 prefill/decode backend 均为 disabled，scheduler overlap 关闭，
speculative algorithm 为 None，prefix cache 关闭。shared-expert 多流重叠保留，
与预测里的 `shared_expert_overlap=true` 一致。

node20 的 `youxingye` 账号已配置到 node25 的专用 SSH 公钥；部署脚本在
`sudo -E` 下也显式使用该密钥。SSH 免密不等于远端 sudo 免密码。
其他节点通过脚本交互读取 SSH/sudo 密码；密码不写入配置或采集结果。

## 测试口径

用户确定预热为两个请求，使用当前镜像内的 `sglang.benchmark.serving`。

| 项目 | 设置 |
| --- | --- |
| 正式输入 / 输出 | 8192 / 1024 tokens |
| 输入生成 | random-ids、tokenize-prompt、seed=910、range ratio=1 |
| 输出条件 | streaming、ignore_eos=true、temperature=0 |
| 并发阶梯 | 4、8、16、32、64、128 |
| 正式请求数 | 每档 N=2C，即 8、16、32、64、128、256 |
| 预热 | 2 个请求；镜像 benchmark 将预热输出限制为最多 32 tokens |
| 指标采样 | 各物理节点 /metrics，每秒一次，保留标签与原始数值 |
| 失败标准 | 请求不全成功、实际 token 长度不符，均停止该配置继续升压 |

每个配置均使用同一口径。N=2C 是有限请求的测试，不能直接与此前 N=256 的曲线
混用，也不能把短测试的最佳点当作不限时长的稳定最优点。两个预热请求不计入正式
TTFT/TPOT/吞吐；它们不是两个完整的 1K 输出请求。

首档推荐配置 C4/N8 已完成：平均 TTFT=6147.63 ms，平均 TPOT=328.15 ms，
输入吞吐=95.65 tok/s，输出吞吐=11.9564 tok/s，完成 8/8。
Decode 采样 running requests 峰值为 4、非零时均值 3.977，覆盖全部 32 个 attention
DP rank，单 rank 峰值为 1。此处仅报告实测；精确误差需要同 N、C 的预测重放。

## 运行与保存

```bash
cd '/home/youxingye/youxingye_data/910c_script/PD disaggregation'
./check/check-config.sh
./service/show-launch-plan.sh
sudo -E ./check/check-pd-env.sh
sudo -E ./service/start-pd.sh
```

启动时自动将服务脚本复制到其他节点的 `/home/youxingye/glm52-pd-aisim`，
不依赖共享 `youxingye_data` 文件夹。

单点测试：

```bash
NUM_PROMPTS=8 MAX_CONCURRENCY=4 RANDOM_INPUT_LEN=8192 RANDOM_OUTPUT_LEN=1024 \
  WARMUP_REQUESTS=2 ./tools/performance/run-benchmark.sh
```

采集当前拓扑的整条并发曲线：

```bash
CONFIG_LABEL=recommended-2p1d-dtp32 CONCURRENCIES='4 8 16 32 64 128' \
  WARMUP_REQUESTS=2 ./tools/performance/run-aisim-comparison.sh
```

本次矩阵文件为 `tools/performance/comparison-configs.json`。矩阵控制程序
`run-serving-matrix.py` 在一组全部完成并归档 Worker 日志后才停止旧拓扑、启动下一组。
它不在测试期间做离线搜索，避免占用同机 CPU 干扰实测。

| 标签 | P 节点 | D 节点与实例 | P TP/attention-DP/EP | D TP/attention-DP/EP |
| --- | --- | --- | --- | --- |
| recommended-2p1d-dtp32 | 20、25 | 31+32 一个实例 | 16/4/16 | 32/32/32 |
| control-1p3d-tp16 | 20 | 25、31、32 各一个实例 | 16/4/16 | 16/4/16 |
| control-2p2d-tp16 | 20、25 | 31、32 各一个实例 | 16/4/16 | 16/4/16 |
| control-3p1d-tp16 | 20、25、31 | 32 一个实例 | 16/4/16 | 16/4/16 |

这些对照是固定配置，不宣称是各个 PD ratio 的全局最优。完成矩阵后控制程序恢复
推荐配置。实际运行中的配置以结果目录 `runtime-config.json` 为准，控制程序使用环境
变量覆盖默认配置以切换拓扑；默认 `pd-config.sh` 始终保留推荐方案。

本次结果根目录：
`/home/youxingye/youxingye_data/910c_script/PD disaggregation/results/aisim-comparison`。

- `matrix-progress.json`：当前组、阶段、完成点数、错误。
- `matrix-summary.csv`：已完成点的吞吐、TTFT/TPOT、running batch 汇总。
- `配置名-20261101/cC-nN/bench_*.jsonl`：逐请求输入输出长度、TTFT、ITL 和错误。
- `metrics.jsonl.gz`：按端点保存的指标原始采样，含 DP/TP 标签。
- `metrics-summary.json`：对同实例/同 attention DP rank 的 TP 副本去重后统计。
- `runtime-config.json` 与压缩 server-info：服务返回的有效配置。
- `worker-logs/`：切换部署前归档的各节点完整日志。

运行批次是离散采样；全局并发、全局 running batch、每个 attention DP rank 的
running batch 是不同指标。P 上同一 DP rank 的多个 TP 进程会重复暴露请求数，
不能直接求和。跨节点 D 要采集两台节点，按全局 DP rank 合并。

实测发现，C8 第一波请求结束后，部分空闲 DP rank 的 running gauge 仍保留 1，
第二波直接累计得到 16。原始 token counter 显示旧 rank 没有继续输出，因此新版
汇总只累计当前一秒采样间隔 decode token counter 增长的 rank，C8 的过滤后峰值
为 8。原始 gauge 峰值也保留，早期汇总另存 `metrics-summary-raw-gauge.json`，
不会覆盖或伪造原始采样。该方法给出活跃 rank 的离散近似值，可能包含系统探活，
不能代替逐轮 forward trace；客户端 TTFT/TPOT/吞吐不依赖这项修正。

后续对比保留原始预测，再单独替换实际 KV 容量做诊断，以区分算子/通信耗时偏差、
容量偏差与排队调度偏差。未完成该对比之前，不宣称预测准确性已通过验收。
