# 四节点 GLM5.2 W8A8：实测与 AISimulate 对照

更新日期：2026-11-02，Asia/Shanghai。目标是修复实际 PD 启动、采集相同工作负载下的
多配置实测，再与算子级预测比较。推荐组、1P3D、2P2D 各六档已完成，
3P1D 的 C4/C8/C16/C32/C64 也已完成，共 23/24 点、1760 条正式请求全部成功。
3P1D C128 未完成，连接 Router 失败，该点不计入结果。原始预测的 46 次运行
（23 点各包含默认内存、实测 KV 容量两种口径）均已完成，保留在
`aisimulate-output/four-node-serving-matched-20261101`。
完整矩阵的完成情况以 `matrix-progress.json` 和各点 `.complete` 为准。

**逐轮 running batch 已补齐（2026-11-02）：**现在可从 Rust replay 的已完成
forward 读取真实batch，并分别统计逻辑worker总量和各attention-DP rank。
8192/1024、C4/N8双节点验证的decode平均batch：实测3.9805、预测3.9728，
误差-0.19%，峰值均为4；prefill平均batch误差为-14.29%，仍有分批差异。
9组延迟/吞吐回放与上一版数值完全一致；6组具有完整逐轮实测，另外3组末轮
异步计时缺少idle rank记录，不伪造补齐。详见
[逐轮batch对照](../benchmarks/evidence/ascend910c-glm52-w8a8/production/prediction-repair-20261102/forward-batch-comparison.md)。
运行镜像为`aisimulate-910c:forward-batch-repair-20261102`，用`predict --detail time`
将统计保存到`prediction.json`的`forward_pass_batches`。原23点的TTFT超标项仍需修复。

**最新修复结果（2026-11-02，活跃DP开销修复）：**原23点中，TPOT的MAPE为1.69%，
最大绝对误差3.65%；输入/输出吞吐的MAPE为2.49%，最大5.20%。
TTFT有15/23点进入15%以内，MAPE为14.73%，仍有8点超标、最差53.42%。
上一版TTFT为16/23；Decode时长改善改变闭环到达和Prefill分批，个别TTFT变差，
因此整体准确性目标继续推进，不能把吞吐改善说成所有指标都已达标。
新增同镜像1P1D、8192/1024、C4/N8的独立验证：TTFT实测/预测为
5424.56/4998.83ms（-7.85%），TPOT为424.13/413.49ms（-2.51%），
输出吞吐为9.310/9.560 token/s（+2.68%）。
完整23点、新增9组82条请求、训练与验证划分、冷编译证据见
[活跃DP修复总表](../benchmarks/evidence/ascend910c-glm52-w8a8/production/prediction-repair-20261102/active-rank-serving-comparison.md)。
上一版结果保留在[请求时序修复总表](../benchmarks/evidence/ascend910c-glm52-w8a8/production/prediction-repair-20261102/request-timing-repair-comparison.md)。
本页后续记录保留各阶段历史结果；使用当前修复需显式选择
`aisimulate-910c:forward-batch-repair-20261102`（保留活跃DP修复的全部时序参数），
原默认镜像和各阶段修复镜像均保留。

在上一版参数下完成24条新的冷启动/重复访问控制请求：8K/128的稳态C4/N16
TTFT误差为-13.86%、TPOT为-6.93%、吞吐为+9.07%。C1/N8包含首次访问rank的
慢请求，整体TTFT仍低估22.16%；其中重复访问六条均值与预测相差-3.68%，
该子集只用于定位，不能替代整体误差。失败的32K请求因实际KV容量不足被排除，
详见[冷启动与重复请求对照](../benchmarks/evidence/ascend910c-glm52-w8a8/production/prediction-repair-20261102/cold-warm-validation.md)。

按当前已完成数据整理的总表见
[22 个已完成点的即时对照](../benchmarks/evidence/ascend910c-glm52-w8a8/production/four-node-serving-20261101/available-comparison-22points.md)。
该表不等剩余测试，只列已核验实测；尚未运行的同口径预测标为“—”。

## 2026-11-02 的修复与误差定位

已修复 Decode 的 LM-head 并行组：实际 D 使用全局 TP 分片，P 使用 attention TP
分片；补上全局隐藏态汇集、logits 汇集与等长请求条件下的 all-to-all 分支。
eager 的 logits 行数改为实际请求数，与 MLP 为 TP 对齐而填充的 token 行数分别处理。
这些开关由 Rust 配置持有，保存在角色 timing 配置中。

在当前 node20 补采全局 TP32 的 BF16 LM-head：30 个 shape/执行模式组合均完成，
29 点通过数值及稳定性检查，1 点因 CV=26.6% 被排除。原始记录及失败记录均保留；
性能表从 2888 行增加到 2917 行，原有曲线未替换。

相同 C4/N8 的重跑说明，这项修复的主要价值是执行语义正确，不能消除当前主要误差：

| 配置 | 实测 TPOT ms | 修复后预测 ms | 相对误差 |
| --- | ---: | ---: | ---: |
| 2P1D，D 全局 TP32/ADP32 | 328.146 | 188.763 | -42.48% |
| 1P3D，D 全局 TP16/ADP4 | 405.080 | 196.155 | -51.58% |
| 2P2D，D 全局 TP16/ADP4 | 410.018 | 197.216 | -51.90% |
| 3P1D，D 全局 TP16/ADP4 | 420.997 | 197.216 | -53.16% |

新的同节点 TP16/ADP4/EP16 诊断服务已用同一镜像完成三轮 profiler。
轨迹显示 eager decode 存在大量 CPU 提交及核间空闲区间，不能以 kernel 时长之和
表示整轮 forward。Profiler 自身会增加耗时，其时间不能直接作为未插桩的校准值。
因此继续使用镜像内置 Forward Pass Metrics 的 NPU 事件计时，采集独立的逐轮
shape/耗时，再区分校准请求与验证请求。该服务为单节点合并部署的诊断服务，
其 TTFT 不代替原四节点 PD 测量。

当前完成 54 项 Ascend Rust 测试、420 项模型及两套 GPU parity 测试；
serving 精度尚未验收，仍需修复执行开销和核对 TTFT。完整证据位于
`benchmarks/evidence/ascend910c-glm52-w8a8/production/prediction-repair-20261102`。

随后完成 TP16/ADP4/EP16 的独立 eager 开销校准，并保留原始算子预测：在每轮
decode 上单独加入约197.609ms的执行残差。校准只使用同镜像单节点C1的稳态
forward记录，其他并发、4K/16K输入及原来的17个PD实测点用于核对，未用于求该参数。
17个PD点的 TPOT 误差降至约 -2.8%～-7.9%，输入/输出吞吐误差约 +5.9%～+8.8%。
低并发TTFT仍低估约36%～65%；高并发主要受Decode排队限制的点已降至约6%～10%。
完整前后对照见
[17点 eager 修复结果](../benchmarks/evidence/ascend910c-glm52-w8a8/production/prediction-repair-20261102/eager-overhead-serving-comparison.md)。

这是限于指定镜像及TP16/ADP4/EP16的经验残差，默认不启用，独立标记为estimated；
不能直接套给TP32、graph或其他模型。逐轮计时的事件异步回收可能与相邻batch错配，
因此这里只采用C1稳态区间校准Decode，Prefill将补充携带轮次标识的计时记录。
本轮55项Ascend Rust测试、421项模型与GPU parity测试通过；准确性目标仍未完成。

之后又完成七组按实际forward轮次配对的验证：保留相同197.609ms残差，不再拟合，
各工况Decode平均绝对误差为3.13%～9.30%。这次直接将batch快照绑定到计时事件，
解决原生FPM异步回收时可能配到相邻batch的问题；完整结果见
[逐轮配对验证](../benchmarks/evidence/ascend910c-glm52-w8a8/production/prediction-repair-20261102/paired-forward-validation.md)。
Prefill和PD握手开销继续单独检查，不把这些逐轮结果等同于端到端TTFT已经达标。

node20+25 的独立1P1D补测已完成五组、42条正式请求。完整8192/1024、C4/N8、
P/D均TP16/ADP4/EP16时，实测/预测TPOT为421.270/394.827ms（-6.28%），
输出吞吐为9.354/10.065 token/s（+7.59%），输入吞吐为74.836/80.516 token/s。
TTFT仍为6282.183/3066.704ms（-51.18%），尚未满足准确性目标。
完整表格、running batch和逐请求分解见
[双节点PD独立验证](../benchmarks/evidence/ascend910c-glm52-w8a8/production/prediction-repair-20261102/paired-pd-validation.md)。

这次记录直接关联P/D的bootstrap_room，确认首次KV分配、首次shape执行、错峰入队
以及客户端“首个非空text”口径都会影响TTFT。已修复多attention-DP的SGLang
源端登记顺序：先在P调度边界登记DP rank，再允许D预留；tokenization只执行一次，
取消请求不会遗留预留KV。470项Rust回放测试通过。该修复没有消除本组完全同步
合成到达造成的TTFT差距，仍保留修复前后结果，不宣称全局搜索已经准确。

node19、31、32被其他服务占用，本次使用此前授权且空闲的node25。
node20+25 的 TP32/ADP32/EP32 补测已完成6组、106条正式请求；仅C1用于拟合，
额外每轮decode残差为132.119ms，不复用TP16参数。其余独立工况的单轮MAPE为
2.52%～12.94%；原四节点推荐组的6个点TPOT误差降至-2.21%～-12.56%，
吞吐误差为+3.44%～+15.56%。完整23点修复前后结果见
[TP32补测及23点对照](../benchmarks/evidence/ascend910c-glm52-w8a8/production/prediction-repair-20261102/tp32-paired-validation.md)。
TTFT仍未验收；诊断TP32服务已停止，继续使用20+25采集带请求ID的PD客户端、
Tokenizer与Scheduler时间，核对网关等待Prefill返回及错峰到达造成的延迟。

## 推荐配置与默认部署

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

推荐配置的 504 条正式请求全部成功，逐请求长度均为 8192/1024，六档 metrics
端点错误均为零，已归档并提交 `55ef2b6c`：

| 并发 / 请求数 | 平均 TTFT 秒 | 平均 TPOT ms | 输入 tok/s | 输出 tok/s | D running 峰值 | 单 DP rank 峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 / 8 | 6.148 | 328.146 | 95.651 | 11.956 | 4 | 1 |
| 8 / 16 | 5.987 | 334.322 | 187.511 | 23.439 | 8 | 1 |
| 16 / 32 | 7.655 | 357.397 | 349.105 | 43.638 | 16 | 1 |
| 32 / 64 | 9.945 | 367.752 | 672.951 | 84.119 | 32 | 1 |
| 64 / 128 | 13.879 | 365.605 | 1327.188 | 165.899 | 64 | 3 |
| 128 / 256 | 21.674 | 367.077 | 2543.811 | 317.976 | 128 | 5 |

在切换配置、没有正式压测请求的时间段，已验证 C4/N8 的两种同口径预测：
原推荐预测 TTFT=2.937 秒、TPOT=187.437 ms、输出吞吐=21.039 tok/s；
相对实测分别偏低 52.22%、偏低 42.88%、偏高 75.96%。替换实际 KV 容量后该点
结果完全相同，因此低并发误差不能靠容量修正解决。其余点的预测在采集结束后运行；
当前不能宣称精度已通过验收，也不能据此提前确定四组实测排序。

1P3D 对照六档已全部完成，504 条正式请求长度均为 8192/1024，客户端错误和指标
端点错误均为零，归档提交为 `a037e908`。C128 使用下文说明的延长等待时限。

| 并发 / 请求数 | 平均 TTFT 秒 | 平均 TPOT ms | 输入 tok/s | 输出 tok/s | D running 峰值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 / 8 | 6.916 | 405.080 | 75.449 | 9.431 | 4 |
| 8 / 16 | 7.374 | 412.321 | 151.171 | 18.896 | 8 |
| 16 / 32 | 9.593 | 415.661 | 296.827 | 37.103 | 16 |
| 32 / 64 | 14.504 | 415.590 | 584.001 | 73.000 | 33 |
| 64 / 128 | 273.378 | 416.707 | 587.427 | 73.428 | 36 |
| 128 / 256 | 843.909 | 417.698 | 593.641 | 74.205 | 36 |

本轮 N=2C 的测试中，C32 升到 C128，输出吞吐只增加约 1.65%，平均 TTFT 却从
14.504 秒升到 843.909 秒。C128 的 p99 TTFT 为 1353.290 秒，整档耗时 3532.697 秒。
这体现了容量受限时增加客户端并发主要增加排队。C128 的推荐组为 TTFT 21.674 秒、
TPOT 367.077 ms、输出 317.976 tok/s；这只比较当前两种固定策略，不代表所有
1P3D 策略都更差。3P1D 尚未完成，当前不是完整候选排序。完整 1P3D 摘要见
`control-1p3d-completed-summary.json`；早期 C64 摘要作为历史证据继续保留。

2P2D 对照六档全部完成，均无客户端错误或指标端点错误。完整原始客户端数据、
配置和 Worker 日志已归档为 `measurements/control-2p2d-tp16.tar.gz`；各源文件
哈希见对应 manifest。早期 `control-2p2d-through-c16-summary.json` 继续作为历史证据保留。

| 并发 / 请求数 | 平均 TTFT 秒 | 平均 TPOT ms | 输入 tok/s | 输出 tok/s | D running 峰值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 / 8 | 8.379 | 410.018 | 75.157 | 9.395 | 4 |
| 8 / 16 | 6.041 | 416.535 | 149.660 | 18.708 | 8 |
| 16 / 32 | 7.701 | 416.575 | 299.855 | 37.482 | 17 |
| 32 / 64 | 118.019 | 419.151 | 395.991 | 49.499 | 24 |
| 64 / 128 | 554.432 | 420.306 | 395.374 | 49.422 | 24 |
| 128 / 256 | 1420.092 | 417.993 | 432.572 | 54.072 | 24 |

C16 的采样 running 峰值 17 保留原值，不能据此认为客户端实际突破了并发限制；
异步 gauge 和服务级流量的限制见下文。这些短测试也不足以给出两种 PD 策略的
稳定性能差异或置信区间。

3P1D 已完成的四档如下，正式请求均为 8192/1024 且零错误。C16 起出现明显等待，
D running 峰值约 12；提高客户端并发未使输出吞吐继续按比例增长。C64/C128 尚未完成。

| 并发 / 请求数 | 平均 TTFT 秒 | 平均 TPOT ms | 输入 tok/s | 输出 tok/s | D running 峰值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 / 8 | 8.374 | 420.997 | 74.420 | 9.303 | 4 |
| 8 / 16 | 5.645 | 425.964 | 148.018 | 18.502 | 8 |
| 16 / 32 | 116.177 | 424.747 | 198.446 | 24.806 | 12 |
| 32 / 64 | 561.187 | 429.047 | 197.030 | 24.629 | 12 |

### 为什么并发不等于 running batch

1P3D 对照三个 D 实例的实际 KV 池分别为每个 attention-DP rank
32512、32384、32512 tokens，每个实例 ADP4。每个 rank 的请求数量上限虽然是 32，
但一条完整 8K/1K 请求需要约 9216 tokens；按初始 8192+512 的预留量计算也只能
容纳约 3 条。因此 3 个 D × 4 个 DP rank × 3 条，约为 36 条驻留请求。

C64 测试中的额外只读采样确实观察到各 D rank 运行约 3 条，其他请求进入预分配
队列，KV 使用率接近 85%。这是当前固定长度下的容量解释，不是所有 workload 的
通用上限；原始标签、实际配置和来源哈希保存在
`control-1p3d-c64-admission-window.json`。后续 `observed-kv` 预测正是用来区分
这种容量/准入偏差与 forward 耗时偏差。

显存源码核对也排除了一个猜测：EP16/32 下，共享专家的 2048 intermediate
维度按每个 rank 完整计入权重，没有再除以 attention TP。当前容量模型的限制
是运行时/通信开销仍来自固定系统常数：`nccl_mem=0`，SGLang 的
`other_mem × 1.2` 约为 1.048 GiB；它没有按本次 HCCL 缓冲区、通信组或
DeepEP 布局逐项推算。这不等于这些开销都应再次从 KV 预算扣除，必须区分
分配发生的阶段以及 `mem_fraction_static` 已预留的空间，避免重复扣减。

LM head 布局差异也影响驻留权重：TP16/ADP4 的 D 实际按 TP16 分片，
冻结模型按 TP4 分片，仅这块 BF16 矩阵的模型估计就多约 0.332 GiB/rank。
这是矩阵形状推导的差值，不是已测出的总显存误差或 TPOT 归因；实际 allocator
保留空间与其他张量还需分别核对。上述源码哈希和启动日志证据记录于
`memory-capacity-source-audit.json`，本轮保持原容量预测与实测容量诊断分开。

### C128 的等待超时与重测

2026-11-01 06:13（Asia/Shanghai），1P3D C128 的 P 日志明确报告等待 Decode KV
索引 600 秒后在 `KVPoll.Bootstrapping` 失败。原尝试已中止，指标文件、启动配置、
基准日志和五个服务容器的日志保存在该组 `failed-attempts/bootstrap-timeout-*`。
它是未完成的失败尝试，不计作正式性能点。
重启前的容器日志同时覆盖此前成功的 C4～C64；重启后的日志另存，按启动时间和
各点 runtime 配置对应，不能仅看重启后的日志分析旧测试。

为测量长队列下的完成时间，重启后将 `PD_BOOTSTRAP_TIMEOUT` 设为 7200 秒，
`ROUTER_REQUEST_TIMEOUT` 设为 21600 秒；两项只改变等待时限，PD/TP/ADP/EP 与算子
执行开关沿用原配置。已经完成的 11 点仍保留原来的 600/1800 秒时限，C128 重测与
后续配置使用延长的时限，比较时必须保留这个区别。这不意味着吞吐或 KV 容量提高。

已核对镜像源码中的另一项 300 秒 KV 等待超时：当前 Decode allocation policy
为 `early`，在 KV 地址发布后才开始传输等待计时，预分配队列中的等待不启动这项
计时。此次没有统一放大所有超时；C128 重测的全部 256 条正式请求已经成功完成。

这两项现已参数化到 `service/pd-config.sh`，普通启动默认仍为 600/1800 秒。
继续本次矩阵时需在启动控制脚本的环境中传入 7200/21600；仅修改 benchmark 进程的
环境不能改变已经启动的服务。恢复过程与新旧参数保存在
`bootstrap-timeout-recovery.json`，实际进程核验保存在 `timeout-extension-validation.json`。

### 已确认的执行路径差异

冻结模拟器版本 `db5f2e721670d3eb2a10785fe55ad2d28af81349` 的
`ascend_glm52.py` 按 attention-local TP 划分 LM head 词表和 logits gather。
实际镜像的 `DeepseekV2ForCausalLM` 与 `LogitsProcessor` 则依据
`enable_dp_lm_head` 选择 attention TP 或全局 TP。原始 server-info 确认 P 为 true、
D 为 false；因而推荐 D 是实际 TP32 对模拟 TP1，对照 D 是实际 TP16 对模拟 TP4。
这不是调度排队差异，而是最后输出层的算子与通信路径尚未对齐。

推荐组 D 的 `enable_tp_lm_head_all_to_all` 为 true，对照组 D 为 false。
该选项还受每轮 token 布局等条件限制，当前没有逐轮分支记录，不能声称每轮都执行
all-to-all。源码副本已与运行中镜像的文件哈希核对一致，证据见
`benchmarks/evidence/ascend910c-glm52-w8a8/production/four-node-serving-20261101/logits-layout-audit.json`。

当前数据仍按冻结启动配置完整采集，用于复现原推荐和定位误差。不会在中途切换
该开关、覆盖已有数据，或把该路径差异的影响伪装成纯算子耗时误差。
每个预测输入会从已保存的完整 server-info 校验哈希并写出 `logits_layout_audit`；
后续应支持真实的输出层分组，或另做开启 D DP LM-head 的独立对照，才能归因其影响。

## 运行与保存

最终报告还会按并发列出 Decode 预分配/传输队列峰值、KV 使用率峰值和调度
`run_batch` 墙钟占比。分析脚本从原始逐秒指标计算，不增加服务端 profiling；
缺失指标与计数器重置会明确标记。调度汇总包含空闲实例，不能当成单个请求的关键
路径，也不能等同于 NPU kernel 时间。详细结果保存在预测归档中的
`serving-stage-diagnostics.json`，包含来源哈希、有效时间窗口和直方图观测数量。
这些服务级指标没有按本轮压测 request ID 过滤，可能包含探活等流量；因此直方图
均值不是正式请求的阶段耗时均值，不能把各项相加作为客户端 TTFT。

诊断 JSON 还分别保留各副本的流式/非流式请求与 token 计数增量，用来核对额外流量。
这些服务端窗口计数没有逐个关联客户端请求 ID，不能替代客户端成功数；非流式请求
也不能直接全部认定为健康检查。缺少窗口边界的计数系列不会假定初始值为零。

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

使用 sudo 采集时，后处理也会把结果文件归属交回仓库用户，方便直接查看。
此处理只修正文件归属，不改测量内容或 SHA256；已有私有权限位保持原样。

- `matrix-progress.json`：当前组、阶段、完成点数、错误。
- `comparison-progress.json`、`live-summary.csv`：后处理进程每 30 秒更新的逐点进度与结果。
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

例如 1P3D 的 C32 测试中，有两次采样汇总为 33，客户端的 64 条正式请求则全部
成功且并发上限为 32。不同 rank 的指标刷新与一秒窗口不是同一个瞬时快照，也可能
涉及系统探活。这里保留 33 的原值，不能据此声称有 33 条正式请求同时运行；当前
未用逐轮 trace 归因这两个采样点，也不会把峰值强行截成客户端并发。

`collection_version=2` 另外采集镜像实际导出的 PD prealloc/transfer/bootstrap 队列、
KV 传输直方图的 sum/count、DP 同步等待、scheduler 分阶段累计时间和回退 gauge。
这项扩展从推荐组 C128 及后续对照组生效；先前 C4～C64 的客户端结果和 running/KV
采样仍有效，但不能据此补造当时未采集的分阶段时间。
`scheduler_stage_seconds_total` 是各进程的互斥调度循环墙钟分类，包含同步等待，
不能直接当成 NPU 算子执行时间。镜像在此环境没有导出有效的 forward 设备计时序列，
指标缺失不能解释为 forward 时间为零。

推荐组 C64 的额外 30.45 秒观察窗口中，32 个 Decode rank 平均 `run_batch` 时间
增加 29.98 秒，占已归类调度墙钟时间的 98.53%。它提示后续需要核对 forward 内部
的算子、通信、CPU 提交和同步开销，而不应把全部 TPOT 偏差都归因于外层请求排队。
该观察不能代替设备侧 profiler，也不能外推为其他并发/配置的时间占比；首尾原始
计数器保存在推荐组目录的 `mid-c64-scheduler-window.json`。

后续对比保留原始预测，再单独替换实际 KV 容量做诊断，以区分算子/通信耗时偏差、
容量偏差与排队调度偏差。未完成该对比之前，不宣称预测准确性已通过验收。

## 同口径预测和结果归档

对照工具位于
`benchmarks/evidence/ascend910c-glm52-w8a8/production/four-node-serving-20261101/`：

- `compare_serving_predictions.py`：核对实际运行的 PD 实例数、TP/attention-DP/EP、
  graph/overlap、量化、分页和调度参数。每个实测点各运行一次原推荐容量预测与
  实际 KV 容量诊断，正式请求数都使用 `N=2C`。模型路径、模拟器镜像 ID、配置文件
  和测量文件哈希写入执行记录，不复用输入已经改变的旧预测。
- `report_serving_comparison.py`：生成一张覆盖四组配置、24 个点的实测/预测总表，
  同列 TTFT、TPOT、输入与输出吞吐的实测值、原预测值和带正负号的百分比误差，
  同时列出实际 D running 峰值/非零均值。比较同一并发下四个候选
  的吞吐排序，并记录预测所选配置相对实测最佳候选的吞吐损失。
- `finish_serving_comparison.py`：独立于终端连接等待采集完成；每组结束即归档，
  24 个实测点全部完成后再执行 48 个预测/诊断作业。开启 `--commit-measurements`
  时只提交本次归档的指定文件。任一阶段失败会写入 `comparison-progress.json`，
  不会把未完成的点标记为成功。

所有正式请求都是完整的 8K/1K；每档只测两波请求、每点一次，因此该结果包含
填充与排空阶段，不能作为长时间稳态吞吐或统计置信区间。此前固定 256 请求的
推荐表不能直接用于计算这里低并发点的误差。

归档目录为本阶段的 `measurements/`。Git 保存客户端完整请求明细、基准日志、
参数、指标摘要和 Worker 日志；较大的原始逐秒 `metrics.jsonl.gz` 留在结果根目录，
每份的来源、大小和 SHA256 写入 manifest，便于核验与再次分析。
最终对照写入本阶段的 `comparison-report.md`、`comparison.csv` 和 `ranking.csv`。
总表每个性能单元格固定为“实测 / 原预测 / 误差”，误差为
`100 × (预测值 / 实测值 - 1)`。TTFT 单位为秒、TPOT 为毫秒、吞吐为 token/s。
主表使用未经此次实测调参的 `stock` 预测；替换实际 KV 容量的 `observed-kv`
另作容量诊断，不覆盖原预测。`comparison.csv` 保留两种口径的独立数值列，方便筛选。
生成这些文件之前，采集与精度验收均不能视为已经完成。

## 中断恢复与本机锁

本机项目挂载目录的独立测试出现了 flock 无法重新获取的问题，包括显式解锁后；
相同测试放到 `/tmp` 正常。控制锁因此统一通过 `service/pd_lock.py` 放入
`/tmp/aisim-pd-locks-<规范化目录哈希>/`，目录和文件归属保持与对应工作目录一致。
已验证正常退出、失败退出后均可再次取得锁，活跃锁仍会拒绝第二个进程。

首次推荐组采集期间修改活动 shell 文件，导致全部六档完成后的收尾出现 EOF；
六档客户端与采样器均已各自成功退出。原日志和锁迁移记录保留，没有重新生成或
覆盖这些实测数据。此后更新活动 shell 脚本使用原子替换，避免改变其打开文件的
读取位置。1P3D 采集进程在控制器迁移时保持运行，模型与压测均未重启。

矩阵程序支持 `--resume`：保留已完成配置及其日志，跳过成功点，将不完整点移入
`failed-attempts/` 后重试。单组默认超时为 12 小时：TP16/ADP4 对照的 D KV pool
约为每 rank 32K tokens，高并发的 8K 请求会受到容量限制并排队，不能用低并发
耗时估算整组完成时间。可通过 `--campaign-timeout` 显式调整。
