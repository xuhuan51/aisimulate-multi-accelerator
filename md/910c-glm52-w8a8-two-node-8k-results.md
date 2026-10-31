# 两节点 910C / GLM-5.2 W8A8：8K/1K 并发与配置预测

运行日期：2026-10-31。以下为当前 AISimulate 算子模型和请求调度模拟的**离线预测**，没有启动 SGLang 服务，也没有使用 NPU 采集。
416 个候选全部成功；本轮覆盖的三种布局中，最高预测吞吐出现在跨两节点的单个合并实例、并发 128，为 **507.58 output token/s**，平均 TTFT **37.943 s**、TPOT **215.33 ms/token**。这只是当前候选菜单内、没有延迟 SLA 的吞吐第一名，不能当作实测最优。

## 1. 预测条件

| 项目 | 本轮设置 |
| --- | --- |
| 模型 | `/data_lib/data/models/glm-q`，当前 GLM-5.2 ModelSlim W8A8 元数据 |
| 硬件预算 | **恰好两个节点**；每节点 16 个运行时逻辑 device，共 32；配置字段仍使用 `gpus` 命名 |
| 输入/输出 | 每请求严格 8192 输入、1024 输出 token，`ignore_eos: true` |
| 客户端并发 C | 1、2、4、8、16、32、64、128 |
| 请求数 N | 每档 `N=2*C`；闭环并发，请求完成后补发；无预热、无 think time、无 prefix cache 命中 |
| 目标 | 整个两节点部署的输出吞吐；没有 TTFT/TPOT SLA |
| 调度与执行 | SGLang `0.5.22` 对齐路径；同步调度、eager Decode，关闭 overlap 调度和 graph |
| 并行搜索范围 | 当前默认合法并行结构，PP=1、CP=1；不含 eager DCP |
| Attention / KV / 通信精度 | BF16 / BF16 / half；GEMM 为 W8A8，MoE 为 W8A8 dynamic |
| 上下文上限 | `context_length=180000`；实际请求仍为 8192/1024 |
| Scheduler | max sequences=32，max batched tokens=8192；P/合并 chunk=16384、max prefill tokens=32768；D reserved tokens=512 |
| KV 配置 | page=128；P memory fraction=0.88，D/合并=0.90；按模型驻留和可用内存推导容量 |
| 算子模型 | `op_level`、`fallback_policy=deny`、`shape_policy=anchored`、balanced 专家路由 |
| PD 传输 | 实测传输曲线，`timing_mode=full_prompt`、contiguous page layout、`replica_round_robin` |

`shared_expert_overlap: true` 保留当前共享专家算子建模设置；它和关闭的 **scheduler overlap** 不是同一开关。
客户端并发 C 也不等于某个 DP rank 的 running batch：请求会分到不同副本和 attention DP rank，并受 KV 容量、排队和调度限制。
这里请求数随并发变化，属于有限请求的冷启动模拟，不是充分预热后的稳态压测。

## 2. 三种部署布局

| 布局 | 两节点如何使用 | 每档合法候选数 |
| --- | --- | ---: |
| PD：1P1D | 一个节点用于 P、一个节点用于 D；每个实例全局 TP=16 | 36 |
| 合并：2×单节点 | 两个独立合并实例，每个全局 TP=16，各占一节点 | 6 |
| 合并：1×双节点 | 一个合并实例跨两节点，全局 TP=32 | 10 |

当前模型在本轮内存和并行约束下，PD 的最小 P/D 实例各占一节点，所以两个节点只能比较 1P1D；2P1D、1P2D 需要增加节点。
每档总共 52 个合法候选，8 档共 416 个。搜索虽使用 random 算法，但 16 个独立菜单均记录了 `finite search space exhausted`，因此当前声明的有限候选全部评估完毕。
这里没有搜索所有可能的 chunk size、max-running-requests、内存比例或调度策略，只搜索这些固定设置下的合法并行结构。

下表每一行选择该布局、该并发下**输出吞吐最高**的并行配置；不同并发可能换用不同配置。
并行四元组统一写作 **`(Attention TP, Attention DP, MoE TP, EP)`**。
对本轮 PP=1/CP=1 的配置，SGLang 全局 TP = Attention TP × Attention DP；不能把 Attention TP 直接当成启动参数的全局 TP。

## 3. PD：一节点 P + 一节点 D

| C / N | P 四元组 | D 四元组 | 输出吞吐 token/s | mean TTFT s | mean TPOT ms/token | P99 TTFT s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| [1 / 2](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-pd-1p1d-c1.yaml) | (16, 1, 1, 16) | (16, 1, 16, 1) | 6.82 | 2.269 | 144.60 | 2.269 |
| [2 / 4](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-pd-1p1d-c2.yaml) | (8, 2, 1, 16) | (16, 1, 16, 1) | 13.41 | 2.463 | 146.93 | 2.463 |
| [4 / 8](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-pd-1p1d-c4.yaml) | (4, 4, 1, 16) | (16, 1, 16, 1) | 26.35 | 3.080 | 148.92 | 3.080 |
| [8 / 16](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-pd-1p1d-c8.yaml) | (4, 4, 1, 16) | (16, 1, 16, 1) | 53.01 | 4.583 | 145.72 | 6.208 |
| [16 / 32](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-pd-1p1d-c16.yaml) | (4, 4, 1, 16) | (4, 4, 16, 1) | 80.25 | 5.468 | 191.98 | 12.279 |
| [32 / 64](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-pd-1p1d-c32.yaml) | (4, 4, 1, 16) | (4, 4, 16, 1) | 77.21 | 101.036 | 207.55 | 225.675 |
| [64 / 128](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-pd-1p1d-c64.yaml) | (4, 4, 1, 16) | (4, 4, 16, 1) | 87.31 | 364.568 | 207.56 | 652.468 |
| [128 / 256](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-pd-1p1d-c128.yaml) | (4, 4, 1, 16) | (4, 4, 16, 1) | 93.41 | 882.205 | 207.56 | 1304.942 |

点击 C / N 可打开该行可以直接 `predict` 的 YAML。

C16→C32 的吞吐从 80.25 降到 77.21，而 TTFT 从 5.468 s 增到 101.036 s；并发增加并不保证吞吐更好。
逐请求记录显示，C32 的平均首次 admission 等待为 97.961 s，C128 为 879.133 s。高 TTFT 主要体现请求入场等待；并不能用客户端并发直接推断 D 端能同时运行多少请求。
C128 的 93.41 token/s 虽是本菜单 PD 的吞吐第一名，但平均 TTFT 达 882.205 s。若优先控制等待，本次菜单里 C8/C16 更值得先做实测验证，具体仍取决于延迟要求。

## 4. 合并：两个单节点副本

| C / N | 合并实例四元组 | 输出吞吐 token/s | mean TTFT s | mean TPOT ms/token | P99 TTFT s |
| --- | --- | ---: | ---: | ---: | ---: |
| [1 / 2](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-agg-2x1node-c1.yaml) | (16, 1, 16, 1) | 6.81 | 2.510 | 144.60 | 2.510 |
| [2 / 4](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-agg-2x1node-c2.yaml) | (16, 1, 16, 1) | 13.61 | 2.510 | 144.60 | 2.510 |
| [4 / 8](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-agg-2x1node-c4.yaml) | (16, 1, 16, 1) | 26.39 | 4.884 | 146.93 | 4.884 |
| [8 / 16](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-agg-2x1node-c8.yaml) | (16, 1, 16, 1) | 50.53 | 7.325 | 151.31 | 9.767 |
| [16 / 32](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-agg-2x1node-c16.yaml) | (16, 1, 16, 1) | 97.22 | 12.209 | 152.81 | 19.534 |
| [32 / 64](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-agg-2x1node-c32.yaml) | (4, 4, 16, 1) | 153.45 | 10.720 | 198.26 | 17.152 |
| [64 / 128](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-agg-2x1node-c64.yaml) | (4, 4, 16, 1) | 144.28 | 115.565 | 216.45 | 248.834 |
| [128 / 256](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-agg-2x1node-c128.yaml) | (4, 4, 16, 1) | 161.74 | 403.288 | 216.59 | 712.200 |

点击 C / N 可打开该行可以直接 `predict` 的 YAML。

C1～C16 的第一名使用每实例全局 TP16、ADP1、MoE-TP16、EP1；C32 起换为全局 TP16、ADP4、MoE-TP16、EP1。
两副本合并部署在 C16/C32 给出 97.22/153.45 token/s；继续提高并发后，这一布局同样出现明显入场等待。

## 5. 合并：一个跨双节点实例

| C / N | 合并实例四元组 | 输出吞吐 token/s | mean TTFT s | mean TPOT ms/token | P99 TTFT s |
| --- | --- | ---: | ---: | ---: | ---: |
| [1 / 2](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-agg-1x2node-c1.yaml) | (1, 32, 1, 32) | 4.89 | 15.177 | 189.66 | 15.177 |
| [2 / 4](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-agg-1x2node-c2.yaml) | (1, 32, 1, 32) | 9.79 | 15.177 | 189.66 | 15.177 |
| [4 / 8](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-agg-1x2node-c4.yaml) | (1, 32, 1, 32) | 19.58 | 15.177 | 189.66 | 15.177 |
| [8 / 16](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-agg-1x2node-c8.yaml) | (1, 32, 1, 32) | 39.16 | 15.177 | 189.66 | 15.177 |
| [16 / 32](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-agg-1x2node-c16.yaml) | (1, 32, 1, 32) | 78.32 | 15.177 | 189.66 | 15.177 |
| [32 / 64](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-agg-1x2node-c32.yaml) | (1, 32, 1, 32) | 156.63 | 15.177 | 189.66 | 15.177 |
| [64 / 128](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-agg-1x2node-c64.yaml) | (1, 32, 1, 32) | 281.03 | 22.766 | 205.70 | 30.355 |
| [128 / 256](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-agg-1x2node-c128.yaml) | (1, 32, 1, 32) | 507.58 | 37.943 | 215.33 | 60.709 |

点击 C / N 可打开该行可以直接 `predict` 的 YAML。

八档第一名均为全局 TP32、ADP32、Attention TP1、MoE-TP1、EP32，使用跨节点 DeepEP 路径。
它在低并发下不能充分利用所有 DP rank，而较高并发时可以把请求分散到更多 rank；本轮 C32/C64/C128 的吞吐分别为 156.63/281.03/507.58 token/s。
C128 的平均 TTFT 为 37.943 s、P99 TTFT 为 60.709 s，平均 TPOT 为 215.33 ms/token；是否可用应按业务 SLA 判断，不能只看吞吐。

## 6. 固定 TP16 / ADP2 / EP16 的参考

为便于与之前的并行设置比较，额外列出 P、D 都采用 `(8, 2, 1, 16)`、各一个节点的固定 1P1D 配置。其调度、精度、内存和请求设置与本轮搜索一致；并不声称其他启动参数与历史服务完全相同。

| C | 输出吞吐 token/s | mean TTFT s | mean TPOT ms/token |
| ---: | ---: | ---: | ---: |
| 1 | 4.92 | 2.391 | 201.22 |
| 2 | 9.83 | 2.455 | 201.22 |
| 4 | 19.53 | 3.210 | 201.26 |
| 8 | 38.58 | 4.523 | 201.35 |
| 16 | 39.28 | 96.446 | 201.35 |
| 32 | 44.81 | 350.472 | 201.36 |
| 64 | 48.20 | 849.300 | 201.36 |
| 128 | 48.29 | 1851.560 | 201.36 |

这说明本轮不仅改变客户端并发，也比较了 attention DP 和 MoE 并行路径。EP16 的固定配置不能替代全部并行配置搜索。

## 7. 校验、解释边界与文件

- 16 个菜单全部耗尽有限候选空间；416/416 成功，0 failed、0 timed out、0 unsupported。
- 每个成功候选均校验请求完成数、输入/输出总 token 和吞吐分母；没有请求拒绝、失败或取消。
- 三类布局×八档并发，共 24 份保存配置重新运行 `predict --capture-per-request`。1530 条请求均为 8192/1024 并正常完成；逐请求首次/末次 token 时间顺序检查通过。
- 复算与搜索的最大绝对数值差为 `8.15e-10`，属于浮点求和顺序差异；本轮 24 个布局优胜者没有 readmission。
- C4/C8/C16/C32 的 144 个两节点 PD 候选与之前四节点预算搜索中的同配置结果一致。
- 这些校验证明保存配置可复现、统计口径一致，**不能证明与真实 SGLang 服务同样准确**。
- 仍未校准的因素包括 host KV 备份/恢复耗时、CPU/host 开销、跨请求发送争用和真实专家负载偏斜；shape 外推仍采用 anchored 策略。
- 本轮没有进行与相同 eager / 无 overlap 调度部署对应的服务实测，不能给出实测误差百分比。低并发中仅几个百分点的排名差异不宜直接当作部署结论。

- [416 个候选的全部指标和配置坐标 CSV](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/candidates.csv)
- [汇总与模型元数据 SHA256](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/report.json)
- [24 份配置复算及入场等待统计](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/replay-validation.json)
- [完整搜索原始输出存档](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/search-evidence.tar.gz)
- [逐请求复算存档](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/replay-evidence.tar.gz)
- [PD 搜索配置](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/search-pd.yaml) / [合并搜索配置](../benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/search-agg.yaml)

运行镜像：`aisimulate-910c:glm52-w8a8`。

- Image ID：`sha256:b3c49d92579f638228c66076901af4768278c8fa309a90d0a002eed79e653cd9`
- Runtime source：`db5f2e721670d3eb2a10785fe55ad2d28af81349`
- Native SHA256：`5ee56d5c3b3096d63582604baf040b2cb07d68f73755752de709db80d156cbad`

## 8. 如何复现

例如重新搜索 C32、N64 下两节点 PD 的全部当前候选：

```bash
cd /home/youxingye/youxingye_data/aisimulate-910c
AISIMULATE_910C_MODEL=/data_lib/data/models/glm-q bash scripts/run-ascend910c.sh recommend \
  --stack engine \
  --config benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/search-pd.yaml \
  --set traffic.load.concurrency=32 \
  --set traffic.stop.requests=64 \
  --output-dir aisimulate-output/two-node-pd-c32-recheck
```

将配置换成 `search-agg.yaml` 即搜索两种合并布局；保持 `min_candidate_gpus=max_candidate_gpus=32` 才是本报告的恰好两节点条件。

例如复算吞吐最高的跨两节点合并部署 C128：

```bash
AISIMULATE_910C_MODEL=/data_lib/data/models/glm-q bash scripts/run-ascend910c.sh predict \
  --stack engine \
  --config benchmarks/evidence/ascend910c-glm52-w8a8/production/two-node-8k-eager-20261031/best-agg-1x2node-c128.yaml \
  --capture-per-request \
  --output-dir aisimulate-output/two-node-agg-c128-recheck
```

每次使用一个尚不存在的输出目录。只改变并发时，同时按目标压测方式设置请求总数；不必始终使用 `N=2*C`，但不同 N/预热/缓存条件的结果不能直接混为一组。

适配检查点仍处于 stage 2 / in_progress：本轮完成的是离线预测和配置复算，真实 serving 精度尚未验收。既有历史启动脚本路径缺失告警不影响本轮使用模型元数据、当前镜像和性能表进行离线预测。
检查点：`benchmarks/evidence/ascend910c-glm52-w8a8/onboarding-checkpoint.json`。
