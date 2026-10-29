# 910C 算子 shape 扩充与实测偏差排查

更新日期：2026-10-29。硬件是当前 node20，沿用此前固定的 SGLang-Ascend 0.5.22 镜像和计时协议。

**当前 8K/1K、32K/1K 的首要问题已经不是 shape 数量不足。**这轮新增 2,165 个有效测量点并入库，
重新运行了 DOCX 中对应的 50 组 PD 配置。新增的大范围覆盖几乎不改变这 50 组结果；现有算子诊断均为
`silicon`，没有报告范围外估计。后续应优先核对实际调度参数、MoE/通信执行与重叠关系，再针对缺口补测。

`silicon` 包括测量范围内插值，不表示每轮的每个坐标都直接实测过；组合算子的汇总来源也不是完整的内部查询追踪。
“能预测任意长度”“对应 shape 有实测数据”“整机预测准确”是三个不同的验收项目。

## 1. 这轮实际增加了什么

| 数据表 | 扩充前 | 新增有效点 | 扩充后 |
|---|---:|---:|---:|
| Attention / Indexer | 2,577 | 1,638：先补 268，再补 1,370 | 4,215 |
| Linear / GEMM | 2,645 | 243 | 2,888 |
| MLA BMM | 366 | 68 | 434 |
| Vector / 归一化等 | 1,194 | 216 | 1,410 |
| 合计 | 6,782 | **2,165** | **8,947** |

新增范围主要包括：

- Attention / Indexer 的长 prefix、长上下文和 32K prefill token 规模；decode 上下文测到 132,096，即 128K 输入加 1K 输出。
- 稀疏注意力 local heads 为 4/8/16/32/64，沿用对应 TP 布局；Indexer 维度按模型结构固定。
- 本次长上下文 eager prefill 选择 active batch 1/2/4；decode 沿用图桶 8/16/24/32 及原有活跃请求数网格。
- Linear、BMM、Vector 增加较大的 token/M 维度，包括 12K、24K、32K 附近缺点，具体范围以各表完整键为准。

这些范围是多个维度的覆盖汇总，**不是完整笛卡尔积**。“有 128K 点”和“有 batch32 点”，
不能推断任意 TP、任意执行模式下都存在 `128K × batch32` 的实测。
图桶仍最大为 32，本轮没有将更大的 decode 图桶变成已验证配置。

测量使用固定镜像摘要 `5f384cab0e16…`、框架提交 `baa9befa3815…`，保持原有 dtype、量化方式、布局、
collector 主体和计时协议。不同协议的 Linear 曲线分别扩充，没有混用计时条件。

后续批次共 1,897 个新坐标，首次有 3 个点 CV 超过 10%：两个 Attention/Indexer 点、一个 Vector 点。
保留全部原始记录，只对三个缺口单独复测一次，复测 CV 分别约 8.96%、5.66%、0.28%，合格后补入。
没有从多个合格重复样本中挑最快值，也没有覆盖既有合格测量。

四张表全部 8,947 行通过 Rust 原生算子查询验证，实测坐标查询的最大绝对误差为 0。
这是数据接入检查，不是整轮 forward 或端到端准确率为 100%。
原始样本、失败记录、负对照、方案、质量报告和入库前后数据见
[采集清单](../benchmarks/evidence/ascend910c-glm52-w8a8/production/shape-expansion-node20-20261029/manifest.json)
与同目录 `raw-evidence.tar.gz`，清单包含归档及逐文件 SHA256。

## 2. DOCX 基线如何对比

源文件是用户提供的 `不同 PD Ratio的Baseline.docx`，SHA256 为
`db38316f640ddcb0880a651063edb8b59874a5d569daf192b59ab4b7d5fe082f`。
直接读取 24 个嵌入 Excel 表，提取 107 条记录，没有使用截图 OCR。
表头标的是 **mean / p99**，因此不能把这些 mean 指标与预测 p50 直接比较。

本轮比较五种 PD 部署的 8K/1K、32K/1K，共 50 个点：

- 8K/1K 的请求数为并发的两倍；32K/1K 的请求数等于并发，是有限的一轮请求。
- P/D 都采用 attention TP8、ADP2、MoE TP1、EP16；一个 replica 是一个 16-device 节点。
- P/D 显存比例为 0.88/0.90，上下文 180,000，调度上限 64，关闭前缀复用。
- 输入输出长度按 workload 名称取固定 token 数；历史逐请求 tokenized trace 不存在。

**发现了一个必须纠正的配置解释：**公共字段 `scheduler.max_batched_tokens` 不控制当前 SGLang replay 的实际 chunk。
SGLang replay 使用原生默认全局 chunk 8,192，ADP2 后每 rank 为 4,096。
DOCX 部署写的是全局 chunk 16,384，另有 `max-prefill-tokens=32768`。
仅把 `max_batched_tokens` 设成 16,384，不能声称已对齐部署。
此差异已记录在对比脚本并修正使用说明；原生参数的独立公共配置仍需继续补齐。

其他未对齐项包括历史镜像/CANN/HCCL 精确版本、请求到达时刻、P 的 cache-aware 路由、
P 的 `TASK_QUEUE_ENABLE=1`、启动时真实 KV 容量、每个 rank 的 batch 时间线。
这些对比可以定位问题，但不是同环境精度验收。

## 3. 结果说明了什么

以下为新增全部 shape 后、保持原有 admission 行为的 1P1D 结果：

| workload / 并发 | 实测 TTFT/s | 预测 TTFT/s | 实测 TPOT/ms | 预测 TPOT/ms | 实测输出tok/s | 预测输出tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 8K/1K / 1 | 6.305 | 2.771 | 52.47 | 108.50 | 17.07 | 9.00 |
| 8K/1K / 32 | 123.417 | 69.168 | 56.91 | 234.28 | 141.90 | 86.06 |
| 32K/1K / 1 | 12.918 | 11.500 | 53.80 | 108.58 | 15.07 | 8.35 |
| 32K/1K / 2 | 16.414 | 11.500 | 55.17 | 108.58 | 28.10 | 16.71 |
| 32K/1K / 4 | 52.770 | 72.789 | 55.38 | 108.58 | 28.11 | 16.71 |
| 32K/1K / 8 | 125.314 | 195.367 | 55.37 | 108.58 | 27.65 | 16.71 |

50 点的绝对相对误差中位数约为 TTFT 54.14%、TPOT 99.93%、吞吐 41.02%。
完成第一批 268 点后，再增加 1,897 点，50 组配置的五项对比指标最大变化仅为浮点舍入量级。
新覆盖对更长上下文和更大矩阵有价值，但不是现有两倍 TPOT 偏差的解决办法。

在 TP8 / ADP2 / EP16、decode batch1 的原生拆分查询中，按 75 层 MoE 累计：

- dispatch 约 23.36 ms；
- combine 约 22.21 ms；
- expert compute 约 17.46 ms。

三项已约 63.03 ms，而历史实测整轮 TPOT 常在 55 ms 左右，完整 MoE 组合还包含其他操作。
这把检查重点指向实际 MoE/通信调用、图捕获、输入负载和重叠关系，
但**不能据此断言哪一项恰好多算一倍**。没有给原始算子数据或最终 TPOT 乘拟合系数。

完整结果见[对比报告](../benchmarks/evidence/ascend910c-glm52-w8a8/production/baseline-comparison-node20-20261029/report.json)
和[CSV](../benchmarks/evidence/ascend910c-glm52-w8a8/production/baseline-comparison-node20-20261029/comparison.csv)。

![1P1D并发与TTFT、TPOT、吞吐的对比](../benchmarks/evidence/ascend910c-glm52-w8a8/production/baseline-comparison-node20-20261029/comparison.png)

## 4. 并发为什么不能直接当作 running batch

客户端并发包括排队、P 计算、等待 D 接收、KV 传输、D 计算等状态。
单个 D rank 的 running batch 还受 KV 容量、ADP 分流、接收预算限制。

这组模拟配置的 D rank 可用 KV 容量约为 `406 × 128 = 51,968` token。
两个 32K 输入、1K 输出请求需要约 `2 × (32768 + 1024) = 67,584` token，超过一个 rank 容量。
因此 ADP2 的一个 D worker 同时容纳的长请求数量很小；客户端并发从 2 增到 8，主要可能增加等待。
这是当前模拟配置的容量估计，不是 DOCX 那次运行已测得的真实 KV pool。

实测 1P1D 的 32K/1K 在并发 2/4/8 时，吞吐约 28 token/s、TPOT 约 55 ms，而 TTFT 显著增长，与此现象相符。
`吞吐 × TPOT × (输出长度−1)/输出长度` 可以辅助估计整个有限测试区间的平均 decode 占用，
但它包含填充与排空，不能当作某个 rank 某一时刻的实际 running batch。

## 5. 已补的 PD decode 接收预算

这轮增加可选字段：

```yaml
engine:
  workers:
    decode:
      scheduler:
        decode_reserved_tokens: 512
```

它依据固定框架版本的 decode admission 行为，为运行、传输、等待激活的请求保留增长预算，
考虑被撤回请求及单请求完成空间限制。它是逻辑接收门槛，不会增加 KV 传输字节。
也补了撤回请求优先恢复，防止待接收请求与撤回请求相互等待。
默认 0 保持此前行为；推荐搜索、保存候选 YAML、再 predict 都保留该值。

同样 50 点开启 512 后全部完成。1P1D 的 8K/1K、并发 32：

| 指标 | 旧接收行为 | 开启预算512 | DOCX实测 |
|---|---:|---:|---:|
| 请求重新进入调度的总次数 | 26 | 0 | 无该项记录 |
| mean TPOT/ms | 234.28 | 108.66 | 56.91 |
| mean TTFT/s | 69.17 | 194.19 | 123.42 |
| 输出token/s | 86.06 | 81.87 | 141.90 |

预算避免了这组模拟中的频繁撤回，使等待更多发生在接收前，**不能解决单步耗时偏差**。
整体误差没有全面改善：50 点 TPOT 绝对相对误差中位数仍约 96.97%，TTFT 55.58%，吞吐 42.66%。
不能把行为补齐当成精度校准完成。

## 6. 接下来需要补什么

| 优先级 | 工作 | 原因 |
|---|---|---|
| 最高 | 对齐有效chunk、max-prefill、调度、KV容量、运行版本 | 先消除配置差异，尤其TTFT和排队误差 |
| 最高 | 同版本实际路径测decode batch1/2/4的MoE整块、整轮forward | 低batch已有两倍TPOT偏差，需要核对组合边界与重叠 |
| 次高 | 采真实专家token分布、热点专家及dispatch/combine消息规模 | balanced/single-expert等合成分布不等于真实路由 |
| 按需 | 补新workload查询的estimated、缺失键、不稳定插值区域 | 按实际查询频率和耗时占比选择，不均匀加密所有维度 |
| 按需 | 新图桶、更大active batch、新TP/EP/拓扑、新内核版本 | 执行身份变化不能用输入输出长度外推代替实测 |

现有 MoE chain 为 225 点，张量并行 MoE 为 126 点，DeepEP 为 143 点，HCCL 为 449 点。
具体维度见采集清单，也不构成任意部署的完整实测网格。

目前 node20 的 `/mnt/raid/user/data/models/glm-q/config.json` 不存在，本轮离线回放使用仓库保存的真实模型元数据。
离线预测不需要完整权重；启动完整模型、采真实路由或完整 forward 时需要可访问的权重和部署。
本次没有把合成算子微基准当作新采到的整机实测。

## 7. 运行与证据位置

仍使用 `predict` / `recommend`。在仓库根目录执行，缺少原模型目录时可用元数据快照：

```bash
model_metadata="$PWD/benchmarks/evidence/ascend910c-glm52-w8a8/production/deepep-contract-node19-20261024/model-metadata"
AISIMULATE_910C_MODEL="$model_metadata" bash scripts/run-ascend910c.sh predict \
  --stack engine \
  --config examples/cli/ascend910c-glm52-pd-8k1k.yaml \
  --set "engine.model=$model_metadata" \
  --set engine.context_length=180000 \
  --set traffic.source.input_tokens=32768 \
  --set traffic.source.output_tokens=1024 \
  --set traffic.load.concurrency=2 \
  --set traffic.stop.requests=2 \
  --set engine.workers.decode.scheduler.decode_reserved_tokens=512 \
  --detail memory,time,source \
  --output-dir aisimulate-output/my-32k1k-check
```

这是使用示例；逐点基线对比以 `production/replay_pd_baseline.py` 生成的完整配置为准。
采集证据归档采用仓库相对路径，可以先按清单校验 SHA256，再解压到空目录。

阶段提交：`73cbb896` 扩展采集与增量接入，`263c518b` 可选PD预算，
`b39d8f57` 正式入库2,165点及归档原始证据。
本次算子扩充已完成；完整 serving 准确率和正式 FPM onboarding 验收仍未完成。

## 8. 2026-10-30：按固定 SGLang 版本修复调度对齐

后续以 `v0.5.22-cann9.1.0-a3` 为主基线，参考 `v0.5.20-cann9.0.0-a3`。
已查询官方镜像标签，并核对本地镜像的源码：前者 manifest digest 为
`sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443`，
SGLang commit 为 `baa9befa381511cd4d56f83980951e5a45fffc6d`，与算子采集版本一致。
后者标签的 digest 为 `sha256:0756b534a0379863c435ec8b7896ba17a18fccda2c8889326fb1dc9453de9be4`，
本地同名镜像源码 commit 为 `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`；
其本地镜像未记录 RepoDigest，因此不将标签查询单独作为本地二进制匹配证明。

实验性的 SGLang Simulator 桥接已撤出运行代码，并保留在
`build/sglang-simulator-deferred-20261029/experimental-bridge.tar.gz`。
此前仅完成接口测试，完整调度进程未跑通，也未接管现有 PD 搜索。
当前继续使用 AISimulate 的 Rust 调度、PD 事件和统一算子预测器。
该选择避免先扩展单进程模拟器的多 P/D、多 DP orchestration，再重复维护现有搜索路径。
没有“接入 SGLang Simulator 后准确率已提高”的测量结论。

首轮修复覆盖配置入口到实际执行的传递：

| 项目 | 修复前 | 本轮行为 |
| --- | --- | --- |
| SGLang chunk | 修改公开 `max_batched_tokens` 不改变 chunk，仍为全局 8192 | 新增 `scheduler.chunked_prefill_size`，显式传入全局值，由原生调度器按 ADP 除一次 |
| Prefill 上限 | 公开配置不能传入 `max_prefill_tokens`，使用原生默认 16384 | 新增同名字段，可与启动命令的 32768 对齐 |
| 最大请求数 | 基线脚本把全局 64 填成每 rank 64 | ADP2 时填每 rank 32；公开 `max_sequences` 的每 rank 含义不变 |
| 推荐配置保存 | 无法保留上述显式调度预算 | 搜索候选、部署参数、保存 YAML、重新 `predict` 均保留 |
| D 端准入预留 | 基线脚本默认 0，需手动指定 512 | 默认 512，与上述版本参数默认值一致；可显式指定 0 做消融 |

`replay_pd_baseline.py --scheduler-alignment runtime` 使用修复后的预算。
`--scheduler-alignment legacy --decode-reserved-tokens 0` 可重建旧脚本的预算设置。
已有历史报告保持原样；新报告必须写入新目录。

原生执行回归使用 ADP2、两条 8192-token 请求和固定每轮 prefill 10 ms：
全局 chunk 8192 时 TTFT 为 20 ms，改成 16384 后为 10 ms，decode TPOT 均为 1 ms。
这验证参数确实改变 Rust 调度轮数；它不是模型性能准确率测试。

仍需继续处理的差异：固定长度请求的 `ignore_eos` 准入规则、P/D 交接和路由、
每轮真实 batch 与算子耗时组合，以及 graph/eager 执行模式和内存预算。
本节测试时 GLM decode 算子图仍固定选取 graph 数据；随后加入的显式 eager 模式见第 9 节。
`--disable-overlap-schedule` 与 MoE shared-expert 多流重叠也是不同开关。
新关闭 overlap/graph 实测到位前，本节只确认源码和调度配置修复，不声明 TTFT、TPOT、吞吐已达到准确率目标。

本轮 50/50 组历史 workload 回放均完成，decode readmission 总数为 0。
按“每组相对误差绝对值再取中位数”统计：

| 版本 | TTFT | TPOT | 输出吞吐 |
| --- | ---: | ---: | ---: |
| 原始已扩充 shape | 54.14% | 99.93% | 41.02% |
| 仅 D 预留 512 | 55.58% | 96.97% | 42.66% |
| 本轮再对齐 chunk、prefill 上限、每 rank 请求数 | 55.81% | 96.97% | 42.30% |

这些结果说明配置修复没有消除主体误差，不能声称整体准确率改善。
例如 1P1D、8K/1K、C1 的新预测 TTFT 2.564 s、TPOT 108.50 ms，历史实测仍为
6.305 s、52.47 ms；32K/1K、C1 为 10.671 s、108.58 ms，对照实测 12.918 s、53.80 ms。
50 组逐项结果、参数及原始回放输出归档位于
`benchmarks/evidence/ascend910c-glm52-w8a8/production/scheduler-comparison-node20-20261030/`。

## 9. 显式选择关闭 graph 的预测

设置 `engine.estimator_config.op_level.ascend.decode_execution_mode=eager` 后，
decode 的线性、注意力、MoE 和通信使用 eager 性能记录，取消 graph batch bucket 填充。
默认仍为 `graph`，保持既有报告可复现。Prefill 仍使用 eager 数据。
没有对应 eager shape 时明确报错，不借用 graph 耗时冒充已覆盖。
当前 DCP>1 没有 eager DCP 数据，因此该组合也明确拒绝。

```bash
AISIMULATE_910C_MODEL="$model_metadata" bash scripts/run-ascend910c.sh predict \
  --stack engine \
  --config examples/cli/ascend910c-glm52-pd-8k1k.yaml \
  --set "engine.model=$model_metadata" \
  --set engine.context_length=180000 \
  --set engine.estimator_config.op_level.ascend.decode_execution_mode=eager \
  --set engine.estimator_config.op_level.ascend.shape_policy=anchored \
  --set traffic.load.concurrency=2 \
  --set traffic.stop.requests=2 \
  --output-dir aisimulate-output/my-eager-8k1k-check
```

对 TP1/2/4/8/16、全局规模16/32、EP或tensor-MoE、两种上下文、四种batch的
320次统一预测器查询，graph为160/160成功，eager为144/160成功。
16次失败均为EP32的DeepEP发送token数超出已测范围，错误原样保留。
这只是该网格的可查询性，不表示覆盖任意shape，也不表示性能准确率。
探针脚本及报告位于 `production/probe_decode_execution_modes.py` 和
`production/scheduler-audit-node20-20261030/decode-mode-probe.json`（相对本模型的证据目录）。

原生 eager PD 回放完成2条8K/1K请求，TTFT为2655.76ms、TPOT为200.52ms。
这是预测值，尚无相同启动参数下的真实基线。eager微基准并不一定更快，
也不能将原先开graph的52ms左右TPOT直接作为eager误差评价。
本阶段通过2063项Rust单元测试（1项忽略）、390项双向parity测试和19项模型组成测试。
旁侧KV测试中有一项仍期待DCP2不受支持，但旧镜像已支持；已单独复现为既存测试问题。

关闭CPU overlap调度不等于关闭shared-expert算子多流重叠。
`shared_expert_overlap` 需独立按实际部署设置；这里只切换graph/eager执行模式。
