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

截至2026-10-30，MoE chain为225点，张量并行MoE为126点，DeepEP为150点，HCCL为449点。
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

## 10. 搜索使用 workload 长度做 KV 可行性检查

修复了一个会直接阻止搜索的问题：模型配置允许180K上下文，
不代表预测8K/1K时每个rank的KV池必须装得下180K请求。
对单轮合成文本负载，现在按实际 `input_tokens + output_tokens` 筛选容量，
保存配置和真实回放仍保留 `context_length=180000`。
trace、多轮、多模态、AFD和直接FPM profile等无法安全确定长度的路径仍采用原来的保守检查。
workload超过显式上下文限制时仍按原校验路径处理。

修复后，8K/1K、C8、8请求的小规模eager搜索完成12个候选；
第一名保存YAML重新预测后，完成请求数、TTFT、TPOT、吞吐与搜索报告完全相同。
32K/1K也完成TP8/ADP2/EP16下、最多4节点的6种合法配比搜索。
这些短回放只验证搜索、保存、重新执行连通，不能当作稳态吞吐基准或真实最优配比结论。

两个实际使用时的区别：

- 8K示例中的TP4/ADP4在当前内存预算下装不下32K请求；修改输入长度时须同步调整并行候选。
- `parallelism.preset` 是显式承诺所有组合合法的清单；P、D各列1/2/3个副本会产生超过4节点的组合并报错。
  希望按总节点预算自动筛选时，改用独立搜索轴：

```yaml
parallelism:
  preset: false
  tensor: 8
  attention_data: 2
  moe_tensor: 1
  moe_expert: 16
  pipeline: 1
  replicas: {choices: [1, 2, 3]}
```

将该块分别用于P、D，并设置 `optimization.constraints.max_candidate_gpus: 64`，
即可包含1P1D、1P2D、1P3D、2P1D、2P2D、3P1D。
这里一个副本占16个模拟器device（一个910C节点）；64表示4节点。
搜索目标、并发、请求数和SLA必须与实测一致，最大吞吐目标与每device吞吐目标可能给出不同排名。

搜索相关205项测试通过；额外5项依赖HuggingFace模型元数据下载的测试在无网络容器中失败。
独立运行本次修改的 `tests/sweeper/test_search.py` 为45/45通过。
原始成功与失败回放、配置和一致性检查保存在
`production/scheduler-audit-node20-20261030/search-validation-evidence.tar.gz`。

## 11. 固定输出长度的准入规则

新增 `workers.<role>.scheduler.ignore_eos: true`，对齐固定版本0.5.22的同名请求语义。
它仅在客户端确实发送 `ignore_eos=true` 时启用；未设置时保留旧行为。
可以在P、D分别设置；基线回放脚本增加了 `--ignore-eos` / `--no-ignore-eos`。
历史DOCX没有完整保存该采样参数，因此不自动把历史50组结果改成此模式。

原来正在decode的请求统一按自适应比例估算剩余输出，
这对固定1K输出的请求会低估后续KV需求。无prefix cache时，
新路径还会在开始chunk之前检查整段待计算prompt，并按请求完成顺序估算KV释放；
不是简单把所有请求的最长输出同时相加。额外的一页准入预留和首条prefill跨越计算预算的行为也已对齐。
chunk续算保留原来已拥有的KV，避免将续算当成新请求重复匹配或重复记录准入。

参考依据为镜像中的 `schedule_policy.py` 与 `mem_cache/prefill_budget.py`。
`production/check_fixed_output_admission.py` 对这两个文件先校验SHA256，
然后直接执行其中原始的准入方法，用确定性请求和单KV池对象生成103个参考案例。
Rust逐项检查实际准入和chunk终点；103例全部相符。另有5项边界回归，
完整Rust库为2069通过、1忽略。该oracle不启动NPU，不证明完整SGLang调度进程或实测性能完全等价。

0.5.20的running-request offset仍使用自适应比例，与0.5.22存在源码差异。
因此本开关明确采用0.5.22语义，不能用版本号字符串推断已经同时精确复现两个版本。
当前开关是每role统一设置，未表示混合 `ignore_eos` 请求流，SWA、Mamba、DCP专用准入也不在这组oracle范围内。

## 12. PD 路由需要与入口服务一致

现有原生PD默认将全部副本的attention-DP rank展平再轮询。
例如两个D副本、每副本ADP2，前四条请求落在rank `[0,1,2,3]`：
前两条同时占用第一个副本。实际入口若先轮询D副本，再由各副本独立轮询内部rank，
对应顺序应为 `[0,2,1,3]`。并发很小时，这会直接改变使用几个副本，影响PD搜索排名。

新增固定配置 `engine.pd_routing: replica_round_robin`，P、D分别执行两级轮询。
`rank_round_robin` 或省略字段可重现旧顺序；此字段随候选一起保存。
它只适用于离线engine栈，不接管Dynamo等适配器的路由，不代表已模拟cache-aware路由。

历史DOCX写明了P入口 `--prefill-policy cache_aware`、D入口 `--decode-policy round_robin`。
P进程显式 `--load-balance-method round_robin`；所核验版本的D默认也解析为round_robin。
因此历史P端的cache-aware行为尚未完全重现，不能宣称换了这个字段便完全等价于历史部署。
为了建立可归因的新基线，建议实测时P、D入口都显式使用round_robin，
P、D进程都显式使用 `--load-balance-method round_robin`，并关闭prefix cache。

对应一组明确的预测设置为：

```yaml
engine:
  pd_routing: replica_round_robin
  estimator_config:
    op_level:
      ascend:
        decode_execution_mode: eager
        shape_policy: anchored
        shared_expert_overlap: true
  workers:
    prefill:
      scheduler: {chunked_prefill_size: 16384, max_prefill_tokens: 32768, max_sequences: 32, ignore_eos: true}
      kv_cache: {prefix_caching: false}
    decode:
      scheduler: {max_sequences: 32, decode_reserved_tokens: 512, ignore_eos: true}
      kv_cache: {prefix_caching: false}
```

这是对已有完整YAML的配置片段，不能单独作为完整输入。
`max_sequences: 32` 对应全局max-running64、ADP2；改变ADP时需重算每rank上限。
`ignore_eos: true` 对应客户端固定输出设置，shared-expert重叠开关仍须与实际设备流执行一致。

0.5.22中关闭两阶段graph的显式参数为
`--cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled`；
旧的 `--disable-cuda-graph` 已是兼容别名。P、D均加 `--disable-overlap-schedule`。
这些开关应记录到最终启动命令，并从启动日志核对解析后的配置，而不只记录镜像标签。

路由修正的验证：Rust库2072通过、1忽略，runtime/replay/artifact合同34通过，
Python配置与搜索473通过、runner513通过，两套必须的parity测试合计390通过。
真实原生引擎的双请求测试确认了两种不同的D rank顺序。
8K/1K和32K/1K各6种PD组合也完成了功能搜索，但这不是准确性验收。
随后源码审计发现 eager 仍会按attention TP对齐token数；此前eager建模漏掉了该padding，
这些搜索结果仅作为修正前的功能证据保存，不能用来选择实际部署的最优配比。
正在补齐对应的算子shape与测量，详见后续修正记录。

## 13. Eager decode 的 TP 对齐与补采

固定0.5.22源码的 `prepare_mlp_sync_batch` 先将各rank的token数向上对齐attention TP，
然后选择MAX_LEN/SUM_LEN；关闭graph不会取消这层对齐。
例如TP8、ADP2、每rank一个decode请求，矩阵输入仍有8行，attention只有1条有效KV序列，
另外7行的KV长度为0。实际请求数、补齐后的token数和分片后的MoE输入必须分别建模。
新增eager decode采集case覆盖这种active/padded组合，复用现有列定义，保留未补齐case。

2026-10-30在空闲的node20使用原固定镜像采集首轮306个case，全部通过数值检查，
其中65个因计时CV超过10%而未通过质量门槛。原采集器先预热再做CPU数值校验，
计时前经历较长的CPU校验，首个样本普遍偏高。修正为数值校验后再执行完整计时预热，
并预热计时事件；没有删除首个实测样本，也没有降低10%的门槛。
修正后的全量eager decode计划为1872个case，全部执行并通过数值检查，1835个通过CV门槛。
对37个CV不合格点预先约定只做一次串行复测，36个通过，合计1871个合格点。
剩余1点为local-heads4、active/padded8/8、KV长度16384；保留原始异常样本，不纳入性能表。
旧数据和失败记录保留在 `production/eager-padding-node20-20261030`，不会混入同一测量协议。
新采集覆盖context128～132096、active batch1～32、attention TP1/2/4/8/16，
包括8K/1K、32K/1K对应的KV长度端点。未采集坐标仍受measured/anchored策略约束。

原生模型现已区分eager TP对齐与graph bucket，并将对齐后的形状传给线性层、BMM、向量、
HCCL、MoE和attention/indexer。Tensor-MoE且ADP1、无A2A时不进入MLP同步，仍用未补齐eager行数。
attention的未补齐曲线与padded曲线分开保存；不会把实际请求数增加为补齐后的行数。
新增算子字段使EngineSpec二进制schema升级为30，旧原生库需要随Python代码一起重建。

第一批发布用1871条新eager decode记录替换419条旧协议记录，3796条graph/prefill原始记录逐项保持不变。
Rust全量2073通过、1忽略，必须的两套parity390通过，模型与接口检查38通过。
TP8/ADP2/EP16下两种workload各6种PD组合均完成；保存的最佳候选重新predict，
TTFT、TPOT、吞吐、完成请求数和模拟总时长均与搜索结果逐项相同。
这是8条请求、并发8的功能回归，不能作为稳态性能排名或实测准确性证据。

随后补齐TP2等布局的中间batch所需padded曲线：新增1072个case，首次23个CV超标，
一次串行复测全部通过。现有attention表共有6739条，其中本次新协议eager decode为2943条；
原graph/prefill观测值保持不变。补采覆盖记录和原始样本分别保存在
`padding-plane-collection-validation.json` 与 `padding-plane-raw-evidence.tar.gz`。

扩大探针到所有batch1～32、KV长度8193/32769、两种MoE布局、全局宽度16/32、
local TP1/2/4/8/16、graph/eager，共2560个整轮forward查询。
2394个成功，166个失败全部属于EP32 eager DeepEP的sent-token轴超出实测1～3范围；
EP16 eager、tensor-MoE eager和所有graph测试均成功。
这是uniform DP batch的接口与覆盖验证，尚未证明非均匀DP负载、实际MoE路由或整轮耗时准确。

## 14. 本轮对齐状态与未完成验收

已修复：global chunk预算被遗漏、固定输出准入预留、PD两级轮询、
搜索时误用完整180K context做workload可行性检查、eager TP padding与对应算子profile。
另外将非固定输出模式的初始new-token ratio封顶为1，与0.5.22一致；
此前提高schedule conservativeness可能预留超过请求输出上限的KV。默认参数不受该边界修正影响。

以下仍不能仅凭源码修正宣称准确：

- PD的attention-DP算子组合已按各rank实际batch推导MAX_LEN/SUM_LEN，见第16节。
  不同rank混合prefill/decode、非均匀DeepEP传输曲线及真实MoE路由仍需继续核验。
- 历史P router使用cache-aware，新基线选择round-robin；两者请求分布不完全相同。
- P释放资源及D返回首token的边界已按源码修复，见第19、20节。
  PD多路同时传输的带宽竞争、chunk发送、CPU调度和HTTP开销尚需继续核验。
- EP32 eager DeepEP的balanced曲线已补齐1～32 token范围，见第18节；
  DCP eager仍缺少实测profiles，不能借用graph时间。
- 历史DOCX的overlap/graph及采样参数没有完全固定，不能作为关闭两者后的准确性验收。

因此目前能提交的是可复现的修正、算子数据和功能验证；最终TTFT/TPOT/吞吐误差以及
“最优PD配比”是否与实测相符，仍需要同版本、同启动参数、同workload的服务结果。
之前历史50组回放的误差仍大，不能用本轮测试通过数代替性能误差下降的证据。

## 15. 新实测必须固定客户端负载

固定镜像的 `python/sglang/benchmark/serving.py` 中，`--random-range-ratio` 默认是0，
`datasets/common.py:56-64` 会在1到目标长度之间随机采样。只设置input/output长度，
并不等于每条请求都是8192/1024。`--random-range-ratio 1` 才固定为目标长度。
另外，文本decode后重新tokenize可能改变长度；`--tokenize-prompt` 直接发送token IDs。
这不是在断言历史DOCX一定用了错误参数，而是新对照实验需要排除的歧义。

在已启动并按第12节固定参数的真实服务客户端环境中，可执行：

```bash
python3 -m sglang.benchmark.serving \
  --backend sglang --base-url http://127.0.0.1:30000 \
  --model /mnt/raid/user/data/models/glm-q \
  --tokenizer /mnt/raid/user/data/models/glm-q \
  --dataset-name random-ids --tokenize-prompt \
  --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1 \
  --num-prompts 128 --max-concurrency 8 --request-rate inf \
  --warmup-requests 1 --seed 910 --output-details \
  --output-file baseline-eager-8k1k-c8.jsonl
```

将base URL改成实际router地址；32K实验把input长度改成32768并更换输出文件。
这个版本SGLang generate客户端默认发送 `ignore_eos=true`，不要加 `--disable-ignore-eos`。
保留默认streaming，不能加 `--disable-stream` 后仍按同样方式解读TTFT/TPOT。
离线侧对应 `traffic.stop.requests: 128`、并发8、固定input/output长度以及
P/D `scheduler.ignore_eos: true`。预热请求不计入这128条；prefix cache关闭时不会引入预热命中。

同时保存P/D和router完整启动命令、镜像digest、解析后的max-running/chunk/max-prefill参数、
每rank `max_total_num_tokens`、请求实际input/output tokens和完整结果JSONL。
尤其核对profile采集使用的 `TASK_QUEUE_ENABLE=0`、CPU亲和性及共享专家多流设置。
`random-ids`是便于复现长度的合成负载，不保证真实专家路由与balanced profile完全相同。

客户端源码SHA和直接执行原始长度生成函数的结果保存于
`production/eager-padding-node20-20261030/benchmark-client-contract.json`。
第15节验证时，本机运行镜像更新为源码 `e53c91d3` 的schema30版本，
安装后的公开predict和PD搜索均通过；最终二进制两套parity共390通过、原生边界5通过，
最后一次准入边界修改另有84项SGLang调度回归通过。运行包身份与全部证据位于同目录
`runtime-publication.json`。后续schema31更新见第17节。这些结果验证实现与安装状态，准确性仍未验收。

## 16. 用真实DP batch推导算子shape

本阶段继续沿用AISimulate原生调度器，对齐固定0.5.22镜像的执行规则。
之前的实验性sglang-simulator桥接已经撤下并保留归档，当前预测不依赖它。

此前每个attention-DP rank分别估时，再取最大值作为同步完成时间。
虽然有同步屏障，dense/MoE/通信仍按“本rank token数乘DP”计算，无法表达不均匀负载。
现在专用P/D worker先完成各rank的调度选择，将包括空闲rank在内的整组batch交给
canonical Rust性能模型，然后确定共同完成时间。单rank估时不会重复计入这轮耗时。

以TP4/ADP4、某轮decode有效batch为`[1,0,0,0]`为例：

1. 先按attention TP对齐，得到`[4,0,0,0]`；attention仍只有1条有效KV序列。
2. 固定版本的eager规则比较`2 × sum`与`max × DP`，此例为8和16，选择SUM_LEN。
3. 全局dense/tensor-MoE buffer为4行。此前独立rank近似会按4×4=16行计算。
4. gather/combine选择SUM_LEN的all-reduce分支，不沿用MAX_LEN的all-gather/reduce-scatter组合。
5. EP16、top-k8的balanced模型使用`ceil(4×8/16)=2`作为平均专家接收token数。
   这仍是路由分布假设，不等于每次真实模型推理都恰好均匀。

相反，TP8/ADP2的`[1,9]`先对齐为`[8,16]`，eager选择MAX_LEN，
两个rank均以16行执行，全局buffer为32行。attention分别保留1和9条有效序列。
P阶段使用SUM_LEN；例如只有一个rank计算8192个新token，另一个空闲，
全局buffer就是8192行。graph开启时使用所有rank共同选择的capture bucket；
超出bucket范围后整个算子组合回到eager。

另外修正了decode统计：KV不足导致请求回退后，forward metrics只记录实际执行的
survivor batch；回退请求计入等待队列，避免并发升高时把它们算进本轮算子shape。

适用边界：专用P/D角色、DCP1、无投机、固定拓扑、已审计的DeepEP v1或tensor-MoE。
PD搜索自动使用此路径。直接传入多个rank的forward metrics时，必须提供完整DP向量；
单条metrics仍保留原来的uniform查询语义。PD合并部署的跨rank混合prefill/decode，
以及更复杂的gatherv/elastic规则不在此阶段的支持范围内。

源码oracle脚本`production/check_dp_padding.py`直接执行经过SHA验证的原始
`DpPaddingMode`和对齐函数，306个参考case写入
`production/scheduler-audit-node20-20261030/dp-padding-oracle.json`。
`check_dp_shape_coverage.py`通过公开canonical接口检查非均匀batch，结果文件位于同目录。
测试验证调度边界、shape选择和代码一致性，不能代替真实forward耗时验证。

剩余近似包括：DeepEP使用最大sender的实测曲线，尚未采集各rank发送量不同的完整传输曲线；
专家负载按balanced建模；一个rank内的KV长度仍有平均化；多路PD传输竞争和host开销未完全校准。
算子证据对应关键rank，不能据此宣称已计算所有设备的总能耗。
缺失profile仍明确报错，anchored外推结果也不等于该shape已经实测。

本阶段EngineSpec升级为schema31，需要同步更新原生库与Python组合代码。

验证记录为`production/scheduler-audit-node20-20261030/dp-synchronization-validation.json`：
306组源码oracle、2080项Rust核心测试通过；回退统计修改另跑84项SGLang测试；
最终原生库通过28项GLM测试、390项必需parity和5项原生调度边界测试。
包含关闭/开启诊断输出时预测一致的回归，防止搜索快速路径丢失性能模型。
扩展探针1264次查询中1140次成功，72次缺eager补齐plane，52次超出EP32 eager DeepEP范围，
这些缺口会明确报错并作为下一阶段的采集目标。

固定TP8/ADP2/EP16、最多4节点、并发8和8条完整请求的功能回放结果：

| 输入/输出 | 完成PD配比数 | 此小样本搜索首选 | 平均TTFT(ms) | 平均TPOT(ms) | 输出tok/s |
| --- | --- | --- | --- | --- | --- |
| 8192/1024 | 6/6 | 3P1D | 3260.178 | 201.353 | 38.815 |
| 32768/1024 | 6/6 | 2P2D | 119017.881 | 201.119 | 18.918 |

保存后的首选YAML再次predict，完成请求数、duration、TTFT、TPOT和吞吐与搜索完全一致。
上述8请求结果用于验证功能和回放一致性；样本少，部分候选分数接近，不作为实测最优配比结论。

## 17. 2026-10-30：补齐非均匀 DP 的 eager 注意力形状

第16节修正调度后，注意力查询会区分实际请求数和MAX_LEN补齐行数。例如两个DP
rank分别有1和3个decode请求时，均需按3行执行部分算子，而只有1个请求的rank不能
把另两行算作真实KV访问。原有只围绕均匀batch采集的数据不足以覆盖这种情况。

本次冻结新增3360个case，沿用同一镜像、驱动、采集实现和8次计时、5次预热、7组重复
的协议，在node20执行。补齐行数覆盖到32，active batch选择1、补齐数的一半向上取整、
补齐数减1等锚点，KV长度覆盖128至132096；实际GLM配置的注意力头数为64。
原有坐标通过精确比对去重，声明生成器也已包含这些shape。

| 处理结果 | 数量 |
| --- | ---: |
| 首次通过数值和CV检查 | 3256 |
| 数值检查失败，隔离 | 2 |
| 首次CV超10%，按预定协议串行复测一次 | 102 |
| 复测后通过 | 98 |
| 复测后仍超过CV上限，隔离 | 4 |
| 新增合格数据 | 3354 |
| 发布后的注意力数据总行数 | 10093 |

两次数值失败均为TP1的64头稀疏注意力，各有一个元素超过既定误差阈值；不放宽阈值，
不把这些结果写入性能表。旧6739行的耗时保留，原有3796行graph/prefill数据完全不变。
部分曲线的来源标识更新为可追溯的组合身份，不表示旧数据被重新测量。

原始计划、每次计时、数值失败、复测记录、合格导出和合并证明保存在
`production/eager-padding-node20-20261030/dp-max-len/`；归档及摘要见同级
`dp-max-len-evidence.tar.gz`和`dp-max-len-validation.json`。
正式表SHA256为`e1993aedd3ece9a3ad79f696d782d124e56499e8b0988b03edfdb1f5ff9aeafd`。

最终原生库的1264次非均匀batch查询有1198次成功。此前72次注意力补齐缺口均已消除；
其中14次查询继续执行后才暴露出DeepEP范围缺口，因此当前66次失败全部是EP32 eager
通信曲线范围不足。这不代表任意并行方式和任意shape均已覆盖：DCP eager、非均匀
发送量的通信建模、真实服务准确性仍需后续验证。对旧DOCX的误差不能用本轮测试通过数
或新增profile数量来替代衡量。

本机公开运行镜像已更新至提交`56d40b26`，EngineSpec为schema31，原生库SHA256为
`1e728bf4b37a5f1fd3328c44c688f7b597207efdd262c0e26d9f66bd515f7f99`。
2278个打包源码、数据及collector文件与提交内容逐项一致；49项针对性测试、390项parity通过。
安装后另外执行8K输入predict和32K输入、最多4节点的六种PD配比搜索；均成功。
安装检查仅使用2请求、32输出token，不替代第16节完整1K输出的功能回放或真实准确性验证。
运行包身份与命令记录见同级`dp-runtime-publication.json`。

## 18. 2026-10-30：EP32 eager通信扩展与搜索验证

在node19和node20上使用同一固定镜像、26.1.1驱动、bond4通信接口和原CPU亲和性，
采集balanced low-latency dispatch/combine。冻结的每rank token数为1、2、3、4、8、16、32，
协议保持256次计时、10次预热、7组重复。每个计时边界都检查所有rank的有效payload，
仍以CV不超过10%作为入表条件。

第一次连续测七个shape，在32-token combine计时阶段出现设备访存错误和rank0 SIGSEGV。
前六个shape的原始结果和失败日志全部保留；这份未完成的observations没有导入性能表。
之后用不变的collector和协议，在独立进程中单独复现32-token，再逐个采集其余六个shape，
七个shape的14个计时边界全部通过数值和CV检查。这只确认独立采集成功，
原来连续执行的崩溃原因仍未确认，不能据此宣称已修复运行时故障。

其中7个坐标已有合格历史数据，继续保留历史值；新增7个坐标：dispatch的8/16 token，
combine的2/4/8/16/32 token。DeepEP表由143行增至150行，其他算子表不变。
新旧重叠坐标的耗时变化为-12.01%至+13.13%，完整对照保留在发布报告。
同型号节点仍存在运行状态与测量波动，不能默认换节点后误差为零。

合并工具现在支持DeepEP坐标扩展：要求runtime、协议、serving setup完全一致，
拒绝重复坐标替换，保留原始耗时和来源，再生成整张表统一的可追溯身份。
正式表SHA256为`7bab1a8aa5bc69c283846d23d5aee36e39bf26f2838a65b1093bceb57268259c`。
数据和复现脚本位于`production/deepep-eager-pair-20261030/`，摘要为`validation.json`，
原始记录归档为`evidence.tar.gz`。

150条数据的Rust逐点读取全部与表一致；45项模型、导出和合并测试通过。
第17节的1264次查询全部通过。再加入所有uniform batch1～32并去重，共3576次查询全部通过，
覆盖全局16/32、TP1/2/4/8/16、两种MoE布局、graph/eager和KV8193/32769。
这些包含插值/anchored估计，不能把3576次成功查询当成3576个实测shape。

以P为TP8/ADP2/EP16、D为TP8/ADP4/EP32、最多4节点为例，每个P实例占1节点，
每个D实例占2节点。两个可行候选是1P1D（3节点）与2P1D（4节点）：

| 输入/输出 | 并发/请求数 | 此有限候选集的首选 | TTFT(ms) | TPOT(ms) | 输出tok/s |
| --- | --- | --- | ---: | ---: | ---: |
| 8192/1024 | 32/32 | 2P1D | 11263.694 | 223.101 | 132.086 |
| 32768/1024 | 8/8 | 2P1D | 16132.604 | 223.066 | 32.810 |

这里2P1D按实例计，节点分配实际为2个P节点、2个D节点。它们不是搜索全部并行策略后的
全局最优结论，也不是实测最优结论。两组首选YAML重新predict后，完成请求数、总时长、
TTFT、TPOT、吞吐均与搜索完全一致。

在已更新的运行包中，可从eager搜索示例扩展D实例为两个节点：

```bash
AISIMULATE_910C_MODEL="$model_metadata" bash scripts/run-ascend910c.sh recommend \
  --stack engine --config examples/cli/ascend910c-glm52-pd-eager-search.yaml \
  --set "engine.model=$model_metadata" \
  --set engine.workers.decode.parallelism.attention_data=4 \
  --set engine.workers.decode.parallelism.moe_expert=32 \
  --set traffic.load.concurrency=32 --set traffic.stop.requests=32 \
  --output-dir aisimulate-output/eager-ep32-search
```

1～32 token曲线扩展解决了此前已发现的EP32 eager覆盖错误，未解决非均匀专家路由、
多路PD通信竞争、chunk KV发送与计算的重叠、host开销等误差来源。
当前真实部署的完整权重路径在node19/20均不可用，匹配的关闭graph/overlap实测仍需补齐。
因此准确率尚未验收，旧DOCX的误差数字也没有被本轮功能测试取代。

本机镜像现已安装提交`0d3ce2e0`的数据和代码。安装后重复上述完整1024输出的两组搜索，
每组两个候选的完成请求数、总时长、TTFT、TPOT、吞吐均与源码运行结果完全一致。
实际镜像、wheel和算子表身份保存在本节目录的`runtime-publication.json`。

## 19. PD 传输完成后的 P 端资源释放（2026-10-30）

固定版本的 SGLang `process_disagg_prefill_inflight_queue` 在 P 端轮询到
`KVPoll.Success` 后就释放源 KV；D 端在自己的调度边界处理接收完成，两者独立。
原模拟器等待 D 端激活成功才允许释放 P 端，因此 D 正在计算时会额外占用 P 端容量。
关闭 overlap 调度不会消除这个时序差异。

现已让 SGLang 的目的端激活和源端释放独立推进；两边操作仍分别等待本端调度边界。
P 端路由中的请求计数在源释放时同步结束，整个交接仍需两端完成才结束。
取消、重复确认和另一端尚未完成的情况保留独立的清理状态。
VLLM/TRT 的既有源端优先交接次序不受这项 SGLang 修复影响。

确定性用例设置 P 每轮1ms、D 每轮10ms，同时提交两个请求：第二个请求的 P KV 在2ms
释放，D 要到11ms才激活。旧实现把 P 释放推迟到11ms。
Rust 核心测试2083通过、1忽略；固定源码的7个轮询用例验证成功、等待和混合状态，
并检查再次轮询不会重复释放。证据位于
`production/scheduler-audit-node20-20261030/pd-release-validation.json`。
这是调度因果顺序验证，不能视作整机准确性验收；运行包将在本轮调度修复完成后更新。

## 20. PD 首 token 与 decode 轮数（2026-10-30）

固定运行时在接收元数据时已保存 P 端采样出的首 token。D 端把请求加入 prebuilt batch
后，先执行 `process_batch_result_prebuilt` 返回这个 token，再执行
`update_running_batch` 准备下一轮 decode。v0.5.20 与 v0.5.22 的这段选批函数 AST 一致；
结果处理函数只在释放 KV 时是否显式传入 `checkpoint=True` 上有差异。

此前模拟器虽然接收了提示词 KV，却从零输出开始执行 D forward，导致首 token 多等一轮
decode，1024 输出也多计算一轮。现已在调度轮次开始时报告 prebuilt 输出，普通 decode
输出仍在计算结束时报告；输出1个 token 的请求可在 prebuilt 准入时直接完成。
一般输出 N 个 token 需要 N-1 轮普通 decode，不包括另行配置的投机解码等情况。

这项修复同时处理两点：P 已生成的首 token 尚未拥有计算过的 KV，不能提前增加 KV 用量；
第一轮 D forward 的上下文应包含提示词与这个首 token。对应的 KV 页面增长和最终缓存
只按已物化 token 计算。prebuilt 输出也不计入 decode acceptance 的 forward 次数。

确定性验证以 P=1ms、D=10ms 为例：输出1/2/4个 token 的 TTFT 均为1ms，完成时间分别为
1/11/31ms。另一个 DP rank 正在执行 decode 时，当前 rank 的单 token 完成仍发生在
1ms，不能被推迟到11ms。D 的 running batch 已满时仍需先等到准入空位。
24个原始源码函数用例覆盖单/多请求、已有 running batch、是否启用 radix cache 和短输出。
源码 oracle 与版本比较保存在 `production/scheduler-audit-node20-20261030/pd-prebuilt-*.json`。

这一修正消除了确定的一轮时序偏差，不代表能解释此前 DOCX 中全部 TTFT/TPOT 差距。
chunk KV 传输、多个发送队列间的竞争、host 开销以及同配置实测仍需继续对齐。

重复第18节相同并行布局、相同有限候选集的完整1024输出搜索，并重放保存的首选配置：

| 输入/输出 | 并发/请求数 | 首选 | 修复前TTFT(ms) | 修复后TTFT(ms) | 修复后TPOT(ms) | 输出tok/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 8192/1024 | 32/32 | 2P1D | 11263.694 | 11040.638 | 223.100 | 132.205 |
| 32768/1024 | 8/8 | 2P1D | 16132.604 | 15909.741 | 223.066 | 32.840 |

两个搜索均2/2候选可行，保存配置的重放与搜索主要指标一致。这里每个D实例仍占两个节点，
2P1D实际使用2个P节点和2个D节点。TTFT减少约223ms，TPOT基本不变；不能据此声称原有
实测差距已经解决，更不能把两候选的首选当作全部并行策略的最优解。

最终原生核心测试2086通过、1忽略，两套GPU/模型parity共390通过，原生库SHA为
`a53db3bb92d56c7128e9cde7d075cbefae7ccc0c2a29a4a8d460e177af5c6def`。
GLM与公共API测试140通过、1失败：既有签名断言漏写`op_level_options`，已核对旧运行包
与当前代码的签名完全一致，本阶段未改Python公共API。保留该失败记录，不把测试全绿
作为结论。验证摘要与完整日志归档分别为`pd-prebuilt-validation.json`、`pd-prebuilt-evidence.tar.gz`。

本阶段将公开运行镜像更新至源码`d28b47d1`，包含第19、20节两项修复；安装后的8K/1K、
32K/1K搜索均为2/2候选可行，各候选主要指标与源码完全一致。镜像ID为
`sha256:c53639a123c74cdb63491280cf6189bb59340efd5d28537c80facfd119a998e5`。
运行包身份、10张算子表SHA和安装验证见本节目录的`pd-boundary-runtime-publication.json`；
之前的镜像和二进制已保留，当前仍不声明完成实测准确性验收。

## 21. 分块 prefill 与 KV 传输重叠（2026-10-30）

固定版本 `baa9befa381511cd4d56f83980951e5a45fffc6d` 的
`process_prefill_chunk` 表明：即使用 `--disable-overlap-schedule` 关闭
调度重叠，完成的 prefill chunk 仍会调用 `send_kv_chunk`，随后可以继续计算下一个
chunk。因此，关闭 overlap 不等于等待整个 prompt 算完后才开始传 KV。

原实现直到整个 prompt 结束才提交一次传输。现在每轮 forward 完成后发布可发送的
KV 范围，同一请求的传输按 FIFO 排队，与后续 prefill 计算共用模拟时间轴。
慢链路保留每个 chunk 的边界；每段分别查询原有性能曲线，不把多段合成一个大查询，
也不通过固定折扣系数模拟重叠。源端 KV 的释放和 D 端激活仍需等待最终一段传完。

这次同时分开了三个容易混淆的长度：逻辑 prompt 长度、D 端已缓存的前缀长度、
需要复制的物理 KV 页容量。例如 257-token prompt、128-token 页、D 命中前128token，
剩余有效内容为129token，但复制会触及两页、共256个token槽位。不能用
`257 - 256` 反推缓存前缀。现在由真实的缓存预留结果直接提供前缀长度，最后一段
保留页尾填充的传输量；原有整请求传输的计量和越界报错继续保留。

以下是用于验证事件时序的人工测试，**不是 GLM 实测**：3个128-token chunk，
每个 prefill forward 固定1ms，每个请求复制严格按序执行。

| 每段复制耗时 | 原来的最终 KV 到达时间 | 现在的最终 KV 到达时间 |
| --- | --- | --- |
| 0.5ms | 4.5ms | 3.5ms |
| 1ms | 6ms | 4ms |
| 2ms | 9ms | 7ms |

另外核对了目标 NPU 的 `NPUMLATokenToKVPool`：K、V、index K缓冲区属于同一组
逐页传输，不会在最终 chunk 再重复传一遍 DSA 状态。已有910C性能曲线测量的是这组
缓冲区的复制；无需为这项调度修复重采算子。源码身份和核对位置保存在本节同目录的
`pd-chunks-source-audit-v2.json`，直接执行固定上游函数体的7组CPU检查保存在
`pd-chunks-oracle.json`。

v0.5.20 的对应代码也保留了关闭 overlap 时直接发送 chunk、非最终 chunk 向下取整
到完整页的行为；函数名和请求字段有所变化，没有把两个版本的整个函数宣称为一致。

同一有限候选集的搜索与保存配置回放结果如下。P 为TP8/ADP2/EP16，每实例1节点；
D 为TP8/ADP4/EP32，每实例2节点；预算4节点，只包含1P1D和2P1D两种可行组合。
两种workload仍推荐2P1D，即2个P节点加2个D节点；这不是所有并行策略的全局最优证明。

| workload | 并发/请求数 | 原TTFT | 修复后TTFT | 修复后TPOT | 修复后输出吞吐 |
| --- | --- | --- | --- | --- | --- |
| 8192/1024 | 32/32 | 11040.638ms | 11040.638ms | 223.100ms | 132.205 token/s |
| 32768/1024 | 8/8 | 15909.741ms | 15890.210ms | 223.066ms | 32.842 token/s |

这组32K测试的TTFT只减少约19.531ms，TPOT未变。8K请求没有跨越此配置的prefill
chunk边界，因此结果不变。搜索和保存YAML后直接回放的主要指标完全一致。
原DOCX中的大幅预测误差仍需继续定位，不能用这项小幅改善代替同配置实测验证。

验证结果：Rust核心2094通过、1忽略；GPU/模型冻结回归390通过；GLM与公共API
140通过，排除1个上一阶段已确认的既有签名断言问题。7组上游函数体检查通过，
2278个打包源码/数据文件与工作区一致，10张910C算子表未改动。汇总与日志归档为
`pd-chunks-validation.json`、`pd-chunks-evidence.tar.gz`，包含修复过程中发现并解决的
页容量回归记录，没有通过修改冻结golden掩盖偏差。

本机 `aisimulate-910c:glm52-w8a8` 现已更新到本节代码 `a63007f2`，镜像ID为
`sha256:a15e49a0bc8af505b8d67c3f444c5ca9f6c59d56caffe81bdf4958fb001faff2`。
安装后的8K/1K和32K/1K搜索均为2/2候选可行，各候选主要指标与源码一致。
旧镜像保留为 `aisimulate-910c:before-pd-chunks-20261030`。身份、表SHA和安装证据见
`pd-chunks-runtime-publication.json`、`pd-chunks-runtime-evidence.tar.gz`。

当前边界仍需明确：同一请求的分块顺序已建模，不同请求争用发送队列、链路和内存
带宽的行为尚未校准；bootstrap、最终元数据处理与host开销也没有被这项修复覆盖。
staging网格、DCP打包分段和可选的缓存前缀提前发送不在当前DCP1实测传输契约内。
这能修正一类系统性时序误差，但不能据此宣布TTFT、TPOT或最优PD配比已与实测吻合。

## 22. 普通 prefill 的首 token 与 KV 分配（2026-10-30）

继续检查合并部署路径时，发现普通 prefill 将刚采样的首个输出 token 当成已经完成
forward 的输入，提前给它分配了 KV。目标 v0.5.22 的实际顺序是：prefill 只计算输入
prompt，随后采样并返回首 token；下一轮 decode 消费这个 token 时，才为它分配 KV。
本轮修复沿用这个顺序，首 token 不再等待未来 decode 所需的空闲页。

以页大小4、输入4 token、输出3 token为例：

| 时刻 | 已返回输出数 | 可见序列长度 | 已计算 KV 的 token 数 | 请求的页容量 |
| --- | --- | --- | --- | --- |
| prefill 完成 | 1 | 5 | 4 | 4 |
| 第1轮 decode 完成 | 2 | 6 | 5 | 8 |
| 第2轮 decode 完成 | 3 | 7 | 6 | 完成后释放请求占用，完整前缀页可缓存 |

原逻辑会在第一行就申请第二页；多个请求同时到达页边界时，可能把内存压力、请求
准入或首 token 延迟算错。现在允许本轮已采样的首 token 全部返回，下轮 decode 再
根据真正需要计算的输入检查内存，必要时回退请求。终止输出本身不会产生额外 KV。

用固定镜像中的 `process_batch_result_prefill` 和 `prepare_for_decode` 原函数体
执行12组CPU替身检查，覆盖3/4 token输入、1/2/4 token输出、1/2个请求，确认首
token返回与下一轮分配的先后顺序。源码hash、输入和事件保存在
`benchmarks/evidence/ascend910c-glm52-w8a8/production/scheduler-audit-node20-20261030/prefill-token-kv-oracle.json`。
这些是源码行为检查，不是NPU内存实测或完整服务精度验收。

对应Rust回归覆盖页边界、多请求首token输出、下一轮内存回退及缓存命中。旧测试中
依赖提前分配的压力阈值也按真实KV边界调整：DP准入测试增大第二个prompt，取消测试
先实际执行一轮decode后再检查请求独占页释放。没有修改算子耗时、实测数据或冻结golden。

最终回归：Rust核心2096通过、1忽略；冻结GPU/模型回归390通过；GLM与公共API
140通过，仍排除第21节已说明的1个既有签名断言。12组上游函数检查通过，打包的
2278个源码/数据文件与工作区一致，10张910C算子表保持原hash。修复过程中失败的
旧边界测试日志和最终通过日志一并保留，没有只保存最后的成功结果。

重新执行第21节的8K/1K、32K/1K有限PD搜索，各2个候选均可行，保存配置回放与
搜索指标一致，也与修复前一致。另跑合并部署TP8/ADP2/EP16、8K/1K、并发8、8个
请求，关闭graph和overlap调度：TTFT6118.746ms、TPOT204.942ms、输出吞吐37.966
token/s，与修复前一致。这些例子没有触发新增的内存压力分支，不能把它们当作误差
改善证明；边界差异由上述受控回归覆盖。

本节验证汇总和完整日志归档分别为同目录下的 `prefill-token-kv-validation.json`
和 `prefill-token-kv-evidence.tar.gz`。仍需核对跨请求发送队列争用，并获取同配置的
graph/overlap关闭实测，才能判断端到端预测误差是否收敛。
