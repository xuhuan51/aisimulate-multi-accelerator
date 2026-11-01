# 910C + GLM-5.2 W8A8：离线预测与配置搜索

更新时间：2026-11-02。本文面向当前仓库和本机运行镜像；历史32节点命令仍可复用。
适配原理、源码位置和与GPU版的区别见 [实现说明](910c-glm52-w8a8-implementation.md)。
最多四节点、64-device预算下的两组完整PD搜索已运行，结果见
[8K/1K与32K/1K搜索报告](910c-glm52-w8a8-four-node-pd-results.md)。
恰好两节点、32-device下的8K/1K搜索也已完成，覆盖并发1～128、PD和两种合并布局，
416个候选全部成功，见[两节点并发与配置对比](910c-glm52-w8a8-two-node-8k-results.md)。

| 想完成什么 | 阅读位置 |
|---|---|
| 先运行一次预测，再搜索配置 | 第1节 |
| 修改输入、输出长度，理解shape外推 | 第2节 |
| 修改TP、ADP、EP或副本数 | 第3节和第7节 |
| 预测PD分离或DCP | 第4节 |
| 理解结果文件、失败候选和“最优” | 第8节 |
| 从搜索结果映射回United部署 | 第9节 |
| 排查镜像、元数据、shape和超时问题 | 第10节 |

使用标准 `predict/recommend --stack engine`：910C实测算子表 → GLM完整forward组合 →
KV容量推导 → 请求调度模拟 → PD传输 → 配置搜索。
8K/1K是示例负载，不是接口限制。**新版通过 `shape_policy: anchored` 支持数值shape外推，主要示例已开启。**

2026-10-29补入2,165个测量点，长上下文覆盖到128K附近；现有8K/1K、32K/1K仍有明显整机精度偏差。
新增覆盖、DOCX对比和下一步工作见[shape扩充与实测偏差排查](910c-glm52-w8a8-baseline-alignment.md)。
原始实测表不变；范围内沿用原始结果，范围外使用同身份测量锚点和计算/访存工作量模型。
长度、并发仍须满足模型、KV容量和目标运行时限制，不代表任意硬件配置都能执行。
当前正在对齐0.5.22调度和执行模式；完整forward、TTFT、TPOT仍需匹配部署的实测验收。

## 1. 在当前节点运行

2026-11-02的精度修复已另建本机镜像，原默认镜像保留用于对照。需要复现最新
同配置结果时，先使用下面这组已验证配置；其中包含HTTP PD网关等待P响应、
tokenization时序，以及分别校准的TP16/TP32 eager forward开销和真实活跃DP数量。

```bash
cd /home/youxingye/youxingye_data/aisimulate-910c

sudo env AISIMULATE_910C_IMAGE=aisimulate-910c:active-rank-repair-20261102 \
  AISIMULATE_910C_MODEL=/data_lib/data/models/glm-q \
  bash scripts/run-ascend910c.sh predict --stack engine \
  --config benchmarks/evidence/ascend910c-glm52-w8a8/production/prediction-repair-20261102/active-rank-configs/recommended-2p1d-dtp32-8k1k-c8.yaml \
  --capture-per-request --output-dir aisimulate-output/current-repair-c8
```

该配置占4台机器：2个独立P各TP16/ADP4/EP16，1个D跨2台TP32/ADP32/EP32；
8192/1024，C8/N16。同目录还有1P3D、2P2D、3P1D的C8对照，以及
`paired-1p1d-8k1k-c4.yaml`双节点验证配置。修改并发时同时指定
`--set traffic.load.concurrency=32 --set traffic.stop.requests=64`，并使用新的输出目录。

`engine.tokenization`现在可作为固定参数进入`recommend`，保存的候选YAML也保留它；
`engine.pd_response: sglang_gateway`使客户端首字等待P响应，默认`direct`则直接观察D输出。
forward残差只用于对应的P/TP16-D/TP32-D拓扑，不应把固定TP16残差用于其他ADP/EP搜索候选。
这些校准目前限于固定v0.5.22/CANN9.1镜像、graph/调度overlap/投机解码关闭、相同模型。
完整对照和仍未解决的低并发TTFT偏差见
[活跃DP修复结果](../benchmarks/evidence/ascend910c-glm52-w8a8/production/prediction-repair-20261102/active-rank-serving-comparison.md)。

新增`decode_per_extra_active_rank_ms`按真实非空attention-DP rank计数，padding出来的
空rank不收费。原C1基值保持不变；两个额外逐轮工况用于拟合该项，原23个端到端点
未参与拟合。目前23点TPOT最大绝对误差3.65%、吞吐最大5.20%，TTFT仍有8点超出15%。
已测到首次4K/8K KV分页分配的Triton编译约2秒，后续allocator约0.5ms；该冷状态
尚未进入模拟，两个warmup不能保证所有副本/rank/shape预热。不能据此承诺低并发冷请求的TTFT SLO。
前一版`request-timing-configs/`和`aisimulate-910c:request-timing-repair-20261102`保留用于对照。

以下保留通用适配入口和原始镜像用法，供未启用上述校准的基线对照：

以下命令都在仓库根目录运行，需要本机Docker及已构建的
`aisimulate-910c:glm52-w8a8` 镜像。使用新的输出目录保存每次实验。
只从Git拉取仓库不会下载这个本地Docker镜像；镜像重建见第6节。

```bash
cd /home/youxingye/youxingye_data/aisimulate-910c

sudo bash scripts/run-ascend910c.sh predict \
  --stack engine \
  --config examples/cli/ascend910c-glm52-8k1k.yaml \
  --output-dir aisimulate-output/910c-united

sudo bash scripts/run-ascend910c.sh recommend \
  --stack engine \
  --config examples/cli/ascend910c-glm52-auto-search-8k1k.yaml \
  --output-dir aisimulate-output/910c-auto-search
```

第二个命令自动选择并行结构、合并/PD部署、P:D副本数和并发量。
默认预算64device、并发8/16/32、64条8K/1K请求，共1800个候选。
最优只针对当前负载、候选范围和目标，不代表所有配置的全局最优或实测服务排名。
目标是每device输出吞吐，默认没有延迟SLA；选型还需检查TTFT、TPOT。
1800是该8K/1K示例的候选数量；改变负载、预算或并行范围后，应重新检查搜索空间与trial上限。
快速尝试可加 `--set optimizer.max_trials=12`，但只探索部分候选；
也可用 `--set optimizer.algorithm=bayesian`。

若实测将关闭overlap调度和graph，使用新增的eager示例，并让实际P、D入口和各副本内部都使用round-robin。
它显式设置固定输出准入、两级PD路由、无prefix cache及0.5.22调度预算。
已有graph示例保留。关闭graph不代表预测误差已经校准，详见
[对齐说明第9～13节](910c-glm52-w8a8-baseline-alignment.md)。
eager仍按attention TP补齐token；例如TP8/ADP2的单请求decode使用8行矩阵输入，
但只有1条有效KV序列。算子级预测已区分这两种shape；更新时需要同时重建原生库和运行镜像。

```bash
model_metadata="$PWD/benchmarks/evidence/ascend910c-glm52-w8a8/production/deepep-contract-node19-20261024/model-metadata"
AISIMULATE_910C_MODEL="$model_metadata" bash scripts/run-ascend910c.sh predict \
  --stack engine --config examples/cli/ascend910c-glm52-pd-eager-8k1k.yaml \
  --set "engine.model=$model_metadata" \
  --output-dir aisimulate-output/eager-pd-check

AISIMULATE_910C_MODEL="$model_metadata" bash scripts/run-ascend910c.sh recommend \
  --stack engine --config examples/cli/ascend910c-glm52-pd-eager-search.yaml \
  --set "engine.model=$model_metadata" \
  --set traffic.source.input_tokens=32768 \
  --output-dir aisimulate-output/eager-pd-32k-search
```

第二条命令固定TP8/ADP2/EP16，搜索最多4节点内的6种P:D配比和并发8/16/32；
先用这组已具备eager覆盖的配置核验调度。8K搜索改为8192输入即可。
扩大TP/ADP/EP候选仍使用第3节的并行字段，缺失的eager shape会报错；
EP32 balanced eager通信的1～32 token查询范围已补齐，见对齐说明第18节；
DCP eager仍未采集，不应把graph记录混入这些候选。
元数据快照仅足够离线预测；真实部署仍需要完整模型权重。

2026-10-31已重建本机 `aisimulate-910c:glm52-w8a8`，打包源码为 `db5f2e72`、EngineSpec schema31，
包括真实DP batch形状、新增3354条eager注意力数据、EP32 eager通信扩展，以及PD源KV释放、
首token时序、分块prefill与KV传输重叠、合并部署首token的KV分配时序修复。
本版还修正decode内存回退顺序、最后请求OOM退出，并在搜索排名前排除有拒绝/失败请求的候选。
本版进一步修正PD的D端KV回退恢复：保留已生成token、按FIFO恢复、不再在D端重复prefill，
未恢复完时优先处理回退请求。主机KV复制耗时仍需实测校准。
安装后已验证原6个失败候选、8K/1K与32K/1K原最佳配置、8K/1K合并部署及2个OOM接口案例；
主要指标与源码运行结果完全一致。
上一版镜像保留为 `aisimulate-910c:before-pd-restore-20261031`；仅回放旧结果时使用。
运行包、算子表、镜像SHA及回归记录见
`production/scheduler-audit-node20-20261031/pd-retraction-runtime-publication.json`。
本机没有默认权重路径时，使用上面的 `model_metadata` 参数即可进行离线预测。
当前node20的完整模型路径为 `/data_lib/data/models/glm-q`，也可同时将
`AISIMULATE_910C_MODEL` 和 `engine.model` 指向该路径；离线预测只读取模型元数据。

只想检查环境和命令是否可用，可以先运行65输入、32输出的两条请求：

```bash
sudo bash scripts/run-ascend910c.sh predict \
  --stack engine \
  --config examples/cli/ascend910c-glm52-8k1k.yaml \
  --set traffic.source.input_tokens=65 \
  --set traffic.source.output_tokens=32 \
  --set traffic.stop.requests=2 \
  --set engine.context_length=97 \
  --capture-per-request \
  --output-dir aisimulate-output/910c-quickcheck
```

结果是 `recommendation.json`、`recommendation.csv`，排序后的完整配置位于
`recommendations/0001.yaml` 等。复算第一名：

```bash
sudo bash scripts/run-ascend910c.sh predict \
  --stack engine \
  --config aisimulate-output/910c-auto-search/recommendations/0001.yaml \
  --capture-per-request \
  --output-dir aisimulate-output/910c-best
```

脚本使用本地 `aisimulate-910c:glm52-w8a8` 镜像，禁用网络，只读取config和量化元数据。
不加载完整权重、不启动SGLang、不申请NPU，预测时不要求19/31节点在线。
默认16核CPU、32GiB主机内存，输出归调用sudo的用户所有；可通过
`AISIMULATE_910C_CPUS`、`AISIMULATE_910C_CPUSET`、`AISIMULATE_910C_MEMORY` 调整资源。
模型默认 `/mnt/raid/user/data/models/glm-q`。换目录需同时设置
`AISIMULATE_910C_MODEL` 和 `--set engine.model=新路径`。

## 2. 修改输入输出长度

例如搜索4K输入、2K输出：

```bash
sudo bash scripts/run-ascend910c.sh recommend \
  --stack engine \
  --config examples/cli/ascend910c-glm52-auto-search-8k1k.yaml \
  --set traffic.source.input_tokens=4096 \
  --set traffic.source.output_tokens=2048 \
  --set engine.context_length=6144 \
  --output-dir aisimulate-output/910c-search-4k2k
```

`context_length` 是输入加输出的上下文上限。修改负载时也要修改它，避免保留示例9216而截断输出。
同样的参数可用于predict。输入输出不必等于采集点：模型推导每轮batch的算子shape后插值。
eager prefill会在已测batch之间插值，遇到不齐整的序列网格时通过同身份锚点补齐。
范围外算子结果标记为 `estimated`，所选策略写入保存配置和PD元数据。
需要只接受原测量范围时，加 `--set engine.estimator_config.op_level.ascend.shape_policy=measured`。
未设置该字段的旧配置仍默认 `measured`；新版四个主要示例明确选择 `anchored`。

本轮已经验证合并部署、PD、DCP三种模式各8组负载，共24组：
1/32、65/32、4K/2K、16K/1K、32K/1K、64K/1K、128K/1K、8K/16K，
每组4条请求，输入输出token总数均检查。64K/128K使用全局32的配置满足KV容量；
不能在单节点容量不足时仅增加 `context_length`。

4K/2K、65/32和32K/1K分别通过64候选联合搜索，共192个成功候选。
安装后的新版镜像还通过了短输入、128K/1K PD、32K/1K DCP和4K/2K的16候选搜索，
搜索第一名重新predict的64条请求与搜索指标一致。
64K/1K在并发8/16/32的扩展搜索中出现候选CPU耗时超过120秒，本轮诊断已中断并保留日志；
其根因尚未定位，不能将这个搜索视为已通过。128K高并发搜索没有完成验证。
这些长度在并发2的指定合并/PD/DCP配置已通过，暂不能把低并发成功外推成高并发调度也已完全验证。

外推把测量延迟分为固定开销和随shape变化的部分，概念上为：

```text
预测耗时 = 固定开销 + (锚点耗时 - 固定开销) × 目标工作量 / 锚点工作量
```

固定开销从对应原测量曲线估计，不随长度成倍放大；attention使用曲线最小延迟，
共用轴外推helper使用最小坐标延迟并与锚点取较小值，细节见实现说明。
Linear/BMM/MoE依据计算量和权重/激活访存；Indexer考虑完整因果Q×K工作量，
稀疏注意力考虑top2048上限；DCP另计缓存写入和通信；PD按相同rank映射的KV页数增长。
不跨dtype、硬件、内核、MoE布局或graph bucket套用数据。DeepEP已知执行边界继续保留。

下面保留旧 `measured` 策略的边界对照，说明本次补齐的内容。

已用TP8/ADP2/EP16、4K prefill chunk、并发2、每组4条请求验证：

| 输入/输出 | 合并部署 | PD部署 |
|---|---|---|
| 1024 / 128 | 通过 | 通过 |
| 2048 / 512 | 通过 | 通过 |
| 4096 / 2048 | 通过 | 通过 |
| 8192 / 4096 | 通过 | 通过 |
| 12288 / 1024 | 通过 | 通过 |
| 16384 / 1024 | 缺少prefix=12288的数据 | 同样缺少 |
| 32768 / 1024 | 缺少prefix=12288的数据 | 同样缺少 |
| 65 / 32 | decode序列低于实测下界128 | 同样缺少 |

旧版安装包的4K/2K试搜得到15个可行、1个数据覆盖失败；失败配置为
TP16/ADP1、MoE-TP16/EP1、并发32。它需要batch3、每请求2731个prefill token，
而batch4插值端点的采样上界是2048。新增batch插值仍要求两个端点都覆盖序列长度，
`measured` 因此拒绝该候选，`anchored` 现在可以为这些缺失数值坐标估计耗时。
原测量边界为prefill prefix锚点0/8192、decode序列128～16384；batch、head数、图模式和MoE分布还有各自范围。
12K输入分成三个4K chunk时，prefix为0/4096/8192；16K的第四个chunk需要prefix=12288，超出已有表。
输出变长也会逐轮增加decode的KV长度，不能只把1K输出时间乘一个倍数。

PD多数曲线覆盖128～16384token；4→4、8→8碎片页上界为8192，
16→8、16→16碎片页范围1024～8192，16→16连续页下界1024。
因此输入长度、页布局和P/D映射共同决定哪些结果是范围内插值、哪些需要外推。

GPU部分算子同样使用解析计算/访存模型和实测锚点进行范围外外推。
**不需要为每一种输入/输出组合重新测整个模型**。本轮另采集66个验证点和42个后续复核点，
检验NPU执行与外推趋势；这些点保持独立，不作为新测量行偷偷填入原表。
后续42点复核中，四组误差中位数约1.7%～7.3%，最大约24.9%；
这是算子点对照，不是forward、TTFT或TPOT的误差承诺。
未知算子身份或不支持的graph bucket仍报错，不使用A100数据或固定8K耗时代替。

## 3. 支持的并行结构

| 模式 | Attention TP | 全局FFN宽度 | Attention DP | MoE参数 |
|---|---|---|---|---|
| DeepEP | 1/2/4/8/16 | 16或32 | 全局宽度 / TP | MoE-TP1、EP16/32 |
| 张量MoE | 1/2/4/8/16 | 16或32 | 全局宽度 / TP | MoE-TP16/32、EP1 |

共20种结构声明。8K/1K、内存比例0.90时留下16种；全局16下TP1/TP2四种结构不满足当前模型和上下文容量。
修改上下文或内存比例后会重新检查可行性。

模拟器 `tensor` 是attention组内TP；SGLang的 `--tp-size` 是全局TP，等于
`tensor × attention_data`，`--attn-dp-size` 对应 `attention_data`。
例如United全局TP16、ADP2、EP16：

```yaml
parallelism:
  tensor: 8
  attention_data: 2
  moe_tensor: 1
  moe_expert: 16
  pipeline: 1
  replicas: 1
```

每副本占 `tensor × attention_data × pipeline` 个device，EP不额外相乘。
DeepEP对应 `--moe-a2a-backend deepep`；张量MoE对应 `none --ep-size 1`。
每节点16个运行时device。`max_candidate_gpus`、`used_gpus`、`throughput_per_gpu` 保留原字段名，
选ascend910c后按NPU device计数。P/D独立选结构，并按预算生成1P:3D、3P:1D等副本组合。
超过两节点的模拟依赖同硬件、同软件、同互联条件的数据复用；共享链路争用尚未校准。

## 4. PD与DCP

```bash
sudo bash scripts/run-ascend910c.sh predict \
  --stack engine \
  --config examples/cli/ascend910c-glm52-pd-8k1k.yaml \
  --capture-per-request \
  --output-dir aisimulate-output/910c-pd

sudo bash scripts/run-ascend910c.sh predict \
  --stack engine \
  --config examples/cli/ascend910c-glm52-dcp2-8k1k.yaml \
  --output-dir aisimulate-output/910c-dcp2
```

PD计时来自真实SGLang AscendKVManager、MemFabric1.2.1/SDMA，包含rank映射和234组分层KV缓冲区搬运。
KV为BF16，每个完整KV副本每token占109824字节；8K共899678208字节。
W8A8不意味着KV、归一化、路由器和通信全部INT8。

`engine.estimator_config.op_level.ascend.kv_transfer_page_layout` 可选 `contiguous` 或 `fragmented`。
`engine.kv_transfer` 使用 `{type: measured, timing_mode: full_prompt}`；
也可用 `destination_missing` 只计目标缺失token。

DCP2/4/8/16必须整除attention TP。latent/RoPE KV分片，indexer KV仍完整保留；
物理页128token，模拟器 `kv_cache.block_size` 必须是 `128 × DCP`。
38种DCP配置已通过8K/1K预测，包括TP2/ADP16的两种MoE路径。
DCP用于显式合并部署predict；本仓库GPU的recommend也不搜索CP/DCP。
固定NPU运行时不支持PD+DCP，该组合仍明确拒绝。

GPU Replay自身限制PP1；NPU CP2存在缺失方法的执行错误；
混合MoE-TP/EP路径存在实际分组维度错误，DeepEP又强制EP等于全局TP。
部分PP2真实部署可启动，并不代表AISimulate已经具备PP请求模拟。
执行证据与源码位置见 [并行扩展进展](910c-glm52-w8a8-parallel-expansion.md)。

## 5. 数据与解释范围

| 数据族 | 当前源码树已发布记录数（2026-10-30） |
|---|---:|
| Linear | 2888 |
| 向量/归一化/缓存操作 | 1410 |
| Attention/Indexer | 10093 |
| MLA BMM | 434 |
| DeepEP专家计算链 | 225 |
| 张量MoE计算链 | 126 |
| DeepEP通信 | 150 |
| HCCL | 449 |
| PD传输 | 227，组成50条曲线 |
| DCP存储/注意力/合并链 | 2994 |

记录数不是独立内核数；MoE链中8条是已验证的数学等价布局别名。
失败和CV超限记录保留在evidence中，未混入性能表。
发布身份绑定硬件、软件、shape、dtype、计时协议和源码；节点数据复用必须满足相同条件。

Rust模型按shape查表，组合78层：3层dense、75层MoE，21层完整indexer、57层复用。
调度器依据chunk、并发和KV容量决定batch，prefill后逐轮生成输出；
PD通过目标KV预留、传输和源端释放衔接decode，再汇总TTFT、TPOT、吞吐和排队。

算子数值正确、表读取正确、请求完成属于功能证据，完整服务误差尚未校准。
当前使用均衡attention-DP和专家路由，PD bootstrap、逐层流式传输和多请求争用尚未校准。
功耗数据未采集，能耗输出明确不可用。
基线固定为SGLang0.5.22（提交 `baa9befa381511cd4d56f83980951e5a45fffc6d`）、
CANN/HCCL9.1.0.B243、驱动26.1.1。升级需核对执行边界并采集变化部分。

## 6. 文件与重建

- 硬件：`python/aisimulate/src/aisimulate_core/systems/ascend910c.yaml`
- 数据：`python/aisimulate/src/aisimulate_core/systems/data/ascend910c/`
- 组合模型：`python/aisimulate/src/aisimulate_core/sdk/models/ascend_glm52.py`
- 原始证据：`benchmarks/evidence/ascend910c-glm52-w8a8/production/`
- 阶段历史：[生产适配进展](910c-glm52-w8a8-production-progress.md)

`build/ascend910c-runtime/` 为Git忽略的构建产物，含统一wheel、原生库、SHA256身份与离线依赖。
保留它可重建镜像：`sudo bash scripts/build-ascend910c-runtime.sh`。
修改Python、Rust或性能表后必须重建wheel和镜像，挂载源码不会替换已安装的包。

在与基础镜像匹配的ARM64/Linux、Python3.11～3.13和支持edition2024的Rust环境中，
安装 `maturin==1.15.0`，在 `python/aisimulate` 目录执行：

```bash
python3 -m maturin build --release --auditwheel skip \
  --out ../../build/ascend910c-runtime
python3 -m pip download --dest ../../build/ascend910c-runtime \
  ../../build/ascend910c-runtime/aisimulate-*.whl
```

提取wheel中的 `aisimulate/_runtime.abi3.so`，更新 `native-identity.json` 的源码提交、库与wheel的SHA256、
release/extension-module配置，再执行构建脚本。延迟SLA与搜索字段见 `docs/sweeper/`。

## 7. 最常修改的配置

表中的路径可以用于 `--set 路径=值`；`workers.<角色>` 替换为
`workers.aggregated`、`workers.prefill` 或 `workers.decode`。
下面只解释已经存在的字段，完整规则见 [部署配置](../docs/replay/engine/README.md)、
[worker配置](../docs/replay/engine/workers.md) 和 [搜索空间](../docs/sweeper/search-space.md)。

| 配置路径 | 含义与使用方法 |
|---|---|
| `traffic.source.input_tokens` / `output_tokens` | 每条合成请求的输入/输出token数；此处8K是8192，不是8000 |
| `traffic.stop.requests` | 本次总请求数；不是batch大小 |
| `traffic.load.concurrency` | 闭环并发；predict填整数，recommend可以使用 `{choices: [8, 16, 32]}` |
| `engine.context_length` | 数值上限应覆盖输入加输出；显式设置更便于复现实验 |
| `engine.mode` | 合并部署 `aggregated` 或PD分离 `disaggregated`；搜索示例提供两者的choices |
| `engine.workers.<角色>.scheduler.max_batched_tokens` | 通用调度预算；当前SGLang replay不使用它改变chunk，而使用默认全局8192再除以ADP；不是请求总长度 |
| `engine.workers.decode.scheduler.decode_reserved_tokens` | 可选SGLang PD接收增长预算；默认0保持原行为，512对齐本次固定运行时的预算规则；不会增加KV传输量 |
| `engine.workers.<角色>.scheduler.max_sequences` | 调度器同时容纳的序列上限；不等于客户端总并发 |
| `engine.workers.<角色>.parallelism` | 当前角色的TP/ADP/MoE布局及副本数；各字段需一起满足第3节关系 |
| `engine.workers.<角色>.parallelism.replicas` | 该角色独立worker的副本数；PD的P:D比例由两侧副本数决定 |
| `engine.workers.<角色>.kv_cache.capacity.memory_fraction` | 用于容量推导的静态显存比例；不能通过随意调大它消除真实内存不足 |
| `engine.estimator_config.op_level.ascend.shape_policy` | `measured`仅接受已有数值覆盖；`anchored`允许同身份shape外推 |
| `optimization.constraints.max_candidate_gpus` | 搜索中一个部署的总device预算；910C下按NPU运行时device计数 |
| `optimization.target` | 如 `throughput_per_gpu`、`throughput`、`ttft`；决定怎样排序，不会自动加延迟限制 |
| `optimizer.max_trials` | 搜索尝试次数上限；较小值只能代表部分候选 |
| `optimizer.parallelism` | 主机上同时评估几个候选，与模型TP、EP无关 |
| `optimizer.candidate_timeout_seconds` | 单候选主机评估时间预算；超时不是预测的TTFT |

例如，固定United结构、并发8、总请求16；该ADP2配置的SGLang默认chunk为每rank4096：

```bash
sudo bash scripts/run-ascend910c.sh predict \
  --stack engine \
  --config examples/cli/ascend910c-glm52-8k1k.yaml \
  --set traffic.load.concurrency=8 \
  --set traffic.stop.requests=16 \
  --set engine.context_length=9216 \
  --capture-per-request \
  --output-dir aisimulate-output/910c-united-c8-chunk4k
```

只搜索PD、总预算32个device时，可以在自动搜索命令中增加：

```text
--set engine.mode=disaggregated
--set engine.workers.aggregated=null
--set optimization.constraints.max_candidate_gpus=32
```

这三个参数应作为同一条命令的参数使用。自动搜索示例原本包含合并角色，
固定为PD模式时还要将该角色置为null，否则配置校验会拒绝。
当前每个worker至少16个device，
32预算下的PD通常只能容纳1P:1D；64预算才有容纳1P:3D等布局的空间。
如果P或D选择全局32的结构，应按该worker的实际device数重新计费。

## 8. 怎样阅读预测和搜索结果

### 8.1 predict结果

| 文件或字段 | 解读 |
|---|---|
| `prediction.json` | 整个模拟负载的汇总指标和来源信息 |
| `completed_requests` / `num_requests` | 先检查请求是否完整结束 |
| `total_input_tokens` / `total_output_tokens` | 固定长度且无截断时，应等于请求数乘对应长度 |
| `mean_ttft_ms` / `p99_ttft_ms` | 从到达到首token的时间，包括排队和调度等影响 |
| `mean_tpot_ms` / `p99_tpot_ms` | 请求的平均后续输出token延迟及其分布；不是单个算子耗时 |
| `output_throughput_tok_s` | 整个部署的输出token吞吐；包含该负载的启动与排空影响 |
| `duration_ms` | 模拟出的服务时间跨度 |
| `wall_time_ms` | Replay运行消耗的主机时间，不含全部容器/包启动成本；不要用来当作NPU服务时延 |
| `prediction_provenance` | 所用性能模型、数据和预测来源，供复现与排查 |
| `requests.jsonl` | 加 `--capture-per-request` 后保存逐请求记录，包括到达、首/末token、终态与路由信息 |
| `execution-events.jsonl` / `resource-runtime.json` | 执行与资源诊断文件；不是实机NPU profiler记录 |

例如65输入、32输出、2条请求的快速检查，预期完成2条请求，
输入总数130，输出总数64。结果中带 `gpu` 的历史字段在本硬件下仍表示NPU device。
功耗尚无已采集的数据，不能根据空值、零值或未覆盖的字段得出能耗结论。

### 8.2 recommend结果

先看 `recommendation.json` 的 `counts`，再看排名：

| 状态 | 含义 |
|---|---|
| `feasible` | 模拟成功，满足当前启用的约束 |
| `infeasible` | 容量、预算或已启用的约束不满足 |
| `failed` | 配置生成、查表或模拟执行发生错误，需要看原因 |
| `timed_out` | 候选主机评估超时，不能据此判断其真实NPU性能差 |
| `unsupported` | 后端或拓扑能力不支持 |
| `resource_limited` | 主机资源不足，未能完成评估 |

`recommendation.csv` 便于表格查看；`recommendations/0001.yaml` 是保存出的第一名完整配置。
JSON中 `views.top_n` 给出排名的candidate ID，不能假设 `candidates[0]` 就是第一名。
固定硬件数量的部署里，`throughput_per_gpu` 等于输出吞吐除以占用device数。
使用动态资源伸缩时，通用评分按时间平均设备数计算，详见
[目标与约束](../docs/sweeper/optimization-goals.md)。

默认目标没有延迟SLA，所以吞吐第一名可能TTFT较大。应把它作为给定负载、
候选范围和预测假设下的推荐，复算后再结合实际服务要求选型。
本轮完整8K/1K搜索1800个候选全通过；64K高并发搜索仍有第2节记录的未解决超时，
不能把部分候选的排名描述成该搜索已经完整完成。

## 9. 搜索结果怎样对应United部署

首先打开具体的 `recommendations/0001.yaml`，按角色读取下面的字段：

| 模拟器配置 | 当前固定SGLang-NPU运行时的对应关系 |
|---|---|
| `tensor × attention_data` | 全局 `--tp-size` |
| `attention_data` | `--attn-dp-size`；大于1时还需与原United脚本的DP-attention开关保持一致 |
| `moe_expert` | `--ep-size`；DeepEP路径要求它等于全局TP |
| `moe_tensor=1, moe_expert=16/32` | DeepEP路径，`--moe-a2a-backend deepep` |
| `moe_tensor=16/32, moe_expert=1` | 张量MoE路径，`--moe-a2a-backend none` |
| P/D的 `replicas` | P/D独立worker数量；每个worker还可能跨两个节点 |

例如 `tensor=8, attention_data=2, moe_tensor=1, moe_expert=16` 对应
United的全局TP16、ADP2、EP16，而不是SGLang的TP8。
离线结果YAML描述模拟配置，**不是直接可执行的SGLang启动脚本**。
执行时仍需把所选角色、节点、rank、端口、网卡及容器环境写回部署脚本，
并使用相同的量化、融合和通信路径。

模拟器不生成实际文本，也不证明回答质量；它使用请求长度预测时间和资源。
升级SGLang/CANN、改变融合开关、换量化产物后，应核对执行边界，
不能只修改 `backend_version` 字符串继续使用旧表。

## 10. 常见问题

| 现象 | 处理方式 |
|---|---|
| `Runtime image ... is unavailable` | 当前主机缺少本地镜像；第6节说明构建产物和重建条件，Git拉取本身不会安装镜像 |
| `Missing GLM ModelSlim metadata` | 检查模型目录中的 `config.json` 和 `quant_model_description.json`；改路径时同时改脚本挂载和 `engine.model` |
| 更改源码后结果没有变化 | 脚本运行镜像内安装的wheel；需要重新构建wheel和镜像 |
| `outside measured` | 核对是否选了 `anchored`、是否确为同身份的数值越界；DeepEP等仍有实际执行边界 |
| `exceeds graph buckets` | 当前decode图桶最大32，且按attention TP筛选；不能靠shape外推增加图桶 |
| KV容量不足或没有合法候选 | 同时检查上下文、并发、并行结构、静态内存比例及device预算；增加预算或使用可行布局 |
| DCP配置被拒绝 | `decode_context`须整除attention TP，逻辑block为128×DCP，且本适配不支持PD+DCP |
| 输入输出改变后token总数不对 | 先检查 `context_length`、终止条件、请求终态和是否发生截断 |
| `timed_out` / `resource_limited` | 查看候选原因与主机日志；减少搜索评估并行度可缓解主机资源压力，但不能代替定位模拟器问题 |

当前已交付的变长负载结果在本机 `aisimulate-output/shape-delivery-20261025/`。
可随Git查阅的验证证据在
[shape交付记录](../benchmarks/evidence/ascend910c-glm52-w8a8/production/shape-delivery-20261025/summary.json)。
其中的性能数值是预测/算子验证结果，完整服务的TTFT、TPOT误差校准仍未完成。
