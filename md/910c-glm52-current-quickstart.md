# 当前版本使用：910C + GLM-5.2 W8A8 离线预测与PD搜索

更新日期：2026-11-04。本文对应当前代码和本机镜像；新用户从本文开始，完整字段解释见[详细使用说明](910c-glm52-w8a8-usage.md)。

## 1. 先明确预测模式

预测与搜索在CPU上运行，使用已经采集的910C性能数据。无需为每次预测启动SGLang服务或占用NPU显存。

| 模式 | 配置值 | 计时来源 | 当前边界 |
|---|---|---|---|
| K：独立算子 | `isolated_kernels` | Prefill和Decode由独立算子组合 | 候选并行范围较广，但eager运行时耗时覆盖不完整 |
| L：层计时 | `runtime_layers` | Prefill仍用独立算子，纯Decode内部按真实decoder层调用组合 | 仅覆盖global TP16/ADP4/EP16和TP32/ADP32/EP32；具体batch还需测量锚点 |

两种模式均为离线预测，不使用目标workload的实测TTFT/TPOT作为预测输入。L替换其覆盖的内部kernel项，避免重复计时；它不等同于纯独立算子预测。

最新39条历史实测对照中，K完成39条，TTFT/TPOT/吞吐MAPE为26.85%/51.01%/98.58%；L完成36条，为21.57%/6.88%/5.17%。成功子集不同，完整报告另列共同成功子集。当前不能保证任意workload、PD比例或并行方式的误差低于20%，搜索第一名需要实测验证。

结果及完整配置见[两节点、四节点K/L对照](910c-glm52-2node-4node-rerun-20261104.md)。本文的通用示例采用均衡专家工作量；该报告额外使用独立采集的`prefill_expert_workload`，精确复现其数值应使用报告保存的完整配置。

## 2. 在当前节点准备运行环境

需要Docker、当前ARM64离线模拟器镜像，以及模型目录中的`config.json`和`quant_model_description.json`。实际模型是当前78层GLM ModelSlim检查点，不能仅修改模型名称套用其他权重结构。

```bash
cd /home/youxingye/youxingye_data/aisimulate-910c

export AISIMULATE_910C_IMAGE=aisimulate-910c:maxlen-population-cli-20261104
export AISIMULATE_910C_MODEL=/data_lib/data/models/glm-q
export AISIMULATE_910C_CPUS=4
RUN_ID=$(date +%Y%m%d-%H%M%S)

sudo docker image inspect "$AISIMULATE_910C_IMAGE" --format '{{.Id}}'
test -f "$AISIMULATE_910C_MODEL/config.json"
test -f "$AISIMULATE_910C_MODEL/quant_model_description.json"

aisim910c() {
  sudo env \
    AISIMULATE_910C_IMAGE="$AISIMULATE_910C_IMAGE" \
    AISIMULATE_910C_MODEL="$AISIMULATE_910C_MODEL" \
    AISIMULATE_910C_CPUS="$AISIMULATE_910C_CPUS" \
    bash scripts/run-ascend910c.sh "$@"
}

aisim910c --help
```

本机镜像ID为`sha256:fc65d388984af172b461d63f38ddbb265b0b4436df8a1e057d2bc549f7694fe7`。这是AISimulate镜像；实测基线的SGLang0.5.22/CANN9.1镜像是另一套运行环境。

后续命令在同一个终端执行。重复整轮操作时重新生成`RUN_ID`，使用新的输出目录。`AISIMULATE_910C_MODEL`负责挂载目录，命令中的`--set engine.model=...`负责修改配置；两者应一致。

## 3. 单节点预测8K/1K

下面为单节点合并部署，global TP16、Attention DP4、EP16，客户端并发4，总请求8，使用L模式：

```bash
aisim910c predict --stack engine \
  --config examples/cli/ascend910c-glm52-eager-client.yaml \
  --set engine.model="$AISIMULATE_910C_MODEL" \
  --capture-per-request --detail time \
  --output-dir "aisimulate-output/quickstart-$RUN_ID/single-L"
```

只切换到K模式，其他条件保持一致：

```bash
aisim910c predict --stack engine \
  --config examples/cli/ascend910c-glm52-eager-client.yaml \
  --set engine.model="$AISIMULATE_910C_MODEL" \
  --set engine.estimator_config.op_level.ascend.decode_timing_scope=isolated_kernels \
  --capture-per-request --detail time \
  --output-dir "aisimulate-output/quickstart-$RUN_ID/single-K"
```

当前示例关闭graph、调度overlap及投机解码，`nextn=0`，不启用整轮forward残差。`shared_expert_overlap=true`表示模型内部shared expert的多流执行，与调度overlap不同，保留该真实部署设置。

## 4. 更换输入、输出和并发

例如12K输入、4K输出，并发4、总请求8：

```bash
aisim910c predict --stack engine \
  --config examples/cli/ascend910c-glm52-eager-client.yaml \
  --set engine.model="$AISIMULATE_910C_MODEL" \
  --set traffic.source.input_tokens=12288 \
  --set traffic.source.output_tokens=4096 \
  --set traffic.load.concurrency=4 \
  --set traffic.stop.requests=8 \
  --set engine.context_length=16384 \
  --capture-per-request --detail time \
  --output-dir "aisimulate-output/quickstart-$RUN_ID/single-12k4k"
```

8K=8192、1K=1024。`concurrency`为客户端并发，`requests`为本次请求总数，均不同于每轮running batch。输出变长会逐轮增加KV长度，模拟器会重新生成相应shape。

`shape_policy=anchored`允许基于相同算子身份的测量锚点进行插值/外推，结果会标记为estimated。它不允许跨硬件、量化精度或未知执行路径随意套用数据。改为`measured`时，只接受测量覆盖内的查询。

修改长度后需同时检查`context_length`和KV容量。单节点示例固定为254页/rank，PD示例为P135、D253页/rank，每页128token。这些是8K基线的容量：32K/1K单请求至少需要P容纳256页输入、D约264页完整序列，再加运行时预留，不能直接沿用P135/D253。

## 5. 两节点1P1D及固定四节点配置

新增的具体配置[ascend910c-glm52-pd-eager-client.yaml](../examples/cli/ascend910c-glm52-pd-eager-client.yaml)使用1P1D，每个实例占1节点，P/D均为global TP16、ADP4、EP16，8K/1K、C4/N8：

```bash
aisim910c predict --stack engine \
  --config examples/cli/ascend910c-glm52-pd-eager-client.yaml \
  --set engine.model="$AISIMULATE_910C_MODEL" \
  --capture-per-request --detail time \
  --output-dir "aisimulate-output/quickstart-$RUN_ID/pd-1p1d-L"
```

PD配置为两个角色分别设置计时参数；切换Decode到K，需要修改角色内的字段：

```bash
aisim910c predict --stack engine \
  --config examples/cli/ascend910c-glm52-pd-eager-client.yaml \
  --set engine.model="$AISIMULATE_910C_MODEL" \
  --set engine.workers.decode.timing.estimator_config.op_level.ascend.decode_timing_scope=isolated_kernels \
  --capture-per-request --detail time \
  --output-dir "aisimulate-output/quickstart-$RUN_ID/pd-1p1d-K"
```

四节点2P2D、并发16、总请求32：

```bash
aisim910c predict --stack engine \
  --config examples/cli/ascend910c-glm52-pd-eager-client.yaml \
  --set engine.model="$AISIMULATE_910C_MODEL" \
  --set engine.workers.prefill.parallelism.replicas=2 \
  --set engine.workers.decode.parallelism.replicas=2 \
  --set traffic.load.concurrency=16 \
  --set traffic.stop.requests=32 \
  --capture-per-request --detail time \
  --output-dir "aisimulate-output/quickstart-$RUN_ID/pd-2p2d-c16"
```

把P/D副本数改为1/3或3/1，即对应1P3D或3P1D。这里每个实例均为16个device；若一个D实例使用global TP32，它占2节点，因此2P1D也可能占4节点。

## 6. 搜索四节点PD比例和并发

下面固定每个P/D实例为global TP16/ADP4/EP16，搜索1P3D、2P2D、3P1D及并发1/4/8/16/32。目标为输出吞吐，`requests_per_load_unit=2`表示每个候选总请求数为并发的2倍。

```bash
aisim910c recommend --stack engine \
  --config examples/cli/ascend910c-glm52-pd-client-search.yaml \
  --set engine.model="$AISIMULATE_910C_MODEL" \
  --set optimization.constraints.min_candidate_gpus=64 \
  --set optimization.constraints.max_candidate_gpus=64 \
  --output-dir "aisimulate-output/quickstart-$RUN_ID/search-pd-4node"
```

每节点16个运行时device，字段名中的`gpus`在910C下按NPU device计数。上下限都设64表示恰好4节点；只设置`max_candidate_gpus=64`表示最多4节点，可能推荐使用更少节点。两节点预算将上下限都改为32；该实例布局下仅有1P1D。

该搜索只改变PD副本配比和并发。若要搜索更多并行策略，使用下一节的K模式候选搜索；L的层数据目前没有覆盖所有拓扑。

## 7. 独立算子模式搜索并行策略

下面使用默认并行候选集，允许合并部署和PD，重新推导KV容量，以恰好4节点为预算，并显式设置eager、K模式和关闭投机/在线修正：

```bash
aisim910c recommend --stack engine \
  --config examples/cli/ascend910c-glm52-auto-search-8k1k.yaml \
  --set engine.model="$AISIMULATE_910C_MODEL" \
  --set engine.nextn=0 \
  --set engine.estimator_config.correction.enabled=false \
  --set engine.estimator_config.op_level.ascend.decode_execution_mode=eager \
  --set engine.estimator_config.op_level.ascend.decode_timing_scope=isolated_kernels \
  --set optimization.target=throughput \
  --set optimization.constraints.min_candidate_gpus=64 \
  --set optimization.constraints.max_candidate_gpus=64 \
  --set optimizer.max_trials=180 \
  --set optimizer.parallelism=2 \
  --output-dir "aisimulate-output/quickstart-$RUN_ID/search-parallel-K"
```

此例为探索搜索，与固定实测KV容量、客户端准备参数的历史回放具有不同假设。180为尝试次数上限，并不保证遍历全部候选；结合当前K模式较大的Decode误差，不能把排名直接当作真实部署最优配置。

代码声明的并行候选为：global TP16/32，Attention组内TP1/2/4/8/16，ADP=global TP/组内TP；MoE使用EP=global TP、MoE-TP1，或EP1、MoE-TP=global TP。每个候选仍需通过内存和数据覆盖检查。

PP/CP目前只支持1；不支持单实例TP64/EP64及MoE-TP、EP同时大于1。当前eager方案不支持DCP，PD+DCP也不支持。完整范围见[并行说明](910c-glm52-w8a8-usage.md#3-支持的并行结构)。

## 8. 阅读结果与复算推荐配置

预测结果在各输出目录的`prediction.json`中：

| 字段 | 含义 |
|---|---|
| `completed_requests`、`failed_requests` | 检查请求是否完整结束 |
| `total_input_tokens`、`total_output_tokens` | 固定长度请求应等于请求数乘输入/输出长度 |
| `mean_ttft_ms`、`mean_tpot_ms` | 平均TTFT和TPOT，单位ms |
| `input_throughput_tok_s`、`output_throughput_tok_s` | 整个部署的输入/输出吞吐 |
| `forward_pass_batches` | Prefill/Decode/Mixed逐轮batch汇总；`dp_rank=null`为整个logical worker |
| `duration_ms` | 模拟出的服务时间跨度 |
| `wall_time_ms` | 本机运行模拟器的时间，不能当作NPU服务时间 |

running batch的算术平均为`request_instances / passes`；峰值为`max_batch_size`。不要把等待队列、padding或客户端总并发当成running batch。`--capture-per-request`还会保存`requests.jsonl`。

搜索先看`recommendation.json`中的候选状态，再看排名。`failed`、`timed_out`、`unsupported`、`resource_limited`不等于真实吞吐为0；应检查失败原因。`recommendation.csv`便于表格查看，`recommendations/0001.yaml`为第一名的具体配置，可重新运行：

```bash
aisim910c predict --stack engine \
  --config "aisimulate-output/quickstart-$RUN_ID/search-pd-4node/recommendations/0001.yaml" \
  --capture-per-request --detail time \
  --output-dir "aisimulate-output/quickstart-$RUN_ID/recheck-top1"
```

吞吐目标没有自动加入TTFT/TPOT的业务SLA。对比实测时，要对齐镜像版本、PD实例数、TP/ADP/EP、输入输出、并发、总请求数、KV容量、chunk、路由策略、graph/overlap/投机设置及客户端行为。两个warmup不保证所有shape已预热，当前冷编译过程也未完整建模。

## 9. 复现历史表格与在其他机器运行

历史结果及K/L定义见[39条对照报告](910c-glm52-2node-4node-rerun-20261104.md)，其末尾给出CSV、预测配置和原始日志位置。完整重跑产物已归档在：

```text
benchmarks/evidence/ascend910c-glm52-w8a8/production/prediction-repair-20261102/historical-modes-rerun-evidence-archive-20261104/evidence.tar.gz
```

GitHub保存代码、性能表、文档和已归档证据。模型权重与本机Docker镜像需要在运行机器上准备。相同ARM64环境可通过`docker save`/`docker load`迁移当前镜像；从源码重建见[文件与重建](910c-glm52-w8a8-usage.md#6-文件与重建)。

修改Python、Rust或性能表后，需要重新打包wheel并构建镜像，单纯挂载更新后的Git源码不会替换镜像中已安装的AISimulate包。旧默认镜像保留历史版本，因此本文始终显式指定当前镜像标签。

## 10. 本文命令的验证范围

2026-11-04在当前节点执行了本文9条预测/搜索命令，均正常结束。单节点K/L、12K/4K、两节点1P1D K/L及四节点2P2D均完成全部请求；四节点PD搜索完成全部15个候选，第一名保存的YAML重新预测后，TTFT、TPOT和吞吐与搜索结果一致。

第7节的并行策略搜索此次以`optimizer.max_trials=12`完成12个候选的冒烟验证，未运行文档所示的180次完整预算。验证使用同一镜像，每个CLI进程限制2个CPU，未启动NPU服务。这是命令和结果一致性验证，不是新增实测精度验收。

命令、退出状态、结果摘要和原始输出归档见[验证记录](../benchmarks/evidence/ascend910c-glm52-w8a8/production/prediction-repair-20261102/current-quickstart-validation-20261104/validation.json)。
