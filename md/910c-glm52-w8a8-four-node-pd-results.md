# 四节点910C：8K/1K与32K/1K的PD搜索结果

## 2026-10-31：PD回退修复前的eager基线

本节保存 `2c1503d4` 的完整搜索结果。当前默认运行包已更新为 `db5f2e72`，
修复了这里6个失败配置的D端KV恢复流程；6个配置及原两组最佳配置均通过独立回放。
完整菜单复查进行中，完成后将补充新结果。修复与验证见对齐说明第25节。

使用本机已安装源码 `2c1503d4` 的离线运行包，模型路径为
`/data_lib/data/models/glm-q`。搜索于2026-10-30启动，以下汇总于2026-10-31核验。
预算最多4节点、每节点16个逻辑device，目标为整个部署的输出吞吐；仅比较PD，
无TTFT/TPOT SLA。关闭graph，使用同步调度；shared-expert内部重叠仍按原模型开启。
均为固定长度、无prefix cache、`ignore_eos=true`，P/D显存比例0.88/0.90。

8K搜索并发4/8/16/32，各N=2C；32K搜索并发1/2/4/8，各N=C，与用户脚本的
正式请求数量一致。没有复现脚本的2条预热，也没有启动真实NPU服务。
P/D分别遍历当前声明的DeepEP和张量MoE结构、全局16/32设备、副本数量及KV容量筛选。
8组搜索均耗尽各自的有限空间：共4024候选，4018成功、6失败、0超时。

| Workload | 成功候选中的最优并发 / 请求数 | P:D副本数 | P:D节点数 | 输出吞吐 token/s | mean TTFT s | mean TPOT ms/token |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 8192/1024 | 32 / 64 | 2:1 | 2:2 | **162.29** | **5.579** | **189.67** |
| 32768/1024 | 8 / 8 | 2:1 | 2:2 | **38.91** | **17.899** | **188.29** |

这里两个P副本各占1节点；一个D副本跨2节点，不能把“2P:1D副本”理解为只用3节点。
具体并行配置如下，attention TP是模拟器字段，SGLang的全局TP为attention TP×ADP：

| Workload/角色 | 副本 | 每副本节点 | Attention TP | ADP | 全局TP | MoE-TP | EP | 路径 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8K P | 2 | 1 | 4 | 4 | 16 | 1 | 16 | DeepEP |
| 8K D | 1 | 2 | 1 | 32 | 32 | 1 | 32 | DeepEP |
| 32K P | 2 | 1 | 4 | 4 | 16 | 16 | 1 | 张量MoE |
| 32K D | 1 | 2 | 1 | 32 | 32 | 1 | 32 | DeepEP |

各并发档位的最高输出吞吐为：

| Workload | 并发 | 请求数 | 评估 / 成功 / 失败 | 最高吞吐 token/s | mean TTFT s | mean TPOT ms |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 8K/1K | 4 | 8 | 556 / 556 / 0 | 26.81 | 2.463 | 146.93 |
| 8K/1K | 8 | 16 | 556 / 556 / 0 | 53.87 | 3.080 | 145.65 |
| 8K/1K | 16 | 32 | 556 / 556 / 0 | 106.02 | 4.583 | 145.72 |
| 8K/1K | 32 | 64 | 556 / 550 / 6 | 162.29 | 5.579 | 189.67 |
| 32K/1K | 1 | 1 | 450 / 450 / 0 | 6.51 | 9.409 | 144.48 |
| 32K/1K | 2 | 2 | 450 / 450 / 0 | 13.03 | 9.409 | 144.48 |
| 32K/1K | 4 | 4 | 450 / 450 / 0 | 25.43 | 10.551 | 147.14 |
| 32K/1K | 8 | 8 | 450 / 450 / 0 | 38.91 | 17.899 | 188.29 |

每行对应的并行结构可能不同。特别是32K只搜索到C8，不能据此断言C16/C32一定更差。
6个失败都发生在8K/C32：D为TP8/ADP2/MoE-TP16/EP1、两个D副本，P结构不同；
报错为 `Ascend synchronized operator timing requires a pure prefill or pure decode group`。
失败项未参与排名，详细坐标和异常均保留。该问题列为下一阶段源码审计和修复对象，
因此8K结果应称为“当前成功评估候选中的最优”，不能声称所有声明组合已有效比较。

所有4018个成功候选均核对了正式请求数、完成数、零拒绝/失败/取消、输入/输出token总数。
两个第一名的保存配置重新predict后，64/64和8/8条请求均正常完成、每条输出1024token；
各指标与搜索一致，仅有小于1e-9的浮点舍入差异。复算并不等于真实服务精度验收。
跨请求发送争用、host时序及同配置eager实测仍未完成校准。

结果和复现入口：

- [本轮汇总与失败原因](../benchmarks/evidence/ascend910c-glm52-w8a8/production/four-node-pd-eager-20261031/report.json)
- [4024个候选CSV](../benchmarks/evidence/ascend910c-glm52-w8a8/production/four-node-pd-eager-20261031/candidates.csv)
- [8K第一名配置](../benchmarks/evidence/ascend910c-glm52-w8a8/production/four-node-pd-eager-20261031/best-8192-1024.yaml)
- [32K第一名配置](../benchmarks/evidence/ascend910c-glm52-w8a8/production/four-node-pd-eager-20261031/best-32768-1024.yaml)

```bash
cd /home/youxingye/youxingye_data/aisimulate-910c
AISIMULATE_910C_MODEL=/data_lib/data/models/glm-q bash scripts/run-ascend910c.sh predict \
  --stack engine \
  --config benchmarks/evidence/ascend910c-glm52-w8a8/production/four-node-pd-eager-20261031/best-8192-1024.yaml \
  --capture-per-request --output-dir aisimulate-output/eager-best-8k-recheck
AISIMULATE_910C_MODEL=/data_lib/data/models/glm-q bash scripts/run-ascend910c.sh predict \
  --stack engine \
  --config benchmarks/evidence/ascend910c-glm52-w8a8/production/four-node-pd-eager-20261031/best-32768-1024.yaml \
  --capture-per-request --output-dir aisimulate-output/eager-best-32k-recheck
```

## 历史结果：2026-10-25旧运行包

以下保留旧版本、旧workload设置下的结果，不是当前eager配置的推荐。

运行日期：2026-10-25。模型为当前 `/mnt/raid/user/data/models/glm-q` 的GLM-5.2 ModelSlim W8A8。
本次运行的是已安装AISimulate镜像中的离线预测，没有启动真实四节点推理服务。

## 1. 结果

两组搜索均耗尽当前配置的有限合法候选空间，全部候选完成，无失败、超时或主机资源不足。
8K/1K评估984个候选，32K/1K评估825个候选。两组并行运行，搜索分别约10分21秒和9分48秒。

| 负载 | P节点:D节点 | P worker:D worker | 最优并发 | 输出吞吐 | 平均TTFT | 平均TPOT |
|---|---|---|---:|---:|---:|---:|
| 8192输入/1024输出 | **1:3** | **1:3**，每worker一个节点 | 32 | **314.78 token/s** | 10.10 s | 85.91 ms/token |
| 32768输入/1024输出 | **2:2** | **1:1**，每worker两个节点 | 32 | **228.58 token/s** | 37.84 s | 95.75 ms/token |

这里的P:D节点比例和worker比例需要区分。32K第一名不是两个独立P worker加两个独立D worker，
而是一个跨两节点的P worker，以及一个跨两节点的D worker。
两个第一名在当前评分下均无并列；排名分别为 `candidate-000874` 和 `candidate-000721`。

上述结果是**给定候选结构、并发范围和吞吐目标下的预测最优**。
没有设置TTFT或TPOT的SLA，完整服务误差及四节点共享链路争用尚未校准。

## 2. 最优并行参数

下表TP是模拟器的attention TP；SGLang启动参数中的全局TP等于TP×ADP。
两种负载的PP均为1，CP/DCP均为1。

| 负载/角色 | Worker数 | 节点总数 | 每worker设备数 | Attention TP | ADP | 全局TP | MoE-TP | EP | MoE路径 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8K P | 1 | 1 | 16 | 4 | 4 | 16 | 1 | 16 | DeepEP |
| 8K D | 3 | 3 | 16 | 8 | 2 | 16 | 16 | 1 | 张量MoE |
| 32K P | 1 | 2 | 32 | 2 | 16 | 32 | 1 | 32 | DeepEP |
| 32K D | 1 | 2 | 32 | 8 | 4 | 32 | 32 | 1 | 张量MoE |

8K第一名的核心配置是：

```yaml
prefill:
  parallelism: {tensor: 4, attention_data: 4, moe_tensor: 1, moe_expert: 16, pipeline: 1, replicas: 1}
decode:
  parallelism: {tensor: 8, attention_data: 2, moe_tensor: 16, moe_expert: 1, pipeline: 1, replicas: 3}
```

32K第一名的核心配置是：

```yaml
prefill:
  parallelism: {tensor: 2, attention_data: 16, moe_tensor: 1, moe_expert: 32, pipeline: 1, replicas: 1}
decode:
  parallelism: {tensor: 8, attention_data: 4, moe_tensor: 32, moe_expert: 1, pipeline: 1, replicas: 1}
```

这些是 `engine.workers` 下的片段，完整可直接predict的配置保存在下列文件：

- [8K第一名完整YAML](../benchmarks/evidence/ascend910c-glm52-w8a8/production/four-node-pd-20261025/searches/8k1k/recommendations/0001.yaml)
- [32K第一名完整YAML](../benchmarks/evidence/ascend910c-glm52-w8a8/production/four-node-pd-20261025/searches/32k1k/recommendations/0001.yaml)

与当前固定SGLang-NPU运行时的关键参数对应如下；它们不是完整的启动命令，
多节点地址、rank、端口、DP-attention开关及其余United环境仍需按实际部署脚本配置。

| 负载/角色 | `--tp-size` | `--attn-dp-size` | `--ep-size` | `--moe-a2a-backend` |
|---|---:|---:|---:|---|
| 8K P | 16 | 4 | 16 | `deepep` |
| 8K D，每个副本 | 16 | 2 | 1 | `none` |
| 32K P | 32 | 16 | 32 | `deepep` |
| 32K D | 32 | 4 | 1 | `none` |

## 3. 本次搜索条件

| 项目 | 设置 |
|---|---|
| 总资源 | 最少64、最多64个运行时device；16个device/节点，恰好四节点 |
| 部署模式 | 只搜索PD；原合并角色设为null |
| 目标 | 最大化整个部署的输出token吞吐 |
| 负载 | 固定输入输出长度，闭环并发候选8/16/32，每候选64条请求 |
| 上下文上限 | 8K负载9216；32K负载33792 |
| P调度 | `max_batched_tokens=4096`，`max_sequences=1` |
| D调度 | `max_batched_tokens=8192`，`max_sequences=32` |
| KV | BF16，128-token页，静态内存比例0.90，关闭prefix cache |
| 算子预测 | `op_level`，`fallback_policy=deny`，`shape_policy=anchored` |
| PD传输 | 实测曲线，连续页，`timing_mode=full_prompt` |
| 并行候选 | 当前已发布的DeepEP和张量MoE结构，单worker全局16或32，容量检查后枚举副本组合 |
| 搜索算法 | random，尝试上限10000，不放回遍历有限配置；空间耗尽后结束 |
| 候选执行 | 每组同时评估4个候选，单候选预算120秒；两组CPU搜索并行执行 |
| 延迟SLA | 未设置 |

10000不是本次实际候选数量。8K有限空间包含984个候选，32K包含825个，
每个并发分别覆盖328和275个部署配置。
32K的上下文更大，容量检查额外排除了不能容纳一条33792-token上下文的结构，
因此候选数有所减少。每个成功候选均核对了64条请求的完成数量和输入输出token总数。

## 4. 并发变化会改变推荐

下面分别在固定并发下选出该子集中的吞吐第一名；各行的并行结构可能不同。

| 负载 | 并发 | P节点:D节点 | 输出吞吐 token/s | 平均TTFT s | 平均TPOT ms |
|---|---:|---|---:|---:|---:|
| 8K/1K | 8 | 3:1 | 103.60 | 2.69 | 74.35 |
| 8K/1K | 16 | 2:2 | 202.56 | 3.81 | 74.35 |
| 8K/1K | 32 | 1:3 | 314.78 | 10.10 | 85.91 |
| 32K/1K | 8 | 2:2 | 81.43 | 12.29 | 85.63 |
| 32K/1K | 16 | 2:2 | 139.15 | 15.88 | 95.38 |
| 32K/1K | 32 | 2:2 | 228.58 | 37.84 | 95.75 |

因此，不能仅根据“输入8K/输出1K”就给出与实际并发无关的唯一最优配比。
如果线上固定并发或有明确请求到达率，应按该负载重新搜索，而不是直接采用上表跨并发的第一名。

再按P/D节点分配挑选各子集的吞吐第一名，可以看到本轮不同节点配比的差异：

| 负载 | P节点:D节点 | 所选并发 | 该子集最高输出吞吐 token/s | 对应平均TTFT s |
|---|---|---:|---:|---:|
| 8K/1K | 1:3 | 32 | 314.78 | 10.10 |
| 8K/1K | 2:2 | 32 | 307.32 | 6.29 |
| 8K/1K | 3:1 | 32 | 152.94 | 46.24 |
| 32K/1K | 1:3 | 16 | 108.11 | 48.22 |
| 32K/1K | 2:2 | 32 | 228.58 | 37.84 |
| 32K/1K | 3:1 | 32 | 41.73 | 523.39 |

例如8K/1K的2:2节点方案，吞吐较第一名低约2.37%，但平均TTFT从10.10秒降到6.29秒。
如果业务优先首token延迟，这个方案值得与吞吐第一名一起评估；本次没有加入SLA自动替用户做该取舍。
各子集的具体并行参数见证据摘要中的 `best_by_concurrency` 和 `best_by_node_split`。

## 5. 第一名复算与指标核对

搜索结束后，直接读取两组 `recommendations/0001.yaml` 再次执行predict，没有修改并发、模型或shape策略。
两组各完成64条请求，逐请求的终态、输入长度和1024个输出token均检查通过。

| 核对项 | 8K/1K | 32K/1K |
|---|---:|---:|
| 完成请求数 | 64 | 64 |
| 输入token总数 | 524288 | 2097152 |
| 输出token总数 | 65536 | 65536 |
| 平均TTFT | 10099.423947737 ms | 37839.659424005 ms |
| P99 TTFT | 27965.784073974 ms | 60044.549666380 ms |
| 平均TPOT | 85.914940782 ms | 95.753152895 ms |
| P99 TPOT | 85.933312738 ms | 95.753786500 ms |
| 模拟服务时间跨度 | 208.194432532 s | 286.703125457 s |

完成请求数、token总数、吞吐、平均/P99 TTFT、平均/P99 TPOT、平均端到端延迟、
服务时间跨度这10项指标与搜索结果一致。
这是保存配置的复现验证，不是与真实四节点服务的准确性对照。

在32节点上可复算冻结的第一名配置，输出目录应使用尚未存在的新目录：

```bash
cd /home/youxingye/youxingye_data/aisimulate-910c

sudo bash scripts/run-ascend910c.sh predict \
  --stack engine \
  --config benchmarks/evidence/ascend910c-glm52-w8a8/production/four-node-pd-20261025/searches/8k1k/recommendations/0001.yaml \
  --capture-per-request \
  --output-dir aisimulate-output/4node-pd-8k1k-recheck

sudo bash scripts/run-ascend910c.sh predict \
  --stack engine \
  --config benchmarks/evidence/ascend910c-glm52-w8a8/production/four-node-pd-20261025/searches/32k1k/recommendations/0001.yaml \
  --capture-per-request \
  --output-dir aisimulate-output/4node-pd-32k1k-recheck
```

## 6. 完整结果与复现材料

本机完整输出包含所有候选配置，不随Git提交：

- [8K完整搜索结果](../aisimulate-output/4node-pd-8k1k/)，以及 [第一名复算](../aisimulate-output/4node-pd-8k1k-best/)。
- [32K完整搜索结果](../aisimulate-output/4node-pd-32k1k/)，以及 [第一名复算](../aisimulate-output/4node-pd-32k1k-best/)。

随Git保存的证据位于
[four-node-pd-20261025](../benchmarks/evidence/ascend910c-glm52-w8a8/production/four-node-pd-20261025/)：

- `plan.json`：两条完整搜索命令、输入文件SHA256、镜像身份及CPU资源配置。
- `summary.json`：完整计数、前五名、各并发和节点配比的第一名、复算对照。
- `searches/`：完整候选JSON/CSV及日志的压缩副本、前五名YAML。
- `replays/`：第一名复算的汇总指标和逐请求记录。
- `manifest.json`：上述证据文件的大小与SHA256。

本次镜像为 `aisimulate-910c:glm52-w8a8`，镜像ID为
`sha256:19e7b55d1190797455f97073a8f97dce032fed540eae901e168d20a1ea9d78c2`，
运行时代码源提交为 `bf61026f`。采集基线仍为SGLang0.5.22固定提交、CANN/HCCL9.1.0.B243与驱动26.1.1。

本次未修改算子表、性能模型或调度实现。它补充了四节点固定预算下8K/1K与32K/1K完整有限空间搜索的功能证据。
真实路由不均衡、重叠效率、共享链路争用及完整服务误差仍需校准；
此前64K高并发搜索问题也没有因本次两组成功而被标记为解决。
