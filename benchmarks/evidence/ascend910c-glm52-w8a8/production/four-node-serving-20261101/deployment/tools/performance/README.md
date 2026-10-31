# 性能测试与采样

本目录中的脚本都读取 `../../service/pd-config.sh`，请求通过当前 Router，或直接选择指定 Worker 进行采样。

六组固定长度测试使用 `run-workload-suite.sh`，逐组并发扫描使用 `run-workload-concurrency-sweep.sh`。

## AISimulate 8K/1K 对照采集

服务启动成功后运行：

```bash
NUM_PROMPTS=8 MAX_CONCURRENCY=4 RANDOM_INPUT_LEN=8192 RANDOM_OUTPUT_LEN=1024 \
  WARMUP_REQUESTS=2 ./tools/performance/run-benchmark.sh
```

同时记录四节点运行批次、队列、KV 占用和逐请求结果：

```bash
CONFIG_LABEL=recommended-2p1d-dtp32 CONCURRENCIES='4 8 16 32 64 128' \
  WARMUP_REQUESTS=2 ./tools/performance/run-aisim-comparison.sh
```

每个并发点默认请求数为并发的两倍，固定 8192 输入和 1024 输出。种子为 910，使用
`random-ids --tokenize-prompt`，忽略 EOS，保留流式输出。预热为两个输入 8192、
输出最多 32 token 的请求，由此版本 benchmark 的预热实现决定，不计入正式吞吐。
正式结果逐项校验 `input_lens` 和 `output_lens`；任何请求失败或长度不符即停止本组。

结果写入 `results/aisim-comparison/配置名_时间戳/`。各并发子目录保留 JSONL、
`benchmark.log`、`runtime-config.json`、`metrics.jsonl.gz` 和 `metrics-summary.json`。
后两者包括一秒采样的 running requests、排队和 KV 指标。跨节点 Decode 采集两台
节点；同一实例、同一 attention DP rank 的 TP 副本去重。DP 协同执行时，空闲 rank
的 gauge 可能保留旧值，因此汇总只累计本采样间隔 decode token counter 增长的 rank，
同时保留 `raw_decode_running_peak`。运行批次峰值及非零时均值属于离散采样统计，
不等于精确的每轮 forward batch trace，可能包含系统探活请求。

比较预测时必须匹配 PD 实例数、各角色 TP/attention DP/EP、请求数、并发、KV 容量、
chunk、graph/overlap/speculative/prefix-cache 设置。`FIXED_NUM_PROMPTS=256` 可改为
每档固定请求数，但不能直接与默认 `N=2C` 的预测曲线比较。改变启动配置前先结束
本组采集并停止当前拓扑，重新启动后使用另一个 `CONFIG_LABEL` 保存结果。

## 六组统一测试

```bash
./tools/performance/run-workload-suite.sh
```

结果统一保存到 `results/workload-suite_时间戳/`。第 2 组固定使用 320 个请求、客户端并发 32；第 5 组长输出默认使用 16 个请求、并发 4。

确认第 5 组资源充足后，可让长输出用例使用全局请求数和并发：

```bash
FULL_SCALE_LONG_OUTPUT=1 ./tools/performance/run-workload-suite.sh
```

第 2 组不再使用 1024 并发；需要观察机器上限时使用下面的并发扫描脚本逐级加压。

## 六组工作负载并发扫描

```bash
./tools/performance/run-workload-concurrency-sweep.sh
```

默认并发阶梯：

```text
1. 3.5K/1.5K：8 16 32 64
2. 4K/1K：    8 16 32 64
3. 8K/1K：    4 8 16 32
4. 32K/1K：   1 2 4 8
5. 1K/32K：   1 2
6. 前缀缓存：  4 8 16 32
```

只扫描部分用例或覆盖某组并发列表：

```bash
CASES='1 2 3' ./tools/performance/run-workload-concurrency-sweep.sh
CASE4_CONCURRENCIES='1 2 4' ./tools/performance/run-workload-concurrency-sweep.sh
```

普通用例每个并发点默认发送并发数 2 倍的请求，32K 输入和 32K 输出发送并发数 1 倍的请求，缓存用例固定为 20 个请求；所有并发点统一预热 2 个请求。某组出现失败后默认跳过该组更高并发并继续下一组。

结果保存在 `results/workload-concurrency-sweep_时间戳/`，并生成 `status.tsv`、`summary.tsv` 和 `summary.md`。

## 单次 serving benchmark

```bash
NUM_PROMPTS=300 \
MAX_CONCURRENCY=16 \
RANDOM_INPUT_LEN=1024 \
RANDOM_OUTPUT_LEN=128 \
./tools/performance/run-benchmark.sh
```

默认使用 `random-ids`，在 `Prefill[0]` 容器内运行客户端，不需要从 Hugging Face 下载数据。

## 并发扫描

```bash
START_CONCURRENCY=1 END_CONCURRENCY=32 \
./tools/performance/run-concurrency-sweep.sh
```

脚本会检查扫描上限是否超过 `PREFILL_MAX_RUNNING_REQUESTS` 和 `DECODE_MAX_RUNNING_REQUESTS` 中较小的值。它只改变客户端并发，不会自动改变服务端并发限制。修改服务端限制后必须重启 PD 服务再测。

默认不在每轮之间重启服务。需要严格冷启动对比时可使用 `sudo -E RESTART_SERVICE_EACH_RUN=1 ...`，但这仍不会修改服务端配置。

## Worker 性能采样

先在 `service/pd-config.sh` 设置：

```bash
SGLANG_NPU_PROFILING=1
```

重启服务后执行：

```bash
PROFILE_SECONDS=10 ./tools/performance/profile-worker.sh prefill 0 prefill_sample
PROFILE_SECONDS=10 ./tools/performance/profile-worker.sh decode 0 decode_sample
```

采样脚本本身不生成请求。也可用 `LOAD_COMMAND` 在采样期间运行负载：

```bash
LOAD_COMMAND='./tools/performance/run-benchmark.sh' \
./tools/performance/profile-worker.sh decode 0 with_load
```
