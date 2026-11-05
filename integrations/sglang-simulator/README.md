# SGLang 原生调度与 AISimulate 负载回放

原生 SGLang 在 CPU 上执行请求排队、组批和 KV 管理，AISimulate 为每轮 forward 提供耗时预测。更换模型时，通过配置选择模型、精度、拓扑和请求负载，使用对应硬件的测量数据。

## 安装

在 Linux 和 Python 3.12 环境中，从仓库根目录安装 AISimulate 及回放插件。AISimulate 的 native 扩展需要 Rust 构建工具链。

```bash
git clone https://github.com/xuhuan51/aisimulate-multi-accelerator.git
cd aisimulate-multi-accelerator
python -m pip install -e ./python/aisimulate
python -m pip install -e ./integrations/sglang-simulator
aisimulate recommend --help
```

原生调度进程还需要兼容的 SGLang 源码及其 Python 导入依赖。已测 GLM 调度源码为 SGLang `76f9213a411018547f4fd6a75f36feaa4d6bed58`。AISimulate 和 SGLang 可使用两个独立解释器。每次运行会记录实际源码 hash。

## 配置与运行

[GLM Graph 示例](examples/glm52-graph.json) 枚举 TP4/DP4、TP8/DP2 和 TP16/DP1，均为 EP16。将 `engine.model` 替换为本地 GLM ModelSlim 目录，其中需要 `config.json` 和 `quant_model_description.json`。运行模拟只读取模型元数据。

```bash
export SGLANG_SIMULATOR_PYTHON=/path/to/sglang-env/bin/python
export SGLANG_SIMULATOR_SGLANG_ROOT=/path/to/pinned-sglang
export SGLANG_SIMULATOR_KERNEL_ROOT=/path/to/sgl-kernel/python
export SGLANG_SIMULATOR_RUN_ROOT=/path/to/fresh-run-evidence
export SGLANG_SIMULATOR_RUN_TIMEOUT=7000
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
export JAX_PLATFORMS=cpu
aisimulate recommend --stack sglang-simulator \
  --config integrations/sglang-simulator/examples/glm52-graph.json \
  --output-dir /path/to/fresh-recommendation
```

`SGLANG_SIMULATOR_NATIVE_ARGS` 可指定已解析的原生 ServerArgs JSON。复用真机配置时，会按候选拓扑重新计算 DP chunk。精确重放语料可设置 `SGLANG_SIMULATOR_CORPUS_FILE` 与 `SGLANG_SIMULATOR_CORPUS_SHA256`。模型和系统数据路径由运行者提供。

负载位于 `traffic`，包含输入输出长度、并发和请求数。候选拓扑位于 `engine.workers.aggregated.parallelism.preset`。`tensor` 和 `attention_data` 分别表示 Attention TP 与 DP，专家 TP/EP 单独指定。输出目录包含逐请求指标、逐轮 batch、KV 配置、源码身份及耗时查询证据。

## 当前验证范围

| 项目 | 状态 |
| --- | --- |
| GLM-5.2 W8A8 | 六种单机 Graph 回放完成，四种有正式真机对照 |
| GLM 精度 | TP4/DP4 与 TP8/DP2 吞吐档位一致，EP16 吞吐仍低估约38%至40% |
| Qwen3.5-397B-A17B W8A8 | Graph 模拟0/9，缺少匹配 Graph 与 FP32 GDN 状态的数据及执行覆盖 |
| 单实例拓扑 | PP1、普通生成、纯专家 TP 或全宽 EP |
| 相同双副本 | 对称闭环负载，零时刻到达，同长度，并发与请求数均分，prefix cache关闭 |

GLM 测试使用每DP组32768 KV token，因此DP更多的配置拥有更大的节点总KV容量。调度源码与0.5.22算子表存在版本差异。已有结果限于记录的模型、负载和配置。

## 测试

```bash
python -m pytest integrations/sglang-simulator/tests -q
```

集成测试覆盖配置传递、拓扑检查、请求指标和上下文预留槽。完整回放与性能精度分别由运行证据和真机对照评估。

## 来源

本目录基于 SGLang `tools/sglang-simulator`，保留 Apache-2.0 许可，并增加 Ascend 与 AISimulate 集成。完整来源及修改范围见仓库 [第三方声明](../../THIRD_PARTY_NOTICES.md) 和 [源码清单](source-manifest.json)。
