# KV分配冷编译与同配置独立验证（2026-11-02）

使用node20 P与node25 D，固定镜像、模型元数据与启动参数核对一致。两端均TP16/ADP4/EP16，graph、调度overlap、投机解码关闭，8192输入/128输出，各工况仅2条benchmark warmup。共16条正式请求全部成功；随后4条512/32请求只用于排空异步计时，未计入正式性能。全部诊断服务已停止，两节点均记录8张物理NPU无进程。

该轮没有拟合或改动预测参数。实测与当前AISimulate相同配置回放：

| 指标 | C1/N8实测 | C1预测 | 误差 | C4/N8实测 | C4预测 | 误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TTFT ms | 3990.432 | 3052.417 | -23.51% | 5092.606 | 4999.049 | -1.84% |
| TPOT ms | 400.619 | 392.638 | -1.99% | 424.985 | 410.189 | -3.48% |
| 输入 token/s | 149.295 | 154.807 | +3.69% | 550.007 | 569.369 | +3.52% |
| 输出 token/s | 2.333 | 2.419 | +3.69% | 8.594 | 8.896 | +3.52% |

该8192/128补测用于快速分离准备开销，不能冒充完整8192/1024验证。此前完整8K/1K C4/N8独立验证仍为TTFT-7.85%、TPOT-2.51%、吞吐+2.68%，本轮没有改动这些值。

## 两条冷请求的直接证据

| 请求 | 发生位置 | 分配耗时ms | 内嵌compile API耗时ms | 客户端TTFT ms |
| --- | --- | ---: | ---: | ---: |
| `aisim-6a67369cfb8145daac7505e7d2183994` | decode DP2 | 2134.52 | 2132.78 | 5719.53 |
| `aisim-0bf02dbb24d649678c2a6fec0b3b4282` | prefill DP3 | 2163.33 | 2160.13 | 7012.73 |

第二条请求Prefill耗时中，run_batch之外有2188.28ms，页分配内的compile API占2160.13ms；此前尚未归因的约2.17秒间隙得到定位。第一条请求D的分配/metadata总计2137.41ms，内嵌compile API为2132.78ms。没有compile调用的正式分配中位数为0.577ms。

关键不是“每个rank首次请求都加2秒”：同一个`alloc_extend_kernel`的编译身份还包括free_pages指针是否16字节对齐。4K aligned与unaligned是两个不同特化；相同特化在其他rank调用compile API时可仅耗约2ms，不能把每次compile API调用都当成重新编译。实际Triton源码的cache key还包含backend选项和环境，现有采集记录的signature/constants/attrs并不是完整cache key。

HTTP健康检查会生成1-token请求并分配页，可能改变后续free_pages指针对齐，也会触发自身的新特化。已核对镜像实际DP controller：health_round_robin_counter与用户round_robin_counter独立；不能归咎于健康检查挤占用户轮询序号。两条warmup覆盖不了所有页对齐与cache状态。

TTFT还按首个非空text计时；本镜像benchmark对后续累计文本包按新增token数展开ITL，完整成功流的output_tokens减ITL个数能恢复首个非空包时累计生成的token数。新C1的8条分别为1/3/1/1/1/1/3/1。这不等于每条请求都额外等两个decode，也不能把观察到的间隔直接加到预测表。

## 逐轮batch完整性

本轮在正式测试结束后额外排空异步计时，两个工况都取得全部rank的逐轮数据，并核验decode实例数N×(O−1)。C1的P/D平均batch均为1，与预测一致；C4的P平均batch实测/预测均2.2857，D为3.7770/3.7910（+0.37%）。证明C1的23.51% TTFT误差不是runningbatch或graph/overlap配置不同造成。

## 尚未解决

冷编译状态尚未进入模拟器的请求准备事件；原23点TTFT仍有8点超过15%，不能标为goal完成。下一步要在请求准备/批次提交边界处理cache miss，并描述共享编译缓存、初始cache状态、对齐特化与health请求；不能使用每rank固定常数，也不能向稳态算子表注入冷编译时间。部分首字还受文本流可见性影响，预测必须保留这一口径差异。

原始数据在`pd-allocator-1p1d/`；远端41个regular files逐一SHA256验证。时序在`pd-allocator-timeline-analysis.json.gz`，逐请求编译归因在`allocator-cold-attribution.json.gz`，完整模拟输出在`allocator-validation-replays/`，文件清单为`allocator-validation-manifest.json`。
