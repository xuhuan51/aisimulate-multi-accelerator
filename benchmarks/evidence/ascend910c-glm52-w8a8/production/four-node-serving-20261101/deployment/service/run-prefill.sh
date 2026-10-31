#!/usr/bin/env bash
# 用途：在 Prefill 容器内设置 Ascend/SGLang 环境并启动 GLM-5.2 Prefill 服务。

set -Eeuo pipefail

# shellcheck source=pd-config.sh
source "$(dirname "$0")/pd-config.sh"

# 避免宿主机代理影响节点间的 HCCL、Gloo 和 MemFabric 通信。
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING

# 镜像内必须包含 CANN Toolkit 与 ATB 环境脚本。
# Ascend 官方环境脚本会读取未定义变量，加载期间临时关闭 nounset。
if [[ ${DRY_RUN:-0} != 1 ]]; then
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
fi

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32
export ASCEND_MF_STORE_URL="tcp://${STORE_IP}:${STORE_PORT}"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export TRANSFORMERS_VERBOSITY=error
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
export HCCL_BUFFSIZE=$PREFILL_HCCL_BUFFSIZE
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export HCCL_SOCKET_IFNAME=$COMM_IFNAME
export GLOO_SOCKET_IFNAME=$COMM_IFNAME
export HCCL_IF_BASE_PORT
export TASK_QUEUE_ENABLE=0
export SGLANG_NPU_USE_MULTI_STREAM=1
[[ $ENABLE_SPECULATIVE_DECODING == 0 ]] || { echo "本对照配置仅支持关闭投机解码" >&2; exit 1; }

command=(
  python3 -m sglang.launch_server
  --model-path /models/glm-q
  --disaggregation-mode prefill
  --disaggregation-transfer-backend ascend
  --disaggregation-bootstrap-port "$WORKER_BOOTSTRAP_PORT"
  --host "$WORKER_IP"
  --port "$WORKER_PORT"
  --context-length "$CONTEXT_LENGTH"
  --nnodes 1
  --node-rank 0
  --tp-size "$PREFILL_TP_SIZE"
  --attn-dp-size "$PREFILL_DP_SIZE"
  --ep-size "$PREFILL_EP_SIZE"
  --mem-fraction-static "$PREFILL_MEM_FRACTION_STATIC"
  --max-running-requests "$PREFILL_MAX_RUNNING_REQUESTS"
  --attention-backend ascend
  --device npu
  --quantization modelslim
  --served-model-name "$SERVED_MODEL_NAME"
  --chunked-prefill-size "$PREFILL_CHUNK_SIZE"
  --max-prefill-tokens "$PREFILL_MAX_TOKENS"
  --disable-shared-experts-fusion
  --dtype bfloat16
  --load-balance-method round_robin
  --page-size "$KV_PAGE_SIZE"
  --enable-dp-lm-head
  --enable-cache-report
  --enable-metrics
  --enable-metrics-for-all-schedulers
  --trust-remote-code
)

if ((PREFILL_EP_SIZE > 1)); then
  command+=(--moe-a2a-backend deepep --deepep-mode normal)
else
  command+=(--moe-a2a-backend none)
fi
[[ $DISABLE_GRAPH != 1 ]] || command+=(--cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled)
[[ $DISABLE_OVERLAP != 1 ]] || command+=(--disable-overlap-schedule)
[[ $DISABLE_PREFIX_CACHE != 1 ]] || command+=(--disable-radix-cache)

printf '执行 Prefill 命令：'
printf '%q ' "${command[@]}"
printf '\n'
[[ ${DRY_RUN:-0} != 1 ]] || exit 0
exec "${command[@]}"
