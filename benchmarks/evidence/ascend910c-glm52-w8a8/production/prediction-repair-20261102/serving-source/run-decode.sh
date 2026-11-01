#!/usr/bin/env bash
# 用途：在 Decode 容器内设置 Ascend/SGLang 环境并启动 GLM-5.2 Decode 服务。

set -Eeuo pipefail

# shellcheck source=pd-config.sh
source "$(dirname "$0")/pd-config.sh"

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
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
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=$PD_BOOTSTRAP_TIMEOUT
export TRANSFORMERS_VERBOSITY=error
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
export HCCL_BUFFSIZE=$DECODE_HCCL_BUFFSIZE
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export TASK_QUEUE_ENABLE=0
export HCCL_SOCKET_IFNAME=$COMM_IFNAME
export GLOO_SOCKET_IFNAME=$COMM_IFNAME
export HCCL_IF_BASE_PORT
export SGLANG_NPU_USE_MULTI_STREAM=1

read -r -a decode_graph_batches <<<"$DECODE_GRAPH_BATCHES"
[[ $ENABLE_SPECULATIVE_DECODING == 0 ]] || { echo "本对照配置仅支持关闭投机解码" >&2; exit 1; }

command=(
  python3 -m sglang.launch_server
  --model-path /models/glm-q
  --disaggregation-mode decode
  --disaggregation-transfer-backend ascend
  --host "$WORKER_IP"
  --port "$WORKER_PORT"
  --nnodes "${WORKER_NNODES:-$DECODE_NNODES}"
  --node-rank "${WORKER_NODE_RANK:-0}"
  --tp-size "$DECODE_TP_SIZE"
  --attn-dp-size "$DECODE_DP_SIZE"
  --ep-size "$DECODE_EP_SIZE"
  --mem-fraction-static "$DECODE_MEM_FRACTION_STATIC"
  --max-running-requests "$DECODE_MAX_RUNNING_REQUESTS"
  --attention-backend ascend
  --device npu
  --quantization modelslim
  --served-model-name "$SERVED_MODEL_NAME"
  --watchdog-timeout 9000
  --context-length "$CONTEXT_LENGTH"
  --tokenizer-worker-num 4
  --disable-shared-experts-fusion
  --dtype bfloat16
  --load-balance-method round_robin
  --page-size "$KV_PAGE_SIZE"
  --num-reserved-decode-tokens "$DECODE_RESERVED_TOKENS"
  --enable-cache-report
  --enable-metrics
  --enable-metrics-for-all-schedulers
  --trust-remote-code
)

if ((${WORKER_NNODES:-$DECODE_NNODES} > 1)); then
  command+=(--dist-init-addr "${WORKER_DIST_INIT_ADDR:?跨节点D需要WORKER_DIST_INIT_ADDR}")
fi
if ((DECODE_EP_SIZE > 1)); then
  command+=(--moe-a2a-backend deepep --deepep-mode low_latency)
else
  command+=(--moe-a2a-backend none)
fi
if [[ $DISABLE_GRAPH == 1 ]]; then
  command+=(--cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled)
else
  command+=(--cuda-graph-backend-decode full --cuda-graph-max-bs-decode "$DECODE_GRAPH_MAX_BS" --cuda-graph-bs-decode "${decode_graph_batches[@]}")
fi
[[ $DISABLE_OVERLAP != 1 ]] || command+=(--disable-overlap-schedule)
[[ $DISABLE_PREFIX_CACHE != 1 ]] || command+=(--disable-radix-cache)

printf '执行 Decode 命令：'
printf '%q ' "${command[@]}"
printf '\n'
[[ ${DRY_RUN:-0} != 1 ]] || exit 0
exec "${command[@]}"
