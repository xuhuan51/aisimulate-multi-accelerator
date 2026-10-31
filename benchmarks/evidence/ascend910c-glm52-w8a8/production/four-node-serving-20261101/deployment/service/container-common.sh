#!/usr/bin/env bash
# 用途：构造 Prefill 和 Decode 共用的 Ascend Docker 参数，统一设备、驱动和模型挂载方式。

# 调用本函数后，Docker 参数保存在全局数组 DOCKER_ARGS 中。
build_ascend_docker_args() {
  local container_name=$1
  local entrypoint=$2
  local index
  local device

  [[ -d $MODEL_DIR ]] || die "模型目录不存在：$MODEL_DIR"
  [[ -d /usr/local/Ascend/driver ]] || die "未找到 Ascend 驱动目录：/usr/local/Ascend/driver"

  DOCKER_ARGS=(
    run -d
    --name "$container_name"
    --init
    --privileged=true
    --net=host
    --shm-size="$SHM_SIZE"
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver
    -v "$MODEL_DIR":/models/glm-q:ro
    -v "$PD_SCRIPT_DIR":/apps:ro
  )

  # 这些路径在不同 CANN/驱动版本中不一定都存在，因此按实际环境挂载。
  [[ -d /var/queue_schedule ]] && DOCKER_ARGS+=(-v /var/queue_schedule:/var/queue_schedule)
  [[ -f /etc/ascend_install.info ]] && DOCKER_ARGS+=(-v /etc/ascend_install.info:/etc/ascend_install.info:ro)
  [[ -d /usr/local/sbin ]] && DOCKER_ARGS+=(-v /usr/local/sbin:/usr/local/sbin:ro)
  [[ -d /usr/local/Ascend/firmware ]] && DOCKER_ARGS+=(-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro)

  for ((index = 0; index < NPU_DEVICE_COUNT; index++)); do
    device="/dev/davinci${index}"
    [[ -e $device ]] || die "缺少 NPU 设备：$device；请检查 NPU_DEVICE_COUNT"
    DOCKER_ARGS+=(--device="$device")
  done

  for device in /dev/davinci_manager /dev/hisi_hdc; do
    [[ -e $device ]] || die "缺少 Ascend 管理设备：$device"
    DOCKER_ARGS+=(--device="$device")
  done

  # 部分 Ascend 软件栈使用 devmm_svm；存在时挂载，不存在时保持原部署行为。
  [[ -e /dev/devmm_svm ]] && DOCKER_ARGS+=(--device=/dev/devmm_svm)

  DOCKER_ARGS+=(
    -e WORKER_IP -e WORKER_PORT -e WORKER_BOOTSTRAP_PORT
    -e WORKER_NNODES -e WORKER_NODE_RANK -e WORKER_DIST_INIT_ADDR
    -e STORE_IP -e STORE_PORT -e COMM_IFNAME
    -e HCCL_IF_BASE_PORT -e PREFILL_HCCL_BUFFSIZE -e DECODE_HCCL_BUFFSIZE -e CONTEXT_LENGTH
    -e TP_SIZE -e DP_SIZE -e EP_SIZE
    -e MEM_FRACTION_STATIC
    -e PREFILL_MAX_RUNNING_REQUESTS -e DECODE_MAX_RUNNING_REQUESTS
    -e PREFILL_CHUNK_SIZE -e PREFILL_MAX_TOKENS
    -e DECODE_GRAPH_MAX_BS -e DECODE_GRAPH_BATCHES
    -e SERVED_MODEL_NAME -e SGLANG_NPU_PROFILING
    -e PREFILL_TP_SIZE -e PREFILL_DP_SIZE -e PREFILL_EP_SIZE
    -e DECODE_TP_SIZE -e DECODE_DP_SIZE -e DECODE_EP_SIZE
    -e PREFILL_MEM_FRACTION_STATIC -e DECODE_MEM_FRACTION_STATIC
    -e DISABLE_GRAPH -e DISABLE_OVERLAP -e DISABLE_PREFIX_CACHE
    -e ENABLE_SPECULATIVE_DECODING -e KV_PAGE_SIZE -e DECODE_RESERVED_TOKENS
    --entrypoint "$entrypoint"
    "$IMAGE"
  )
}
