#!/usr/bin/env bash
# 用途：在第一台 Prefill 节点一键启动多 Prefill、多 Decode、共享 MemFabric Store 和统一 Router。

set -Eeuo pipefail

# shellcheck source=pd-common.sh
source "$(dirname "$0")/pd-common.sh"

exec 9>"$PD_ROOT_DIR/.start-pd.lock"
flock -n 9 || die "已有 start-pd.sh 正在运行"

for command_name in docker ssh scp curl python3 ip flock ss awk; do
  require_command "$command_name"
done
require_controller_node
[[ $EUID -eq 0 ]] || die "启动过程需要设置系统参数，请使用 sudo -E $0"
prepare_remote_access

"$PD_ROOT_DIR/check/check-pd-env.sh"

mapfile -t WORKER_IPS < <(all_worker_ips | awk '!seen[$0]++')

sync_worker_scripts() {
  local node_ip
  local file
  local service_dir_q
  local check_dir_q
  local files=(
    pd-config.sh
    pd-common.sh
    container-common.sh
    tune-system.sh
    run-prefill.sh
    run-prefill-container.sh
    run-decode.sh
    run-decode-container.sh
  )

  for node_ip in "${WORKER_IPS[@]}"; do
    local_has_ip "$node_ip" && continue
    service_dir_q=$(shell_quote "$REMOTE_DIR/service")
    check_dir_q=$(shell_quote "$REMOTE_DIR/check")
    echo "同步脚本到 $(node_target "$node_ip"):$REMOTE_DIR"
    node_exec "$node_ip" "mkdir -p $service_dir_q $check_dir_q && chown $(shell_quote "$SSH_USER") $service_dir_q $check_dir_q"
    for file in "${files[@]}"; do
      node_copy "$PD_SCRIPT_DIR/$file" "$node_ip" "$REMOTE_DIR/service/"
    done
    node_copy "$PD_ROOT_DIR/check/check-ports.py" "$node_ip" "$REMOTE_DIR/check/"
    node_exec "$node_ip" "chmod +x $service_dir_q/*.sh $check_dir_q/check-ports.py"
  done
}

apply_system_tuning() {
  local node_ip
  local command_text

  if [[ $ENABLE_SYSTEM_TUNING != 1 ]]; then
    echo "ENABLE_SYSTEM_TUNING=$ENABLE_SYSTEM_TUNING，跳过系统调优"
    return 0
  fi

  echo "设置所有节点的 CPU/NUMA 参数"
  for node_ip in "${WORKER_IPS[@]}"; do
    if local_has_ip "$node_ip"; then
      ENABLE_SYSTEM_TUNING=1 "$PD_SCRIPT_DIR/tune-system.sh"
    else
      printf -v command_text 'cd %q && ENABLE_SYSTEM_TUNING=1 ./tune-system.sh' "$REMOTE_DIR/service"
      node_exec "$node_ip" "$command_text"
    fi
  done
}

clean_old_containers() {
  local index
  local spec
  local container

  echo "清理旧的同名 PD 容器"
  docker rm -f "$ROUTER_CONTAINER" >/dev/null 2>&1 || true

  for index in "${!PREFILL_NODE_LIST[@]}"; do
    spec=${PREFILL_NODE_LIST[$index]}
    parse_prefill_node "$spec"
    container=$(prefill_container_name "$index")
    node_exec "$NODE_IP" "docker rm -f $(shell_quote "$container") $(shell_quote "$PREFILL_CONTAINER_PREFIX") >/dev/null 2>&1 || true"
  done

  for index in "${!DECODE_NODE_LIST[@]}"; do
    spec=${DECODE_NODE_LIST[$index]}
    parse_decode_node "$spec"
    container=$(decode_container_name "$index")
    node_exec "$NODE_IP" "docker rm -f $(shell_quote "$container") $(shell_quote "$DECODE_CONTAINER_PREFIX") >/dev/null 2>&1 || true"
  done
}

check_node_ports() {
  local node_ip=$1
  shift
  local command_text
  local port_args=''
  local port

  for port in "$@"; do
    printf -v port_args '%s %q' "$port_args" "$port"
  done

  if local_has_ip "$node_ip"; then
    python3 "$PD_ROOT_DIR/check/check-ports.py" "$node_ip" "$@"
  else
    printf -v command_text 'python3 %q %q%s' "$REMOTE_DIR/check/check-ports.py" "$node_ip" "$port_args"
    node_exec "$node_ip" "$command_text"
  fi
}

check_service_ports() {
  local index
  local spec
  local -a ports

  echo "检查所有节点的服务端口"
  for index in "${!PREFILL_NODE_LIST[@]}"; do
    spec=${PREFILL_NODE_LIST[$index]}
    parse_prefill_node "$spec"
    ports=("$NODE_PORT" "$NODE_BOOTSTRAP")
    [[ $NODE_IP == "$STORE_IP" ]] && ports+=("$STORE_PORT")
    [[ $NODE_IP == "$ROUTER_IP" ]] && ports+=("$ROUTER_PORT")
    check_node_ports "$NODE_IP" "${ports[@]}"
  done

  for index in "${!DECODE_NODE_LIST[@]}"; do
    spec=${DECODE_NODE_LIST[$index]}
    parse_decode_node "$spec"
    if ((index % DECODE_NNODES == 0)); then
      if ((DECODE_NNODES > 1)); then
        check_node_ports "$NODE_IP" "$NODE_PORT" "$DECODE_DIST_PORT"
      else
        check_node_ports "$NODE_IP" "$NODE_PORT"
      fi
    fi
  done
}

runtime_assignments() {
  local name
  local variables=(
    STORE_IP STORE_PORT IMAGE MODEL_DIR COMM_IFNAME HCCL_IF_BASE_PORT
    PREFILL_HCCL_BUFFSIZE DECODE_HCCL_BUFFSIZE
    CONTEXT_LENGTH TP_SIZE DP_SIZE EP_SIZE NPU_DEVICE_COUNT SHM_SIZE
    MEM_FRACTION_STATIC PREFILL_MAX_RUNNING_REQUESTS DECODE_MAX_RUNNING_REQUESTS
    PREFILL_CHUNK_SIZE PREFILL_MAX_TOKENS DECODE_GRAPH_MAX_BS DECODE_GRAPH_BATCHES
    SERVED_MODEL_NAME SGLANG_NPU_PROFILING
    PREFILL_TP_SIZE PREFILL_DP_SIZE PREFILL_EP_SIZE DECODE_TP_SIZE DECODE_DP_SIZE DECODE_EP_SIZE
    PREFILL_MEM_FRACTION_STATIC DECODE_MEM_FRACTION_STATIC
    DISABLE_GRAPH DISABLE_OVERLAP DISABLE_PREFIX_CACHE ENABLE_SPECULATIVE_DECODING
    KV_PAGE_SIZE DECODE_RESERVED_TOKENS DECODE_NNODES DECODE_DIST_PORT
  )
  for name in "${variables[@]}"; do
    printf '%s=%s\n' "$name" "${!name}"
  done
}

run_worker() {
  local role=$1
  local index=$2
  local node_ip=$3
  local node_port=$4
  local bootstrap_port=$5
  local container=$6
  local script="run-${role}-container.sh"
  local assignment
  local quoted_assignment
  local remote_command
  local -a assignments=(
    "WORKER_IP=$node_ip"
    "WORKER_PORT=$node_port"
    "WORKER_BOOTSTRAP_PORT=$bootstrap_port"
    "WORKER_CONTAINER=$container"
  )
  if [[ $role == decode ]]; then
    decode_distribution "$index"
    assignments+=("WORKER_NNODES=$WORKER_NNODES" "WORKER_NODE_RANK=$WORKER_NODE_RANK" "WORKER_DIST_INIT_ADDR=$WORKER_DIST_INIT_ADDR")
  else
    assignments+=("WORKER_NNODES=1" "WORKER_NODE_RANK=0" "WORKER_DIST_INIT_ADDR=")
  fi

  while IFS= read -r assignment; do
    assignments+=("$assignment")
  done < <(runtime_assignments)

  echo "启动 ${role^}[$index]：$node_ip:$node_port"
  if local_has_ip "$node_ip"; then
    env "${assignments[@]}" "$PD_SCRIPT_DIR/$script"
  else
    printf -v remote_command 'cd %q && env' "$REMOTE_DIR/service"
    for assignment in "${assignments[@]}"; do
      printf -v quoted_assignment '%q' "$assignment"
      remote_command+=" $quoted_assignment"
    done
    printf -v quoted_assignment '%q' "./$script"
    remote_command+=" bash $quoted_assignment"
    node_exec "$node_ip" "$remote_command"
  fi
}

start_prefill() {
  local index=$1
  local spec=${PREFILL_NODE_LIST[$index]}
  parse_prefill_node "$spec"
  run_worker prefill "$index" "$NODE_IP" "$NODE_PORT" "$NODE_BOOTSTRAP" "$(prefill_container_name "$index")"
}

start_decode() {
  local index=$1
  local spec=${DECODE_NODE_LIST[$index]}
  parse_decode_node "$spec"
  run_worker decode "$index" "$NODE_IP" "$NODE_PORT" 0 "$(decode_container_name "$index")"
}

wait_for_store() {
  local deadline=$((SECONDS + STORE_TIMEOUT))
  local primary_container
  primary_container=$(prefill_container_name 0)
  echo "等待共享 MemFabric Store：$STORE_IP:$STORE_PORT"

  while ((SECONDS < deadline)); do
    require_running_container "$STORE_IP" "$primary_container" || return 1
    if ss -H -ltn "sport = :$STORE_PORT" | grep -q .; then
      echo "MemFabric Store 已就绪"
      return 0
    fi
    sleep 5
  done
  echo "等待 MemFabric Store 超时" >&2
  return 1
}

wait_for_http() {
  local name=$1
  local node_ip=$2
  local node_port=$3
  local container=$4
  local deadline=$((SECONDS + HEALTH_TIMEOUT))
  local url="http://$node_ip:$node_port/health"

  echo "等待 $name：$url"
  while ((SECONDS < deadline)); do
    require_running_container "$node_ip" "$container" || return 1
    if curl -fsS --max-time 30 "$url" >/dev/null 2>&1; then
      echo "$name 已就绪"
      return 0
    fi
    sleep 10
  done
  echo "等待 $name 超时" >&2
  return 1
}

start_router() {
  local spec
  local index
  local router_command=(
    docker run -d
    --name "$ROUTER_CONTAINER"
    --init
    --net=host
    -e TORCH_DEVICE_BACKEND_AUTOLOAD=0
    -v "$MODEL_DIR":/models/glm-q:ro
    --entrypoint python3
    "$IMAGE"
    -m sglang_router.launch_router
    --pd-disaggregation
    --policy round_robin
  )

  for spec in "${PREFILL_NODE_LIST[@]}"; do
    parse_prefill_node "$spec"
    router_command+=(--prefill "http://$NODE_IP:$NODE_PORT" "$NODE_BOOTSTRAP")
  done
  for index in "${!DECODE_NODE_LIST[@]}"; do
    ((index % DECODE_NNODES == 0)) || continue
    parse_decode_node "${DECODE_NODE_LIST[$index]}"
    router_command+=(--decode "http://$NODE_IP:$NODE_PORT")
  done
  router_command+=(--host "$ROUTER_IP" --port "$ROUTER_PORT")

  echo "启动 Router：$ROUTER_IP:$ROUTER_PORT"
  "${router_command[@]}" >/dev/null
}

sync_worker_scripts
apply_system_tuning
clean_old_containers
check_service_ports

# 首个 Prefill 负责建立共享 Store，必须最先启动。
start_prefill 0
wait_for_store

for index in "${!PREFILL_NODE_LIST[@]}"; do
  ((index == 0)) && continue
  start_prefill "$index"
done
for index in "${!DECODE_NODE_LIST[@]}"; do
  start_decode "$index"
done

for index in "${!PREFILL_NODE_LIST[@]}"; do
  parse_prefill_node "${PREFILL_NODE_LIST[$index]}"
  wait_for_http "Prefill[$index]" "$NODE_IP" "$NODE_PORT" "$(prefill_container_name "$index")"
done
for index in "${!DECODE_NODE_LIST[@]}"; do
  parse_decode_node "${DECODE_NODE_LIST[$index]}"
  require_running_container "$NODE_IP" "$(decode_container_name "$index")"
  if ((index % DECODE_NNODES == 0)); then
    wait_for_http "Decode[$((index / DECODE_NNODES))]" "$NODE_IP" "$NODE_PORT" "$(decode_container_name "$index")"
  fi
done

start_router
wait_for_http Router "$ROUTER_IP" "$ROUTER_PORT" "$ROUTER_CONTAINER"

echo
printf '%dP%dD 已启动：API=http://%s:%s\n' \
  "${#PREFILL_NODE_LIST[@]}" "$((${#DECODE_NODE_LIST[@]} / DECODE_NNODES))" "$ROUTER_IP" "$ROUTER_PORT"
echo "查看状态：$PD_SCRIPT_DIR/status-pd.sh"
echo "测试请求：$PD_ROOT_DIR/tools/test-pd.sh"
echo "停止服务：$PD_SCRIPT_DIR/stop-pd.sh"
