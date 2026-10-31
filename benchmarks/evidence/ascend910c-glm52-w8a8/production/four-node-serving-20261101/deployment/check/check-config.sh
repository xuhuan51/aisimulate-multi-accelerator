#!/usr/bin/env bash
# 用途：离线检查多 Prefill/多 Decode 拓扑格式、节点重复、端口冲突和并行参数。
#
# 本脚本不连接远端、不启动容器，修改 service/pd-config.sh 后应先执行它。

set -Eeuo pipefail

# shellcheck source=../service/pd-common.sh
source "$(dirname "$0")/../service/pd-common.sh"

errors=0
warnings=0
declare -A used_ips=()
declare -A used_endpoints=()
declare -A prefill_ips=()

error() {
  echo "错误：$*" >&2
  errors=$((errors + 1))
}

warn() {
  echo "警告：$*" >&2
  warnings=$((warnings + 1))
}

is_uint() {
  [[ $1 =~ ^[0-9]+$ ]]
}

valid_port() {
  is_uint "$1" && ((10#$1 >= 1 && 10#$1 <= 65535))
}

valid_ipv4() {
  local ip=$1
  local octet
  local -a parts
  IFS=. read -r -a parts <<<"$ip"
  ((${#parts[@]} == 4)) || return 1
  for octet in "${parts[@]}"; do
    is_uint "$octet" || return 1
    ((10#$octet >= 0 && 10#$octet <= 255)) || return 1
  done
}

register_ip() {
  local ip=$1
  local role=$2
  if [[ -n ${used_ips[$ip]:-} ]]; then
    error "节点 $ip 同时或重复配置为 ${used_ips[$ip]} 和 $role；当前脚本每个节点独占全部 NPU"
  else
    used_ips[$ip]=$role
  fi
}

register_endpoint() {
  local ip=$1
  local port=$2
  local purpose=$3
  local key="$ip:$port"
  if [[ -n ${used_endpoints[$key]:-} ]]; then
    error "端口冲突 $key：${used_endpoints[$key]} 与 $purpose"
  else
    used_endpoints[$key]=$purpose
  fi
}

((${#PREFILL_NODE_LIST[@]} > 0)) || error "PREFILL_NODES 不能为空"
((${#DECODE_NODE_LIST[@]} > 0)) || error "DECODE_NODES 不能为空"

for index in "${!PREFILL_NODE_LIST[@]}"; do
  spec=${PREFILL_NODE_LIST[$index]}
  if ! parse_prefill_node "$spec"; then
    error "Prefill[$index] 格式错误：$spec；应为 IP:HTTP端口:Bootstrap端口"
    continue
  fi
  valid_ipv4 "$NODE_IP" || error "Prefill[$index] IP 无效：$NODE_IP"
  valid_port "$NODE_PORT" || error "Prefill[$index] HTTP 端口无效：$NODE_PORT"
  valid_port "$NODE_BOOTSTRAP" || error "Prefill[$index] Bootstrap 端口无效：$NODE_BOOTSTRAP"
  register_ip "$NODE_IP" "Prefill[$index]"
  register_endpoint "$NODE_IP" "$NODE_PORT" "Prefill[$index] HTTP"
  register_endpoint "$NODE_IP" "$NODE_BOOTSTRAP" "Prefill[$index] Bootstrap"
  prefill_ips[$NODE_IP]=1
done

for index in "${!DECODE_NODE_LIST[@]}"; do
  spec=${DECODE_NODE_LIST[$index]}
  if ! parse_decode_node "$spec"; then
    error "Decode[$index] 格式错误：$spec；应为 IP:HTTP端口"
    continue
  fi
  valid_ipv4 "$NODE_IP" || error "Decode[$index] IP 无效：$NODE_IP"
  valid_port "$NODE_PORT" || error "Decode[$index] HTTP 端口无效：$NODE_PORT"
  register_ip "$NODE_IP" "Decode[$index]"
  register_endpoint "$NODE_IP" "$NODE_PORT" "Decode[$index] HTTP"
done

valid_ipv4 "$ROUTER_IP" || error "ROUTER_IP 无效：$ROUTER_IP"
valid_ipv4 "$STORE_IP" || error "STORE_IP 无效：$STORE_IP"
valid_port "$ROUTER_PORT" || error "ROUTER_PORT 无效：$ROUTER_PORT"
valid_port "$STORE_PORT" || error "STORE_PORT 无效：$STORE_PORT"

first_prefill=${PREFILL_NODE_LIST[0]:-}
if [[ -n $first_prefill ]] && parse_prefill_node "$first_prefill"; then
  [[ $ROUTER_IP == "$NODE_IP" ]] || error "ROUTER_IP 必须是第一台 Prefill：$NODE_IP"
  [[ $STORE_IP == "$NODE_IP" ]] || error "STORE_IP 必须是第一台 Prefill：$NODE_IP"
fi
register_endpoint "$ROUTER_IP" "$ROUTER_PORT" "Router"
register_endpoint "$STORE_IP" "$STORE_PORT" "MemFabric Store"

for value_name in PREFILL_TP_SIZE PREFILL_DP_SIZE PREFILL_EP_SIZE DECODE_TP_SIZE DECODE_DP_SIZE DECODE_EP_SIZE DECODE_NNODES NPU_DEVICE_COUNT CONTEXT_LENGTH DECODE_GRAPH_MAX_BS PREFILL_MAX_RUNNING_REQUESTS DECODE_MAX_RUNNING_REQUESTS PREFILL_CHUNK_SIZE PREFILL_MAX_TOKENS PREFILL_HCCL_BUFFSIZE DECODE_HCCL_BUFFSIZE PD_BOOTSTRAP_TIMEOUT ROUTER_REQUEST_TIMEOUT; do
  value=${!value_name}
  [[ $value =~ ^[1-9][0-9]*$ ]] || error "$value_name 必须是正整数：$value"
done

((errors == 0)) || { echo "配置包含非法整数或节点，停止进一步检查" >&2; exit 1; }
valid_port "$DECODE_DIST_PORT" || error "DECODE_DIST_PORT 无效：$DECODE_DIST_PORT"
(( ${#DECODE_NODE_LIST[@]} % DECODE_NNODES == 0 )) || error "Decode 节点数必须能被 DECODE_NNODES 整除"
for role in PREFILL DECODE; do
  tp_name=${role}_TP_SIZE; dp_name=${role}_DP_SIZE; ep_name=${role}_EP_SIZE
  cap_name=${role}_MAX_RUNNING_REQUESTS
  tensor=${!tp_name}; attention_dp=${!dp_name}; expert=${!ep_name}; capacity=${!cap_name}
  nodes=1
  [[ $role != DECODE ]] || nodes=$DECODE_NNODES
  ((tensor <= NPU_DEVICE_COUNT * nodes)) || error "$role TP=$tensor 超过 $nodes 个节点的设备数"
  ((tensor % nodes == 0)) || error "$role TP 必须能被节点数整除"
  ((tensor % attention_dp == 0)) || error "$role TP=$tensor 必须能被 attention DP=$attention_dp 整除"
  ((tensor % expert == 0)) || error "$role TP=$tensor 必须能被 EP=$expert 整除"
  ((capacity >= attention_dp && capacity % attention_dp == 0)) || error "$role max-running-requests=$capacity 必须为 attention DP=$attention_dp 的正整数倍"
done
for index in "${!DECODE_NODE_LIST[@]}"; do
  parse_decode_node "${DECODE_NODE_LIST[$index]}"
  leader_index=$((index / DECODE_NNODES * DECODE_NNODES))
  leader_port=${DECODE_NODE_LIST[$leader_index]##*:}
  [[ $NODE_PORT == "$leader_port" ]] || error "同一跨节点 Decode 实例必须使用相同 HTTP 端口"
  if ((DECODE_NNODES > 1 && index % DECODE_NNODES == 0)); then
    register_endpoint "$NODE_IP" "$DECODE_DIST_PORT" "Decode[$((index / DECODE_NNODES))] rendezvous"
  fi
done
if is_uint "$PREFILL_CHUNK_SIZE" && is_uint "$PREFILL_MAX_TOKENS" && is_uint "$CONTEXT_LENGTH"; then
  ((PREFILL_CHUNK_SIZE <= PREFILL_MAX_TOKENS)) || error "PREFILL_CHUNK_SIZE 不能大于 PREFILL_MAX_TOKENS"
  ((PREFILL_MAX_TOKENS <= CONTEXT_LENGTH)) || error "PREFILL_MAX_TOKENS 不能大于 CONTEXT_LENGTH"
fi
for value_name in PREFILL_MEM_FRACTION_STATIC DECODE_MEM_FRACTION_STATIC; do
  [[ ${!value_name} =~ ^0\.[0-9]*[1-9][0-9]*$ ]] || error "$value_name 必须在 0 与 1 之间"
done
for value_name in DISABLE_GRAPH DISABLE_OVERLAP DISABLE_PREFIX_CACHE REMOTE_SUDO SGLANG_NPU_PROFILING; do
  [[ ${!value_name} == 0 || ${!value_name} == 1 ]] || error "$value_name 必须是 0 或 1"
done
[[ $ENABLE_SPECULATIVE_DECODING == 0 ]] || error "此对照脚本仅支持关闭投机解码"

if [[ $DISABLE_GRAPH == 0 ]]; then
  if ((DECODE_TP_SIZE % DECODE_DP_SIZE != 0)); then
    error "Decode TP 必须能被 attention DP 整除"
  else
    attention_tp=$((DECODE_TP_SIZE / DECODE_DP_SIZE))
    for batch in $DECODE_GRAPH_BATCHES; do
      valid_port "$batch" || { error "无效 Decode Graph batch：$batch"; continue; }
      ((batch % attention_tp == 0)) || error "Decode Graph batch=$batch 必须是 attention TP=$attention_tp 的整数倍"
      ((batch <= DECODE_GRAPH_MAX_BS)) || error "Decode Graph batch=$batch 大于 DECODE_GRAPH_MAX_BS=$DECODE_GRAPH_MAX_BS"
    done
  fi
fi

[[ $REMOTE_DIR != *[[:space:]]* ]] || error "REMOTE_DIR 不能包含空白字符：$REMOTE_DIR"

if ((errors > 0)); then
  echo "配置检查失败：$errors 个错误，$warnings 个警告" >&2
  exit 1
fi

echo "拓扑明细："
for index in "${!PREFILL_NODE_LIST[@]}"; do
  parse_prefill_node "${PREFILL_NODE_LIST[$index]}"
  printf "  Prefill[%s] %s:%s bootstrap=%s container=%s\n" "$index" "$NODE_IP" "$NODE_PORT" "$NODE_BOOTSTRAP" "$(prefill_container_name "$index")"
done
for index in "${!DECODE_NODE_LIST[@]}"; do
  parse_decode_node "${DECODE_NODE_LIST[$index]}"
  printf "  Decode[%s] node-rank=%s/%s %s:%s container=%s\n" "$((index / DECODE_NNODES))" "$((index % DECODE_NNODES))" "$DECODE_NNODES" "$NODE_IP" "$NODE_PORT" "$(decode_container_name "$index")"
done
printf "  Router/Store %s router=%s store=%s\n" "$ROUTER_IP" "$ROUTER_PORT" "$STORE_PORT"

printf '配置检查通过：%d Prefill + %d Decode，Router/Store=%s\n' \
  "${#PREFILL_NODE_LIST[@]}" "$((${#DECODE_NODE_LIST[@]} / DECODE_NNODES))" "$ROUTER_IP"
printf '  Prefill: TP=%s attention-DP=%s EP=%s HCCL_BUFFSIZE=%s MB\n' "$PREFILL_TP_SIZE" "$PREFILL_DP_SIZE" "$PREFILL_EP_SIZE" "$PREFILL_HCCL_BUFFSIZE"
printf '  Decode:  TP=%s attention-DP=%s EP=%s HCCL_BUFFSIZE=%s MB\n' "$DECODE_TP_SIZE" "$DECODE_DP_SIZE" "$DECODE_EP_SIZE" "$DECODE_HCCL_BUFFSIZE"
printf '  等待时限：PD bootstrap=%s s，Router request=%s s\n' "$PD_BOOTSTRAP_TIMEOUT" "$ROUTER_REQUEST_TIMEOUT"
((warnings == 0)) || echo "配置包含 $warnings 个警告"
