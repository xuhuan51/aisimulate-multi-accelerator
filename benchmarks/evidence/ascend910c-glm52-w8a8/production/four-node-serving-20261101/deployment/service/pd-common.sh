#!/usr/bin/env bash
# 用途：为多 P/多 D 管理脚本提供拓扑解析、SSH、节点执行和容器检查函数。

PD_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PD_ROOT_DIR=$(cd -- "$PD_SCRIPT_DIR/.." && pwd)
# shellcheck source=pd-config.sh
source "$PD_SCRIPT_DIR/pd-config.sh"

read -r -a PREFILL_NODE_LIST <<<"$PREFILL_NODES"
read -r -a DECODE_NODE_LIST <<<"$DECODE_NODES"

SSH_OPTIONS=(
  -o ConnectTimeout="$SSH_CONNECT_TIMEOUT"
  -o StrictHostKeyChecking=accept-new
)
[[ ! -f $SSH_IDENTITY_FILE ]] || SSH_OPTIONS+=(-i "$SSH_IDENTITY_FILE")

prepare_remote_access() {
  [[ ${PD_REMOTE_AUTH_READY:-0} != 1 ]] || return 0
  if [[ -z ${PD_SSH_PASSWORD:-} && -t 0 ]]; then
    read -r -s -p "SSH 密码（${SSH_USER}；免密登录可直接回车）：" PD_SSH_PASSWORD
    printf '\n'
  fi
  if [[ $REMOTE_SUDO == 1 && $SSH_USER != root && -z ${PD_SUDO_PASSWORD:-} ]]; then
    PD_SUDO_PASSWORD=${PD_SSH_PASSWORD:-}
    if [[ -z $PD_SUDO_PASSWORD && -t 0 ]]; then
      read -r -s -p "远端 sudo 密码：" PD_SUDO_PASSWORD
      printf '\n'
    fi
  fi
  export PD_SSH_PASSWORD PD_SUDO_PASSWORD
  if [[ -n ${PD_SSH_PASSWORD:-} ]]; then
    require_command sshpass
  fi
  export PD_REMOTE_AUTH_READY=1
}

remote_ssh() {
  if [[ -n ${PD_SSH_PASSWORD:-} ]]; then
    SSHPASS=$PD_SSH_PASSWORD sshpass -e ssh "${SSH_OPTIONS[@]}" "$@"
  else
    ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes "$@"
  fi
}

die() {
  echo "错误：$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "缺少命令：$1"
}

local_has_ip() {
  ip -o -4 addr show | awk '{print $4}' | cut -d/ -f1 | grep -Fxq "$1"
}

require_controller_node() {
  local_has_ip "$ROUTER_IP" || die "请在控制节点/Router 节点 $ROUTER_IP 上执行本脚本"
}

parse_prefill_node() {
  local spec=$1
  local extra=${spec#*:*:*}
  IFS=: read -r NODE_IP NODE_PORT NODE_BOOTSTRAP <<<"$spec"
  [[ $extra != *:* ]] || return 1
  [[ -n $NODE_IP && -n $NODE_PORT && -n $NODE_BOOTSTRAP ]]
}

parse_decode_node() {
  local spec=$1
  local extra
  IFS=: read -r NODE_IP NODE_PORT extra <<<"$spec"
  [[ -n $NODE_IP && -n $NODE_PORT && -z ${extra:-} ]]
}

prefill_container_name() {
  printf '%s-%s' "$PREFILL_CONTAINER_PREFIX" "$1"
}

decode_container_name() {
  printf '%s-%s' "$DECODE_CONTAINER_PREFIX" "$1"
}

node_target() {
  printf '%s@%s' "$SSH_USER" "$1"
}

node_exec() {
  local node_ip=$1
  local command_text=$2
  if local_has_ip "$node_ip"; then
    bash -c "$command_text"
  else
    if [[ $REMOTE_SUDO == 1 && $SSH_USER != root ]]; then
      if [[ -n ${PD_SUDO_PASSWORD:-} ]]; then
        printf '%s\n' "$PD_SUDO_PASSWORD" | remote_ssh "$(node_target "$node_ip")" "sudo -S -p '' bash -c $(shell_quote "$command_text")"
      else
        remote_ssh "$(node_target "$node_ip")" "sudo -n bash -c $(shell_quote "$command_text")"
      fi
    else
      remote_ssh "$(node_target "$node_ip")" "$command_text"
    fi
  fi
}

node_copy() {
  local source_path=$1
  local node_ip=$2
  local destination=$3
  local_has_ip "$node_ip" && return 0
  if [[ -n ${PD_SSH_PASSWORD:-} ]]; then
    SSHPASS=$PD_SSH_PASSWORD sshpass -e scp -q "${SSH_OPTIONS[@]}" "$source_path" "$(node_target "$node_ip"):$destination"
  else
    scp -q "${SSH_OPTIONS[@]}" -o BatchMode=yes "$source_path" "$(node_target "$node_ip"):$destination"
  fi
}

decode_distribution() {
  local index=$1
  local first_index=$((index / DECODE_NNODES * DECODE_NNODES))
  local leader=${DECODE_NODE_LIST[$first_index]%%:*}
  WORKER_NNODES=$DECODE_NNODES
  WORKER_NODE_RANK=$((index % DECODE_NNODES))
  WORKER_DIST_INIT_ADDR="$leader:$DECODE_DIST_PORT"
}

all_worker_ips() {
  local spec
  for spec in "${PREFILL_NODE_LIST[@]}"; do
    parse_prefill_node "$spec" || continue
    printf '%s\n' "$NODE_IP"
  done
  for spec in "${DECODE_NODE_LIST[@]}"; do
    parse_decode_node "$spec" || continue
    printf '%s\n' "$NODE_IP"
  done
}

container_is_running() {
  local node_ip=$1
  local container=$2
  local running
  running=$(node_exec "$node_ip" "docker inspect -f '{{.State.Running}}' '$container'" 2>/dev/null || true)
  [[ $running == true ]]
}

show_container_logs() {
  local node_ip=$1
  local container=$2
  local log_dir="$PD_ROOT_DIR/results/startup-failures"
  local log_file="$log_dir/$(date +%Y%m%d_%H%M%S)-$node_ip-$container.log"
  mkdir -p "$log_dir"
  node_exec "$node_ip" "docker logs $(shell_quote "$container")" >"$log_file" 2>&1 || true
  echo "完整日志：$log_file；首先出现的异常：" >&2
  awk '/Scheduler hit an exception|RuntimeError:|ValueError:|AssertionError:|HCCL_BUFFSIZE.*SMALL/ {print; count++; if (count >= 12) exit}' "$log_file" >&2
  tail -30 "$log_file" >&2
}

require_running_container() {
  local node_ip=$1
  local container=$2
  if ! container_is_running "$node_ip" "$container"; then
    echo "节点 $node_ip 上的容器 $container 已退出或无法查询，最近日志如下：" >&2
    show_container_logs "$node_ip" "$container"
    return 1
  fi
}

shell_quote() {
  printf '%q' "$1"
}
