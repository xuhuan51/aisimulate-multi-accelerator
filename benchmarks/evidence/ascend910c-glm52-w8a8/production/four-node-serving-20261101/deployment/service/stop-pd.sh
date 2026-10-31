#!/usr/bin/env bash
# 用途：从控制节点停止并删除多 P/多 D 的 Router、全部 Prefill 和全部 Decode 容器。

set -Eeuo pipefail

# shellcheck source=pd-common.sh
source "$(dirname "$0")/pd-common.sh"

require_command docker
require_command ssh
require_controller_node
"$PD_ROOT_DIR/check/check-config.sh"
prepare_remote_access

docker rm -f "$ROUTER_CONTAINER" >/dev/null 2>&1 || true

for index in "${!PREFILL_NODE_LIST[@]}"; do
  parse_prefill_node "${PREFILL_NODE_LIST[$index]}"
  container=$(prefill_container_name "$index")
  node_exec "$NODE_IP" "docker rm -f $(shell_quote "$container") $(shell_quote "$PREFILL_CONTAINER_PREFIX") >/dev/null 2>&1 || true"
done

for index in "${!DECODE_NODE_LIST[@]}"; do
  parse_decode_node "${DECODE_NODE_LIST[$index]}"
  container=$(decode_container_name "$index")
  node_exec "$NODE_IP" "docker rm -f $(shell_quote "$container") $(shell_quote "$DECODE_CONTAINER_PREFIX") >/dev/null 2>&1 || true"
done

printf '%dP%dD 服务已停止\n' "${#PREFILL_NODE_LIST[@]}" "$((${#DECODE_NODE_LIST[@]} / DECODE_NNODES))"
