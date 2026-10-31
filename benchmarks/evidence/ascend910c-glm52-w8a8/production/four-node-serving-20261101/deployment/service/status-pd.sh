#!/usr/bin/env bash
# 用途：查看多 P/多 D 所有 Worker 与 Router 的容器状态和 HTTP 健康状态。

set -uo pipefail

# shellcheck source=pd-common.sh
source "$(dirname "$0")/pd-common.sh"

http_status() {
  local code
  code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 "$1" 2>/dev/null || true)
  printf '%s' "${code:-000}"
}

"$PD_ROOT_DIR/check/check-config.sh" || exit 1
prepare_remote_access

for index in "${!PREFILL_NODE_LIST[@]}"; do
  parse_prefill_node "${PREFILL_NODE_LIST[$index]}"
  container=$(prefill_container_name "$index")
  echo "===== Prefill[$index] $NODE_IP:$NODE_PORT ====="
  node_exec "$NODE_IP" "docker ps -a --filter name=$(shell_quote "$container") --format 'table {{.Names}}\\t{{.Status}}\\t{{.Image}}'" || true
  printf 'HTTP: %s\n\n' "$(http_status "http://$NODE_IP:$NODE_PORT/health")"
done

for index in "${!DECODE_NODE_LIST[@]}"; do
  parse_decode_node "${DECODE_NODE_LIST[$index]}"
  container=$(decode_container_name "$index")
  echo "===== Decode[$((index / DECODE_NNODES))] node-rank=$((index % DECODE_NNODES)) $NODE_IP:$NODE_PORT ====="
  node_exec "$NODE_IP" "docker ps -a --filter name=$(shell_quote "$container") --format 'table {{.Names}}\\t{{.Status}}\\t{{.Image}}'" || true
  printf 'HTTP: %s\n\n' "$(http_status "http://$NODE_IP:$NODE_PORT/health")"
done

echo "===== Router $ROUTER_IP:$ROUTER_PORT ====="
docker ps -a --filter "name=$ROUTER_CONTAINER" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
printf 'HTTP: %s\n' "$(http_status "http://$ROUTER_IP:$ROUTER_PORT/health")"
