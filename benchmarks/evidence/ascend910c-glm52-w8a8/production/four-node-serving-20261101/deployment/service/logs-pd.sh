#!/usr/bin/env bash
# 用途：按角色和索引查看多 P/多 D 的 Worker 或 Router 容器日志。
#
# 用法：
#   ./service/logs-pd.sh prefill 0
#   ./service/logs-pd.sh decode 1 --tail 200
#   ./service/logs-pd.sh router
#   ./service/logs-pd.sh list

set -Eeuo pipefail

# shellcheck source=pd-common.sh
source "$(dirname "$0")/pd-common.sh"

role=${1:-}
[[ -n $role ]] || die "用法：$0 prefill|decode <索引> [docker logs 参数]，或 router/list"
shift

if [[ $role == list ]]; then
  for index in "${!PREFILL_NODE_LIST[@]}"; do
    parse_prefill_node "${PREFILL_NODE_LIST[$index]}"
    printf 'prefill %s -> %s:%s (%s)\n' "$index" "$NODE_IP" "$NODE_PORT" "$(prefill_container_name "$index")"
  done
  for index in "${!DECODE_NODE_LIST[@]}"; do
    parse_decode_node "${DECODE_NODE_LIST[$index]}"
    printf 'decode  %s -> %s:%s (%s)\n' "$index" "$NODE_IP" "$NODE_PORT" "$(decode_container_name "$index")"
  done
  printf 'router    -> %s:%s (%s)\n' "$ROUTER_IP" "$ROUTER_PORT" "$ROUTER_CONTAINER"
  exit 0
fi

if [[ $role == router ]]; then
  (($# > 0)) || set -- -f --tail 100
  exec docker logs "$@" "$ROUTER_CONTAINER"
fi

index=${1:-0}
[[ $index =~ ^[0-9]+$ ]] || die "Worker 索引必须是非负整数：$index"
(($# > 0)) && shift
(($# > 0)) || set -- -f --tail 100

case "$role" in
  prefill)
    ((index < ${#PREFILL_NODE_LIST[@]})) || die "Prefill 索引越界：$index"
    parse_prefill_node "${PREFILL_NODE_LIST[$index]}"
    container=$(prefill_container_name "$index")
    ;;
  decode)
    ((index < ${#DECODE_NODE_LIST[@]})) || die "Decode 索引越界：$index"
    parse_decode_node "${DECODE_NODE_LIST[$index]}"
    container=$(decode_container_name "$index")
    ;;
  *)
    die "未知角色：$role；可选 prefill、decode、router、list"
    ;;
esac

printf -v log_args '%q ' "$@"
prepare_remote_access
node_exec "$NODE_IP" "docker logs $log_args $(shell_quote "$container")"
