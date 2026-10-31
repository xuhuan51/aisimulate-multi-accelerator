#!/usr/bin/env bash
# 用途：在线检查多 P/多 D 所有节点的 SSH、root、模型、镜像、网卡、驱动和 NPU 数量。

set -Eeuo pipefail

# shellcheck source=../service/pd-common.sh
source "$(dirname "$0")/../service/pd-common.sh"

for command_name in docker ssh ip awk; do
  require_command "$command_name"
done
require_controller_node
"$PD_ROOT_DIR/check/check-config.sh"
prepare_remote_access

mapfile -t worker_ips < <(all_worker_ips | awk '!seen[$0]++')

check_local_node() {
  local node_ip=$1
  local device_count

  [[ $EUID -eq 0 ]] || die "控制节点需要以 root 执行环境检查"
  [[ -f $MODEL_DIR/config.json ]] || die "$node_ip 缺少模型文件：$MODEL_DIR/config.json"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || die "$node_ip 缺少镜像：$IMAGE"
  ip link show "$COMM_IFNAME" >/dev/null 2>&1 || die "$node_ip 缺少通信网卡：$COMM_IFNAME"
  [[ -d /usr/local/Ascend/driver ]] || die "$node_ip 缺少 Ascend 驱动目录"
  device_count=$(find /dev -maxdepth 1 -type c -name 'davinci[0-9]*' | wc -l)
  ((device_count >= NPU_DEVICE_COUNT)) || die "$node_ip 仅发现 $device_count 个 NPU 设备，需要 $NPU_DEVICE_COUNT 个"
  echo "$node_ip：环境检查通过，NPU=$device_count"
}

check_remote_node() {
  local node_ip=$1
  local command_text
  printf -v command_text '%s' \
    "set -Eeuo pipefail; " \
    "[[ \$(id -u) -eq 0 ]] || { echo '必须使用 root' >&2; exit 1; }; " \
    "test -f $(shell_quote "$MODEL_DIR/config.json") || { echo '缺少模型文件' >&2; exit 1; }; " \
    "docker image inspect $(shell_quote "$IMAGE") >/dev/null 2>&1 || { echo '缺少镜像' >&2; exit 1; }; " \
    "ip link show $(shell_quote "$COMM_IFNAME") >/dev/null 2>&1 || { echo '缺少通信网卡' >&2; exit 1; }; " \
    "test -d /usr/local/Ascend/driver || { echo '缺少 Ascend 驱动目录' >&2; exit 1; }; " \
    "count=\$(find /dev -maxdepth 1 -type c -name 'davinci[0-9]*' | wc -l); " \
    "((count >= $NPU_DEVICE_COUNT)) || { echo \"NPU 数量不足：\$count/$NPU_DEVICE_COUNT\" >&2; exit 1; }; " \
    "echo '$node_ip：环境检查通过，NPU='\$count"
  node_exec "$node_ip" "$command_text"
}

for node_ip in "${worker_ips[@]}"; do
  echo "===== 节点 $node_ip ====="
  if local_has_ip "$node_ip"; then
    check_local_node "$node_ip"
  else
    check_remote_node "$node_ip"
  fi
done

echo "全部 ${#worker_ips[@]} 个 Worker 节点环境检查通过"
