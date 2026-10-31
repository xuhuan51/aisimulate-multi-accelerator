#!/usr/bin/env bash
# 用途：为 Ascend 推理节点设置 CPU performance 模式及推荐的 NUMA/调度内核参数。
#
# 该脚本会修改当前系统配置，需要 root 权限；修改在重启后通常会恢复系统默认值。

set -Eeuo pipefail

[[ ${EUID:-$(id -u)} -eq 0 ]] || {
  echo "错误：系统调优需要 root 权限" >&2
  exit 1
}

if [[ ${ENABLE_SYSTEM_TUNING:-1} != 1 ]]; then
  echo "已通过 ENABLE_SYSTEM_TUNING=${ENABLE_SYSTEM_TUNING:-0} 跳过系统调优"
  exit 0
fi

# 某些虚拟机没有 cpufreq 节点；此时只跳过 governor，不影响 sysctl 设置。
shopt -s nullglob
governor_files=(/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor)
if ((${#governor_files[@]} > 0)); then
  printf 'performance\n' | tee "${governor_files[@]}" >/dev/null
  echo "CPU governor 已设置为 performance"
else
  echo "未发现 CPU governor 文件，跳过该项"
fi

sysctl -w vm.swappiness=0 >/dev/null
sysctl -w kernel.numa_balancing=0 >/dev/null
sysctl -w kernel.sched_migration_cost_ns=50000 >/dev/null
echo "NUMA 与调度参数已设置"
