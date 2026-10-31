#!/usr/bin/env bash
# 用途：在指定 Prefill 节点创建并后台启动一个 GLM-5.2 Prefill Ascend 容器。

set -Eeuo pipefail

# shellcheck source=pd-common.sh
source "$(dirname "$0")/pd-common.sh"
# shellcheck source=container-common.sh
source "$PD_SCRIPT_DIR/container-common.sh"

: "${WORKER_IP:?必须设置 WORKER_IP}"
: "${WORKER_PORT:?必须设置 WORKER_PORT}"
: "${WORKER_BOOTSTRAP_PORT:?必须设置 WORKER_BOOTSTRAP_PORT}"
: "${WORKER_CONTAINER:?必须设置 WORKER_CONTAINER}"

require_command docker
local_has_ip "$WORKER_IP" || die "当前节点不包含 Prefill IP：$WORKER_IP"

docker rm -f "$WORKER_CONTAINER" >/dev/null 2>&1 || true
build_ascend_docker_args "$WORKER_CONTAINER" /apps/run-prefill.sh

echo "启动 Prefill 容器：$WORKER_CONTAINER ($WORKER_IP:$WORKER_PORT)"
docker "${DOCKER_ARGS[@]}" >/dev/null
