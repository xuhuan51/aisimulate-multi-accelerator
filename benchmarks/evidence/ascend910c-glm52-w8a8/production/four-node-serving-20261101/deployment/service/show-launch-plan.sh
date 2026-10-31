#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/pd-common.sh"
"$PD_ROOT_DIR/check/check-config.sh"
for index in "${!PREFILL_NODE_LIST[@]}"; do
  parse_prefill_node "${PREFILL_NODE_LIST[$index]}"
  DRY_RUN=1 WORKER_IP=$NODE_IP WORKER_PORT=$NODE_PORT WORKER_BOOTSTRAP_PORT=$NODE_BOOTSTRAP \
    "$PD_SCRIPT_DIR/run-prefill.sh"
done
for index in "${!DECODE_NODE_LIST[@]}"; do
  parse_decode_node "${DECODE_NODE_LIST[$index]}"
  decode_distribution "$index"
  DRY_RUN=1 WORKER_IP=$NODE_IP WORKER_PORT=$NODE_PORT WORKER_NNODES=$WORKER_NNODES \
    WORKER_NODE_RANK=$WORKER_NODE_RANK WORKER_DIST_INIT_ADDR=$WORKER_DIST_INIT_ADDR \
    "$PD_SCRIPT_DIR/run-decode.sh"
done
