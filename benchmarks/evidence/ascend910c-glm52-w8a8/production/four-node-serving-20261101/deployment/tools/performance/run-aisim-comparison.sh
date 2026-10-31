#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../service/pd-common.sh"
CONCURRENCIES=${CONCURRENCIES:-"4 8 16 32 64 128"}
PROMPT_MULTIPLIER=${PROMPT_MULTIPLIER:-2}
CONFIG_LABEL=${CONFIG_LABEL:-recommended-2p1d-dtp32}
RUN_DIR=${RUN_DIR:-$PD_ROOT_DIR/results/aisim-comparison/${CONFIG_LABEL}_$(date +%Y%m%d_%H%M%S)}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-2}
BENCHMARK_SEED=${BENCHMARK_SEED:-910}
RESUME=${RESUME:-1}
FIXED_NUM_PROMPTS=${FIXED_NUM_PROMPTS:-}
[[ $PROMPT_MULTIPLIER =~ ^[1-9][0-9]*$ ]] || die "PROMPT_MULTIPLIER 必须是正整数"
mkdir -p "$RUN_DIR"
exec 8>"$PD_ROOT_DIR/.aisim-benchmark.lock"
flock -n 8 || die "已有对照采集正在运行"
cp "$PD_SCRIPT_DIR/pd-config.sh" "$RUN_DIR/pd-config.sh"
cp "$PD_SCRIPT_DIR/run-prefill.sh" "$RUN_DIR/run-prefill.sh"
cp "$PD_SCRIPT_DIR/run-decode.sh" "$RUN_DIR/run-decode.sh"
endpoints=()
for index in "${!PREFILL_NODE_LIST[@]}"; do
  parse_prefill_node "${PREFILL_NODE_LIST[$index]}"
  endpoints+=(--endpoint "P$index=http://$NODE_IP:$NODE_PORT")
done
for index in "${!DECODE_NODE_LIST[@]}"; do
  parse_decode_node "${DECODE_NODE_LIST[$index]}"
  endpoints+=(--endpoint "D$((index / DECODE_NNODES))=http://$NODE_IP:$NODE_PORT")
done
printf 'CONFIG_LABEL=%s\nCONCURRENCIES=%s\nPROMPT_MULTIPLIER=%s\nFIXED_NUM_PROMPTS=%s\nWARMUP_REQUESTS=%s\nBENCHMARK_SEED=%s\nINPUT_TOKENS=8192\nOUTPUT_TOKENS=1024\n' \
  "$CONFIG_LABEL" "$CONCURRENCIES" "$PROMPT_MULTIPLIER" "$FIXED_NUM_PROMPTS" "$WARMUP_REQUESTS" "$BENCHMARK_SEED" >"$RUN_DIR/protocol.env"
for concurrency in $CONCURRENCIES; do
  [[ $concurrency =~ ^[1-9][0-9]*$ ]] || die "非法并发：$concurrency"
  num_prompts=${FIXED_NUM_PROMPTS:-$((concurrency * PROMPT_MULTIPLIER))}
  point_dir="$RUN_DIR/c${concurrency}-n${num_prompts}"
  if [[ $RESUME == 1 && -f $point_dir/.complete ]]; then
    continue
  fi
  mkdir -p "$point_dir"
  printf 'NUM_PROMPTS=%s\nMAX_CONCURRENCY=%s\nWARMUP_REQUESTS=%s\n' "$num_prompts" "$concurrency" "$WARMUP_REQUESTS" >"$point_dir/config.env"
  SNAPSHOT_OUTPUT="$point_dir" python3 "$SCRIPT_DIR/snapshot-serving-config.py" "${endpoints[@]}"
  echo "开始 $CONFIG_LABEL：8K/1K，并发=$concurrency，请求=$num_prompts，预热=$WARMUP_REQUESTS"
  NUM_PROMPTS=$num_prompts MAX_CONCURRENCY=$concurrency \
    RANDOM_INPUT_LEN=8192 RANDOM_OUTPUT_LEN=1024 RANDOM_RANGE_RATIO=1 \
    WARMUP_REQUESTS=$WARMUP_REQUESTS BENCHMARK_SEED=$BENCHMARK_SEED \
    TOKENIZE_PROMPT=1 DATASET_NAME=random-ids RESULT_DIR="$point_dir" \
    python3 "$SCRIPT_DIR/sample-serving-metrics.py" --output "$point_dir" "${endpoints[@]}" \
    -- "$SCRIPT_DIR/run-benchmark.sh" --disable-tqdm
  touch "$point_dir/.complete"
  python3 "$SCRIPT_DIR/recompute-serving-metrics.py" "$RUN_DIR"
done
echo "采集完成：$RUN_DIR"
