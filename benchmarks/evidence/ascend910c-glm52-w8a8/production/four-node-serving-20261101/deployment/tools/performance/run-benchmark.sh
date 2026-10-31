#!/usr/bin/env bash
# 用途：通过当前 PD Router 运行 SGLang serving 性能测试，并把 JSONL 结果保存到宿主机。
#
# 示例：NUM_PROMPTS=100 MAX_CONCURRENCY=16 RANDOM_INPUT_LEN=4096 ./tools/performance/run-benchmark.sh
# 可在命令末尾继续传入当前 SGLang benchmark serving 支持的其他参数。

set -Eeuo pipefail

# shellcheck source=pd-common.sh
source "$(dirname "$0")/../../service/pd-common.sh"

require_command docker
require_command python3
require_controller_node
container_is_running "$ROUTER_IP" "$ROUTER_CONTAINER" || die "Router 容器未运行：$ROUTER_CONTAINER"

parse_prefill_node "${PREFILL_NODE_LIST[0]}" || die "无法解析 Prefill[0] 配置"
BENCH_CONTAINER=$(prefill_container_name 0)
container_is_running "$NODE_IP" "$BENCH_CONTAINER" || die "benchmark 执行容器未运行：$BENCH_CONTAINER"

NUM_PROMPTS=${NUM_PROMPTS:-300}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-16}
RANDOM_INPUT_LEN=${RANDOM_INPUT_LEN:-1024}
RANDOM_OUTPUT_LEN=${RANDOM_OUTPUT_LEN:-128}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-1.0}
REQUEST_RATE=${REQUEST_RATE:-inf}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-2}
DATASET_NAME=${DATASET_NAME:-random-ids}
BENCHMARK_SEED=${BENCHMARK_SEED:-910}
TOKENIZE_PROMPT=${TOKENIZE_PROMPT:-1}
ALLOW_PARTIAL_RESULTS=${ALLOW_PARTIAL_RESULTS:-0}
[[ $ALLOW_PARTIAL_RESULTS == 0 || $ALLOW_PARTIAL_RESULTS == 1 ]] || die "ALLOW_PARTIAL_RESULTS 必须是 0 或 1"
RESULT_DIR=${RESULT_DIR:-$PD_ROOT_DIR/results/performance/benchmark}

timestamp=$(date +%Y%m%d_%H%M%S)
result_name="bench_${timestamp}_c${MAX_CONCURRENCY}_$$.jsonl"
container_result="/tmp/$result_name"
host_result="$RESULT_DIR/$result_name"

mkdir -p "$RESULT_DIR"

benchmark_command=(
  python3 -m sglang.benchmark.serving
  --backend sglang
  --host "$ROUTER_IP"
  --port "$ROUTER_PORT"
  --model /models/glm-q
  --served-model-name "$SERVED_MODEL_NAME"
  --dataset-name "$DATASET_NAME"
  --num-prompts "$NUM_PROMPTS"
  --max-concurrency "$MAX_CONCURRENCY"
  --random-input-len "$RANDOM_INPUT_LEN"
  --random-output-len "$RANDOM_OUTPUT_LEN"
  --random-range-ratio "$RANDOM_RANGE_RATIO"
  --request-rate "$REQUEST_RATE"
  --warmup-requests "$WARMUP_REQUESTS"
  --seed "$BENCHMARK_SEED"
  --output-details
  --output-file "$container_result"
)
if [[ $DATASET_NAME == random-ids && $TOKENIZE_PROMPT == 1 ]]; then
  benchmark_command+=(--tokenize-prompt)
fi
benchmark_command+=("$@")

cleanup() {
  docker exec "$BENCH_CONTAINER" rm -f "$container_result" >/dev/null 2>&1 || true
}
trap cleanup EXIT

printf '执行基准测试（容器=%s，数据集=%s）：' "$BENCH_CONTAINER" "$DATASET_NAME"
printf '%q ' "${benchmark_command[@]}"
printf '\n'
docker exec -e PYTHONUNBUFFERED=1 "$BENCH_CONTAINER" "${benchmark_command[@]}"
docker cp "$BENCH_CONTAINER:$container_result" "$host_result" >/dev/null

completed_requests=$(python3 -c 'import json,sys; lines=[line for line in open(sys.argv[1], encoding="utf-8") if line.strip()]; print(json.loads(lines[-1]).get("completed", ""))' "$host_result")
[[ $completed_requests =~ ^[0-9]+$ ]] || die "结果中缺少有效 completed 字段：$host_result"
if ((completed_requests != NUM_PROMPTS)); then
  echo "错误：请求未全部完成：成功 $completed_requests / 计划 $NUM_PROMPTS；该结果不能作为完整性能数据" >&2
  [[ $ALLOW_PARTIAL_RESULTS == 1 ]] || exit 2
  echo "警告：ALLOW_PARTIAL_RESULTS=1，保留不完整结果"
fi

if [[ $DATASET_NAME == random-ids && $TOKENIZE_PROMPT == 1 ]]; then
  python3 - "$host_result" "$RANDOM_INPUT_LEN" "$RANDOM_OUTPUT_LEN" "$NUM_PROMPTS" "$RANDOM_RANGE_RATIO" <<'PY'
import json
import sys

path, input_length, output_length, requests, ratio = sys.argv[1:]
result = json.loads([line for line in open(path, encoding="utf-8") if line.strip()][-1])
if float(ratio) == 1.0:
    for key, expected in [("input_lens", int(input_length)), ("output_lens", int(output_length))]:
        values = result.get(key, [])
        if len(values) != int(requests) or any(value != expected for value in values):
            raise SystemExit(f"固定长度校验失败：{key}，期望 {requests} 条 × {expected}，实际长度集合 {sorted(set(values))}")
print("逐请求 token 长度校验通过")
PY
fi

echo "基准测试结果：$host_result"
