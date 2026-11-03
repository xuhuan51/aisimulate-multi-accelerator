#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
image_name=${AISIMULATE_910C_IMAGE:-aisimulate-910c:glm52-w8a8}
model_dir=${AISIMULATE_910C_MODEL:-/mnt/raid/user/data/models/glm-q}
runtime_uid=${SUDO_UID:-$(id -u)}
runtime_gid=${SUDO_GID:-$(id -g)}
runtime_cpuset=${AISIMULATE_910C_CPUSET:-$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:16])))')}

if [[ ! -f "$model_dir/config.json" || ! -f "$model_dir/quant_model_description.json" ]]; then
    printf 'Missing GLM ModelSlim metadata in %s\n' "$model_dir" >&2
    exit 1
fi
if ! docker image inspect "$image_name" >/dev/null 2>&1; then
    printf 'Runtime image %s is unavailable; see md/910c-glm52-w8a8-usage.md\n' "$image_name" >&2
    exit 1
fi

exec docker run --rm --network none \
    --tmpfs /tmp:rw,nosuid,nodev,size=1g,mode=1777 \
    --user "$runtime_uid:$runtime_gid" \
    --cpus "${AISIMULATE_910C_CPUS:-16}" \
    --cpuset-cpus "$runtime_cpuset" \
    --memory "${AISIMULATE_910C_MEMORY:-32g}" \
    --mount "type=bind,src=$repo_dir,dst=$repo_dir" \
    --mount "type=bind,src=$model_dir,dst=$model_dir,readonly" \
    --workdir "$repo_dir" \
    "$image_name" "$@"
