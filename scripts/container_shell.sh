#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${PROJECT_ROOT}/upstream/sglang"
CONTAINER_IMAGE="$(awk -F= '$1 == "image" {print $2}' "${PROJECT_ROOT}/CONTAINER.lock")"
CONTAINER_DIGEST="$(awk -F= '$1 == "digest" {print $2}' "${PROJECT_ROOT}/CONTAINER.lock")"

if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
  echo "Missing ${PROJECT_ROOT}/.env; copy configs/host.env.example and edit model paths." >&2
  exit 1
fi
if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  echo "Missing pinned SGLang checkout; run scripts/bootstrap_remote.sh first." >&2
  exit 1
fi

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

exec docker run --rm "${TTY_ARGS[@]}" \
  --gpus all \
  --ipc=host \
  --network=host \
  --shm-size=32g \
  --env-file "${PROJECT_ROOT}/.env" \
  -e PYTHONPATH=/workspace/sglang/python:/workspace/agentperf/src \
  -v "${PROJECT_ROOT}:/workspace/agentperf" \
  -v "${UPSTREAM_DIR}:/workspace/sglang" \
  -v /data:/data \
  -w /workspace/agentperf \
  "${CONTAINER_IMAGE}@${CONTAINER_DIGEST}" \
  "$@"
