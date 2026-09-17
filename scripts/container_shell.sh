#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${PROJECT_ROOT}/upstream/sglang"
CONTAINER_IMAGE="$(awk -F= '$1 == "image" {print $2}' "${PROJECT_ROOT}/CONTAINER.lock")"
CONTAINER_MIRROR_IMAGE="$(awk -F= '$1 == "mirror_image" {print $2}' "${PROJECT_ROOT}/CONTAINER.lock")"
CONTAINER_DIGEST="$(awk -F= '$1 == "digest" {print $2}' "${PROJECT_ROOT}/CONTAINER.lock")"

if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
  echo "Missing ${PROJECT_ROOT}/.env; copy configs/host.env.example and edit model paths." >&2
  exit 1
fi
ENV_CACHE_DIR="$(awk -F= '$1 == "AGENTPERF_CACHE_DIR" {print substr($0, index($0, "=") + 1)}' "${PROJECT_ROOT}/.env")"
AGENTPERF_CACHE_DIR="${AGENTPERF_CACHE_DIR:-${ENV_CACHE_DIR:-/data/agentperf-cache}}"
if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  echo "Missing pinned SGLang checkout; run scripts/bootstrap_remote.sh first." >&2
  exit 1
fi
mkdir -p "${AGENTPERF_CACHE_DIR}"

RUNTIME_IMAGE="${CONTAINER_IMAGE}@${CONTAINER_DIGEST}"
if ! docker image inspect "${RUNTIME_IMAGE}" >/dev/null 2>&1; then
  RUNTIME_IMAGE="${CONTAINER_MIRROR_IMAGE}@${CONTAINER_DIGEST}"
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
  -v "${AGENTPERF_CACHE_DIR}:/root/.cache" \
  -v "${PROJECT_ROOT}:/workspace/agentperf" \
  -v "${UPSTREAM_DIR}:/workspace/sglang" \
  -v /data:/data \
  -w /workspace/agentperf \
  "${RUNTIME_IMAGE}" \
  "$@"
