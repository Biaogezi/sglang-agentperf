#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_COMMIT="$(awk -F= '$1 == "commit" {print $2}' "${PROJECT_ROOT}/UPSTREAM.lock")"
CONTAINER_IMAGE="$(awk -F= '$1 == "image" {print $2}' "${PROJECT_ROOT}/CONTAINER.lock")"
CONTAINER_MIRROR_IMAGE="$(awk -F= '$1 == "mirror_image" {print $2}' "${PROJECT_ROOT}/CONTAINER.lock")"
CONTAINER_DIGEST="$(awk -F= '$1 == "digest" {print $2}' "${PROJECT_ROOT}/CONTAINER.lock")"

fail() {
  echo "PRECHECK FAILED: $*" >&2
  exit 1
}

command -v git >/dev/null 2>&1 || fail "git is missing"
command -v docker >/dev/null 2>&1 || fail "docker is missing"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is missing"
[[ -d /data ]] || fail "/data is not mounted"

FREE_GIB="$(df -BG --output=avail /data | tail -1 | tr -dc '0-9')"
[[ "${FREE_GIB}" -ge 80 ]] || fail "/data has only ${FREE_GIB} GiB free; at least 80 GiB is required"

ACTUAL_COMMIT="$(git -C "${PROJECT_ROOT}/upstream/sglang" rev-parse HEAD)"
[[ "${ACTUAL_COMMIT}" == "${UPSTREAM_COMMIT}" ]] || \
  fail "SGLang commit ${ACTUAL_COMMIT} does not match ${UPSTREAM_COMMIT}"

RUNTIME_IMAGE="${CONTAINER_IMAGE}@${CONTAINER_DIGEST}"
if ! docker image inspect "${RUNTIME_IMAGE}" >/dev/null 2>&1; then
  RUNTIME_IMAGE="${CONTAINER_MIRROR_IMAGE}@${CONTAINER_DIGEST}"
fi
docker image inspect "${RUNTIME_IMAGE}" >/dev/null 2>&1 || \
  fail "Pinned SGLang image is not present; run bootstrap_remote.sh"

docker run --rm --gpus all \
  "${RUNTIME_IMAGE}" \
  python -c 'import importlib.metadata as m; import torch, triton; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), m.version("sglang"), triton.__version__)'

nvidia-smi --query-gpu=name,driver_version,memory.total,temperature.gpu,pstate \
  --format=csv,noheader
echo "PRECHECK PASSED: ${FREE_GIB} GiB free on /data; upstream ${UPSTREAM_COMMIT}"
