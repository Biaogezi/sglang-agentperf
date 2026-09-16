#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${PROJECT_ROOT}/upstream/sglang"
UPSTREAM_REPO="$(awk -F= '$1 == "repository" {print $2}' "${PROJECT_ROOT}/UPSTREAM.lock")"
UPSTREAM_REF="$(awk -F= '$1 == "ref" {print $2}' "${PROJECT_ROOT}/UPSTREAM.lock")"
UPSTREAM_COMMIT="$(awk -F= '$1 == "commit" {print $2}' "${PROJECT_ROOT}/UPSTREAM.lock")"
CONTAINER_IMAGE="$(awk -F= '$1 == "image" {print $2}' "${PROJECT_ROOT}/CONTAINER.lock")"
CONTAINER_DIGEST="$(awk -F= '$1 == "digest" {print $2}' "${PROJECT_ROOT}/CONTAINER.lock")"

python3 -m venv "${PROJECT_ROOT}/.venv"
source "${PROJECT_ROOT}/.venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${PROJECT_ROOT}[dev]"

mkdir -p "${PROJECT_ROOT}/upstream"
if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  git clone --filter=blob:none "${UPSTREAM_REPO}" "${UPSTREAM_DIR}"
fi
git -C "${UPSTREAM_DIR}" fetch --depth 1 origin "${UPSTREAM_REF}"
git -C "${UPSTREAM_DIR}" checkout --detach "${UPSTREAM_COMMIT}"

if command -v docker >/dev/null 2>&1; then
  docker pull "${CONTAINER_IMAGE}@${CONTAINER_DIGEST}"
else
  echo "Docker is required for the pinned GPU environment." >&2
  exit 1
fi

bash "${PROJECT_ROOT}/scripts/collect_system_info.sh" \
  "${PROJECT_ROOT}/results/bootstrap-environment.txt"
python -m agentperf.cli validate --config "${PROJECT_ROOT}/configs/experiment_matrix.json"

echo "Bootstrap complete. Configure model paths in .env, then run scripts/run_gpu_smoke.sh."
