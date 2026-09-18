#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${PROJECT_ROOT}/upstream/sglang"
UPSTREAM_REPO="$(awk -F= '$1 == "repository" {print $2}' "${PROJECT_ROOT}/UPSTREAM.lock")"
UPSTREAM_REF="$(awk -F= '$1 == "ref" {print $2}' "${PROJECT_ROOT}/UPSTREAM.lock")"
UPSTREAM_COMMIT="$(awk -F= '$1 == "commit" {print $2}' "${PROJECT_ROOT}/UPSTREAM.lock")"
UPSTREAM_PATCHES="$(awk -F= '$1 == "patches" {print $2}' "${PROJECT_ROOT}/UPSTREAM.lock")"
UPSTREAM_PATCHED_TREE="$(awk -F= '$1 == "patched_tree" {print $2}' "${PROJECT_ROOT}/UPSTREAM.lock")"
CONTAINER_IMAGE="$(awk -F= '$1 == "image" {print $2}' "${PROJECT_ROOT}/CONTAINER.lock")"
CONTAINER_MIRROR_IMAGE="$(awk -F= '$1 == "mirror_image" {print $2}' "${PROJECT_ROOT}/CONTAINER.lock")"
CONTAINER_DIGEST="$(awk -F= '$1 == "digest" {print $2}' "${PROJECT_ROOT}/CONTAINER.lock")"

python3 -m venv "${PROJECT_ROOT}/.venv"
source "${PROJECT_ROOT}/.venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${PROJECT_ROOT}[dev]"

mkdir -p "${PROJECT_ROOT}/upstream"
if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  git clone --filter=blob:none --depth 1 --branch "${UPSTREAM_REF}" \
    "${UPSTREAM_REPO}" "${UPSTREAM_DIR}"
  git -C "${UPSTREAM_DIR}" checkout --detach "${UPSTREAM_COMMIT}"
fi
git -C "${UPSTREAM_DIR}" diff --quiet && git -C "${UPSTREAM_DIR}" diff --cached --quiet || {
  echo "SGLang has local edits; refusing to switch or patch this checkout." >&2
  exit 1
}
CURRENT_TREE="$(git -C "${UPSTREAM_DIR}" rev-parse 'HEAD^{tree}')"
if [[ "${CURRENT_TREE}" != "${UPSTREAM_PATCHED_TREE}" ]]; then
  CURRENT_COMMIT="$(git -C "${UPSTREAM_DIR}" rev-parse HEAD)"
  if [[ "${CURRENT_COMMIT}" != "${UPSTREAM_COMMIT}" ]]; then
    echo "Existing SGLang checkout is neither the pinned base nor patched tree; preserve it and use a fresh directory." >&2
    exit 1
  fi
  # Verify content, not commit IDs: git am gives commits new committer timestamps.
  read -r -a PATCH_FILES <<< "${UPSTREAM_PATCHES}"
  for PATCH_FILE in "${PATCH_FILES[@]}"; do
    git -C "${UPSTREAM_DIR}" -c user.name=AgentPerf -c user.email=agentperf@localhost \
      am "${PROJECT_ROOT}/${PATCH_FILE}"
  done
  ACTUAL_PATCHED_TREE="$(git -C "${UPSTREAM_DIR}" rev-parse 'HEAD^{tree}')"
  [[ "${ACTUAL_PATCHED_TREE}" == "${UPSTREAM_PATCHED_TREE}" ]] || {
    echo "Patched source tree ${ACTUAL_PATCHED_TREE} does not match ${UPSTREAM_PATCHED_TREE}." >&2
    exit 1
  }
fi

if command -v docker >/dev/null 2>&1; then
  docker pull "${CONTAINER_IMAGE}@${CONTAINER_DIGEST}" || \
    docker pull "${CONTAINER_MIRROR_IMAGE}@${CONTAINER_DIGEST}"
else
  echo "Docker is required for the pinned GPU environment." >&2
  exit 1
fi

bash "${PROJECT_ROOT}/scripts/collect_system_info.sh" \
  "${PROJECT_ROOT}/results/bootstrap-environment.txt"
python -m agentperf.cli validate --config "${PROJECT_ROOT}/configs/experiment_matrix.json"

echo "Bootstrap complete. Configure model paths in .env, then run scripts/run_gpu_smoke.sh."
