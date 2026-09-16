#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "${PROJECT_ROOT}/scripts/preflight_gpu.sh"

exec bash "${PROJECT_ROOT}/scripts/container_shell.sh" \
  python -m agentperf.cli run \
    --config configs/experiment_matrix.json \
    --model qwen3_8b_fp16 \
    --profile baseline \
    --suite smoke \
    --output-root results
