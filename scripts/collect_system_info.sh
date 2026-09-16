#!/usr/bin/env bash
set -euo pipefail

OUTPUT_PATH="${1:-results/environment.txt}"
mkdir -p "$(dirname "${OUTPUT_PATH}")"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "kernel=$(uname -a)"
  echo "python=$(python --version 2>&1)"
  echo "git=$(git --version)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,pstate,clocks.max.sm,power.limit \
      --format=csv,noheader
  else
    echo "nvidia-smi=missing"
  fi
  python - <<'PY'
try:
    import torch
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
except Exception as exc:
    print(f"torch_probe_error={exc!r}")
PY
  python -m pip freeze
} > "${OUTPUT_PATH}"

echo "Wrote ${OUTPUT_PATH}"

