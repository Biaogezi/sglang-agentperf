#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 MODEL_PATH OUTPUT_PREFIX [extra launch_server args...]" >&2
  exit 2
fi

MODEL_PATH="$1"
OUTPUT_PREFIX="$2"
shift 2

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys is not installed" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_PREFIX}")"
nsys profile \
  --trace=cuda,nvtx,osrt \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  -o "${OUTPUT_PREFIX}" \
  python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 127.0.0.1 --port 30000 \
    --enable-layerwise-nvtx-marker \
    --disable-cuda-graph \
    "$@"

