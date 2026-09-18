# SGLang code map for the first optimization cycle

Pinned source: `v0.5.19` / `0bcd822377da7b5718e674eaf9c870d349424dd1`.

This map contains the original investigation branches, not a claim that all branches became
accepted optimizations. The selected runtime change is Candidate B's guarded INT8 GEMM.
The inspectable source additions live in [patch 0004](../patches/0004-feat-quant-add-opt-in-SM86-short-prefill-INT8-GEMM.patch),
with bounded JIT refinement in [patch 0005](../patches/0005-perf-quant-bound-prefill-JIT-variants-and-remove-ful.patch).
Bootstrap applies the complete locked series; `upstream/sglang/` itself is intentionally not
vendored into this repository. See [the final report](FINAL_REPORT.zh-CN.md) for actual coverage.

## Measurement path

| Concern | Primary source | Initial inspection target |
|---|---|---|
| Online metrics and workload generation | `python/sglang/benchmark/serving.py` | JSONL schema, shared-prefix dataset, warmup and cache reporting |
| Profiling control | `python/sglang/profiler.py` | live capture arguments and trace output |
| Profile lifecycle | `python/sglang/srt/managers/scheduler_components/profiler_manager.py` | start/stop step boundaries and overhead |
| Step annotations | `python/sglang/srt/model_executor/step_span_utils.py` | batch, query and KV-length metadata |
| Profile helpers | `python/sglang/srt/utils/profile_utils.py` | activities, trace export and NVTX integration |

## Candidate A — CUDA Graph bucket efficiency

| Concern | Primary source |
|---|---|
| Decode capture/replay | `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` |
| Prefill capture/replay | `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` |
| Bucket argument normalization | `python/sglang/srt/arg_groups/cuda_graph_hook.py` |
| Cross-argument validation | `python/sglang/srt/arg_groups/validation_hook.py` |
| User-facing configuration | `python/sglang/srt/server_args.py` |

First measurements: actual batch-size histogram, selected capture bucket, padded rows, eager fallback,
captured-graph HBM, and replay time. Do not change default buckets until these distributions exist.

## Candidate B — quantized shape-aware dispatch

| Concern | Primary source |
|---|---|
| Quantization registry | `python/sglang/srt/layers/quantization/__init__.py` |
| AWQ schemes | `python/sglang/srt/layers/quantization/awq/` |
| GPTQ schemes | `python/sglang/srt/layers/quantization/gptq/` |
| W8A8 path | `python/sglang/srt/layers/quantization/w8a8_int8.py` |
| Compressed-tensors W8A8 path | `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py` |
| Custom guarded GEMM | `python/sglang/kernels/ops/quantization/int8_prefill_gemm.py` |
| Fused normalization + activation quantization | `python/sglang/kernels/ops/quantization/int8_kernel.py` and `python/sglang/srt/layers/layernorm.py` |
| Qwen3 opt-in TP=1 handoff | `python/sglang/srt/models/qwen3.py` |
| NVIDIA AWQ dispatch | `python/sglang/srt/hardware_backend/gpu/quantization/awq_kernels.py` |
| NVIDIA GPTQ dispatch | `python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py` |
| Marlin utilities | `python/sglang/srt/layers/quantization/marlin_utils.py` |

First measurements: every hot GEMM shape in prefill and decode, selected implementation, repack cost,
workspace cost, kernel time and any surrounding cast/dequant kernels. Dispatch overhead must be below
the recovered kernel time.

## Candidate C — SLO-aware chunked prefill

| Concern | Primary source |
|---|---|
| Main scheduling loop | `python/sglang/srt/managers/scheduler.py` |
| Batch construction and states | `python/sglang/srt/managers/schedule_batch.py` |
| Scheduling policies | `python/sglang/srt/managers/schedule_policy.py` |
| Scheduler metrics | `python/sglang/srt/managers/scheduler_components/metrics_reporter.py` |
| Runtime arguments and validation | `python/sglang/srt/server_args.py` and `python/sglang/srt/arg_groups/` |

First measurements: decode step delay during long prefill, queue depth, chunk sizes, running request
count, token usage, TTFT and TPOT distributions. The controller must be compared with existing static
flags and dynamic chunking, not only with an intentionally poor configuration.

## Candidate D — CPU/GPU overlap

Start from gaps in the Nsight timeline, then follow the corresponding scheduler/model-runner region.
Likely entry points include `scheduler.py`, `schedule_batch.py`, tokenizer management, and model runner
metadata preparation. No broad refactor is allowed without a trace showing the blocking region.

## Test boundary

Every runtime patch starts with the narrow unit tests colocated with the touched subsystem, then
`bench_one_batch` for phase isolation, and finally the online suite in this repository. A local
microbenchmark is explanatory evidence only; the acceptance gate remains online serving behavior.
