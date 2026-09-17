# Optimization 003 — W8A8 RMSNorm + activation quantization fusion

Status: **rejected for serving; kernel result retained**.

## Hypothesis

The calibrated W8A8 path runs RMSNorm and dynamic per-token INT8 activation quantization before
QKV and gate/up projections. A Triton kernel was added to combine residual addition, RMSNorm,
absmax reduction, scale calculation, and INT8 conversion in one launch. The downstream
`CompressedTensorsW8A8Int8` linear accepts the pre-quantized tensor/scale tuple, so the original
quantization launch and intermediate FP16 write/read are removed.

The implementation is default-off behind `SGLANG_W8A8_FUSED_RMSNORM_QUANT`, limited to TP=1 and
dynamic symmetric per-token W8A8, and published as
[`patches/0003-feat-quant-fuse-RMSNorm-with-dynamic-INT8-quantizati.patch`](../patches/0003-feat-quant-fuse-RMSNorm-with-dynamic-INT8-quantizati.patch).

## Correctness

- CUDA unit tests cover FP16 and BF16, with and without the residual path. Quantized bytes and
  per-token scales match the existing `sgl_kernel` RMSNorm + `per_token_quant_int8` path; the
  residual buffer also matches exactly (`2 passed`, one unrelated test deselected).
- The fixed 335-token prompt-NLL corpus is bit-for-bit identical at the aggregate and per-document
  level: mean NLL `2.8919478789`, perplexity `18.02839255`, delta `0.0`.
- Serving generations are not used as a strict equivalence gate because benchmark sampling is not
  deterministic across launches; the fixed teacher-forced NLL gate is the numerical criterion.

## Kernel-only result

Seven timing rounds were collected per point on the A10; each round contains 100 ms warmup and
500 ms measurement. Values below are medians. These numbers measure only the adjacent operator
pair and must not be described as end-to-end speedups.

| Rows × hidden | Residual | Unfused ms | Fused ms | Kernel-pair speedup |
|---|---:|---:|---:|---:|
| 1 × 4096 | no | 0.00873 | 0.00625 | 39.7% |
| 1 × 4096 | yes | 0.00890 | 0.00658 | 35.4% |
| 8 × 4096 | no | 0.00936 | 0.00631 | 48.3% |
| 8 × 4096 | yes | 0.00988 | 0.00675 | 46.4% |
| 128 × 4096 | no | 0.01345 | 0.00959 | 40.2% |
| 128 × 4096 | yes | 0.01626 | 0.01236 | 31.5% |
| 1024 × 4096 | no | 0.06453 | 0.03068 | 110.3% |
| 1024 × 4096 | yes | 0.09934 | 0.06458 | 53.8% |

## End-to-end result

Every serving point contains three repetitions and uses the same checkpoint, SGLang commit,
container, request counts, concurrency and workload seed as its control.

| Workload / control | Input tok/s change | p99 E2E change | p99 TTFT change | p99 TPOT change | p99 ITL change |
|---|---:|---:|---:|---:|---:|
| shared prefix, default profile | -0.01% | -0.14% | -0.26% | -0.13% | +0.27% |
| decode 1K/512, default profile | -0.08% | -0.19% | -0.03% | -0.09% | -0.02% |
| prefill 8K/64, default profile | -0.77% | -0.68% | -0.57% | -0.13% | -1.35% |
| prefill 8K/64, static chunk 1024 | -0.59% | -0.43% | -1.52% | -9.21% | -0.06% |

Positive means improvement. The static-chunk candidate also has materially higher p99 TPOT
variance, so its apparent tail regression is not treated as a precise 9.21% effect; it still
provides no evidence of a win.

## Decision

Reject the fusion for production serving. Profiling predicted the result: separate quantization
and RMSNorm account for only 3.53% of batch-one decode and 5.74% of mixed-prefill kernel time,
while INT8 GEMM accounts for 77.92% and 64.71%. The fusion substantially accelerates its local
operator pair but cannot clear the pre-registered end-to-end gate and slightly regresses all
throughput controls.

This is retained as a bounded negative result because it demonstrates the distinction between a
microbenchmark win and a deployable inference optimization.

## GEMM follow-up

An exact-shape comparison tested the current fused CUTLASS `int8_scaled_mm` against
`torch._int_mm` with both a PyTorch and custom Triton scale/cast epilogue for Qwen3-8B QKV, O,
gate/up, and down projections. `_int_mm` does not support `M <= 16` on this stack. At `M=128` and
`M=1024`, the alternative is slower for every projection except `down_proj, M=1024`, where the
Triton epilogue path is 11.27% faster. One isolated large-prefill shape cannot justify a runtime
dispatch and cannot meet the end-to-end acceptance gate; the next candidate is A10-specific
CUTLASS tile/dispatch tuning rather than wholesale backend replacement.

