# Optimization 005 — connect norm fusion to the native W8A8 method

Status: implementation prepared; GPU integration and serving validation pending.

## Hypothesis and controls

The dispatch audit in Optimization 003 showed a missing consumer: the native
`W8A8Int8LinearMethod` did not accept the `(quantized, scale, output_dtype)` tuple.
This follow-up explicitly adds that contract, admits the method in the norm guard,
and combines it with Optimization 004 only after correctness checks. The Qwen3
call sites remain TP=1-only. Other models and TP>1 are not claimed as validated.

The same pinned runtime can run four configurations without editing weights:

| Arm | `SGLANG_A10_INT8_PREFILL` | `SGLANG_W8A8_FUSED_RMSNORM_QUANT` |
|---|---|---|
| Control | false | false |
| GEMM only | true | false |
| Norm fusion only | false | true |
| Combined | true | true |

Use `scripts/run_paired_prefill.py --candidate gemm|fusion|combined` with alternating
OFF/ON server restarts. The original 10% throughput / 15% tail-latency acceptance
threshold is unchanged. Short workloads send native token IDs. A profile trace
must contain the fused kernel in ON and none in OFF; an environment variable alone
does not prove execution. CUDA Graphs remain enabled unless explicitly recorded.

## Implementation

The fused kernel adds the residual, computes RMSNorm, rounds to the original
activation dtype, computes the dynamic token scale and writes INT8 bytes plus
FP32 scales. It updates the residual buffer in place, preserving the model's
residual contract. The native linear consumes that tuple without quantizing again;
both CUTLASS fallback and the shape-guarded custom GEMM consume identical formats.

`test_native_norm_fusion_gpu.py` tests the real RMSNorm dispatch and native linear,
FP16/BF16, residual/no-residual, M=1/16/79/80/96/128/129/1024, and GEMM OFF/ON.
Floating-point RMSNorm reductions may move values near quantization boundaries;
the test reports mismatch fractions and output relative L2 instead of claiming
all fused normalization arithmetic is bitwise invariant. Full-model NLL and
greedy task regressions remain required.

## Results

Pending. Historical Optimization 003 serving numbers must not be reused here.
