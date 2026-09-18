# Optimization 005 — connect norm fusion to the native W8A8 method

Status: numerical/task regression passed; default-overlap serving gain remains below acceptance gate.

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

## GPU integration and execution proof

The exact six-patch source tree is `09b0d2afa4645626040d5bc54592b62bb798bc7f`;
the measured remote commit is `7c699d0083`. All 64 native-fusion integration cases pass.
Maximum observed INT8 element mismatch fraction is `3.0904e-6`; maximum linear output
relative L2 is `2.1581e-4`. The 44-case GEMM, eight-shape method dispatch and eleven-row
JIT-binary suites also pass. Logs and source fingerprints are published in
`evidence/a10/native_fusion_gpu_validation/`.

The paired native-token profile has 216 `_int8_prefill` calls and 360
`_rmsnorm_quant_int8` calls in ON, and zero of either in OFF. The five captured steps
include three 128-token EXTEND steps and two batch-one DECODE steps. Kernel counts
agree with 36 layers × two selected projections × three/five steps respectively.
See `evidence/a10/combined_proof_off/` and `combined_proof_on/`. Profiled timings are
not the performance acceptance data.

## Serving and quality results

Three alternating repetitions (`20260918T041033Z__short`) yield:

| Native input tokens | Throughput change | p99 TTFT reduction |
|---|---:|---:|
| 96 | +3.85% | 3.74% |
| 128 | +9.02% | 8.54% |
| 160 | +0.67% | 0.87% |

All 2,880 requests complete, but generated token strings are not identical. In repetition one,
the exact-match counts are 145/160, 149/160 and 145/160 for 96/128/160 respectively. Random
unstructured prompts are not task accuracy, and these differences cannot be called lossless.
Output lengths and errors match, so timing is not made faster by shorter generations.

The separate quality run (`20260918T042120Z`) scores 8,128 tokens: OFF NLL 2.9490919142,
ON NLL 2.9522406621, delta +0.0031487479 (within the +0.02 gate). Both pass 16/16 JSON and
8/8 long retrieval and fail 16/16 strict arithmetic format. Greedy text is identical on 34/40
tasks, with no lost correct tasks. This is bounded numerical/task regression, not broad quality
equivalence. Evidence is in `combined_native_v1_off/on` and `combined_quality_v1_off/on`.

The combined candidate does not clear the original performance gate on the default overlap
configuration. Keep it default-off. Investigate the scheduler/latency control separately in
Optimization 006, and do not retrofit these results into the invalid historical experiment.

Patch 0007 further gates the Qwen3 call-site hints on the explicit norm-fusion switch. This
preserves original default call sites for other quantization formats instead of inadvertently
enabling an existing FP8 fusion when our INT8 switch is off. The measurements above precede
that call-site safety hardening; final-release validation is tracked separately.
