# Decision log

## 2026-09-16 — Reset project scope

- Retired the standalone mini-engine as the primary resume project.
- Chose upstream SGLang as the runtime under test.
- Kept agent workloads and quantization as the domain, not as claims of novelty.
- Selected NVIDIA A10 as the first reproducible target.
- Deferred the exact runtime optimization until profiling identifies a top bottleneck.
- Deferred Ascend NPU work until real hardware is available.

## Optimization selection gate

The first implementation target will be selected from CUDA Graph bucket efficiency,
shape-aware quantized backend dispatch, SLO-aware chunked prefill, or CPU/GPU overlap. The chosen
item must account for a meaningful fraction of end-to-end time and have a measurable acceptance
criterion. Add the trace evidence and decision here before coding the patch.

## 2026-09-17 — Reject first SLO-aware source candidate

- Profiling logs identified long-prefill/decode interference after AWQ removed KV retractions.
- A real SGLang patch switched from a 2048-token base chunk to 512 or 1024 while decode was active.
- Three-repetition A/B tests included matching static 512 and 1024 controls.
- Static 1024 reduced p99 TPOT by 72.9% versus the default at a 7.8% input-throughput cost.
- The adaptive patch did not beat static 1024 on either fixed-length or heterogeneous input.
- Decision: retain the measured static profile and evidence tooling; reject the source candidate.
- Full evidence: [Optimization 001](OPTIMIZATION_001_SLO_CHUNKING.md).

## 2026-09-17 — Reject Marlin FP16 reduction

- A bounded trace showed AWQ-Marlin variants consuming 77.7% of kernel time.
- Added a default-safe environment switch for FP32 versus FP16 global reduction.
- FP16 reduction improved a tiny smoke by about 0.8%, but reduced 8K-prefill throughput by 0.50%.
- Long-prefill generated text changed in all three repetitions.
- Decision: reject because it provides no hot-workload speedup and is numerically observable.
- Full evidence: [Optimization 002](OPTIMIZATION_002_MARLIN_REDUCTION.md).

## 2026-09-18 — Reject uncalibrated W8A8 launch

- A first W8A8 experiment incorrectly pointed `--quantization w8a8_int8` at the FP16 checkpoint.
- The run appeared fast on 8K prefill, but the new prompt-NLL gate measured mean NLL 18.21 versus
  2.89 for FP16 across the same 335 scored tokens.
- SGLang documents this flag for checkpoints whose weights are already per-channel INT8 and whose
  activations use per-token dynamic quantization; the FP16 checkpoint does not satisfy that contract.
- Decision: invalidate all performance numbers from this configuration and exclude them from
  accepted evidence. Pin and test a calibrated W8A8 checkpoint before rerunning performance.

## 2026-09-18 — Accept calibrated W8A8 for performance evaluation

- Pinned `nytopop/Qwen3-8B.w8a8` at revision
  `13e255a9648ec08d3873bce1c3d9886a76494c43` and verified both shards by size and SHA-256.
- SGLang confirmed that the checkpoint's `compressed-tensors` metadata is compatible with
  `w8a8_int8`; reported weight memory is 8.81 GB and KV capacity is 65,770 tokens on the A10.
- The fixed 335-token regression corpus measured mean NLL 2.89195 versus 2.89436 for FP16,
  a delta of -0.00241 against the pre-registered maximum increase of 0.02.
- Decision: the calibrated W8A8 artifact clears the numerical gate and may proceed to the same
  three-repetition serving matrix as FP16 and AWQ. This gate does not replace downstream task eval.

## 2026-09-18 — Select W8A8 GEMM as the next primary target

- A bounded mixed-prefill trace attributes 64.71% of kernel time to CUTLASS INT8 GEMM variants.
- A second trace with 49/50 batch-one decode steps raises the GEMM share to 77.92%.
- Separate dynamic activation quantization and RMSNorm together account for only 5.74% of mixed
  prefill and 3.53% of batch-one decode kernel time.
- Decision: run the prepared Norm+quant fusion only as a bounded microbenchmark/negative-result
  check; use exact-shape W8A8 GEMM dispatch or tuning as the primary retained-optimization target.
- Full evidence: [Profile 002](PROFILE_002_W8A8_KERNELS.md).

## 2026-09-18 — Reject W8A8 RMSNorm + quantization fusion for serving

- Implemented a default-off Triton fusion for residual add, RMSNorm, dynamic per-token INT8
  quantization, and pre-quantized handoff to QKV/gate-up linear layers.
- GPU unit tests match the existing operator path for FP16/BF16 with and without residuals; the
  335-token prompt-NLL result is exactly unchanged.
- Seven-round kernel microbenchmarks improve the local operator pair by 31.5%–110.3% across the
  tested Qwen3 hidden-size shapes.
- Three-repetition serving controls show no end-to-end win: throughput changes range from -0.01%
  to -0.77%; the static chunk-1024 control also loses 0.59% throughput.
- Decision: reject for serving. The trace correctly showed this pair was too small a share of the
  critical path; continue with A10-specific W8A8 GEMM dispatch/tile tuning.
- Full evidence: [Optimization 003](OPTIMIZATION_003_W8A8_NORM_QUANT_FUSION.md).
