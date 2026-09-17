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
