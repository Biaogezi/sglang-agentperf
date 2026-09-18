# Profiling checklist

## Triage order

1. Confirm request correctness, output lengths, OOM/retry counts and cache state.
2. Compare online serving, offline throughput and one-batch results to locate HTTP/scheduler overhead.
3. Separate prefill, decode and mixed steps using SGLang's detailed annotations.
4. Inspect GPU idle gaps and CPU launch/synchronization stalls in Nsight Systems.
5. Rank kernels by total time, not only average duration.
6. Record tensor shapes and batch/KV-length distributions associated with hot kernels.
7. Check CUDA Graph coverage, padding and eager fallback.
8. For quantized paths, verify selected kernels and account for repack/dequantization/scales.
9. Form one falsifiable optimization hypothesis before modifying runtime code.

## Evidence for a runtime patch

- Before/after timeline screenshots for the same workload.
- Hot-kernel or CPU-region table.
- Unit tests and one-batch correctness test.
- Online serving metrics with three repetitions.
- HBM comparison with measurement method: the current runner retains one-second NVML samples,
  so report sampled maximum whole-device memory, not an exact allocator/instantaneous peak.
- Quality regression result when numerics changed.
- Explanation of losing shapes or workloads, not only winning cases.

## What this release actually collected

The release contains PyTorch CPU/GPU traces, SGLang step annotations, kernel aggregates,
serving JSONL/logs, compiled PTX metadata and one-second NVML telemetry. Nsight Systems and
Nsight Compute remain available follow-up tools, not completed measurements. Do not describe
kernel-duration percentages as roofline utilization or measured memory-bandwidth saturation.
The Nsight triage items above are a checklist, not proof that every item was executed.
