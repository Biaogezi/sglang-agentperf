# Profile 001 — AWQ mixed prefill

Environment: Qwen3-8B AWQ-Marlin, NVIDIA A10 24 GiB, static 1024-token prefill chunk,
2K–8K random input, concurrency 8. The profiler starts after ten forward steps and records thirty
CPU+GPU steps.

## Trace summary

- 231,821 events, including 14,796 CUDA kernels.
- GPU activity spans 3452.74 ms and is active for 3392.32 ms: only 1.75% uncovered time.
- The main large-M INT4 Marlin kernel consumes 2469.68 ms, or 73.10% of summed kernel time.
- A second small-M Marlin variant consumes another 154.11 ms (4.56%).
- FlashInfer paged prefill attention consumes 346.44 ms (10.25%).
- The next individual GPU kernels are each below 2.5%.

The trace therefore rejects CPU/GPU overlap as the next primary target: the GPU is already active
98.25% of the profiled window. `cudaEventSynchronize` accounts for 96.8% of CUDA runtime API time,
and `scheduler.process_batch_result` appears large on the CPU, but both mostly represent the host
waiting for the GPU-bound forward pass. Removing that wait would not remove the Marlin critical path.

## Next falsifiable hypothesis

The next experiment targets the AWQ-Marlin reduction/backend choice for large-M prefill shapes.
It must first demonstrate a kernel-level win on the exact Qwen projection shapes, then pass online
serving and numerical-quality gates. Attention or scheduler refactors are lower priority because
their maximum available share is much smaller in this trace.

Generate the tables with:

```bash
agentperf analyze-trace \
  --trace profiles/<run>/*.trace.json.gz \
  --output-dir profiles/analysis
```

The command writes `summary.json`, `kernels.csv`, `cuda_runtime.csv`, `cpu_ops.csv`, and
`user_annotations.csv`.

