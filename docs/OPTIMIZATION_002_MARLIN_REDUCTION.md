# Optimization 002 — Marlin reduction precision

Status: **rejected**.

## Hypothesis

[Profile 001](PROFILE_001_AWQ_MIXED_PREFILL.md) attributes 77.7% of GPU kernel time to
AWQ-Marlin variants. SGLang's Marlin wrapper uses FP32 global reduction by default and notes that
FP16 reduction can save memory movement. A default-safe environment switch was added so the two
paths could be tested without changing the normal runtime.

## Results

The smoke comparison used the same patched commit and differed only in
`SGLANG_MARLIN_USE_FP32_REDUCE`.

| Workload | Reduction | Output tok/s | p99 TTFT ms | p99 TPOT ms | Output equivalence |
|---|---|---:|---:|---:|---|
| 256/32 smoke | FP32 | 71.90 | 108.94 | 11.68 | reference |
| 256/32 smoke | FP16 | 72.47 | 108.03 | 11.59 | exact, 3/3 repetitions |
| 8K/64, static chunk 1024 | FP32 | 19.63 | 22934.51 | 92.53 | reference |
| 8K/64, static chunk 1024 | FP16 | 19.53 | 22848.07 | 92.94 | different, 3/3 repetitions |

The small smoke improves by about 0.8%, but the trace-relevant long-prefill workload loses 0.50%
throughput and regresses p99 TPOT/ITL by 0.45%/0.71%. The only latency win is a non-actionable
0.38% TTFT change. All three long-prefill repetitions also produce different generated text,
showing that the lower-precision reduction is numerically observable on long sequences.

## Decision

Reject FP16 reduction as an optimization. It does not accelerate the hot workload and weakens
numerical stability. The experiment switch remains default-safe and documents the tested path,
but no resume performance claim uses it.

