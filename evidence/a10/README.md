# NVIDIA A10 evidence snapshot

These directories contain the small, reviewable portion of each accepted run: the exact manifest,
the three-repetition aggregate or quality report, and SHA-256/size records for ignored raw JSONL
and log files. Raw request traces remain local because they are large; their hashes make later
substitutions detectable.

## Environment

- GPU: NVIDIA A10 24 GiB (SM 86)
- Runtime: digest-pinned SGLang v0.5.19 CUDA 12.9 image
- Model family: Qwen3-8B
- Repetitions: three measured runs per reported point
- Concurrency: 16 for decode and shared-prefix; 8 for 8K prefill

The exact container digest, SGLang commit, server command, workload command and request sizes are in
each `manifest.json`.

## Quantization quality gate

The fixed corpus is a fast numerical regression gate, not a replacement for downstream task
evaluation. All three paths score the same 335 tokens.

| Precision path | Mean NLL | Perplexity | NLL delta vs FP16 | Gate |
|---|---:|---:|---:|---|
| FP16 | 2.89436 | 18.0719 | — | baseline |
| AWQ-Marlin INT4 | 2.91211 | 18.3955 | +0.01775 | pass |
| calibrated W8A8 INT8 | 2.89195 | 18.0284 | -0.00241 | pass |

The allowed mean-NLL increase is 0.02. The W8A8 checkpoint is pinned to revision
`13e255a9648ec08d3873bce1c3d9886a76494c43`, and both weight shards are verified by byte size and
SHA-256 before launch. The earlier experiment that interpreted ordinary FP16 weights as W8A8 is
invalidated and is not published here.

## Accepted observations

| Workload / configuration | Input tok/s | Output tok/s | p99 TTFT | p99 TPOT | p99 ITL |
|---|---:|---:|---:|---:|---:|
| FP16, decode 1K/512 | 510.40 | 255.20 | 26.06 s | 60.44 ms | 48.12 ms |
| AWQ-Marlin, decode 1K/512 | 1129.33 | 564.67 | 5.21 s | 27.86 ms | 19.78 ms |
| calibrated W8A8, decode 1K/512 | 1037.52 | 518.76 | 7.99 s | 30.54 ms | 27.17 ms |
| FP16, prefill 8K/64 | 2240.92 | 17.51 | 29.22 s | 76.40 ms | 42.26 ms |
| AWQ default chunk 2048, prefill 8K/64 | 2725.83 | 21.30 | 22.56 s | 342.08 ms | 5620.96 ms |
| calibrated W8A8 default chunk 2048, prefill 8K/64 | 4349.44 | 33.98 | 17.24 s | 185.31 ms | 2848.82 ms |
| AWQ static chunk 1024, prefill 8K/64 | 2512.28 | 19.63 | 22.93 s | 92.53 ms | 343.53 ms |
| calibrated W8A8 static chunk 1024, prefill 8K/64 | 3414.11 | 26.67 | 16.56 s | 68.24 ms | 222.47 ms |

AWQ-Marlin raises decode output throughput by 121.3% versus FP16 and cuts p99 TTFT by 80.0% on
this memory-constrained load. For the interference-heavy 8K case, a static 1024-token chunk cuts
p99 TPOT by 72.9% versus AWQ's default 2048-token chunk at a 7.8% input-throughput cost. This is a
measured deployment operating point, not a claim that static chunking is a new algorithm.

The calibrated W8A8 path exposes a phase-dependent crossover rather than a universally superior
format. AWQ is 8.9% faster in decode output throughput, while W8A8 is 59.6% faster in 8K-prefill
input throughput and reduces that workload's p99 TTFT by 23.6%. On the shared-prefix workload,
W8A8 improves input/output throughput by 4.8% and p99 TTFT by 55.4%, but its p99 ITL is 20.9%
worse. These trade-offs motivate phase-aware deployment rather than a single-format claim.

Within W8A8, the static 1024-token profile trades 21.5% input throughput for a 21.9% p99 E2E
reduction, 63.2% lower p99 TPOT, and 92.2% lower p99 ITL versus its default 2048-token profile.
At the same static 1024-token setting, W8A8 is 35.9% faster in input throughput than AWQ and also
has lower p99 E2E, TTFT, TPOT, and ITL. This is a measured SLO operating point, not a new scheduler
algorithm.

The adaptive source patch and reduced-precision Marlin experiment were both rejected; see the
decision log and optimization reports. Negative results are intentionally retained.

## W8A8 profiler evidence

Compact profiler aggregates are published in `w8a8_profile_mixed_prefill/` and
`w8a8_profile_decode_bs1/`; each directory also records the size and SHA-256 of its ignored raw
trace. The mixed-prefill trace attributes 64.71% of summed kernel time to CUTLASS INT8 GEMM and
the batch-one decode trace attributes 77.92%. Dynamic INT8 activation quantization plus RMSNorm
account for only 5.74% and 3.53%, respectively. See
[`docs/PROFILE_002_W8A8_KERNELS.md`](../../docs/PROFILE_002_W8A8_KERNELS.md) for the decision.
