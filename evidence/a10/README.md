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
| FP16, prefill 8K/64 | 2240.92 | 17.51 | 29.22 s | 76.40 ms | 42.26 ms |
| AWQ default chunk 2048, prefill 8K/64 | 2725.83 | 21.30 | 22.56 s | 342.08 ms | 5620.96 ms |
| AWQ static chunk 1024, prefill 8K/64 | 2512.28 | 19.63 | 22.93 s | 92.53 ms | 343.53 ms |

AWQ-Marlin raises decode output throughput by 121.3% versus FP16 and cuts p99 TTFT by 80.0% on
this memory-constrained load. For the interference-heavy 8K case, a static 1024-token chunk cuts
p99 TPOT by 72.9% versus AWQ's default 2048-token chunk at a 7.8% input-throughput cost. This is a
measured deployment operating point, not a claim that static chunking is a new algorithm.

The adaptive source patch and reduced-precision Marlin experiment were both rejected; see the
decision log and optimization reports. Negative results are intentionally retained.
