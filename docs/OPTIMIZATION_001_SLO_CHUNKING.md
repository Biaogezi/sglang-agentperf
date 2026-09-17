# Optimization 001 — SLO-aware chunked prefill

Status: **source candidate rejected; static 1024-token operating point retained**.

## Trigger

On one NVIDIA A10 24 GiB, Qwen3-8B AWQ increased usable KV capacity from 18,694 to
88,212 tokens and removed the FP16 cache-retraction failure mode. It also changed the
long-prefill operating point: the default 2048-token chunk delivered high input throughput,
but an 8K-input/64-output burst produced 342 ms p99 TPOT and 5.62 s p99 ITL.

The hypothesis was that the scheduler could use the 2048-token base chunk while no decode
request was active, then cap the chunk while decode was active. The implementation added two
validated server arguments and selected the effective chunk in the real SGLang scheduler. It
was disabled by default and packaged as a reproducible patch on top of v0.5.19.

## Fair controls

All points use the same model, AWQ-Marlin kernel, FlashInfer backend, random seed, cold cache,
80 prompts, concurrency 8, and three repetitions. Static 512 and 1024 configurations are
included so an existing CLI tuning opportunity is not attributed to the source patch.

### Fixed 8K input

| Profile | Input tok/s | p99 E2E ms | p99 TTFT ms | p99 TPOT ms | p99 ITL ms |
|---|---:|---:|---:|---:|---:|
| Default static 2048 | 2725.83 | 24842.31 | 22556.87 | 342.08 | 5620.96 |
| Static 512, interval 4 | 2290.74 | 29977.31 | 26755.64 | 51.60 | 169.42 |
| Static 1024, interval 4 | 2512.28 | 28702.62 | 22934.51 | 92.53 | 343.53 |
| Adaptive 2048→512, interval 4 | 2303.72 | 30404.07 | 27127.32 | 52.68 | 172.69 |
| Adaptive 2048→1024, interval 4 | 2484.83 | 28776.37 | 22999.80 | 93.24 | 347.84 |

Static 1024 is the useful SLO point for this workload: versus the default, it reduces p99 TPOT
by 72.9% and p99 ITL by 93.9% at a 7.8% input-throughput cost. The adaptive source candidate is
1.1% slower than the matching static 1024 control and is also slightly worse on every latency
metric.

### Heterogeneous 2K–8K input

| Profile | Input tok/s | p99 E2E ms | p99 TTFT ms | p99 TPOT ms | p99 ITL ms |
|---|---:|---:|---:|---:|---:|
| Static 1024, interval 4 | 2633.41 | 21721.46 | 16649.82 | 90.56 | 328.87 |
| Adaptive 2048→1024, interval 4 | 2634.62 | 21662.93 | 16647.68 | 90.71 | 329.71 |

The apparent throughput gain is 0.046%, far below run-to-run variation. TPOT and ITL regress by
0.17% and 0.26%. The candidate therefore does not pass the repository's retention gate.

## What remains valuable

- The causal diagnosis is retained: quantization relieved KV pressure, which allowed more long
  prefills to coexist and exposed prefill/decode interference hidden by FP16 retractions.
- The 1024-token static profile is retained as a measured deployment operating point, not called
  a novel algorithm.
- The source patch and unit tests are retained as a documented negative experiment. Its numbers
  are not used as a resume speedup claim.
- The next source target must be selected from a profiler trace and must beat the best static
  control, not merely the default configuration.

