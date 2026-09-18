# Quantization crossover on NVIDIA A10

This report compares three Qwen3-8B serving paths on one NVIDIA A10 24 GiB under the same
digest-pinned SGLang v0.5.19 runtime. It is a workload-specific deployment study, not a claim that
one quantization format is universally superior.

## Artifact and quality controls

- FP16: `Qwen/Qwen3-8B`, revision
  `b968826d9c46dd6066d109eabc6255188de91218`.
- AWQ-Marlin INT4: `Qwen/Qwen3-8B-AWQ`, revision
  `4da05a8edb55c6046cce958586c33b61da07bb79`.
- W8A8 INT8: `nytopop/Qwen3-8B.w8a8`, revision
  `13e255a9648ec08d3873bce1c3d9886a76494c43`.

The W8A8 checkpoint metadata specifies static symmetric per-channel INT8 weights and dynamic
symmetric per-token INT8 activations. Both weight shards are checked by exact byte size and
SHA-256 before launch. This matters because passing ordinary FP16 weights to SGLang's
`w8a8_int8` flag can run and produce plausible performance while silently destroying quality.

All quantized paths score the same 335-token multilingual/code prompt-NLL corpus before any
performance number is accepted:

| Path | Mean NLL | Perplexity | NLL delta vs FP16 | Gate (max +0.02) |
|---|---:|---:|---:|---|
| FP16 | 2.89436 | 18.0719 | — | baseline |
| AWQ-Marlin | 2.91211 | 18.3955 | +0.01775 | pass |
| calibrated W8A8 | 2.89195 | 18.0284 | -0.00241 | pass |

This corpus is a fast numerical regression gate, not a substitute for downstream task accuracy.

## Memory and KV capacity

Values below come from SGLang's server initialization log at the same `mem_fraction_static=0.82`.

| Path | Reported weight memory | KV token capacity | Capacity vs FP16 |
|---|---:|---:|---:|
| FP16 | 15.28 GB | 18,694 | 1.00x |
| AWQ-Marlin | 5.73 GB | 88,212 | 4.72x |
| calibrated W8A8 | 8.81 GB | 65,770 | 3.52x |

AWQ maximizes capacity because its weights are smaller. W8A8 still removes the FP16 memory
constraint while retaining an INT8 tensor-core path for large matrix multiplications.

## Three-repetition serving results

**Historical workload note:** these runs used `random-ids` without `--tokenize-prompt`, so
inputs were decoded to text and re-tokenized by the server. 1K/8K labels and input tok/s are
nominal dataset lengths, not exact server token counts. The same text sets were compared across
paths, so measured output-throughput and latency comparisons remain workload observations.
Do not mix their absolute input tok/s with the newer native-ID fixed-shape protocol.

Every cell is the mean of three measured runs after warmup. Decode uses 160 requests at
concurrency 16 with 1,024 input and 512 output tokens. Prefill uses 80 requests at concurrency 8
with 8,192 input and 64 output tokens. Shared-prefix uses ten 4K-token prefix groups at 8 req/s.

| Workload / path | Input tok/s | Output tok/s | p99 TTFT | p99 TPOT | p99 ITL |
|---|---:|---:|---:|---:|---:|
| Decode — FP16 | 510.40 | 255.20 | 26.06 s | 60.44 ms | 48.12 ms |
| Decode — AWQ | 1129.33 | 564.67 | 5.21 s | 27.86 ms | 19.78 ms |
| Decode — W8A8 | 1037.52 | 518.76 | 7.99 s | 30.54 ms | 27.17 ms |
| 8K prefill — FP16 | 2240.92 | 17.51 | 29.22 s | 76.40 ms | 42.26 ms |
| 8K prefill — AWQ | 2725.83 | 21.30 | 22.56 s | 342.08 ms | 5620.96 ms |
| 8K prefill — W8A8 | 4349.44 | 33.98 | 17.24 s | 185.31 ms | 2848.82 ms |
| 8K prefill — AWQ static chunk 1024 | 2512.28 | 19.63 | 22.93 s | 92.53 ms | 343.53 ms |
| 8K prefill — W8A8 static chunk 1024 | 3414.11 | 26.67 | 16.56 s | 68.24 ms | 222.47 ms |
| Shared prefix — AWQ | 12297.41 | 344.30 | 9.17 s | 85.10 ms | 31.65 ms |
| Shared prefix — W8A8 | 12890.07 | 360.90 | 4.09 s | 64.02 ms | 38.25 ms |

All three paths reported zero retraction events in these accepted runs.

## Interpretation

- Decode is weight-bandwidth and small-M dominated. AWQ's smaller INT4 weights beat W8A8 by
  8.9% in output throughput and also have better p99 latency.
- Long prefill presents large matrix multiplications that use INT8 tensor cores effectively.
  W8A8 beats AWQ by 59.6% in input throughput, reduces p99 TTFT by 23.6%, and reduces p99 TPOT by
  45.8%. Its p99 end-to-end latency is still 7.4% worse, so streaming and completion SLOs must be
  evaluated separately.
- Shared-prefix serving is mixed. W8A8 improves input/output throughput by 4.8%, p99 TTFT by
  55.4%, and p99 TPOT by 24.8%, but p99 ITL regresses by 20.9%.
- W8A8's static 1024-token chunk is a latency-oriented operating point: versus its default 2048
  setting, it sacrifices 21.5% input throughput while improving p99 E2E by 21.9%, p99 TPOT by
  63.2%, and p99 ITL by 92.2%. Against AWQ at the same static chunk, it is 35.9% faster in input
  throughput and has lower values for all four reported tail-latency metrics.

The defensible deployment conclusion is phase-aware: prefer AWQ for decode-heavy traffic and W8A8
for long-prefill-heavy traffic. A disaggregated prefill/decode deployment could assign a different
weight format to each worker pool; a single-GPU deployment should choose from its observed traffic
mix and SLO, not from model size alone.

## Reproducibility

Published evidence directories contain each aggregate and manifest plus SHA-256/size records for
the ignored raw request outputs and logs. The model source file pins repository revisions and W8A8
shard hashes. The rejected uncalibrated W8A8 runs are deliberately absent from accepted evidence.
