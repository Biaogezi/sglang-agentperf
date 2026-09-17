# Profile 002 — calibrated W8A8 prefill and decode

Environment: calibrated Qwen3-8B W8A8, NVIDIA A10 24 GiB, FlashInfer attention, static
1024-token prefill chunk. Traces are bounded PyTorch CPU+GPU captures taken after warmup.

## Mixed prefill trace

The 2K–8K-input trace contains 240,387 events and 19,873 GPU activities. GPU activity covers
97.46% of the 2.278 s capture window.

| Kernel family | Kernel-time share |
|---|---:|
| CUTLASS INT8 GEMM variants | 64.71% |
| FlashInfer paged/ragged prefill attention | 19.09% |
| Dynamic per-token INT8 quantization | 3.10% |
| RMSNorm | 2.64% |
| SiLU-and-multiply | 2.81% |

## Decode batch-one trace

The decode capture contains 170,035 events and 30,017 GPU activities. Of its fifty sampled
forward steps, 49 are batch-one decode and one is an extend step. GPU activity covers 99.43% of
the 1.088 s capture window.

| Kernel family | Kernel-time share |
|---|---:|
| CUTLASS INT8 GEMM variants | 77.92% |
| FlashInfer paged decode attention | 2.90% |
| Dynamic per-token INT8 quantization | 2.07% |
| RMSNorm | 1.46% |
| SiLU-and-multiply | 0.86% |

## Decision

The runtime is GPU-bound in both phases. A proposed RMSNorm + dynamic-INT8-quant fusion removes
one launch and one activation read/write at two sites per layer, but the measured upper bound is
only 5.74% of kernel time in mixed prefill and 3.53% in batch-one decode. It is worth a bounded
kernel correctness/microbenchmark experiment, but it cannot be presented as a likely 10% serving
win before measurement.

The main retained-optimization target is shape-aware W8A8 GEMM dispatch or tuning: CUTLASS INT8
GEMM accounts for 64.71% of mixed-prefill kernel time and 77.92% of batch-one decode kernel time.
Any source candidate must compare against the exact current backend on the model's projection
shapes and then pass the numerical and three-repetition serving gates.

Regenerate either table with:

```bash
agentperf analyze-trace --trace profiles/<capture>/*.trace.json.gz \
  --output-dir profiles/<capture>/analysis
```

Raw multi-megabyte traces remain outside Git; the compact analysis tables and trace hashes are
published as evidence snapshots.
