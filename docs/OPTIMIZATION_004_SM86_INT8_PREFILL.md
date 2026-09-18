# Optimization 004 — SM86 shape-gated INT8 prefill GEMM

Status: runtime integration and correctness passed; serving acceptance in progress.

## Hypothesis and scope

The real SGLang W8A8 trace identifies INT8 GEMM as the largest kernel family. A10/SM86 supports
INT8 Tensor Core MMA but has a constrained shared-memory budget. Exhaustive replacement is not
appropriate: matrix shapes have different optima. The candidate keeps the existing activation
quantization and weight format, then selects a 128×128×128 Triton tile for measured shapes.

- Opt-in environment variable: `SGLANG_A10_INT8_PREFILL=true`.
- FP16 or BF16 output, INT8 A/B, FP32 per-row/per-channel scales.
- SM86; 80–128 rows; K=4096; N=6144 (QKV) or 24576 (gate/up).
- Contiguous row-major A and column-major B; supported optional output bias.
- Unsupported shapes, layouts, scaling or hardware retain `sgl_kernel.int8_scaled_mm`.
- Both `W8A8Int8LinearMethod` and the dynamic symmetric compressed-tensors scheme are wired.

The kernel accumulates products in INT32, applies `float(acc) * (row_scale * channel_scale)`
and casts once. Scaling order and unfused bias addition match the existing CUTLASS epilogue.
There is no float atomic reduction, extra precision reduction, new quantization calibration,
or change to KV-cache policy.

## Verification already completed

1. 44 GPU correctness cases: FP16/BF16, bias, non-tile-aligned dimensions, extreme INT8 values,
   exact FP64 integer-dot reference, prototype and shipped-kernel parity.
2. Actual `W8A8Int8LinearMethod` dispatch tests at M=1/64/79/80/96/128/129/160: expected branch
   selected, output byte-identical to the original method in all cases.
3. Serving GPU-trace proof: `_int8_prefill` appears **144 times ON and zero times OFF**.
   These profiled requests are execution evidence only, not performance acceptance points.
4. Complete source patch series reapplies to the pinned base and yields tree
   `68ffb227829efeceff3e2ecd459bdfd43ab0387f`.

See `scripts/test_w8a8_splitk_gpu.py`, `scripts/test_w8a8_dispatch_gpu.py`, and the
`evidence/a10/prefill_dispatch_proof_*` snapshots. Scripts are explicit GPU tests, not silently
skipped CPU CI tests.

## Microbenchmark versus serving

The prototype's held-out M=80/96/112/128 measurements show 1.078–1.189× QKV and 1.259–1.383×
gate/up speedups. All thirty tested adjacent shape outputs matched CUTLASS exactly; unsupported
shapes frequently slowed down, motivating a narrow guard. Continuous CUDA-graph microbenchmarks
do not establish full-model performance, especially under GPU power/frequency changes.

The serving protocol fixes model/requests/seed/concurrency and alternates independent OFF/ON
server launches: AB, BA, AB. Each launch warms up. The primary synthetic prefill cases are
96/128/160 input tokens, one output token, concurrency one, 160 requests per point. M=160 is a
fallback control. Each individual launch manifest and original server log is retained.

The quality extension uses 64 fixed 128-token WikiText windows, plus 40 deterministic arithmetic,
strict JSON extraction and long key-retrieval checks. These are numerical/application regression
tests, not a claim of standard WikiText perplexity or general agent capability.

## Reproduction

Inside the pinned container, with the full locked patch series applied:

```bash
python scripts/test_w8a8_splitk_gpu.py
python scripts/test_w8a8_dispatch_gpu.py
python scripts/run_paired_prefill.py --suite proof --repetitions 1
python scripts/run_paired_prefill.py --suite short --repetitions 3
```

Profiling and quality are separate from performance runs; never combine their timing data.
### First three-round serving result (default breakable prefill graph)

| Input tokens | OFF input tok/s, mean ± sample SD | ON input tok/s, mean ± sample SD | Throughput change | p99 TTFT OFF → ON |
|---|---:|---:|---:|---:|
| 96 | 2215.95 ± 9.57 | 2327.29 ± 2.08 | +5.02% | 48.277 → 48.246 ms |
| 128 | 2440.59 ± 8.87 | 2458.40 ± 2.06 | +0.73% | 55.268 → 55.523 ms |
| 160 (fallback) | 2884.19 ± 8.17 | 2885.82 ± 7.32 | +0.06% | 57.042 → 57.151 ms |

All 2,880 measured requests completed (3 shapes × 160 requests × 3 repetitions × 2 modes),
and all nine paired generated-output records matched exactly. The candidate **does not clear
the 10% serving-throughput gate on this configuration**. Keep it default-off while investigating
the microbenchmark/serving gap; these are not 38% end-to-end gains.

Both modes use upstream worktree commit `997819a602` on the GPU machine. Harness commit IDs
changed during the series only because independent application-test files were added; the
benchmark runner, selected configs and runtime source were unchanged for the measured launches.
Original manifests preserve each ID rather than rewriting history. Snapshots are
`evidence/a10/prefill_paired_v1_off/` and `prefill_paired_v1_on/`.

### Expanded numerical/application regression

On the same 8,128 scored tokens from 64 fixed 128-token windows, OFF NLL is 2.9490910677 and ON
is 2.9490919142: delta +0.0000008466, below the +0.02 gate. This is a tiny numerical difference,
not bitwise NLL identity and not standard WikiText perplexity.

All 40 greedy application outputs are identical across OFF/ON. Both modes pass 16/16 strict JSON
and 8/8 key-retrieval checks (10,988-token inputs). Both fail 16/16 strict arithmetic *format*
checks because they include explanation despite the instruction to return only an integer.
Do not describe this as 100% task accuracy. The short application prompts fall outside the
candidate M range; they are fallback regressions, while the 128-token NLL windows exercise its
target range. Task source and raw-output hashes are retained separately from capability claims.
