# Optimization 006 — account for scheduler overlap in low-concurrency latency

Status: latency-tuned GEMM candidate clears the throughput gate; final-release regression pending.

The combined-kernel execution-proof trace contains three `EXTEND bs=1 toks=128` and
two `DECODE bs=1` steps even though that workload requests only one output token.
The upstream benchmark's warmup also requests only one token, and profiling starts
after warmup. This motivates checking the existing overlap-scheduler setting rather
than treating the entire observed TTFT as prefill compute.

This is a deployment control, **not** a claim to have invented SGLang's overlap scheduler.
No speculative causal percentage is assigned before the controlled measurement.

Four cells are required to separate scheduler configuration from custom-kernel value:

| Configuration | Custom kernels OFF | Custom kernels ON |
|---|---|---|
| Default overlap schedule | existing paired data | existing paired data |
| `--disable-overlap-schedule` | new latency control | same latency control + candidate |

The source-patch gain is computed within one row. The configuration gain is computed
within the OFF column. They must not be added together or attributed to one new kernel.
CUDA Graphs stay enabled in all cells. A separate 32-output-token workload checks TTFT,
TPOT and total throughput; a one-token response has no meaningful TPOT.

```bash
python scripts/run_paired_prefill.py --suite short --candidate gemm --disable-overlap
python scripts/run_paired_prefill.py --suite short_decode --candidate gemm --disable-overlap
```

If disabling overlap helps low-concurrency first-token latency but hurts saturated decode,
report that trade-off and retain the default scheduler for throughput-bound workloads.
Nothing here implies that an existing upstream flag is a novel source optimization.

## First controlled result

The accepted-throughput candidate here is **GEMM only**, leaving norm fusion OFF for both arms.
This choice keeps the numerical change smaller than the combined candidate. On source tree
`09b0d2afa4645626040d5bc54592b62bb798bc7f`, run `20260918T042553Z__short` has three alternating
paired restarts, 160 requests per shape and concurrency one:

| Native input tokens | OFF input tok/s, mean ± SD | ON input tok/s, mean ± SD | Change | p99 TTFT OFF → ON |
|---|---:|---:|---:|---:|
| 96 | 3304.70 ± 24.05 | 3490.95 ± 22.12 | +5.64% | 29.826 → 28.175 ms |
| 128 | 4151.16 ± 32.99 | 4652.61 ± 46.22 | **+12.08%** | 31.765 → 28.036 ms |
| 160, fallback | 3923.06 ± 4.61 | 3904.24 ± 21.34 | -0.48% | 42.004 → 42.090 ms |

All 2,880 measured requests complete, all nine paired output records match, and executed-source
fingerprints agree across all launches. CUDA Graphs remain enabled (breakable prefill backend);
only the two columns' custom GEMM switch differs. One-second NVML telemetry is retained with
each original launch. Snapshots: `evidence/a10/latency_gemm_v1_off/` and `latency_gemm_v1_on/`.

The 128-token case clears the **unchanged 10% throughput gate**. This is not a 12% improvement
for arbitrary prompts, concurrency or long generation, and the latency reduction itself is below
the separate 15% tail-latency gate. The original default-overlap 128-token GEMM result was ~7.9%.
Do not add the upstream configuration benefit to the custom-kernel gain. Final source hardening,
32-output-token, mixed/core, and multi-turn checks remain separate acceptance work.

## Final source retest

On the seven-patch tree `bb70ae7f8fb7cede92b8655fbc503aeb5c42bcbd`, run
`20260918T044732Z__short` repeats the same three-round protocol:

| Native tokens | OFF input tok/s, mean ± SD | ON input tok/s, mean ± SD | Change | p99 TTFT OFF → ON |
|---|---:|---:|---:|---:|
| 96 | 3302.26 ± 28.63 | 3502.28 ± 15.47 | +6.06% | 29.818 → 28.217 ms |
| 128 | 4143.97 ± 20.08 | 4688.91 ± 27.87 | **+13.15%** | **31.705 → 28.115 ms** |
| 160, fallback | 3904.22 ± 17.65 | 3920.42 ± 33.43 | +0.41% | 42.061 → 41.966 ms |

All 2,880 requests complete, nine paired output records match, and all source fingerprints agree.
The final same-checkpoint quality test scores 8,128 tokens: NLL delta +8.466e-7, all 40 greedy
task outputs identical, no lost correct task. Snapshots: `final_short_off/on` and
`final_quality_off/on` under `evidence/a10/`. Long-output/core/agent regression is reported
separately in the [final report](FINAL_REPORT.zh-CN.md); do not extrapolate this table.
