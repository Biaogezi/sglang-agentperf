# Optimization 006 — account for scheduler overlap in low-concurrency latency

Status: trace-backed hypothesis; controlled comparison pending.

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
python scripts/run_paired_prefill.py --suite short --candidate combined --disable-overlap
python scripts/run_paired_prefill.py --suite short_decode --candidate combined --disable-overlap
```

If disabling overlap helps low-concurrency first-token latency but hurts saturated decode,
report that trade-off and retain the default scheduler for throughput-bound workloads.
Nothing here implies that an existing upstream flag is a novel source optimization.
