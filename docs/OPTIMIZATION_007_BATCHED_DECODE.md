# Optimization 007 — test the shape guard under saturated short-context decode

Status: three-repeat graph128 validation shows +7.40% throughput, below the unchanged 10% gate.
This is a reproducible secondary result, not an accepted broad decode speedup. Diagnostic traces
are collected separately from performance measurements.

The candidate is named short-prefill GEMM, but dispatch is based on matrix rows, not forward
mode. A decode step with 80..128 token rows has the same supported QKV/gate-up dimensions.
This suggests a second, distinct use case. It does **not** justify automatically enabling the
kernel for every decode batch or claiming a general decode win.

Protocol: Qwen3-8B calibrated W8A8, same final seven-patch runtime tree, original overlap
scheduler and CUDA Graphs enabled, both arms with norm fusion OFF. 128 native input tokens,
128 requested output tokens, concurrency 128, 640 requests for a bounded screening
run. Per-arm outputs and errors must be retained. This is saturated short-context serving,
not an agent trace or a latency-sensitive low-arrival-rate workload.

Screen once, then only claim performance after three independent alternating repeats. Keep
640 requests (5× concurrency) for both screening and acceptance measurements. The original
>=10% throughput gate and
quality requirements are unchanged. Do not mix screening and acceptance samples. Capture a
separate trace to establish which decode row counts actually execute the custom GEMM.

```bash
python scripts/run_paired_prefill.py --suite batch_decode --candidate gemm --repetitions 1
```

No runtime edits or retuning are made for this experiment. A positive result would extend the
measured use cases; a negative result remains in the report. Long-context and TP>1 generalization
would still require different experiments and hardware capacity.

## Screening result

Run `20260918T063418Z__batch_decode`, unchanged final runtime tree: all 1,280 requests
complete with the required 128 output tokens. OFF output throughput is 2420.20 tok/s;
ON is 2402.17 tok/s (-0.745%). p99 TTFT changes 2286.36→2297.98 ms; p99 TPOT
52.145→53.388 ms. This is **one screening pair**, not a three-repeat performance claim or
a statistically established slowdown. Some generated texts differ even at fixed length.

Decision: do not promote the candidate as high-concurrency decode acceleration. Keep this
negative result and collect a separate dispatch/hotspot trace before explaining its cause:

```bash
python scripts/run_paired_prefill.py --suite batch_decode --candidate gemm \
  --repetitions 1 --trace-start-step 100 --trace-steps 20
```

The profiled result is diagnostic, never a second performance sample. Source manifests and
screening summaries are in `evidence/a10/batch_decode_screen_off/on/`.

## Configuration boundary found before further measurement

The screening server logs show the default decode graph capture list is
`[1, 2, 4, 8, 12, 16, 24]`, while actual decode batches have 128 requests and log
`cuda graph: False`. Prefill graphs remain enabled. Thus “CUDA Graph enabled” did not mean
the tested decode shapes were covered. This is a confirmed configuration boundary; whether
CPU launch overhead hides kernel gains still requires measurement, not assumption.

Add a fair control with the existing upstream `--cuda-graph-max-bs-decode 128` on **both** arms:

```bash
python scripts/run_paired_prefill.py --suite batch_decode --candidate gemm \
  --decode-graph-max-bs 128 --repetitions 1
```

If promising, repeat three times and collect independent trace proof. Compare custom OFF/ON
within the same graph configuration. Any default-OFF → graph128-OFF gain belongs to upstream
configuration/coverage, not our GEMM. Monitor capture-memory costs and do not silently lower
KV capacity or change workload lengths if capture fails.

### Graph128 screening result

Run `20260918T073157Z__batch_decode` completes all 1,280 requests. OFF→ON output throughput is
2461.326→2632.568 tok/s (+6.957%); p99 TPOT is 49.942→47.580 ms (-4.729%); p99 TTFT
is 2282.926→2281.044 ms. Text matches 627/640 pairs. This single screen is below the
pre-registered throughput/latency gates and is not an accepted broad decode speedup.

Logs confirm actual batch-128 decode replay (`cuda graph: True`). In the OFF arm, decode capture
memory is 0.36 GB versus 0.12 GB in the earlier default screen; both retain 74,673 KV token slots
and the same 0.82 memory fraction. These are launch-reported capture allocations, not exact
whole-process peaks. The +1.7% difference between the two OFF screens is not a repeated
configuration-performance result. Public snapshots: `batch_graph128_screen_off/on`.

Three independent alternating graph128 pairs were run next, **without mixing in the screening
pair or changing the gate**. Separate default-coverage and graph128 traces are diagnostic only.

## Independent three-repeat graph128 result

Run `20260918T073808Z__batch_decode`, AB/BA/AB server restarts, same runtime and 640 requests
per arm/repetition. All 3,840 requests complete with 128 output tokens and no recorded KV
retractions; source/command/workload comparability audit passes.

| Metric | OFF, mean ± sample SD | ON, mean ± sample SD | Change |
|---|---:|---:|---:|
| Output tok/s | 2454.395 ± 19.666 | 2636.012 ± 4.273 | +7.400% |
| Per-run p99 TTFT (ms) | 2287.693 ± 17.248 | 2279.409 ± 9.137 | -0.362% |
| Per-run p99 TPOT (ms) | 50.866 ± 0.713 | 47.313 ± 0.072 | -6.985% |

Paired text matches are 629/640, 631/640 and 629/640 (1,889/1,920 total), not universal
bitwise/text equivalence. No throughput or tail-latency threshold is lowered after this result.
The short-prefill acceptance remains separate. The original default-coverage result has only
one pair, so this is not a three-repeat difference-in-differences study of graph configuration.
Public summaries and per-request arrays: `final_batch_graph128_off/on` and
`final_batch_graph128_requests_off/on`. Screening data is retained, not mixed into the table.

The six launches all report 74,673 KV token slots and 0.36 GB decode-graph capture memory.
One-second launch-wide telemetry reaches the same sampled maximum 21,847 MiB and 61°C in both
arms. Sampling does not prove an exact instantaneous allocation peak or absence of all throttling.

## Independent execution/overhead diagnostics

The default-coverage pair is `20260918T074739Z__batch_decode`; graph128 is
`20260918T075146Z__batch_decode`. Each trace contains exactly **20 `DECODE bs=128` steps**.
All 2,560 diagnostic requests complete, but their profiled durations are excluded from serving
acceptance. Trace files reside in profile-ID subdirectories, not directly in the output root.

| Captured mode | Custom GEMM calls OFF / ON | `cudaGraphLaunch` OFF / ON | Captured GPU activity OFF / ON |
|---|---:|---:|---:|
| Default coverage, eager at batch 128 | 0 / 1440 | 0 / 0 | 35.67% / 30.61% |
| Expanded graph128 | 0 / 1440 | 20 / 20 | 99.35% / 99.30% |

Thus the eager result is not a no-op switch: the candidate really runs in both modes. Graph
replay bypasses repeated Python dispatch/launch work inside the captured model. The eager
diagnostic has substantial host-side gaps and more intrusive Python profiling work; graph128
keeps the device busy in this capture. This is consistent with host overhead masking a faster
kernel, **not** a measurement of unprofiled production idle time or a precise causal allocation
of every saved millisecond. CPU step spans in graph mode measure asynchronous enqueue work,
not complete GPU step latency. No Nsight Compute counters were collected.

Compact trace evidence: `batch_eager_proof_off/on`, `batch_graph128_proof_off/on`.
The matching profiled launch/config manifests and explicitly diagnostic timing summaries are
`batch_eager_diagnostic_off/on`, `batch_graph128_diagnostic_off/on`. Raw traces and full server
logs are backed up locally. Decision: preserve the reproducible +7.40% secondary result while
retaining the original acceptance threshold and the narrow primary short-prefill claim.
