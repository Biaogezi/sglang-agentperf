# Optimization 007 — test the shape guard under saturated short-context decode

Status: one-pair screen shows no benefit; not accepted as a decode optimization.

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
