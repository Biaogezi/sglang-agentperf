# Decision log

## 2026-09-16 — Reset project scope

- Retired the standalone mini-engine as the primary resume project.
- Chose upstream SGLang as the runtime under test.
- Kept agent workloads and quantization as the domain, not as claims of novelty.
- Selected NVIDIA A10 as the first reproducible target.
- Deferred the exact runtime optimization until profiling identifies a top bottleneck.
- Deferred Ascend NPU work until real hardware is available.

## Optimization selection gate

The first implementation target will be selected from CUDA Graph bucket efficiency,
shape-aware quantized backend dispatch, SLO-aware chunked prefill, or CPU/GPU overlap. The chosen
item must account for a meaningful fraction of end-to-end time and have a measurable acceptance
criterion. Add the trace evidence and decision here before coding the patch.

## 2026-09-17 — Reject first SLO-aware source candidate

- Profiling logs identified long-prefill/decode interference after AWQ removed KV retractions.
- A real SGLang patch switched from a 2048-token base chunk to 512 or 1024 while decode was active.
- Three-repetition A/B tests included matching static 512 and 1024 controls.
- Static 1024 reduced p99 TPOT by 72.9% versus the default at a 7.8% input-throughput cost.
- The adaptive patch did not beat static 1024 on either fixed-length or heterogeneous input.
- Decision: retain the measured static profile and evidence tooling; reject the source candidate.
- Full evidence: [Optimization 001](OPTIMIZATION_001_SLO_CHUNKING.md).
