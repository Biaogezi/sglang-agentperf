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

