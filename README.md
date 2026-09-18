# SGLang-AgentPerf

[![CI](https://github.com/Biaogezi/sglang-agentperf/actions/workflows/ci.yml/badge.svg)](https://github.com/Biaogezi/sglang-agentperf/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Profiling-driven optimization of quantized agentic LLM serving on the real SGLang runtime.

This repository is the experiment and evidence layer for an upstream SGLang optimization
project. It deliberately does **not** reimplement paging, radix caching, scheduling, or an LLM
runtime. Runtime changes live as reviewable patches against a pinned SGLang checkout; this repository
owns workload definitions, launch configurations, profiling automation, result validation, and
performance reports.

## Research question

For long-prefix, multi-turn agent workloads on a 24 GiB NVIDIA A10, where does time go after
weight quantization, and which SGLang runtime change improves end-to-end latency or throughput
without unacceptable quality loss?

The first profiling pass evaluates three hypotheses:

1. CUDA Graph bucket coverage and padding are inefficient for bursty agent batch shapes.
2. Quantized linear backends have shape-dependent crossovers that static selection misses.
3. Long prefills interfere with latency-sensitive decode and need SLO-aware chunk control.

Only a bottleneck demonstrated in a trace becomes an optimization target.

## Evaluation contract

- Primary GPU: NVIDIA A10 24 GiB (SM 86).
- Primary model: Qwen3-8B.
- Secondary model: Qwen3.5-9B after the baseline is stable.
- Precision paths: FP16, W8A8 INT8, AWQ/GPTQ-Marlin INT4 when compatible.
- Workloads: decode-bound, prefill-bound, generated shared-prefix, and tool-calling replay.
- Metrics: p50/p95/p99 TTFT, TPOT and ITL; request/token throughput; HBM; cache hit rate;
  CUDA Graph coverage; CPU/GPU/kernel time.
- Every reported point uses warmup plus at least three measured repetitions.
- A change is retained only if it improves throughput by >=10%, p99 TTFT/TPOT by >=15%, or
  capacity by >=20%, while keeping the chosen quality metric within 1% of baseline.

## Layout

```text
configs/                 versioned models, server profiles and workload matrix
src/agentperf/           pure-Python plan builder, runner and result summarizer
scripts/                 remote bootstrap, environment capture and Nsight helpers
tests/                   CPU-only validation for orchestration and result logic
docs/                    experiment protocol, profiling checklist and decision log
upstream/sglang/          ignored checkout of the pinned SGLang revision
results/ and profiles/   ignored generated evidence
```

## Experiment flow

```mermaid
flowchart LR
    C[Versioned model/profile/workload matrix] --> R[AgentPerf runner]
    P[Pinned SGLang checkout + patch series] --> R
    R --> S[SGLang server on GPU]
    S --> B[Serving benchmark]
    S --> Q[Prompt-NLL quality gate]
    S --> T[Torch CPU/GPU trace]
    B --> A[Three-run aggregate]
    Q --> G[FP16-relative quality decision]
    T --> H[Kernel/runtime hotspot tables]
    A --> E[Evidence snapshot + raw hashes]
    G --> D[Accept or reject]
    H --> D
    E --> D
```

Each optimization starts from a trace-backed hypothesis. Static controls and the source candidate
run through the same matrix; the patch is retained only when it clears both the performance and
quality gates.

## Local preparation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
agentperf validate --config configs/experiment_matrix.json
agentperf plan --config configs/experiment_matrix.json --suite smoke
pytest
```

On Windows, activate with `.venv\Scripts\Activate.ps1`.

## GPU host bootstrap

After copying this repository to the Ubuntu GPU host, experiments run in a digest-pinned SGLang
CUDA 12.9 development image. The NVIDIA driver remains external to the container:

```bash
bash scripts/bootstrap_remote.sh
cp configs/host.env.example .env
# Edit .env to point to downloaded model directories.
bash scripts/run_gpu_smoke.sh
```

Run the smoke gate before downloading every quantized checkpoint or launching the full matrix.

`w8a8_int8` requires an already calibrated, per-channel INT8 checkpoint. Pointing that flag at
ordinary FP16 weights can produce plausible throughput with unusable output; `run-quality` is the
mandatory gate before any W8A8 performance result is accepted.

To publish a compact, auditable result without committing multi-megabyte request traces, use
`agentperf snapshot-evidence`. It copies the manifest plus either the benchmark aggregate CSV or
the quality JSON, and records the size and SHA-256 of every ignored raw artifact.
`agentperf snapshot-trace-evidence` publishes the compact profiler tables while recording the
size and SHA-256 of the ignored raw trace.

## Evidence policy

No percentage enters the README or resume unless the raw SGLang JSONL output, launch command,
environment manifest, pinned commits, and at least three repetitions are present. Kernel-only
results are labelled kernel-only. Simulator results are never presented as end-to-end latency.

## Current findings

- A revision-pinned, calibrated W8A8 checkpoint passes the pre-registered prompt-NLL gate:
  mean NLL is 2.89195 versus 2.89436 for FP16 across the same 335 scored tokens (delta -0.00241;
  acceptance threshold +0.02). Performance results are accepted only after this gate.
- Quantization has a measured phase crossover on the A10: AWQ-Marlin is 8.9% faster than W8A8 in
  decode output throughput, while calibrated W8A8 is 59.6% faster than AWQ on 8K-prefill input
  throughput and cuts p99 TTFT by 23.6%. The project therefore reports workload-specific operating
  points instead of claiming one format is always best. See the
  [quantization crossover report](docs/QUANTIZATION_CROSSOVER.md).
- AWQ-Marlin removes the A10 KV-capacity failure mode seen in the FP16 load test and materially
  improves decode-bound and shared-prefix serving.
- The extra KV capacity exposes long-prefill/decode interference: default AWQ reaches high input
  throughput but poor streaming tail latency on the 8K workload.
- A measured static 1024-token chunk profile reduces p99 TPOT by 72.9% at a 7.8% input-throughput
  cost. The first adaptive source patch did **not** beat that tuned static control and was rejected.
- On calibrated W8A8, the same static 1024-token operating point reduces p99 E2E by 21.9%, p99
  TPOT by 63.2%, and p99 ITL by 92.2% at a 21.5% input-throughput cost versus W8A8's default.

See [Optimization 001](docs/OPTIMIZATION_001_SLO_CHUNKING.md) for absolute values, controls, and
the negative-result decision.

The first bounded CPU+GPU trace shows 98.25% GPU activity and attributes 77.7% of kernel time to
AWQ-Marlin variants. See [Profile 001](docs/PROFILE_001_AWQ_MIXED_PREFILL.md) for the hotspot table
and next hypothesis.

Calibrated W8A8 traces attribute 64.7% of mixed-prefill and 77.9% of batch-one decode kernel time
to CUTLASS INT8 GEMM. Separate dynamic activation quantization plus RMSNorm account for only 5.7%
and 3.5%, respectively. See [Profile 002](docs/PROFILE_002_W8A8_KERNELS.md) for the phase-separated
analysis and optimization selection decision.

A Triton candidate fuses RMSNorm with dynamic per-token INT8 quantization and improves the
standalone operator pair by 31.5%–110.3%. A subsequent dispatch audit found that the historical
serving runs selected a different quantization method, so their switch was a no-op: those runs
do **not** establish fusion quality or performance. The candidate remains default-off, and new
experiments require positive GPU-trace proof of execution. See
[Optimization 003](docs/OPTIMIZATION_003_W8A8_NORM_QUANT_FUSION.md).

The resulting FP16-reduction experiment did not improve the trace-relevant 8K workload and changed
long-sequence outputs, so it was rejected. See
[Optimization 002](docs/OPTIMIZATION_002_MARLIN_REDUCTION.md).

The SM86 short-prefill INT8 GEMM candidate now has exact GPU correctness, quant-method fallback
tests and positive serving-trace dispatch proof. Its first same-runtime three-round serving test
improves nominal 96-token text-workload throughput by 5.02%, but only 0.73% at nominal 128;
that dataset decoded/re-tokenized inputs and was not fixed-shape. It has **not** cleared
the 10% serving gate. This distinction is preserved in
[Optimization 004](docs/OPTIMIZATION_004_SM86_INT8_PREFILL.md).

For a Chinese walkthrough of architecture, implementation ownership and interview questions, read
[项目讲解与面试准备](docs/PROJECT_GUIDE.zh-CN.md).
