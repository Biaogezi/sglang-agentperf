# SGLang-AgentPerf

Profiling-driven optimization of quantized agentic LLM serving on the real SGLang runtime.

This repository is the experiment and evidence layer for an upstream SGLang optimization
project. It deliberately does **not** reimplement paging, radix caching, scheduling, or an LLM
runtime. Runtime changes live as reviewable commits on a pinned SGLang fork; this repository
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

## Evidence policy

No percentage enters the README or resume unless the raw SGLang JSONL output, launch command,
environment manifest, pinned commits, and at least three repetitions are present. Kernel-only
results are labelled kernel-only. Simulator results are never presented as end-to-end latency.

## Current findings

- AWQ-Marlin removes the A10 KV-capacity failure mode seen in the FP16 load test and materially
  improves decode-bound and shared-prefix serving.
- The extra KV capacity exposes long-prefill/decode interference: default AWQ reaches high input
  throughput but poor streaming tail latency on the 8K workload.
- A measured static 1024-token chunk profile reduces p99 TPOT by 72.9% at a 7.8% input-throughput
  cost. The first adaptive source patch did **not** beat that tuned static control and was rejected.

See [Optimization 001](docs/OPTIMIZATION_001_SLO_CHUNKING.md) for absolute values, controls, and
the negative-result decision.

The first bounded CPU+GPU trace shows 98.25% GPU activity and attributes 77.7% of kernel time to
AWQ-Marlin variants. See [Profile 001](docs/PROFILE_001_AWQ_MIXED_PREFILL.md) for the hotspot table
and next hypothesis.

The resulting FP16-reduction experiment did not improve the trace-relevant 8K workload and changed
long-sequence outputs, so it was rejected. See
[Optimization 002](docs/OPTIMIZATION_002_MARLIN_REDUCTION.md).
