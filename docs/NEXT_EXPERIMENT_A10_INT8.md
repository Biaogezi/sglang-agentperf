# Next experiment: A10 INT8 GEMM tuning

Status: historical build plan, superseded by [Optimization 004](OPTIMIZATION_004_SM86_INT8_PREFILL.md).

The installed CUTLASS wheel was not replaced. The later implementation uses a guarded Triton
kernel in the Python runtime, with GPU correctness, execution traces and serving evidence.
The build checkpoint below describes the earlier abandoned wheel-build route, not current status.

## Motivation and constraints

Two traces identify INT8 GEMM as the largest GPU cost (64.71% mixed prefill, 77.92% batch-one
decode). The pinned source routes SM86 and SM89 through `sm89_dispatch_shape`. This is deliberate:
both have a smaller shared-memory budget than SM80. Reusing SM80 configurations without checking
their shared-memory requirements is unsafe and is not a validated optimization.

The first exact-shape alternative-backend screen has only one positive point and is insufficient
to select a backend. Follow-up timing must include repeated CUDA Graph measurements, alternating
candidate/control order, warmup, correctness assertions, and GPU clock/temperature observations.

## Implementation gate

1. Build an isolated extension containing only the INT8 GEMM and required CUTLASS headers, using
   the pinned source and container. Keep the installed runtime library as the reference.
2. Verify its unchanged dispatch matches the installed reference numerically and in timing.
3. Enumerate a bounded set of SM86-valid tile/stage configurations for QKV/O/gate-up/down shapes.
   Record shared-memory requirements and reject unsupported configurations before benchmarking.
4. Validate full-range INT8 inputs, FP16/BF16 outputs, bias, irregular M, and all supported scale
   layouts against an independent accumulation/scaling reference.
5. Integrate only repeatable winners behind a default-off guard. Run same-commit, interleaved
   on/off controls with fusion disabled before the three-repeat serving and quality matrix.
6. Expand quality evaluation beyond the 335-token smoke corpus, especially if generated text
   changes. Keep measured throughput/latency and task quality as separate acceptance criteria.

## Build checkpoint

The complete upstream wheel build configured successfully inside the pinned container after
selecting `/opt/sglang/bin/python`; the host/harness virtual environment lacks the Torch build
dependencies. The full build was then stopped because it compiles many unrelated kernels and
architectures. No replacement wheel was installed, and no A10-specific dispatch was deployed.
The downloaded build dependencies and build directory remain on the cloud data disk for reuse.

The latest completed serving evidence, quality reports, and kernel measurements are synchronized
locally and published under `evidence/a10/`. No build process is intentionally left running at
this checkpoint.
