#!/usr/bin/env python3
"""Microbenchmark the candidate RMSNorm + dynamic INT8 quantization fusion."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
from sgl_kernel import fused_add_rmsnorm, rmsnorm
from sglang.kernels.ops.quantization.int8_kernel import (
    per_token_quant_int8,
    rmsnorm_quant_int8,
)


def benchmark(
    rows: int, hidden_size: int, residual_path: bool, rounds: int
) -> dict[str, float | int | bool | list[float]]:
    # Zeros keep in-place residual kernels stable across benchmark repetitions; execution time is
    # shape-driven and the randomized numerical parity check lives in the registered GPU test.
    x = torch.zeros((rows, hidden_size), device="cuda", dtype=torch.float16)
    weight = torch.ones((hidden_size,), device="cuda", dtype=torch.float16)
    residual = torch.zeros_like(x) if residual_path else None

    if residual_path:

        def baseline() -> None:
            fused_add_rmsnorm(x, residual, weight, 1e-6)
            per_token_quant_int8(x)

    else:

        def baseline() -> None:
            per_token_quant_int8(rmsnorm(x, weight, 1e-6))

    def candidate() -> None:
        rmsnorm_quant_int8(x, weight, 1e-6, residual=residual)

    baseline_samples = [
        float(triton.testing.do_bench(baseline, warmup=100, rep=500)) for _ in range(rounds)
    ]
    candidate_samples = [
        float(triton.testing.do_bench(candidate, warmup=100, rep=500)) for _ in range(rounds)
    ]
    baseline_ms = statistics.median(baseline_samples)
    candidate_ms = statistics.median(candidate_samples)
    return {
        "rows": rows,
        "hidden_size": hidden_size,
        "residual_path": residual_path,
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "baseline_samples_ms": baseline_samples,
        "candidate_samples_ms": candidate_samples,
        "speedup_pct": (baseline_ms / candidate_ms - 1.0) * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 128, 1024])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()

    results = [
        benchmark(rows, args.hidden_size, residual_path, args.rounds)
        for rows in args.rows
        for residual_path in (False, True)
    ]
    payload = {"device": torch.cuda.get_device_name(), "results": results}
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
