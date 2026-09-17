#!/usr/bin/env python3
"""Benchmark W8A8 GEMM backends on the exact Qwen3-8B projection shapes."""

from __future__ import annotations

import argparse
import json

import torch
import triton
from sgl_kernel import int8_scaled_mm


QWEN3_8B_PROJECTIONS = {
    "qkv": (4096, 6144),
    "o": (4096, 4096),
    "gate_up": (4096, 24576),
    "down": (12288, 4096),
}


def torch_int_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    accumulator = torch._int_mm(a, b)
    return (accumulator.float() * scale_a * scale_b.view(1, -1)).half()


def benchmark(name: str, rows: int, k: int, n: int) -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(rows + k + n)
    a = torch.randint(-16, 17, (rows, k), device="cuda", dtype=torch.int8, generator=generator)
    # Match SGLang's loaded weight layout: logical [K, N], physically transposed from [N, K].
    b = torch.randint(
        -16, 17, (n, k), device="cuda", dtype=torch.int8, generator=generator
    ).t()
    scale_a = torch.rand((rows, 1), device="cuda", generator=generator) * 0.01
    scale_b = torch.rand((n, 1), device="cuda", generator=generator) * 0.01

    def current() -> torch.Tensor:
        return int8_scaled_mm(a, b, scale_a, scale_b, torch.float16, None)

    def native() -> torch.Tensor:
        return torch_int_mm(a, b, scale_a, scale_b)

    expected = current()
    current_ms = float(triton.testing.do_bench(current, warmup=50, rep=200))
    result: dict[str, object] = {
        "projection": name,
        "rows": rows,
        "k": k,
        "n": n,
        "current_ms": current_ms,
    }
    try:
        actual = native()
        native_ms = float(triton.testing.do_bench(native, warmup=50, rep=200))
    except RuntimeError as error:
        result["torch_int_mm_error"] = str(error)
        return result
    result.update(
        {
            "torch_int_mm_ms": native_ms,
            "torch_vs_current_speedup_pct": (current_ms / native_ms - 1.0) * 100.0,
            "max_abs_error": float((expected - actual).abs().max()),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 128, 1024])
    args = parser.parse_args()

    results = []
    for name, (k, n) in QWEN3_8B_PROJECTIONS.items():
        for rows in args.rows:
            results.append(benchmark(name, rows, k, n))
        torch.cuda.empty_cache()
    print(json.dumps({"device": torch.cuda.get_device_name(), "results": results}, indent=2))


if __name__ == "__main__":
    main()
