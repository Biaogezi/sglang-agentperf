#!/usr/bin/env python3
"""Benchmark W8A8 GEMM backends on the exact Qwen3-8B projection shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl
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


@triton.jit
def _scale_int32_epilogue(
    accumulator,
    output,
    scale_a,
    scale_b,
    n_cols,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    values = tl.load(accumulator + row * n_cols + cols, mask=mask).to(tl.float32)
    row_scale = tl.load(scale_a + row)
    col_scale = tl.load(scale_b + cols, mask=mask)
    tl.store(output + row * n_cols + cols, values * row_scale * col_scale, mask=mask)


def triton_epilogue_int_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    accumulator = torch._int_mm(a, b)
    output = torch.empty(accumulator.shape, device=a.device, dtype=torch.float16)
    _scale_int32_epilogue[(a.shape[0], triton.cdiv(b.shape[1], 1024))](
        accumulator,
        output,
        scale_a,
        scale_b,
        b.shape[1],
        BLOCK_N=1024,
        num_warps=8,
    )
    return output


def benchmark(name: str, rows: int, k: int, n: int) -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(rows + k + n)
    a = torch.randint(-16, 17, (rows, k), device="cuda", dtype=torch.int8, generator=generator)
    # Match SGLang's loaded weight layout: logical [K, N], physically transposed from [N, K].
    b = torch.randint(-16, 17, (n, k), device="cuda", dtype=torch.int8, generator=generator).t()
    scale_a = torch.rand((rows, 1), device="cuda", generator=generator) * 0.01
    scale_b = torch.rand((n, 1), device="cuda", generator=generator) * 0.01

    def current() -> torch.Tensor:
        return int8_scaled_mm(a, b, scale_a, scale_b, torch.float16, None)

    def native() -> torch.Tensor:
        return torch_int_mm(a, b, scale_a, scale_b)

    def native_with_triton_epilogue() -> torch.Tensor:
        return triton_epilogue_int_mm(a, b, scale_a, scale_b)

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
    triton_actual = native_with_triton_epilogue()
    triton_ms = float(triton.testing.do_bench(native_with_triton_epilogue, warmup=50, rep=200))
    result.update(
        {
            "torch_int_mm_triton_epilogue_ms": triton_ms,
            "torch_triton_vs_current_speedup_pct": (current_ms / triton_ms - 1.0) * 100.0,
            "triton_epilogue_max_abs_error": float((expected - triton_actual).abs().max()),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 16, 128, 1024])
    parser.add_argument("--output")
    args = parser.parse_args()

    results = []
    for name, (k, n) in QWEN3_8B_PROJECTIONS.items():
        for rows in args.rows:
            results.append(benchmark(name, rows, k, n))
        torch.cuda.empty_cache()
    payload = {"device": torch.cuda.get_device_name(), "results": results}
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
