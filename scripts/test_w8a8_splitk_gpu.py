"""GPU correctness gate, including edge tiles, bias and adversarial integer inputs."""

import itertools
import json

import torch
from sgl_kernel import int8_scaled_mm
from sglang.kernels.ops.quantization.int8_prefill_gemm import a10_int8_prefill_mm
from w8a8_splitk_candidate import splitk_mm


def main():
    torch.manual_seed(20260918)
    tests = 0
    for dtype, splits, bias_enabled, m in itertools.product(
        (torch.float16, torch.bfloat16), (1, 4), (False, True), (1, 17, 127, 129)
    ):
        k, n = 272, 152
        a = torch.randint(-128, 128, (m, k), device="cuda", dtype=torch.int8)
        b = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8).t()
        sa = torch.rand((m, 1), device="cuda") * 0.002
        sb = torch.rand((n, 1), device="cuda") * 0.002
        bias = torch.randn((n,), device="cuda", dtype=dtype) if bias_enabled else None
        integer_dot = (a.double() @ b.double()).float()
        expected = integer_dot * (sa * sb.view(1, -1))
        if bias is not None:
            expected = expected + bias.float()
        expected = expected.to(dtype)
        actual = splitk_mm(a, b, sa, sb, bm=64, splits=splits, dtype=dtype, bias=bias)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            a10_int8_prefill_mm(a, b, sa, sb, dtype, bias), expected, rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual, int8_scaled_mm(a, b, sa, sb, dtype, bias), rtol=0, atol=0
        )
        tests += 1
    # Extreme accumulations are compared using an exact FP64 integer reference.
    for value_a, value_b in [(-128, -128), (-128, 127), (0, 127), (127, 127)]:
        a = torch.full((17, 12288), value_a, device="cuda", dtype=torch.int8)
        b = torch.full((136, 12288), value_b, device="cuda", dtype=torch.int8).t()
        sa = torch.full((17, 1), 0.001, device="cuda")
        sb = torch.full((136, 1), 0.002, device="cuda")
        expected = ((a.double() @ b.double()).float() * (sa * sb.view(1, -1))).half()
        torch.testing.assert_close(
            a10_int8_prefill_mm(a, b, sa, sb, torch.float16), expected, rtol=0, atol=0
        )
        for splits in (1, 4, 8):
            actual = splitk_mm(a, b, sa, sb, bm=32, splits=splits)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            tests += 1
    print(
        json.dumps(
            {"passed": tests, "reference": "exact FP64 integer dot; FP32 scales; CUTLASS parity"}
        )
    )


if __name__ == "__main__":
    main()
