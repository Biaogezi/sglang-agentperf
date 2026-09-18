"""Exercise the actual --quantization w8a8_int8 method, not just its kernel."""

import json
import os
from unittest.mock import patch

import torch
from sglang.kernels.ops.quantization import int8_prefill_gemm as kernels
from sglang.srt.layers.layernorm import _is_dynamic_per_token_int8_linear
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config, W8A8Int8LinearMethod


def main():
    torch.manual_seed(20260918)
    os.environ["SGLANG_A10_INT8_PREFILL"] = "true"
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.randint(-128, 128, (6144, 4096), dtype=torch.int8, device="cuda"),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.rand((6144, 1), device="cuda") * 0.001, requires_grad=False
    )
    method = W8A8Int8LinearMethod(W8A8Int8Config())
    layer.quant_method = method
    method.process_weights_after_loading(layer)
    assert method._a10_prefill
    # Audit the previous norm-fusion experiment: this config does NOT select it.
    assert not _is_dynamic_per_token_int8_linear(layer)
    results = []
    for m in (1, 64, 79, 80, 96, 128, 129, 160):
        x = torch.randn((m, 4096), device="cuda", dtype=torch.float16)
        with patch.object(kernels, "a10_int8_prefill_mm", wraps=kernels.a10_int8_prefill_mm) as spy:
            method._a10_prefill = True
            actual = method.apply(layer, x)
            dispatched = spy.call_count
        method._a10_prefill = False
        expected = method.apply(layer, x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert dispatched == int(80 <= m <= 128)
        results.append({"m": m, "custom_kernel_calls": dispatched, "exact_equal": True})
    print(
        json.dumps(
            {
                "method": type(method).__name__,
                "previous_norm_fusion_eligible": False,
                "cases": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
