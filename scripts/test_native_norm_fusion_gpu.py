"""Exercise the native W8A8 norm->tuple->GEMM path and unfused fallback."""

import json
from unittest.mock import patch

import torch
from sgl_kernel import fused_add_rmsnorm, rmsnorm
from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.layers import layernorm
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config, W8A8Int8LinearMethod


def main():
    torch.manual_seed(20260918)
    results = []
    for dtype in (torch.float16, torch.bfloat16):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.randint(-127, 128, (6144, 4096), dtype=torch.int8, device="cuda"),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.rand((6144, 1), device="cuda") * 0.001, requires_grad=False
        )
        method = W8A8Int8LinearMethod(W8A8Int8Config())
        layer.quant_method = method
        method.process_weights_after_loading(layer)
        norm = layernorm.RMSNorm(4096, eps=1e-6).to(device="cuda", dtype=dtype)
        norm.weight.data.copy_(torch.randn(4096, device="cuda", dtype=dtype))
        assert layernorm._is_dynamic_per_token_int8_linear(layer)
        assert not layernorm._is_dynamic_per_token_int8_linear(None)
        assert not layernorm._is_dynamic_per_token_int8_linear(torch.nn.Linear(4, 4))
        for m in (1, 16, 79, 80, 96, 128, 129, 1024):
            for residual_path in (False, True):
                x = torch.randn((m, 4096), device="cuda", dtype=dtype)
                residual = torch.randn_like(x) if residual_path else None
                expected_x = x.clone()
                expected_r = residual.clone() if residual_path else None
                if residual_path:
                    fused_add_rmsnorm(expected_x, expected_r, norm.weight, 1e-6)
                else:
                    expected_x = rmsnorm(expected_x, norm.weight, 1e-6)
                expected_q, expected_s = per_token_quant_int8(expected_x)
                with patch.object(layernorm, "_enable_w8a8_fused_rmsnorm_quant", True):
                    actual = norm.forward_cuda(x, residual, quant_linear=layer)
                if residual_path:
                    actual, actual_r = actual
                    torch.testing.assert_close(actual_r, expected_r, rtol=0, atol=0)
                assert isinstance(actual, tuple) and len(actual) == 3
                q, scale, out_dtype = actual
                assert out_dtype == dtype
                delta = (q.short() - expected_q.short()).abs()
                mismatch = (delta != 0).float().mean().item()
                assert delta.max().item() <= 1 and mismatch < 0.001
                torch.testing.assert_close(scale, expected_s, rtol=1e-3, atol=1e-7)
                for custom in (False, True):
                    method._a10_prefill = custom
                    # Tuple must not run activation quantization for a second time.
                    with patch(
                        "sglang.srt.layers.quantization.w8a8_int8.per_token_quant_int8",
                        side_effect=AssertionError("Unexpected requantization"),
                    ):
                        output = method.apply(layer, actual)
                    reference = method.apply(layer, expected_x)
                    relative_l2 = (
                        (output.float() - reference.float()).norm()
                        / reference.float().norm().clamp_min(1e-10)
                    ).item()
                    assert relative_l2 < 0.002
                    results.append(
                        {
                            "dtype": str(dtype),
                            "m": m,
                            "residual": residual_path,
                            "custom_gemm": custom,
                            "quant_mismatch_fraction": mismatch,
                            "relative_l2": relative_l2,
                        }
                    )
    print(json.dumps({"passed": len(results), "cases": results}, indent=2))


if __name__ == "__main__":
    main()
