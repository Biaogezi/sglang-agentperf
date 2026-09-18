"""Prove arbitrary supported row counts reuse bounded compiled variants."""

import json
import re

import torch
import triton
from sgl_kernel import int8_scaled_mm
from sglang.kernels.ops.quantization.int8_prefill_gemm import _int8_prefill, a10_int8_prefill_mm


def main():
    torch.manual_seed(21)
    k, n = 4096, 6144
    b = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda").t()
    sb = torch.rand((n, 1), device="cuda") * 0.001
    hashes = {}
    compiled_metadata = {}
    for m in (80, 81, 83, 95, 96, 97, 111, 112, 113, 127, 128):
        a = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda")
        sa = torch.rand((m, 1), device="cuda") * 0.001
        out = a10_int8_prefill_mm(a, b, sa, sb, torch.float16)
        torch.testing.assert_close(out, int8_scaled_mm(a, b, sa, sb, torch.float16), rtol=0, atol=0)
        compiled = _int8_prefill.warmup(
            a,
            b,
            sa,
            sb,
            None,
            out,
            m,
            n,
            k,
            False,
            m == 128,
            grid=(triton.cdiv(m, 128), triton.cdiv(n, 128)),
            num_warps=4,
            num_stages=3,
            enable_fp_fusion=False,
        )
        hashes[m] = compiled.hash
        mma = sorted(set(re.findall(r"mma\.sync\.[^;\s]+", compiled.asm["ptx"])))
        assert any(".s8.s8" in instruction for instruction in mma)
        compiled_metadata[compiled.hash] = {
            "mma_instructions": mma,
            "registers_per_thread": compiled.n_regs,
            "shared_memory_bytes": compiled.metadata.shared,
        }
    assert len({value for m, value in hashes.items() if m < 128}) == 1
    assert len(set(hashes.values())) == 2
    print(
        json.dumps(
            {
                "passed": len(hashes),
                "compiled_variant_count": len(set(hashes.values())),
                "row_count_to_binary_hash": hashes,
                "compiled_metadata": compiled_metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
