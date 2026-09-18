"""Data-sensitive INT8 timing: real layer-0 weights and Gaussian activation quantization."""

import argparse
import json
import statistics
from functools import partial
from pathlib import Path

import torch
import triton
from safetensors import safe_open
from sgl_kernel import int8_scaled_mm
from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.kernels.ops.quantization.int8_prefill_gemm import a10_int8_prefill_mm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/models/Qwen3-8B-W8A8")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.model)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]

    def load(name):
        with safe_open(root / index[name], framework="pt", device="cpu") as handle:
            return handle.get_tensor(name).to("cuda")

    torch.manual_seed(19)
    results = []
    for projection, names in {
        "qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        "gate_up": ["mlp.gate_proj", "mlp.up_proj"],
    }.items():
        b = torch.cat([load(f"model.layers.0.{name}.weight") for name in names]).t()
        # The loader stores channel scales in FP32 even when the checkpoint uses BF16.
        sb = torch.cat([load(f"model.layers.0.{name}.weight_scale") for name in names]).float()
        for m in (96, 128):
            a, sa = per_token_quant_int8(torch.randn((m, 4096), dtype=torch.float16, device="cuda"))
            baseline = partial(int8_scaled_mm, a, b, sa, sb, torch.float16)
            candidate = partial(a10_int8_prefill_mm, a, b, sa, sb, torch.float16)
            torch.testing.assert_close(baseline(), candidate(), rtol=0, atol=0)
            samples = {"baseline": [], "candidate": []}
            for repeat in range(3):
                order = [("baseline", baseline), ("candidate", candidate)]
                if repeat % 2:
                    order.reverse()
                for name, fn in order:
                    samples[name].append(triton.testing.do_bench_cudagraph(fn, rep=100))
            row = {
                "projection": projection,
                "m": m,
                "n": b.shape[1],
                "k": 4096,
                "samples_ms": samples,
                "exact_equal": True,
                "speedup": statistics.median(samples["baseline"])
                / statistics.median(samples["candidate"]),
            }
            results.append(row)
            Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
