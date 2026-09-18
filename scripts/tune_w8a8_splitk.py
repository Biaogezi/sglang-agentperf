"""Screen split-K shapes with CUDA Graph timing and numerical validation."""

import argparse
import hashlib
import itertools
import json
import statistics
import subprocess
from functools import partial
from pathlib import Path

import torch
import triton
from sgl_kernel import int8_scaled_mm
from w8a8_splitk_candidate import splitk_mm

SHAPES = {"qkv": (4096, 6144), "o": (4096, 4096), "gate_up": (4096, 24576), "down": (12288, 4096)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 8, 16])
    parser.add_argument("--projections", nargs="+", default=list(SHAPES))
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-tiles", nargs="+", type=int, default=[16])
    parser.add_argument("--splits", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--column-tiles", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--k-tiles", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--warps", nargs="+", type=int, default=[4])
    parser.add_argument("--stages", nargs="+", type=int, default=[3])
    parser.add_argument(
        "--real-weights",
        help="Local calibrated model: use layer-0 weights and quantized Gaussian activations",
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    payload = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "timing": "CUDA Graph; 3 rounds; median",
        "results": results,
        "rejected_configurations": [],
        "input_distribution": "real layer-0 weights, Gaussian activation quantization"
        if args.real_weights
        else "uniform full-range INT8",
    }
    payload["source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path(__file__).with_name("w8a8_splitk_candidate.py"))
    }
    torch.manual_seed(17)
    if args.real_weights:
        from safetensors import safe_open
        from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8

        model_root = Path(args.real_weights)
        weight_map = json.loads((model_root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        projection_names = {
            "qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            "o": ["self_attn.o_proj"],
            "gate_up": ["mlp.gate_proj", "mlp.up_proj"],
            "down": ["mlp.down_proj"],
        }

        def load_weight(key):
            with safe_open(model_root / weight_map[key], framework="pt", device="cpu") as handle:
                return handle.get_tensor(key).to("cuda")

    for name, m in itertools.product(args.projections, args.rows):
        k, n = SHAPES[name]
        a = torch.randint(-128, 128, (m, k), device="cuda", dtype=torch.int8)
        b = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8).t()
        sa = torch.rand((m, 1), device="cuda") * 0.001
        sb = torch.rand((n, 1), device="cuda") * 0.001
        if args.real_weights:
            a, sa = per_token_quant_int8(torch.randn((m, k), device="cuda", dtype=torch.float16))
            b = torch.cat(
                [load_weight(f"model.layers.0.{part}.weight") for part in projection_names[name]]
            ).t()
            sb = torch.cat(
                [
                    load_weight(f"model.layers.0.{part}.weight_scale")
                    for part in projection_names[name]
                ]
            ).float()
        baseline = partial(int8_scaled_mm, a, b, sa, sb, torch.float16)
        # FP32 exactly represents these input integers and their bounded dot products.
        with torch.no_grad():
            torch.backends.cuda.matmul.allow_tf32 = False
            reference = ((a.float() @ b.float()) * (sa * sb.view(1, -1))).half()
        torch.testing.assert_close(baseline(), reference, rtol=1e-3, atol=1e-4)
        candidates = []
        for bm, bn, bk, splits, warps, stages in itertools.product(
            args.batch_tiles, args.column_tiles, args.k_tiles, args.splits, args.warps, args.stages
        ):
            config = {
                "bm": bm,
                "bn": bn,
                "bk": bk,
                "splits": splits,
                "warps": warps,
                "stages": stages,
            }
            candidate = partial(splitk_mm, a, b, sa, sb, **config)
            try:
                actual = candidate()
                torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-4)
                elapsed = triton.testing.do_bench_cudagraph(candidate, rep=60)
                candidates.append({"config": config, "screen_ms": elapsed})
            except Exception as error:  # noqa: BLE001 - record compiler/resource/numerical rejection
                payload["rejected_configurations"].append(
                    {"shape": [m, n, k], "config": config, "error": str(error)[:1000]}
                )
                print(
                    json.dumps({"shape": [m, n, k], "config": config, "error": str(error)[:300]}),
                    flush=True,
                )
        if not candidates:
            raise RuntimeError(f"No correct configuration for {name} M={m}")
        best = min(candidates, key=lambda x: x["screen_ms"])
        config = best["config"]
        candidate = partial(splitk_mm, a, b, sa, sb, **config)
        base_samples, cand_samples = [], []
        for repeat in range(3):
            order = [(baseline, base_samples), (candidate, cand_samples)]
            if repeat % 2:
                order.reverse()
            for fn, samples in order:
                samples.append(triton.testing.do_bench_cudagraph(fn, rep=200))
        row = {
            "projection": name,
            "m": m,
            "k": k,
            "n": n,
            "best": config,
            "baseline_ms": statistics.median(base_samples),
            "candidate_ms": statistics.median(cand_samples),
            "baseline_samples": base_samples,
            "candidate_samples": cand_samples,
            "screen": candidates,
        }
        row["speedup"] = row["baseline_ms"] / row["candidate_ms"]
        row["gpu_state"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        results.append(row)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({key: value for key, value in row.items() if key != "screen"}), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
