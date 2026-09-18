"""Compare the exact shipped kernel with its prototype and the installed baseline."""

import argparse
import json
import statistics
import subprocess
from functools import partial
from pathlib import Path

import torch
import triton
from sgl_kernel import int8_scaled_mm
from sglang.kernels.ops.quantization.int8_prefill_gemm import a10_int8_prefill_mm
from w8a8_splitk_candidate import splitk_mm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(17)
    results = []
    for m in (96, 128):
        for n in (6144, 24576):
            k = 4096
            a = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda")
            b = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda").t()
            sa, sb = (
                torch.rand((m, 1), device="cuda") * 0.001,
                torch.rand((n, 1), device="cuda") * 0.001,
            )
            functions = {
                "baseline": partial(int8_scaled_mm, a, b, sa, sb, torch.float16),
                "prototype": partial(splitk_mm, a, b, sa, sb, bm=128, bn=128, bk=128, splits=1),
                "production": partial(a10_int8_prefill_mm, a, b, sa, sb, torch.float16),
            }
            expected = functions["baseline"]()
            for fn in functions.values():
                torch.testing.assert_close(fn(), expected, rtol=0, atol=0)
            samples = {key: [] for key in functions}
            states = []
            for repeat in range(3):
                names = list(functions)
                names = names[repeat:] + names[:repeat]
                for name in names:
                    samples[name].append(
                        triton.testing.do_bench_cudagraph(functions[name], rep=300)
                    )
                    states.append(
                        {
                            "variant": name,
                            "round": repeat,
                            "gpu": subprocess.check_output(
                                [
                                    "nvidia-smi",
                                    "--query-gpu=temperature.gpu,clocks.sm,power.draw",
                                    "--format=csv,noheader",
                                ],
                                text=True,
                            ).strip(),
                        }
                    )
            row = {
                "m": m,
                "n": n,
                "k": k,
                "exact_equal": True,
                "samples_ms": samples,
                "medians_ms": {name: statistics.median(values) for name, values in samples.items()},
                "states": states,
            }
            results.append(row)
            Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
