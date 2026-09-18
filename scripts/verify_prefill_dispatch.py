"""Validate one fixed candidate on held-out batch shapes; no per-shape retuning."""

import argparse
import json
import statistics
from functools import partial
from pathlib import Path

import torch
import triton
from sgl_kernel import int8_scaled_mm
from w8a8_splitk_candidate import splitk_mm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(999)
    results = []
    for n, k in [(6144, 4096), (24576, 4096), (4096, 12288)]:
        for m in [32, 48, 64, 80, 96, 112, 128, 144, 192, 256]:
            a = torch.randint(-128, 128, (m, k), device="cuda", dtype=torch.int8)
            b = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8).t()
            sa = torch.rand((m, 1), device="cuda") * 0.001
            sb = torch.rand((n, 1), device="cuda") * 0.001
            base = partial(int8_scaled_mm, a, b, sa, sb, torch.float16)
            candidate = partial(splitk_mm, a, b, sa, sb, bm=128, bn=128, bk=128, splits=1)
            expected, actual = base(), candidate()
            equal = torch.equal(expected, actual)
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
            samples = {"baseline": [], "candidate": []}
            for repeat in range(3):
                order = [("baseline", base), ("candidate", candidate)]
                if repeat % 2:
                    order.reverse()
                for name, fn in order:
                    samples[name].append(triton.testing.do_bench_cudagraph(fn, rep=100))
            row = {
                "m": m,
                "n": n,
                "k": k,
                "exact_equal": equal,
                "samples_ms": samples,
                "speedup": statistics.median(samples["baseline"])
                / statistics.median(samples["candidate"]),
            }
            results.append(row)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
