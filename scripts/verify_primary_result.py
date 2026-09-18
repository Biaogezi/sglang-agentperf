"""Recompute the scoped primary result from public artifacts, without a GPU.

Run verify_evidence.py separately to check the snapshots' byte-level integrity.
This check is not a new experiment or a claim about unmeasured workloads.
"""

import argparse
import csv
import gzip
import json
import math
import statistics
from pathlib import Path

from agentperf.quality import compare_quality


def verify(root: Path) -> dict:
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def read(directory, filename):
        return json.loads((root / directory / filename).read_text())

    def p99(values):
        values = sorted(values)
        position = (len(values) - 1) * 0.99
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] + (position - low) * (values[high] - values[low])

    try:
        rows = {}
        sources, commands, runtime_sources = set(), set(), set()
        metrics = {}
        for arm in ("off", "on"):
            data = json.loads(
                gzip.decompress(
                    (root / f"final_short_requests_{arm}/request_metrics.json.gz").read_bytes()
                )
            )
            rows[arm] = {
                Path(row["file"]).stem.rsplit("__", 2)[-1]: row
                for row in data["runs"]
                if "__short_prefill_128__" in row["file"]
            }
            require(set(rows[arm]) == {"r1", "r2", "r3"}, "Require three primary repetitions")
            require(len(data["runs"]) == 9, "Require complete three-shape regression")
            for row in data["runs"]:
                require(
                    row["completed"] == 160
                    and len(row["errors"]) == 160
                    and not any(row["errors"]),
                    "Missing or failed requests",
                )
                require(
                    row["input_lengths_valid"] and row["output_lens"] == [1] * 160,
                    "Wrong native-input/output-length protocol",
                )
                require(
                    row["duration_s"] > 0 and math.isfinite(row["duration_s"]),
                    "Invalid measurement duration",
                )
                require(
                    row["total_input_tokens"] == sum(row["input_lens"]), "Inconsistent token total"
                )
            for row in rows[arm].values():
                require(
                    row["input_lens"] == [128] * 160 and len(row["ttft_s"]) == 160,
                    "Primary input lengths or timings differ",
                )
            throughput = [
                row["total_input_tokens"] / row["duration_s"] for row in rows[arm].values()
            ]
            ttft = [1000 * p99(row["ttft_s"]) for row in rows[arm].values()]
            metrics[arm] = {
                "input_tok_s_mean": statistics.mean(throughput),
                "input_tok_s_sample_sd": statistics.stdev(throughput),
                "mean_of_run_p99_ttft_ms": statistics.mean(ttft),
            }
            manifest = read(f"final_short_{arm}", "manifest.json")
            require(len(manifest["launches"]) == 3, "Missing source manifests")
            for launch in manifest["launches"]:
                m = launch["manifest"]
                require(
                    bool(m["source_files_sha256"]["runtime_candidate"]), "Missing runtime hashes"
                )
                runtime_sources.add(
                    json.dumps(m["source_files_sha256"]["runtime_candidate"], sort_keys=True)
                )
                sources.add(json.dumps(m["source_files_sha256"], sort_keys=True))
                commands.add(json.dumps(m["server_command"]))
                require(
                    "--disable-overlap-schedule" in m["server_command"],
                    "Primary scheduler control missing",
                )
                require(
                    m["server_environment"]
                    == {
                        "SGLANG_A10_INT8_PREFILL": "true" if arm == "on" else "false",
                        "SGLANG_W8A8_FUSED_RMSNORM_QUANT": "false",
                    },
                    "Candidate switches differ from the selected scope",
                )
                for case in m["cases"]:
                    workload = m["config"]["workloads"][case["workload"]]
                    require(workload["max_concurrency"] == 1, "Wrong primary concurrency")
            with (root / f"final_short_{arm}/summary.csv").open(newline="") as handle:
                summary = next(
                    row
                    for row in csv.DictReader(handle)
                    if row["case"].endswith("__short_prefill_128")
                )
            require(
                math.isclose(
                    float(summary["input_throughput_mean"]),
                    metrics[arm]["input_tok_s_mean"],
                    rel_tol=1e-10,
                ),
                "Published throughput differs from request-level reconstruction",
            )
            require(
                math.isclose(
                    float(summary["ttft_p99_ms_mean"]),
                    metrics[arm]["mean_of_run_p99_ttft_ms"],
                    rel_tol=1e-10,
                ),
                "Published TTFT differs from request-level reconstruction",
            )
        require(len(sources) == len(commands) == 1, "Mixed source or server configuration")
        for repetition in rows["off"]:
            require(
                rows["off"][repetition]["output_text_sha256"]
                == rows["on"][repetition]["output_text_sha256"],
                "Primary outputs differ",
            )
        gain = (metrics["on"]["input_tok_s_mean"] / metrics["off"]["input_tok_s_mean"] - 1) * 100
        require(gain >= 10, "Primary throughput gate failed")
        quality = compare_quality(
            root / "final_quality_off/quality.json",
            root / "final_quality_on/quality.json",
            max_nll_increase=0.02,
        )
        fp16 = compare_quality(
            root / "final_fp16_quality/quality.json",
            root / "final_quality_on/quality.json",
            max_nll_increase=0.02,
        )
        require(quality["passed"] and fp16["passed"], "NLL gate failed")
        tasks_off = read("final_quality_off", "task_outputs.json")
        tasks_on = read("final_quality_on", "task_outputs.json")
        require(len(tasks_off) == len(tasks_on) == 40, "Incomplete task regression")
        require(
            all(
                a["id"] == b["id"]
                and a["text"] == b["text"]
                and not (a["passed"] and not b["passed"])
                for a, b in zip(tasks_off, tasks_on, strict=True)
            ),
            "Task regression differs",
        )
        gpu = read("final_gpu_validation", "manifest.json")
        for directory in (
            "final_gpu_validation",
            "final_quality_off",
            "final_quality_on",
            "final_fp16_quality",
        ):
            manifest = read(directory, "manifest.json")
            runtime_sources.add(
                json.dumps(manifest["source_files_sha256"]["runtime_candidate"], sort_keys=True)
            )
        require(len(runtime_sources) == 1, "Quality/GPU/performance runtime sources differ")
        require(
            len(gpu["tests"]) == 4 and all(t["returncode"] == 0 for t in gpu["tests"]),
            "GPU validation incomplete or failed",
        )
        counts = {}
        for arm in ("off", "on"):
            with (root / f"final_proof_{arm}/kernels.csv").open(newline="") as handle:
                counts[arm] = sum(
                    int(row["count"])
                    for row in csv.DictReader(handle)
                    if "_int8_prefill" in row["name"]
                )
        require(counts == {"off": 0, "on": 360}, "Expected execution proof missing")
        return {
            "passed": True,
            "scope": "A10 W8A8; native 128 input / 1 output token; concurrency 1; no overlap",
            "metrics": metrics,
            "throughput_gain_pct": gain,
            "same_weight_nll_delta": quality["mean_nll_delta"],
            "fp16_relative_nll_delta": fp16["mean_nll_delta"],
            "primary_text_pairs_equal": 480,
            "task_text_pairs_equal": 40,
            "note": "Text agreement is not 40/40 task accuracy or a universal speedup.",
        }
    except (ValueError, KeyError, OSError, StopIteration) as error:
        return {"passed": False, "error": str(error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path("evidence/a10"))
    result = verify(parser.parse_args().root)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
