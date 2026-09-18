"""Alternate source-switch OFF/ON launches and retain every original manifest/log."""

import argparse
import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from agentperf.config import load_config
from agentperf.report import compare_summaries, summarize_run
from agentperf.runner import run_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_matrix.json")
    parser.add_argument(
        "--suite",
        choices=["short", "short_decode", "core", "proof", "agent", "batch_decode"],
        default="short",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-root", default="results/paired")
    parser.add_argument("--prefill-backend", choices=["disabled", "breakable", "tc_piecewise"])
    parser.add_argument("--candidate", choices=["gemm", "fusion", "combined"], default="gemm")
    parser.add_argument("--disable-overlap", action="store_true")
    args = parser.parse_args()
    config = copy.deepcopy(load_config(args.config))
    config["defaults"]["repetitions"] = 1
    for profile, enabled in [("prefill_off", "false"), ("prefill_on", "true")]:
        config["server_profiles"][profile] = copy.deepcopy(config["server_profiles"]["baseline"])
        config["server_profiles"][profile]["env"] = {
            "SGLANG_A10_INT8_PREFILL": enabled if args.candidate != "fusion" else "false",
            "SGLANG_W8A8_FUSED_RMSNORM_QUANT": enabled if args.candidate != "gemm" else "false",
        }
        if args.disable_overlap:
            config["server_profiles"][profile]["args"].append("--disable-overlap-schedule")
        if args.prefill_backend:
            config["server_profiles"][profile]["args"] += [
                "--cuda-graph-backend-prefill",
                args.prefill_backend,
            ]
    for length in (96, 128, 160):
        config["workloads"][f"short_prefill_{length}"] = {
            "dataset": "random-ids",
            "num_prompts": 160,
            "max_concurrency": 1,
            "request_rate": "inf",
            "args": [
                "--tokenize-prompt",
                "--random-input-len",
                str(length),
                "--random-output-len",
                "1",
                "--random-range-ratio",
                "1",
            ],
        }
    config["suites"]["short"] = [f"short_prefill_{length}" for length in (96, 128, 160)]
    for length in (96, 128, 160):
        workload = copy.deepcopy(config["workloads"][f"short_prefill_{length}"])
        workload["num_prompts"] = 80
        workload["args"][workload["args"].index("--random-output-len") + 1] = "32"
        config["workloads"][f"short_decode_{length}"] = workload
    config["suites"]["short_decode"] = [f"short_decode_{length}" for length in (96, 128, 160)]
    config["workloads"]["prefill_dispatch_proof"] = copy.deepcopy(
        config["workloads"]["short_prefill_128"]
    )
    config["workloads"]["prefill_dispatch_proof"]["num_prompts"] = 32
    config["workloads"]["prefill_dispatch_proof"]["args"] += [
        "--profile",
        "--profile-activities",
        "CPU",
        "GPU",
        "--profile-start-step",
        "5",
        "--profile-steps",
        "5",
        "--profile-output-dir",
        "/workspace/agentperf/profiles/torch",
        "--profile-prefix",
        "prefill-dispatch-proof",
    ]
    config["suites"]["proof"] = ["prefill_dispatch_proof"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(args.output_root) / f"{timestamp}__{args.suite}"
    root.mkdir(parents=True, exist_ok=False)
    manifests = {"prefill_off": [], "prefill_on": []}
    for repetition in range(1, args.repetitions + 1):
        order = ["prefill_off", "prefill_on"]
        if repetition % 2 == 0:
            order.reverse()
        for profile in order:
            print(f"PAIRED_BEGIN repetition={repetition} profile={profile}", flush=True)
            run = run_plan(
                config,
                model="qwen3_8b_w8a8",
                profile=profile,
                suite=args.suite,
                output_root=root / "launches",
            )
            aggregate = root / profile
            aggregate.mkdir(exist_ok=True)
            mapping = {}
            for source in run.glob("*.jsonl"):
                new_stem = source.stem.rsplit("__r", 1)[0] + f"__r{repetition}"
                mapping[source.stem] = new_stem
                shutil.copy2(source, aggregate / f"{new_stem}.jsonl")
                shutil.copy2(source.with_suffix(".log"), aggregate / f"{new_stem}.log")
            server_log = (run / "server.log").read_text(errors="replace")
            # Only marker lines change; originals remain in launches/.
            with (aggregate / "server.log").open("a", encoding="utf-8") as handle:
                for line in server_log.splitlines():
                    if line.startswith("AGENTPERF_CASE_"):
                        for old, new in mapping.items():
                            line = line.replace(old, new)
                    handle.write(line + "\n")
            manifest = json.loads((run / "manifest.json").read_text())
            manifests[profile].append(
                {"repetition": repetition, "run": str(run), "manifest": manifest}
            )
            (aggregate / "manifest.json").write_text(
                json.dumps(
                    {
                        "protocol": "alternating OFF/ON server restarts; each launch has warmup",
                        "launches": manifests[profile],
                    },
                    indent=2,
                )
                + "\n"
            )
            summarize_run(aggregate, aggregate / "summary.csv")
            print(f"PAIRED_END repetition={repetition} profile={profile} root={root}", flush=True)
    compare_summaries(
        root / "prefill_off/summary.csv", root / "prefill_on/summary.csv", root / "comparison.csv"
    )
    print(root, flush=True)


if __name__ == "__main__":
    main()
