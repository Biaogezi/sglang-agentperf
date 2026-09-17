from __future__ import annotations

import argparse
import json
from pathlib import Path

from .commands import benchmark_command, server_command, shell_join
from .config import build_plan, load_config
from .report import summarize_run
from .runner import run_plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentperf")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate an experiment config")
    validate.add_argument("--config", required=True)

    plan = subparsers.add_parser("plan", help="print the commands without executing them")
    plan.add_argument("--config", required=True)
    plan.add_argument("--model", default="qwen3_8b_fp16")
    plan.add_argument("--profile", default="baseline")
    plan.add_argument("--suite", default="smoke")

    run = subparsers.add_parser("run", help="launch SGLang and execute a suite")
    run.add_argument("--config", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--profile", default="baseline")
    run.add_argument("--suite", default="smoke")
    run.add_argument("--output-root", default="results")

    summarize = subparsers.add_parser("summarize", help="aggregate repeated SGLang JSONL results")
    summarize.add_argument("--run-dir", required=True)
    summarize.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config) if hasattr(args, "config") else None
    if args.command == "validate":
        print(f"valid: {args.config} (upstream {config['upstream_commit']})")
    elif args.command == "plan":
        print("server:")
        print(f"  {shell_join(server_command(config, args.model, args.profile))}")
        print("benchmarks:")
        for case in build_plan(config, model=args.model, profile=args.profile, suite=args.suite):
            command = benchmark_command(config, case, Path("results") / f"{case.case_id}.jsonl")
            print(f"  {case.case_id}: {shell_join(command)}")
    elif args.command == "run":
        run_dir = run_plan(
            config,
            model=args.model,
            profile=args.profile,
            suite=args.suite,
            output_root=Path(args.output_root),
        )
        summarize_run(run_dir, run_dir / "summary.csv")
        print(run_dir)
    elif args.command == "summarize":
        rows = summarize_run(Path(args.run_dir), Path(args.output))
        print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
