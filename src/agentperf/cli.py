from __future__ import annotations

import argparse
import json
from pathlib import Path

from .commands import benchmark_command, server_command, shell_join
from .config import build_plan, load_config
from .quality import compare_quality, score_corpus
from .report import check_run_equivalence, compare_summaries, summarize_run
from .runner import run_plan
from .trace import analyze_trace


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

    compare = subparsers.add_parser("compare", help="compare matching workloads in two summaries")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--output", required=True)

    trace = subparsers.add_parser("analyze-trace", help="aggregate a Torch profiler trace")
    trace.add_argument("--trace", required=True)
    trace.add_argument("--output-dir", required=True)

    equivalence = subparsers.add_parser(
        "check-equivalence", help="compare deterministic output fields between two runs"
    )
    equivalence.add_argument("--baseline-run", required=True)
    equivalence.add_argument("--candidate-run", required=True)

    quality = subparsers.add_parser(
        "score-corpus", help="score a fixed corpus with SGLang prompt logprobs"
    )
    quality.add_argument("--endpoint", default="http://127.0.0.1:30000")
    quality.add_argument("--corpus", required=True)
    quality.add_argument("--output", required=True)
    quality.add_argument("--timeout-s", type=float, default=120.0)

    quality_compare = subparsers.add_parser(
        "compare-quality", help="gate candidate corpus NLL against a baseline"
    )
    quality_compare.add_argument("--baseline", required=True)
    quality_compare.add_argument("--candidate", required=True)
    quality_compare.add_argument("--max-nll-increase", type=float, default=0.02)
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
    elif args.command == "compare":
        rows = compare_summaries(
            Path(args.baseline), Path(args.candidate), Path(args.output)
        )
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    elif args.command == "analyze-trace":
        summary = analyze_trace(Path(args.trace), Path(args.output_dir))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    elif args.command == "check-equivalence":
        result = check_run_equivalence(
            Path(args.baseline_run), Path(args.candidate_run)
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if not result["equivalent"]:
            raise SystemExit(1)
    elif args.command == "score-corpus":
        result = score_corpus(
            endpoint=args.endpoint,
            corpus_path=Path(args.corpus),
            output_path=Path(args.output),
            timeout_s=args.timeout_s,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "compare-quality":
        result = compare_quality(
            Path(args.baseline),
            Path(args.candidate),
            max_nll_increase=args.max_nll_increase,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
