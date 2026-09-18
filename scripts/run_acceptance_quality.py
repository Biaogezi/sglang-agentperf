"""Same-checkpoint OFF/ON quality: fixed-token NLL, greedy tasks, and long retrieval."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from agentperf.commands import resolve_model_path, server_command
from agentperf.config import load_config, validate_model_artifact
from agentperf.quality import _post_json, compare_quality, score_corpus
from agentperf.runner import _git_head, terminate_process_group, wait_for_server
from agentperf.task_quality import judge, regression_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_matrix.json")
    parser.add_argument("--corpus", default="quality/wikitext2_128tokens.jsonl")
    parser.add_argument("--output-root", default="quality/acceptance")
    parser.add_argument("--model", default="qwen3_8b_w8a8")
    parser.add_argument("--profiles", nargs="+", default=["off", "on"])
    args = parser.parse_args()
    config = copy.deepcopy(load_config(args.config))
    validate_model_artifact(config, args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_model_path(config, args.model), local_files_only=True
    )
    root = Path(args.output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root.mkdir(parents=True, exist_ok=False)
    tasks = regression_tasks()
    taskset_sha = hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()
    all_results = {}
    for name in args.profiles:
        profile = copy.deepcopy(config["server_profiles"]["baseline"])
        profile["args"] += ["--disable-radix-cache"]
        profile["env"] = {
            "SGLANG_A10_INT8_PREFILL": str(name == "on").lower(),
            "SGLANG_W8A8_FUSED_RMSNORM_QUANT": "false",
        }
        config["server_profiles"][name] = profile
        argv = server_command(config, args.model, name)
        directory = root / name
        directory.mkdir()
        manifest = {
            "command": argv,
            "environment": profile["env"],
            "harness_commit": _git_head(),
            "upstream_worktree_commit": _git_head(Path("/workspace/sglang")),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "taskset_sha256": taskset_sha,
            "model": args.model,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        defaults = config["defaults"]
        endpoint = f"http://{defaults['host']}:{defaults['port']}"
        with (directory / "server.log").open("w") as log:
            process = subprocess.Popen(
                argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, **profile["env"]},
                start_new_session=True,
            )
            try:
                wait_for_server(
                    defaults["host"], defaults["port"], defaults["server_ready_timeout_s"], process
                )
                nll = score_corpus(
                    endpoint=endpoint,
                    corpus_path=Path(args.corpus),
                    output_path=directory / "quality.json",
                )
                print(
                    f"QUALITY {name} nll={nll['mean_nll']} tokens={nll['scored_tokens']}",
                    flush=True,
                )
                results = []
                for task in tasks:
                    ids = tokenizer.apply_chat_template(
                        [{"role": "user", "content": task["prompt"]}],
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    response = _post_json(
                        endpoint + "/generate",
                        {
                            "input_ids": ids,
                            "sampling_params": {"temperature": 0, "max_new_tokens": 96},
                        },
                        180,
                    )
                    text = response["text"]
                    results.append(
                        {
                            "id": task["id"],
                            "category": task["category"],
                            "input_tokens": len(ids),
                            "text": text,
                            "passed": judge(task, text),
                            "expected": task["expected"],
                            "finish_reason": response.get("meta_info", {}).get("finish_reason"),
                        }
                    )
                (directory / "task_outputs.json").write_text(json.dumps(results, indent=2) + "\n")
                grouped = defaultdict(list)
                for result in results:
                    grouped[result["category"]].append(result)
                summary = {
                    category: {
                        "correct": sum(row["passed"] for row in rows),
                        "total": len(rows),
                        "min_input_tokens": min(row["input_tokens"] for row in rows),
                        "max_input_tokens": max(row["input_tokens"] for row in rows),
                    }
                    for category, rows in grouped.items()
                }
                (directory / "tasks_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                all_results[name] = results
                print(f"TASKS {name} {summary}", flush=True)
            finally:
                terminate_process_group(process)
    if "off" in all_results and "on" in all_results:
        nll_gate = compare_quality(
            root / "off/quality.json", root / "on/quality.json", max_nll_increase=0.02
        )
        exact = sum(
            a["text"] == b["text"]
            for a, b in zip(all_results["off"], all_results["on"], strict=True)
        )
        lost = [
            a["id"]
            for a, b in zip(all_results["off"], all_results["on"], strict=True)
            if a["passed"] and not b["passed"]
        ]
        comparison = {
            "nll_gate": nll_gate,
            "greedy_exact_match": exact,
            "tasks": len(tasks),
            "lost_correct_tasks": lost,
            "passed": nll_gate["passed"] and not lost,
        }
        (root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        print(json.dumps(comparison), flush=True)
        if not comparison["passed"]:
            raise SystemExit(1)
    print(root, flush=True)


if __name__ == "__main__":
    main()
