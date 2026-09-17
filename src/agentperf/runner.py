from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .commands import benchmark_command, server_command
from .config import build_plan
from .quality import score_corpus


def _git_head(path: Path | None = None) -> str | None:
    repository = path or Path.cwd()
    command = [
        "git",
        "-c",
        f"safe.directory={repository}",
        "-C",
        str(repository),
        "rev-parse",
        "HEAD",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def wait_for_server(host: str, port: int, timeout_s: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + timeout_s
    urls = [f"http://{host}:{port}/health", f"http://{host}:{port}/v1/models"]
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang server exited early with code {process.returncode}")
        if any(_ready(url) for url in urls):
            return
        time.sleep(2)
    raise TimeoutError(f"SGLang server was not ready after {timeout_s}s")


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.terminate()
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_plan(
    config: dict[str, Any],
    *,
    model: str,
    profile: str,
    suite: str,
    output_root: Path,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"{timestamp}__{model}__{profile}__{suite}"
    run_dir.mkdir(parents=True, exist_ok=False)

    plan = build_plan(config, model=model, profile=profile, suite=suite)
    server_argv = server_command(config, model, profile)
    server_environment = {
        str(key): str(value)
        for key, value in config["server_profiles"][profile].get("env", {}).items()
    }
    manifest = {
        "created_at": timestamp,
        "harness_commit": _git_head(),
        "upstream_commit": config["upstream_commit"],
        "upstream_worktree_commit": _git_head(Path("/workspace/sglang")),
        "model": model,
        "profile": profile,
        "suite": suite,
        "cases": [case.__dict__ for case in plan],
        "server_command": server_argv,
        "server_environment": server_environment,
        "benchmark_commands": [
            benchmark_command(config, case, run_dir / f"{case.case_id}.jsonl")
            for case in plan
        ],
        "config": config,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    server_log_path = run_dir / "server.log"
    server_log = server_log_path.open("w", encoding="utf-8")
    process_kwargs: dict[str, Any] = {
        "stdout": server_log,
        "stderr": subprocess.STDOUT,
        "text": True,
        "env": {**os.environ, **server_environment},
    }
    if os.name != "nt":
        process_kwargs["start_new_session"] = True
    process = subprocess.Popen(server_argv, **process_kwargs)

    try:
        defaults = config["defaults"]
        wait_for_server(
            str(defaults["host"]),
            int(defaults["port"]),
            int(defaults["server_ready_timeout_s"]),
            process,
        )
        for case in plan:
            output_file = run_dir / f"{case.case_id}.jsonl"
            log_file = run_dir / f"{case.case_id}.log"
            command = benchmark_command(config, case, output_file)
            server_log.write(f"\nAGENTPERF_CASE_BEGIN {case.case_id}\n")
            server_log.flush()
            with log_file.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            server_log.write(
                f"\nAGENTPERF_CASE_END {case.case_id} returncode={completed.returncode}\n"
            )
            server_log.flush()
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Benchmark {case.case_id} failed with code {completed.returncode}; "
                    f"see {log_file}"
                )
    finally:
        terminate_process_group(process)
        server_log.close()
    return run_dir


def run_quality_plan(
    config: dict[str, Any],
    *,
    model: str,
    profile: str,
    corpus_path: Path,
    output_root: Path,
) -> Path:
    """Launch one configured server and collect a fixed-corpus prompt-NLL result."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"{timestamp}__{model}__{profile}"
    run_dir.mkdir(parents=True, exist_ok=False)

    server_argv = server_command(config, model, profile)
    server_environment = {
        str(key): str(value)
        for key, value in config["server_profiles"][profile].get("env", {}).items()
    }
    manifest = {
        "created_at": timestamp,
        "harness_commit": _git_head(),
        "upstream_commit": config["upstream_commit"],
        "upstream_worktree_commit": _git_head(Path("/workspace/sglang")),
        "model": model,
        "profile": profile,
        "corpus": str(corpus_path),
        "server_command": server_argv,
        "server_environment": server_environment,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    server_log = (run_dir / "server.log").open("w", encoding="utf-8")
    process_kwargs: dict[str, Any] = {
        "stdout": server_log,
        "stderr": subprocess.STDOUT,
        "text": True,
        "env": {**os.environ, **server_environment},
    }
    if os.name != "nt":
        process_kwargs["start_new_session"] = True
    process = subprocess.Popen(server_argv, **process_kwargs)

    try:
        defaults = config["defaults"]
        host = str(defaults["host"])
        port = int(defaults["port"])
        wait_for_server(
            host,
            port,
            int(defaults["server_ready_timeout_s"]),
            process,
        )
        result = score_corpus(
            endpoint=f"http://{host}:{port}",
            corpus_path=corpus_path,
            output_path=run_dir / "quality.json",
        )
        result.update(
            {
                "model": model,
                "profile": profile,
                "harness_commit": manifest["harness_commit"],
                "upstream_worktree_commit": manifest["upstream_worktree_commit"],
            }
        )
        (run_dir / "quality.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    finally:
        terminate_process_group(process)
        server_log.close()
    return run_dir
