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
    manifest = {
        "created_at": timestamp,
        "upstream_commit": config["upstream_commit"],
        "model": model,
        "profile": profile,
        "suite": suite,
        "cases": [case.__dict__ for case in plan],
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
    }
    if os.name != "nt":
        process_kwargs["start_new_session"] = True
    process = subprocess.Popen(server_command(config, model, profile), **process_kwargs)

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
            with log_file.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Benchmark {case.case_id} failed with code {completed.returncode}; "
                    f"see {log_file}"
                )
    finally:
        terminate_process_group(process)
        server_log.close()
    return run_dir
