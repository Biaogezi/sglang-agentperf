from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .config import ConfigError, ExperimentCase


def resolve_model_path(config: dict[str, Any], model_name: str) -> str:
    model = config["models"][model_name]
    env_name = model["path_env"]
    value = os.environ.get(env_name)
    if not value:
        raise ConfigError(f"Environment variable {env_name} is required for {model_name}")
    return value


def server_command(config: dict[str, Any], model_name: str, profile_name: str) -> list[str]:
    defaults = config["defaults"]
    model = config["models"][model_name]
    profile = config["server_profiles"][profile_name]
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        resolve_model_path(config, model_name),
        "--host",
        str(defaults["host"]),
        "--port",
        str(defaults["port"]),
        "--dtype",
        model["dtype"],
        "--mem-fraction-static",
        str(defaults["mem_fraction_static"]),
    ]
    if model.get("quantization"):
        command += ["--quantization", model["quantization"]]
    command += [str(item) for item in profile["args"]]
    return command


def benchmark_command(
    config: dict[str, Any],
    case: ExperimentCase,
    output_file: Path,
) -> list[str]:
    defaults = config["defaults"]
    workload = config["workloads"][case.workload]
    command = [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--host",
        str(defaults["host"]),
        "--port",
        str(defaults["port"]),
        "--model",
        resolve_model_path(config, case.model),
        "--dataset-name",
        workload["dataset"],
        "--num-prompts",
        str(workload["num_prompts"]),
        "--max-concurrency",
        str(workload["max_concurrency"]),
        "--request-rate",
        str(workload["request_rate"]),
        "--warmup-requests",
        str(defaults["warmup_requests"]),
        "--output-file",
        str(output_file),
        "--output-details",
        "--cache-report",
        "--flush-cache",
    ]
    command += [str(item) for item in workload["args"]]
    return command


def shell_join(command: list[str]) -> str:
    """Render a command for display without using it for execution."""
    import shlex

    return shlex.join(command)
