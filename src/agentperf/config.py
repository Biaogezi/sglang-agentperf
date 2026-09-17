from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an experiment configuration is inconsistent."""


@dataclass(frozen=True)
class ExperimentCase:
    model: str
    profile: str
    workload: str
    repetition: int
    case_id: str


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {config_path}: {exc}") from exc
    validate_config(data)
    return data


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "upstream_commit",
        "defaults",
        "models",
        "server_profiles",
        "workloads",
        "suites",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ConfigError(f"Missing top-level keys: {', '.join(missing)}")
    if config["schema_version"] != 1:
        raise ConfigError("Only schema_version=1 is supported")
    if not config["models"]:
        raise ConfigError("At least one model must be configured")
    if not config["server_profiles"]:
        raise ConfigError("At least one server profile must be configured")

    workloads = config["workloads"]
    for name, workload in workloads.items():
        for key in ("dataset", "num_prompts", "max_concurrency", "request_rate", "args"):
            if key not in workload:
                raise ConfigError(f"Workload {name!r} is missing {key!r}")
        if workload["num_prompts"] < 5 * workload["max_concurrency"] and name != "smoke":
            raise ConfigError(
                f"Workload {name!r} needs num_prompts >= 5 * max_concurrency "
                "for a steady-state measurement"
            )
        if int(workload.get("repetitions", config["defaults"]["repetitions"])) < 1:
            raise ConfigError(f"Workload {name!r} needs repetitions >= 1")

    for suite_name, suite_workloads in config["suites"].items():
        unknown = sorted(set(suite_workloads) - workloads.keys())
        if unknown:
            raise ConfigError(
                f"Suite {suite_name!r} references unknown workloads: {', '.join(unknown)}"
            )


def build_plan(
    config: dict[str, Any],
    *,
    model: str,
    profile: str,
    suite: str,
) -> list[ExperimentCase]:
    if model not in config["models"]:
        raise ConfigError(f"Unknown model: {model}")
    if profile not in config["server_profiles"]:
        raise ConfigError(f"Unknown server profile: {profile}")
    if suite not in config["suites"]:
        raise ConfigError(f"Unknown suite: {suite}")

    cases: list[ExperimentCase] = []
    for workload in config["suites"][suite]:
        repetitions = int(
            config["workloads"][workload].get(
                "repetitions", config["defaults"]["repetitions"]
            )
        )
        for repetition in range(1, repetitions + 1):
            case_id = f"{model}__{profile}__{workload}__r{repetition}"
            cases.append(
                ExperimentCase(
                    model=model,
                    profile=profile,
                    workload=workload,
                    repetition=repetition,
                    case_id=case_id,
                )
            )
    return cases


def validate_model_artifact(config: dict[str, Any], model_name: str) -> Path:
    """Validate on-disk metadata that is required by performance-sensitive loaders."""
    model = config["models"].get(model_name)
    if model is None:
        raise ConfigError(f"Unknown model: {model_name}")
    env_name = model["path_env"]
    raw_path = os.environ.get(env_name)
    if not raw_path:
        raise ConfigError(f"Environment variable {env_name} is required for {model_name}")
    model_path = Path(raw_path)
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise ConfigError(f"Model config does not exist: {config_path}")

    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    if model.get("quantization") == "w8a8_int8":
        quantization = metadata.get("quantization_config")
        if not isinstance(quantization, dict):
            raise ConfigError(
                "w8a8_int8 requires a calibrated INT8 checkpoint with quantization_config; "
                f"{model_path} appears to be an unquantized checkpoint"
            )
        groups = quantization.get("config_groups")
        schemes = groups.values() if isinstance(groups, dict) else []
        compatible = False
        for scheme in schemes:
            if not isinstance(scheme, dict):
                continue
            weights = scheme.get("weights", {})
            activations = scheme.get("input_activations", {})
            compatible = compatible or (
                weights.get("type") == "int"
                and weights.get("num_bits") == 8
                and weights.get("strategy") == "channel"
                and activations.get("type") == "int"
                and activations.get("num_bits") == 8
                and activations.get("strategy") == "token"
                and activations.get("dynamic") is True
            )
        if not compatible:
            raise ConfigError(
                "w8a8_int8 requires per-channel INT8 weights and dynamic per-token INT8 "
                f"activations; incompatible metadata in {config_path}"
            )
    return model_path
