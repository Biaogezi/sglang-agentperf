import copy
from pathlib import Path

import pytest

from agentperf.config import ConfigError, build_plan, load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiment_matrix.json"


def test_config_is_valid() -> None:
    config = load_config(CONFIG)
    assert config["schema_version"] == 1


def test_smoke_plan_has_three_repetitions() -> None:
    config = load_config(CONFIG)
    plan = build_plan(config, model="qwen3_8b_fp16", profile="baseline", suite="smoke")
    assert len(plan) == 3
    assert len({case.case_id for case in plan}) == 3


def test_profile_plan_uses_workload_repetition_override() -> None:
    config = load_config(CONFIG)
    plan = build_plan(config, model="qwen3_8b_awq", profile="slo_chunk1024_i4", suite="profile")
    assert len(plan) == 1


def test_unknown_workload_in_suite_is_rejected() -> None:
    config = load_config(CONFIG)
    broken = copy.deepcopy(config)
    broken["suites"]["broken"] = ["missing"]
    with pytest.raises(ConfigError, match="unknown workloads"):
        validate_config(broken)
