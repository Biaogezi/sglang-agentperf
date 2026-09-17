import copy
import json
from pathlib import Path

import pytest

from agentperf.config import (
    ConfigError,
    build_plan,
    load_config,
    validate_config,
    validate_model_artifact,
)

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


def test_w8a8_artifact_rejects_unquantized_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG)
    (tmp_path / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    monkeypatch.setenv("AGENTPERF_MODEL_QWEN3_8B_W8A8", str(tmp_path))
    with pytest.raises(ConfigError, match="calibrated INT8 checkpoint"):
        validate_model_artifact(config, "qwen3_8b_w8a8")


def test_w8a8_artifact_accepts_channel_weight_token_activation_scheme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG)
    metadata = {
        "quantization_config": {
            "config_groups": {
                "group_0": {
                    "weights": {"type": "int", "num_bits": 8, "strategy": "channel"},
                    "input_activations": {
                        "type": "int",
                        "num_bits": 8,
                        "strategy": "token",
                        "dynamic": True,
                    },
                }
            }
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setenv("AGENTPERF_MODEL_QWEN3_8B_W8A8", str(tmp_path))
    assert validate_model_artifact(config, "qwen3_8b_w8a8") == tmp_path
