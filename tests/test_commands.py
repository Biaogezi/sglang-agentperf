from pathlib import Path

from agentperf.commands import benchmark_command, server_command
from agentperf.config import build_plan, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiment_matrix.json"


def test_commands_include_pinned_experiment_parameters(monkeypatch) -> None:
    monkeypatch.setenv("AGENTPERF_MODEL_QWEN3_8B_FP16", "/models/qwen3")
    config = load_config(CONFIG)
    server = server_command(config, "qwen3_8b_fp16", "baseline")
    assert "--enable-metrics" in server
    assert "/models/qwen3" in server

    case = build_plan(config, model="qwen3_8b_fp16", profile="baseline", suite="smoke")[0]
    benchmark = benchmark_command(config, case, Path("out.jsonl"))
    assert benchmark[2] == "sglang.benchmark.serving"
    assert benchmark[benchmark.index("--dataset-name") + 1] == "random-ids"
    assert benchmark[benchmark.index("--num-prompts") + 1] == "8"
    assert "--output-details" in benchmark
    assert "--flush-cache" in benchmark
    assert "--tokenize-prompt" in benchmark

    fp16_reduce = config["server_profiles"]["slo_chunk1024_i4_fp16_reduce"]
    assert fp16_reduce["env"]["SGLANG_MARLIN_USE_FP32_REDUCE"] == "false"


def test_shape_controlled_workloads_send_native_ids():
    config = load_config(CONFIG)
    for workload in config["workloads"].values():
        if workload["dataset"] == "random-ids":
            assert "--tokenize-prompt" in workload["args"]
