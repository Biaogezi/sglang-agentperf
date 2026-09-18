from pathlib import Path

from agentperf.commands import benchmark_command
from agentperf.config import build_plan, load_config
from agentperf.workloads import synthetic_agent_trace


def test_agent_trace_is_deterministic_original_and_multiturn():
    first = synthetic_agent_trace()
    assert first == synthetic_agent_trace()
    assert first["metadata"]["tool_execution"] is False
    assert len(first["conversations"]) == 24
    assert all(len(turns) == 4 for turns in first["conversations"])
    assert all(
        message["role"] != "assistant"
        for turns in first["conversations"]
        for turn in turns
        for message in turn["messages"]
    )  # The upstream client inserts the actual model reply, not a scripted assistant.


def test_agent_trace_uses_chat_backend(monkeypatch):
    monkeypatch.setenv("AGENTPERF_MODEL_QWEN3_8B_W8A8", "/models/qwen")
    config = load_config(Path(__file__).resolve().parents[1] / "configs/experiment_matrix.json")
    case = build_plan(config, model="qwen3_8b_w8a8", profile="baseline", suite="agent")[0]
    command = benchmark_command(config, case, Path("out.jsonl"))
    assert command[command.index("--backend") + 1] == "sglang-oai-chat"
    assert "--dataset-path" in command
    assert "--tokenize-prompt" not in command
