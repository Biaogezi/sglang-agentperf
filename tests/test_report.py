import json
from pathlib import Path

from agentperf.report import summarize_run


def test_summarize_repetitions(tmp_path: Path) -> None:
    for repetition, throughput in enumerate((100.0, 110.0, 120.0), start=1):
        record = {
            "request_throughput": 2.0,
            "output_throughput": throughput,
            "p99_ttft_ms": 50.0,
        }
        path = tmp_path / f"model__profile__workload__r{repetition}.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    output = tmp_path / "summary.csv"
    rows = summarize_run(tmp_path, output)
    assert rows[0]["repetitions"] == 3
    assert rows[0]["output_throughput_mean"] == 110.0
    assert output.exists()
