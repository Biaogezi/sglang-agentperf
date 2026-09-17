import json
from pathlib import Path

import pytest

from agentperf.report import compare_summaries, summarize_run


def test_summarize_repetitions(tmp_path: Path) -> None:
    for repetition, throughput in enumerate((100.0, 110.0, 120.0), start=1):
        record = {
            "completed": 10,
            "request_throughput": 2.0,
            "input_throughput": throughput * 10,
            "output_throughput": throughput,
            "total_throughput": throughput * 11,
            "p99_e2e_latency_ms": 75.0,
            "p99_ttft_ms": 50.0,
        }
        path = tmp_path / f"model__profile__workload__r{repetition}.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    output = tmp_path / "summary.csv"
    (tmp_path / "server.log").write_text(
        "AGENTPERF_CASE_BEGIN model__profile__workload__r1\n"
        "Prefill batch, #new-seq: 1, #new-token: 2048, #cached-token: 0, "
        "token usage: 0.1, #running-req: 0, #queue-req: 1\n"
        "Prefill batch, #new-seq: 1, #new-token: 512, #cached-token: 0, "
        "token usage: 0.2, #running-req: 2, #queue-req: 1\n"
        "KV cache pool is full. Retract requests. #retracted_reqs: 2\n"
        "AGENTPERF_CASE_END model__profile__workload__r1 returncode=0\n",
        encoding="utf-8",
    )
    rows = summarize_run(tmp_path, output)
    assert rows[0]["repetitions"] == 3
    assert rows[0]["completed_mean"] == 10.0
    assert rows[0]["input_throughput_mean"] == 1100.0
    assert rows[0]["output_throughput_mean"] == 110.0
    assert rows[0]["total_throughput_mean"] == 1210.0
    assert rows[0]["e2e_p99_ms_mean"] == 75.0
    assert rows[0]["retracted_requests_mean"] == 2 / 3
    assert rows[0]["mixed_prefill_chunk_size_mean"] == 512.0
    assert output.exists()


def test_compare_summaries_uses_positive_improvement_direction(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    output = tmp_path / "comparison.csv"
    header = "case,input_throughput_mean,tpot_p99_ms_mean\n"
    baseline.write_text(
        header + "model__baseline__prefill,100,200\n", encoding="utf-8"
    )
    candidate.write_text(
        header + "model__candidate__prefill,110,100\n", encoding="utf-8"
    )

    rows = compare_summaries(baseline, candidate, output)

    assert rows[0]["input_throughput_mean_improvement_pct"] == pytest.approx(10.0)
    assert rows[0]["tpot_p99_ms_mean_improvement_pct"] == 100.0
    assert output.exists()
