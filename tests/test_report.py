import json
from pathlib import Path

import pytest

from agentperf.report import check_run_equivalence, compare_summaries, summarize_run


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
        path.with_suffix(".log").write_text("Cache hit rate: 25.0%\n", encoding="utf-8")

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
    assert rows[0]["cache_hit_rate_mean"] == 0.25
    assert rows[0]["mixed_prefill_chunk_size_mean"] == 512.0
    assert output.exists()


def test_compare_summaries_uses_positive_improvement_direction(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    output = tmp_path / "comparison.csv"
    header = "case,input_throughput_mean,tpot_p99_ms_mean\n"
    baseline.write_text(header + "model__baseline__prefill,100,200\n", encoding="utf-8")
    candidate.write_text(header + "model__candidate__prefill,110,100\n", encoding="utf-8")

    rows = compare_summaries(baseline, candidate, output)

    assert rows[0]["input_throughput_mean_improvement_pct"] == pytest.approx(10.0)
    assert rows[0]["tpot_p99_ms_mean_improvement_pct"] == 50.0
    assert output.exists()


def test_check_run_equivalence_matches_workload_and_repetition(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    record = {"generated_texts": ["same"], "output_lens": [4], "errors": [""]}
    (baseline / "model__base__smoke__r1.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    (candidate / "model__candidate__smoke__r1.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )

    result = check_run_equivalence(baseline, candidate)

    assert result == {"matched_repetitions": 1, "equivalent": True, "mismatches": []}


def test_compare_zero_latency_is_undefined_not_error(tmp_path: Path) -> None:
    header = "case,input_throughput_mean,tpot_p99_ms_mean\n"
    baseline, candidate = tmp_path / "baseline.csv", tmp_path / "candidate.csv"
    baseline.write_text(header + "m__off__short,100,0\n", encoding="utf-8")
    candidate.write_text(header + "m__on__short,120,0\n", encoding="utf-8")
    row = compare_summaries(baseline, candidate, tmp_path / "comparison.csv")[0]
    assert row["input_throughput_mean_improvement_pct"] == pytest.approx(20)
    assert row["baseline_tpot_p99_ms_mean"] == 0
    assert row["tpot_p99_ms_mean_improvement_pct"] == ""


def test_equivalence_rejects_missing_fields_and_missing_runs(tmp_path: Path) -> None:
    baseline, candidate = tmp_path / "off", tmp_path / "on"
    baseline.mkdir()
    candidate.mkdir()
    for directory in (baseline, candidate):
        (directory / "model__arm__work__r1.jsonl").write_text('{"completed": 1}\n')
    result = check_run_equivalence(baseline, candidate)
    assert not result["equivalent"]
    assert len(result["mismatches"]) == 3
    record = '{"generated_texts":["a"], "output_lens":[1], "errors":[""]}\n'
    for directory in (baseline, candidate):
        (directory / "model__arm__work__r1.jsonl").write_text(record)
    (baseline / "model__arm__work__r2.jsonl").write_text(record)
    assert not check_run_equivalence(baseline, candidate)["equivalent"]


def test_paired_audit_requires_matching_sources_and_all_requests(tmp_path: Path) -> None:
    from agentperf.report import audit_paired_run

    for arm in ("prefill_off", "prefill_on"):
        directory = tmp_path / arm
        directory.mkdir()
        manifest = {
            "launches": [
                {
                    "repetition": 1,
                    "manifest": {
                        "source_files_sha256": {"runtime_candidate": {"kernel.py": "same"}},
                        "cases": [{"case_id": f"m__{arm}__work__r1", "workload": "work"}],
                        "config": {
                            "workloads": {
                                "work": {
                                    "num_prompts": 1,
                                    "dataset": "random-ids",
                                    "args": [
                                        "--tokenize-prompt",
                                        "--random-input-len",
                                        "128",
                                        "--random-range-ratio",
                                        "1",
                                    ],
                                }
                            }
                        },
                    },
                }
            ]
        }
        (directory / "manifest.json").write_text(json.dumps(manifest))
        (directory / f"m__{arm}__work__r1.jsonl").write_text(
            json.dumps(
                {
                    "completed": 1,
                    "errors": [""],
                    "input_lens": [128],
                    "generated_texts": ["x"],
                    "output_lens": [1],
                }
            )
            + "\n"
        )
    assert audit_paired_run(tmp_path, minimum_repetitions=1)["passed"]
    assert not audit_paired_run(tmp_path)["passed"]
    path = tmp_path / "prefill_on/manifest.json"
    bad = json.loads(path.read_text())
    bad["launches"][0]["manifest"]["source_files_sha256"]["runtime_candidate"] = {"k": "changed"}
    path.write_text(json.dumps(bad))
    assert not audit_paired_run(tmp_path, minimum_repetitions=1)["passed"]
