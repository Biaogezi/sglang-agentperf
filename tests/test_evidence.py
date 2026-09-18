import gzip
import json
from pathlib import Path

import pytest

from agentperf.evidence import snapshot_evidence, snapshot_request_metrics, snapshot_trace_evidence


def test_snapshot_evidence_copies_aggregates_and_hashes_raw_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text('{"model":"test"}\n', encoding="utf-8")
    (run_dir / "summary.csv").write_text("case,value\nsmoke,1\n", encoding="utf-8")
    (run_dir / "raw.jsonl").write_text('{"request":1}\n', encoding="utf-8")

    output_dir = tmp_path / "evidence"
    result = snapshot_evidence(run_dir, output_dir)

    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "summary.csv").is_file()
    assert not (output_dir / "raw.jsonl").exists()
    raw = next(item for item in result["files"] if item["name"] == "raw.jsonl")
    assert raw["published"] is False
    assert len(raw["sha256"]) == 64
    assert json.loads((output_dir / "checksums.json").read_text())["source_run"] == "run"


def test_snapshot_evidence_requires_aggregates(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="summary.csv"):
        snapshot_evidence(run_dir, tmp_path / "evidence")


def test_snapshot_evidence_supports_quality_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "quality-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text('{"model": "w8a8"}\n', encoding="utf-8")
    (run_dir / "quality.json").write_text('{"mean_nll": 2.9}\n', encoding="utf-8")
    (run_dir / "server.log").write_text("raw log\n", encoding="utf-8")

    output_dir = tmp_path / "evidence"
    result = snapshot_evidence(run_dir, output_dir)

    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "quality.json").is_file()
    published = {row["name"]: row["published"] for row in result["files"]}
    assert published["quality.json"] is True
    assert published["server.log"] is False


def test_snapshot_trace_evidence_copies_analysis_and_hashes_trace(tmp_path: Path) -> None:
    trace = tmp_path / "capture.trace.json.gz"
    trace.write_bytes(b"raw trace")
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    for name in (
        "summary.json",
        "kernels.csv",
        "cuda_runtime.csv",
        "cpu_ops.csv",
        "user_annotations.csv",
    ):
        (analysis_dir / name).write_text(name, encoding="utf-8")

    output_dir = tmp_path / "published"
    result = snapshot_trace_evidence(trace, analysis_dir, output_dir)

    assert result["source_trace"] == trace.name
    assert result["files"][0]["published"] is False
    assert len(result["files"][0]["sha256"]) == 64
    assert (output_dir / "summary.json").read_text(encoding="utf-8") == "summary.json"


def test_task_output_publication_is_explicit(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("manifest.json", "quality.json", "tasks_summary.json", "task_outputs.json"):
        (raw / name).write_text("{}\n")
    snapshot_evidence(raw, tmp_path / "default")
    assert (tmp_path / "default/tasks_summary.json").is_file()
    assert not (tmp_path / "default/task_outputs.json").exists()
    snapshot_evidence(raw, tmp_path / "explicit", include_task_outputs=True)
    assert (tmp_path / "explicit/task_outputs.json").is_file()


def test_request_metrics_publish_hashes_not_content(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    record = {
        "completed": 1,
        "duration": 1.0,
        "total_input_tokens": 128,
        "total_output_tokens": 1,
        "input_lens": [128],
        "output_lens": [1],
        "ttfts": [0.02],
        "itls": [[]],
        "generated_texts": ["private response"],
        "errors": [""],
        "server_info": {"secret": "do not publish"},
        "dataset_name": "agentic-trace",
    }
    (raw / "run.jsonl").write_text(json.dumps(record) + "\n")
    snapshot_request_metrics(raw, tmp_path / "public")
    payload = gzip.decompress((tmp_path / "public/request_metrics.json.gz").read_bytes())
    assert b"private response" not in payload
    assert b"do not publish" not in payload
    row = json.loads(payload)["runs"][0]
    assert not row["input_lengths_valid"]
    assert len(row["output_text_sha256"][0]) == 64
    record["dataset_name"] = "random-ids"
    (raw / "run.jsonl").write_text(json.dumps(record) + "\n")
    snapshot_request_metrics(raw, tmp_path / "nominal")
    nominal = json.loads(
        gzip.decompress((tmp_path / "nominal/request_metrics.json.gz").read_bytes())
    )
    assert not nominal["runs"][0]["input_lengths_valid"]
    (raw / "manifest.json").write_text(
        json.dumps({"config": {"workloads": {"run": {"args": ["--tokenize-prompt"]}}}})
    )
    snapshot_request_metrics(raw, tmp_path / "native")
    native = json.loads(gzip.decompress((tmp_path / "native/request_metrics.json.gz").read_bytes()))
    assert native["runs"][0]["input_lengths_valid"]
    record["output_lens"] = []
    (raw / "run.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="Incomplete"):
        snapshot_request_metrics(raw, tmp_path / "bad")
