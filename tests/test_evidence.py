import json
from pathlib import Path

import pytest

from agentperf.evidence import snapshot_evidence


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
