"""Published aggregates must still match their snapshot's byte-level checksums."""

import gzip
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "verify_evidence", Path(__file__).parents[1] / "scripts/verify_evidence.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_published_snapshot_checksums():
    result = module.verify(Path(__file__).resolve().parents[1] / "evidence")
    assert result["passed"], result["errors"]


def test_evidence_verifier_detects_changed_bytes(tmp_path):
    data = b"result\n"
    (tmp_path / "result.csv").write_bytes(data)
    (tmp_path / "checksums.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "name": "result.csv",
                        "published": True,
                        "size_bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                ]
            }
        )
    )
    assert module.verify(tmp_path)["passed"]
    (tmp_path / "result.csv").write_bytes(b"changed\n")
    assert not module.verify(tmp_path)["passed"]


def test_evidence_verifier_rejects_empty_evidence(tmp_path):
    assert not module.verify(tmp_path)["passed"]


def test_primary_claim_reconstructs_and_rejects_tampered_metrics(tmp_path):
    source = Path(__file__).resolve().parents[1]
    primary_spec = importlib.util.spec_from_file_location(
        "verify_primary_result", source / "scripts/verify_primary_result.py"
    )
    primary = importlib.util.module_from_spec(primary_spec)
    primary_spec.loader.exec_module(primary)
    result = primary.verify(source / "evidence/a10")
    assert result["passed"], result
    assert 13.15 < result["throughput_gain_pct"] < 13.16
    assert not primary.verify(tmp_path)["passed"]
    for name in (
        "final_short_requests_off",
        "final_short_requests_on",
        "final_short_off",
        "final_short_on",
    ):
        shutil.copytree(source / "evidence/a10" / name, tmp_path / name)
    target = tmp_path / "final_short_requests_on/request_metrics.json.gz"
    data = json.loads(gzip.decompress(target.read_bytes()))
    for row in data["runs"]:
        row["duration_s"] *= 2
    target.write_bytes(gzip.compress(json.dumps(data).encode()))
    result = primary.verify(tmp_path)
    assert not result["passed"]
    assert "reconstruction" in result["error"]
