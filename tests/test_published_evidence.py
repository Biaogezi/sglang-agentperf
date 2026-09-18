"""Published aggregates must still match their snapshot's byte-level checksums."""

import hashlib
import importlib.util
import json
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
