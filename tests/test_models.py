import hashlib
import json
from pathlib import Path

from agentperf.models import verify_model_files


def test_verify_model_files_checks_size_and_digest(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    content = b"quantized weights"
    (model_dir / "weights.bin").write_bytes(content)
    sources = {
        "models": {
            "test": {
                "revision": "abc123",
                "files": {
                    "weights.bin": {
                        "size_bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                },
            }
        }
    }
    sources_path = tmp_path / "sources.json"
    sources_path.write_text(json.dumps(sources), encoding="utf-8")

    result = verify_model_files(sources_path, "test", model_dir)
    assert result["valid"] is True
    assert result["files"][0]["valid"] is True


def test_verify_model_files_reports_mismatch(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"wrong")
    sources_path = tmp_path / "sources.json"
    sources_path.write_text(
        json.dumps(
            {
                "models": {
                    "test": {
                        "revision": "abc123",
                        "files": {
                            "weights.bin": {"size_bytes": 5, "sha256": "0" * 64}
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert verify_model_files(sources_path, "test", model_dir)["valid"] is False
