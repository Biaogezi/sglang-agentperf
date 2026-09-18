import hashlib
from pathlib import Path

from agentperf.runner import source_fingerprints


def test_source_manifest_identifies_executed_harness_bytes():
    root = Path(__file__).resolve().parents[1]
    fingerprints = source_fingerprints()
    relative = "src/agentperf/runner.py"
    assert (
        fingerprints["harness"][relative]
        == hashlib.sha256((root / relative).read_bytes()).hexdigest()
    )
    assert "scripts/run_paired_prefill.py" in fingerprints["harness"]
