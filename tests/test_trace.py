import gzip
import json
from pathlib import Path

import pytest

from agentperf.trace import analyze_trace


def test_analyze_trace_aggregates_kernels_and_gpu_idle(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json.gz"
    payload = {
        "traceEvents": [
            {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 0, "dur": 100},
            {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 200, "dur": 100},
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "ts": 0,
                "dur": 10,
            },
        ]
    }
    with gzip.open(trace, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    summary = analyze_trace(trace, tmp_path / "out")

    assert summary["top_kernels"][0]["count"] == 2
    assert summary["top_kernels"][0]["total_ms"] == pytest.approx(0.2)
    assert summary["gpu_activity"]["active_ms"] == pytest.approx(0.2)
    assert summary["gpu_activity"]["idle_pct"] == pytest.approx(100 / 3)
    assert (tmp_path / "out" / "kernels.csv").exists()
