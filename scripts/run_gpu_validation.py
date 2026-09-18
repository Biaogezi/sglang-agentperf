"""Run all candidate GPU checks serially and preserve source-identified logs."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from agentperf.runner import source_fingerprints


def main():
    root = Path("results/gpu-validation") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root.mkdir(parents=True, exist_ok=False)
    payload = {
        "source_files_sha256": source_fingerprints(),
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "tests": [],
    }
    for script in (
        "test_w8a8_splitk_gpu.py",
        "test_w8a8_dispatch_gpu.py",
        "test_prefill_jit_variants.py",
        "test_native_norm_fusion_gpu.py",
    ):
        command = [sys.executable, f"scripts/{script}"]
        with (root / f"{script}.log").open("w") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
        payload["tests"].append({"command": command, "returncode": result.returncode})
        payload["passed"] = all(row["returncode"] == 0 for row in payload["tests"])
        (root / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"GPU_TEST {script} returncode={result.returncode} evidence={root}", flush=True)
        if result.returncode:
            print((root / f"{script}.log").read_text(), flush=True)
            raise SystemExit(result.returncode)
    print(root, flush=True)


if __name__ == "__main__":
    main()
