from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_evidence(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Publish small aggregates plus checksums that identify ignored raw artifacts."""
    aggregate_names = [
        name for name in ("summary.csv", "quality.json") if (run_dir / name).is_file()
    ]
    if not aggregate_names:
        raise FileNotFoundError(
            f"Run is missing required aggregate: {run_dir / 'summary.csv'} or "
            f"{run_dir / 'quality.json'}"
        )

    required = [run_dir / "manifest.json", *(run_dir / name for name in aggregate_names)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Run is missing required evidence: {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=False)
    for source in required:
        shutil.copy2(source, output_dir / source.name)

    files = []
    for path in sorted(run_dir.iterdir()):
        if not path.is_file():
            continue
        files.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "published": path.name in {"manifest.json", *aggregate_names},
            }
        )
    index = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run": run_dir.name,
        "files": files,
    }
    (output_dir / "checksums.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def snapshot_trace_evidence(
    trace_path: Path, analysis_dir: Path, output_dir: Path
) -> dict[str, Any]:
    """Publish compact trace aggregates while retaining a hash of the raw trace."""
    aggregate_names = [
        "summary.json",
        "kernels.csv",
        "cuda_runtime.csv",
        "cpu_ops.csv",
        "user_annotations.csv",
    ]
    required = [analysis_dir / name for name in aggregate_names]
    missing = [str(path) for path in [trace_path, *required] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing trace evidence: {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=False)
    for source in required:
        shutil.copy2(source, output_dir / source.name)

    files = [
        {
            "name": trace_path.name,
            "size_bytes": trace_path.stat().st_size,
            "sha256": _sha256(trace_path),
            "published": False,
        }
    ]
    files.extend(
        {
            "name": source.name,
            "size_bytes": source.stat().st_size,
            "sha256": _sha256(source),
            "published": True,
        }
        for source in required
    )
    index = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_trace": trace_path.name,
        "files": files,
    }
    (output_dir / "checksums.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def snapshot_gpu_validation(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Publish bounded GPU-test logs, including failed tests when present."""
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    scripts = [Path(test["command"][-1]).name for test in manifest["tests"]]
    required = [manifest_path, *(run_dir / f"{script}.log" for script in scripts)]
    if not scripts or not all(path.is_file() for path in required):
        raise ValueError("GPU validation is missing its manifest or declared logs")
    output_dir.mkdir(parents=True, exist_ok=False)
    for source in required:
        shutil.copy2(source, output_dir / source.name)
    index = {
        "schema_version": 1,
        "source_run": run_dir.name,
        "files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "published": True,
            }
            for path in required
        ],
    }
    (output_dir / "checksums.json").write_text(json.dumps(index, indent=2) + "\n")
    return index
