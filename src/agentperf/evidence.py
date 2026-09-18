from __future__ import annotations

import gzip
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


def snapshot_evidence(
    run_dir: Path, output_dir: Path, *, include_task_outputs: bool = False
) -> dict[str, Any]:
    """Publish small aggregates plus checksums that identify ignored raw artifacts."""
    aggregate_names = [
        name for name in ("summary.csv", "quality.json") if (run_dir / name).is_file()
    ]
    if not aggregate_names:
        raise FileNotFoundError(
            f"Run is missing required aggregate: {run_dir / 'summary.csv'} or "
            f"{run_dir / 'quality.json'}"
        )
    if "quality.json" in aggregate_names:
        optional = ["tasks_summary.json"]
        if include_task_outputs:
            optional.append("task_outputs.json")
        aggregate_names += [name for name in optional if (run_dir / name).is_file()]

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


def snapshot_request_metrics(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Publish timings/lengths and output hashes, never prompt/generated text or server config."""
    from .report import read_last_json

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    configurations = (
        [launch["manifest"].get("config", {}) for launch in manifest["launches"]]
        if "launches" in manifest
        else [manifest.get("config", {})]
    )
    runs = []
    for source in sorted(run_dir.glob("*.jsonl")):
        record = read_last_json(source)
        workload = source.stem.rsplit("__", 2)[-2] if "__" in source.stem else source.stem
        native_ids = (
            record.get("dataset_name") == "random-ids"
            and bool(configurations)
            and all(
                "--tokenize-prompt" in config.get("workloads", {}).get(workload, {}).get("args", [])
                for config in configurations
            )
        )
        fields = ("input_lens", "output_lens", "ttfts", "itls", "generated_texts", "errors")
        count = record["completed"]
        if count < 1 or any(len(record.get(field, [])) != count for field in fields):
            raise ValueError(f"Incomplete per-request arrays: {source}")
        if any(record["errors"]):
            raise ValueError(f"Cannot publish successful-run metrics with request errors: {source}")
        runs.append(
            {
                "file": source.name,
                "raw_sha256": _sha256(source),
                "dataset_name": record.get("dataset_name"),
                "input_lengths_valid": native_ids,
                "duration_s": record["duration"],
                "completed": count,
                "total_input_tokens": record["total_input_tokens"],
                "total_output_tokens": record["total_output_tokens"],
                "input_lens": record["input_lens"],
                "output_lens": record["output_lens"],
                "ttft_s": record["ttfts"],
                "itl_s": record["itls"],
                "output_text_sha256": [
                    hashlib.sha256(text.encode("utf-8")).hexdigest()
                    for text in record["generated_texts"]
                ],
                "errors": record["errors"],
            }
        )
    if not runs:
        raise ValueError("No raw request records found")
    payload = {
        "schema_version": 1,
        "notes": "Timings in seconds; preserve request order; no prompts/text/server config. "
        "input_lengths_valid only covers this project's native-ID random workloads. "
        "ITL measures streamed chunks; do not reconstruct exact E2E latency from it.",
        "runs": runs,
    }
    encoded = gzip.compress(json.dumps(payload, allow_nan=False).encode("utf-8"), mtime=0)
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact = output_dir / "request_metrics.json.gz"
    artifact.write_bytes(encoded)
    index = {
        "schema_version": 1,
        "source_run": run_dir.name,
        "files": [
            {
                "name": artifact.name,
                "size_bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "published": True,
            }
        ],
    }
    (output_dir / "checksums.json").write_text(json.dumps(index, indent=2) + "\n")
    return index
