from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_files(
    sources_path: Path, model_name: str, model_dir: Path
) -> dict[str, Any]:
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    try:
        model = sources["models"][model_name]
    except KeyError as error:
        raise ValueError(f"Unknown model source: {model_name}") from error
    expected_files = model.get("files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError(f"No file checksums are pinned for {model_name}")

    results = []
    all_valid = True
    for name, expected in expected_files.items():
        path = model_dir / name
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_sha256 = _sha256(path) if exists else None
        valid = (
            exists
            and actual_size == int(expected["size_bytes"])
            and actual_sha256 == str(expected["sha256"]).lower()
        )
        all_valid = all_valid and valid
        results.append(
            {
                "name": name,
                "exists": exists,
                "expected_size_bytes": int(expected["size_bytes"]),
                "actual_size_bytes": actual_size,
                "expected_sha256": str(expected["sha256"]).lower(),
                "actual_sha256": actual_sha256,
                "valid": valid,
            }
        )
    return {
        "model": model_name,
        "revision": model["revision"],
        "model_dir": str(model_dir),
        "valid": all_valid,
        "files": results,
    }
