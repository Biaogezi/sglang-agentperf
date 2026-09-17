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
    required = [run_dir / "manifest.json", run_dir / "summary.csv"]
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
                "published": path.name in {"manifest.json", "summary.csv"},
            }
        )
    index = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run": run_dir.name,
        "files": files,
    }
    (output_dir / "checksums.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    return index
