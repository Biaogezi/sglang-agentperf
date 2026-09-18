"""Published aggregates must still match their snapshot's byte-level checksums."""

import hashlib
import json
from pathlib import Path


def test_published_snapshot_checksums():
    root = Path(__file__).resolve().parents[1] / "evidence"
    checked = 0
    failures = []
    for index in root.rglob("checksums.json"):
        for entry in json.loads(index.read_text(encoding="utf-8"))["files"]:
            if not entry["published"]:
                continue
            path = index.parent / entry["name"]
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
            ):
                failures.append(str(path.relative_to(root)))
            checked += 1
    assert checked > 0
    assert not failures, f"Evidence checksum mismatches: {failures}"
