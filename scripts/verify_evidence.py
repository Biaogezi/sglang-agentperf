"""Verify exact bytes of every published evidence artifact without raw GPU data."""

import argparse
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> dict:
    errors = []
    checked = 0
    indexes = sorted(root.rglob("checksums.json"))
    for index in indexes:
        for entry in json.loads(index.read_text())["files"]:
            if not entry["published"]:
                continue
            target = index.parent / entry["name"]
            if target.parent.resolve() != index.parent.resolve():
                errors.append(f"Unsafe artifact path: {target}")
                continue
            if not target.is_file():
                errors.append(f"Missing artifact: {target}")
                continue
            payload = target.read_bytes()
            if len(payload) != entry["size_bytes"]:
                errors.append(f"Size differs: {target}")
            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                errors.append(f"SHA-256 differs: {target}")
            checked += 1
    if not indexes or not checked:
        errors.append("No published evidence found")
    return {"indexes": len(indexes), "artifacts": checked, "errors": errors, "passed": not errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path("evidence"))
    args = parser.parse_args()
    result = verify(args.root)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
