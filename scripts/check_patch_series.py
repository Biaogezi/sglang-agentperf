"""Verify pinned runtime source content without changing the user's upstream checkout."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="upstream/sglang")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    lock = dict(
        line.split("=", 1)
        for line in (project / "UPSTREAM.lock").read_text().splitlines()
        if "=" in line
    )
    patches = [(project / name).resolve() for name in lock["patches"].split()]
    if not all(path.is_relative_to(project / "patches") and path.is_file() for path in patches):
        raise ValueError("All patches must be existing files inside patches/")
    source = Path(args.source).resolve()
    with tempfile.TemporaryDirectory(prefix="agentperf-patch-check-") as temporary:
        checkout = Path(temporary) / "source"
        subprocess.run(
            ["git", "clone", "--shared", "--no-checkout", str(source), str(checkout)], check=True
        )
        command = ["git", "-C", str(checkout)]
        subprocess.run([*command, "read-tree", lock["commit"]], check=True)
        subprocess.run([*command, "apply", "--cached", *map(str, patches)], check=True)
        actual = subprocess.check_output([*command, "write-tree"], text=True).strip()
    if actual != lock["patched_tree"]:
        raise ValueError(f"Patched tree mismatch: {actual} != {lock['patched_tree']}")
    print(
        json.dumps(
            {
                "base_commit": lock["commit"],
                "patched_tree": actual,
                "passed": True,
                "patches_sha256": {
                    str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in patches
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
