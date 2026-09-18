"""Stratify observed shared-prefix TTFT without treating cache state as randomized."""

import argparse
import json
from pathlib import Path

from agentperf.report import read_last_json, stratify_cache_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.run.glob("*.jsonl")):
        record = read_last_json(path)
        if record.get("dataset_name") != "generated-shared-prefix":
            continue
        rows.append({"file": path.name, "groups": stratify_cache_states(record)})
    if not rows:
        parser.error("No generated-shared-prefix records")
    output = {
        "note": "Observed cache state, not randomized cold/warm trials. TTFT includes queuing. "
        "A positive cached-token count does not mean the whole prompt was cached. "
        "Small zero-cache groups do not establish a stable population p99.",
        "runs": rows,
    }
    (args.run / "cache_states.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
