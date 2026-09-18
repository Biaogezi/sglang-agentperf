"""Audit paired evidence; output differences are reported, not hidden by latency gains."""

import argparse
import json
from pathlib import Path

from agentperf.report import audit_paired_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--minimum-repetitions", type=int, default=3)
    args = parser.parse_args()
    result = audit_paired_run(args.run, minimum_repetitions=args.minimum_repetitions)
    rendered = json.dumps(result, indent=2) + "\n"
    (args.run / "audit.json").write_text(rendered)
    print(rendered)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
