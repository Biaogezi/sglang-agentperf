"""Generate deterministic synthetic conversations for the upstream multi-turn client."""

import argparse
import hashlib
import json
from pathlib import Path

from agentperf.workloads import synthetic_agent_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("configs/synthetic_agent_trace.json"))
    args = parser.parse_args()
    raw = (json.dumps(synthetic_agent_trace(), indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(json.dumps({"path": str(args.output), "sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
