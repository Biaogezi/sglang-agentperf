# Reading and independently checking the evidence

`evidence/a10/` contains immutable snapshots. Original JSONL, server logs, complete traces and
model weights are not ordinary Git source files. Their byte sizes and SHA-256 hashes connect
local raw backups to compact public results. `scripts/verify_evidence.py` checks every published
artifact's exact bytes; CI runs the same check.

There are four different evidence types:

| Artifact | What it proves | What it does not prove |
|---|---|---|
| `manifest.json` + `summary.csv` | Launch/config/source identity and repeated serving metrics | Universal speedup or production SLO |
| `quality.json` + task answers | Fixed-corpus NLL and original synthetic task checks | General model capability or standard full WikiText PPL |
| Trace tables + raw trace hash | The candidate really executes and where captured time goes | Unprofiled production latency |
| GPU test logs + source hashes | Tested arithmetic, boundary dispatch and compiler behavior | Correctness on every GPU/model |

## Request-level metrics without generated text

`final_short_requests_off/on/request_metrics.json.gz` publishes the measured per-request
TTFT/ITL arrays, token lengths, total benchmark duration and SHA-256 of each generated text.
It contains no prompts, generated text or `server_info`. Each record includes its original raw
JSONL hash. The same publisher is available for later regression runs as
`agentperf.evidence.snapshot_request_metrics`.

An example independent reconstruction, from the repository root:

```python
import gzip
import json
import statistics
from pathlib import Path

def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)

for arm in ("off", "on"):
    path = Path(f"evidence/a10/final_short_requests_{arm}/request_metrics.json.gz")
    data = json.loads(gzip.decompress(path.read_bytes()))
    rows = [row for row in data["runs"] if "__short_prefill_128__" in row["file"]]
    assert len(rows) == 3 and all(row["input_lengths_valid"] for row in rows)
    throughput = [row["total_input_tokens"] / row["duration_s"] for row in rows]
    per_run_p99 = [1000 * percentile(row["ttft_s"], 0.99) for row in rows]
    print(arm, statistics.mean(throughput), statistics.stdev(throughput),
          statistics.mean(per_run_p99))
```

This should reproduce approximately `4143.97 ± 20.08` and `4688.91 ± 27.87` input tok/s,
and `31.705` versus `28.115` ms mean-of-run-p99 TTFT. Match OFF/ON records by workload and
repetition suffix, then compare `output_text_sha256`, `output_lens` and `errors` to audit output
agreement. Text-hash equality is not a stored token-ID-by-token-ID comparison.

Do not pool multiple protocols or cherry-pick the fastest run. One-token output has no meaningful
TPOT. Input lengths are marked valid only when a native-ID manifest establishes that protocol;
multi-turn and historical text-roundtrip input metrics must not be reinterpreted as native IDs.
ITL arrays measure streaming events and do not reconstruct exact request E2E completion time.

The seven-patch source tree is verified separately with `scripts/check_patch_series.py`.
Performance and quality launches retain executed-source fingerprints even when the GPU host's
Git commit ID differs from the public documentation commit. Later offline audit/publishing edits
do not change the runtime that produced the retained measurements.
