from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

METRIC_ALIASES = {
    "completed": ("completed",),
    "request_throughput": ("request_throughput",),
    "input_throughput": ("input_throughput", "input_throughput_tok_s"),
    "output_throughput": ("output_throughput", "output_throughput_tok_s"),
    "total_throughput": ("total_throughput", "total_throughput_tok_s"),
    "e2e_p99_ms": ("p99_e2e_latency_ms", "e2e_p99_ms"),
    "ttft_p99_ms": ("p99_ttft_ms", "ttft_p99_ms"),
    "tpot_p99_ms": ("p99_tpot_ms", "tpot_p99_ms"),
    "itl_p99_ms": ("p99_itl_ms", "itl_p99_ms"),
}

CASE_BEGIN = "AGENTPERF_CASE_BEGIN "
CASE_END = "AGENTPERF_CASE_END "
RETRACTED_REQUESTS = re.compile(r"#retracted_reqs:\s*(\d+)")
PREFILL_BATCH = re.compile(
    r"Prefill batch,.*#new-token:\s*(\d+),.*#running-req:\s*(\d+)"
)


def read_server_case_stats(path: Path) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "retraction_events": 0,
            "retracted_requests": 0,
            "prefill_batches": 0,
            "prefill_tokens": 0,
            "mixed_prefill_batches": 0,
            "mixed_prefill_tokens": 0,
        }
    )
    if not path.exists():
        return {}

    current_case: str | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(CASE_BEGIN):
            current_case = line.removeprefix(CASE_BEGIN).strip()
            continue
        if line.startswith(CASE_END):
            current_case = None
            continue
        if current_case and (match := RETRACTED_REQUESTS.search(line)):
            stats[current_case]["retraction_events"] += 1
            stats[current_case]["retracted_requests"] += int(match.group(1))
        if current_case and (match := PREFILL_BATCH.search(line)):
            new_tokens = int(match.group(1))
            running_requests = int(match.group(2))
            stats[current_case]["prefill_batches"] += 1
            stats[current_case]["prefill_tokens"] += new_tokens
            if running_requests > 0:
                stats[current_case]["mixed_prefill_batches"] += 1
                stats[current_case]["mixed_prefill_tokens"] += new_tokens
    return dict(stats)


def _first_number(record: dict[str, Any], names: Iterable[str]) -> float | None:
    for name in names:
        value = record.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def read_last_json(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError(f"No records in {path}")
    return records[-1]


def summarize_run(run_dir: Path, output_csv: Path) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_case_ids: dict[str, list[str]] = defaultdict(list)
    for path in sorted(run_dir.glob("*.jsonl")):
        stem = path.stem
        group = stem.rsplit("__r", 1)[0]
        grouped[group].append(read_last_json(path))
        grouped_case_ids[group].append(stem)

    server_stats = read_server_case_stats(run_dir / "server.log")

    rows: list[dict[str, Any]] = []
    for group, records in sorted(grouped.items()):
        row: dict[str, Any] = {"case": group, "repetitions": len(records)}
        for metric, aliases in METRIC_ALIASES.items():
            values = [
                value for record in records if (value := _first_number(record, aliases)) is not None
            ]
            row[f"{metric}_mean"] = statistics.fmean(values) if values else ""
            row[f"{metric}_stdev"] = (
                statistics.stdev(values) if len(values) >= 2 else 0.0 if values else ""
            )
        for metric in ("retraction_events", "retracted_requests"):
            values = [server_stats.get(case_id, {}).get(metric, 0) for case_id in grouped_case_ids[group]]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_stdev"] = statistics.stdev(values) if len(values) >= 2 else 0.0
        mixed_chunk_sizes = []
        for case_id in grouped_case_ids[group]:
            case_stats = server_stats.get(case_id, {})
            batches = case_stats.get("mixed_prefill_batches", 0)
            if batches:
                mixed_chunk_sizes.append(case_stats.get("mixed_prefill_tokens", 0) / batches)
        row["mixed_prefill_chunk_size_mean"] = (
            statistics.fmean(mixed_chunk_sizes) if mixed_chunk_sizes else ""
        )
        row["mixed_prefill_chunk_size_stdev"] = (
            statistics.stdev(mixed_chunk_sizes)
            if len(mixed_chunk_sizes) >= 2
            else 0.0 if mixed_chunk_sizes else ""
        )
        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["case", "repetitions"]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows
