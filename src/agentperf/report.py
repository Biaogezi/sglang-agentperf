from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

METRIC_ALIASES = {
    "request_throughput": ("request_throughput",),
    "output_throughput": ("output_throughput", "output_throughput_tok_s"),
    "ttft_p99_ms": ("p99_ttft_ms", "ttft_p99_ms"),
    "tpot_p99_ms": ("p99_tpot_ms", "tpot_p99_ms"),
    "itl_p99_ms": ("p99_itl_ms", "itl_p99_ms"),
}


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
    for path in sorted(run_dir.glob("*.jsonl")):
        stem = path.stem
        group = stem.rsplit("__r", 1)[0]
        grouped[group].append(read_last_json(path))

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
        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["case", "repetitions"]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows
