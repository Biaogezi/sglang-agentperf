from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO

GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def _open_trace(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _aggregate(events: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "total_us": 0.0}
    )
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != category:
            continue
        name = str(event.get("name", "<unnamed>"))
        totals[name]["count"] += 1
        totals[name]["total_us"] += float(event.get("dur", 0.0))

    category_total = sum(float(values["total_us"]) for values in totals.values())
    rows = []
    for name, values in totals.items():
        count = int(values["count"])
        total_us = float(values["total_us"])
        rows.append(
            {
                "name": name,
                "count": count,
                "total_ms": total_us / 1000,
                "mean_us": total_us / count,
                "category_time_pct": total_us / category_total * 100 if category_total else 0.0,
            }
        )
    return sorted(rows, key=lambda row: float(row["total_ms"]), reverse=True)


def _gpu_activity(events: list[dict[str, Any]]) -> dict[str, float | int]:
    intervals = sorted(
        (float(event["ts"]), float(event["ts"]) + float(event.get("dur", 0.0)))
        for event in events
        if event.get("ph") == "X" and event.get("cat") in GPU_CATEGORIES
    )
    if not intervals:
        return {"events": 0, "span_ms": 0.0, "active_ms": 0.0, "idle_pct": 0.0}

    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    active_us = sum(end - start for start, end in merged)
    # The last-starting event need not be the last-ending event (multiple streams).
    span_us = merged[-1][1] - merged[0][0]
    return {
        "events": len(intervals),
        "span_ms": span_us / 1000,
        "active_ms": active_us / 1000,
        "idle_pct": (1 - active_us / span_us) * 100 if span_us else 0.0,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else ["name", "count", "total_ms", "mean_us"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_trace(trace_path: Path, output_dir: Path) -> dict[str, Any]:
    with _open_trace(trace_path) as handle:
        payload = json.load(handle)
    events = payload.get("traceEvents", [])
    if not isinstance(events, list):
        raise TypeError("traceEvents must be a list")

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "kernels": _aggregate(events, "kernel"),
        "cuda_runtime": _aggregate(events, "cuda_runtime"),
        "cpu_ops": _aggregate(events, "cpu_op"),
        "user_annotations": _aggregate(events, "user_annotation"),
    }
    for name, rows in tables.items():
        _write_csv(output_dir / f"{name}.csv", rows)

    summary = {
        "trace": str(trace_path),
        "event_count": len(events),
        "gpu_activity": _gpu_activity(events),
        "top_kernels": tables["kernels"][:10],
        "top_cuda_runtime": tables["cuda_runtime"][:10],
        "top_cpu_ops": tables["cpu_ops"][:10],
        "top_user_annotations": tables["user_annotations"][:10],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
