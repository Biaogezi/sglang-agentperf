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

COMPARISON_METRICS = {
    "input_throughput_mean": True,
    "output_throughput_mean": True,
    "e2e_p99_ms_mean": False,
    "ttft_p99_ms_mean": False,
    "tpot_p99_ms_mean": False,
    "itl_p99_ms_mean": False,
}

CASE_BEGIN = "AGENTPERF_CASE_BEGIN "
CASE_END = "AGENTPERF_CASE_END "
RETRACTED_REQUESTS = re.compile(r"#retracted_reqs:\s*(\d+)")
PREFILL_BATCH = re.compile(r"Prefill batch,.*#new-token:\s*(\d+),.*#running-req:\s*(\d+)")
CACHE_HIT_RATE = re.compile(r"Cache hit rate:\s*([0-9.]+)%")


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


def read_cache_hit_rate(path: Path) -> float | None:
    if not path.exists():
        return None
    matches = CACHE_HIT_RATE.findall(path.read_text(encoding="utf-8", errors="replace"))
    return float(matches[-1]) / 100 if matches else None


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
            values = [
                server_stats.get(case_id, {}).get(metric, 0) for case_id in grouped_case_ids[group]
            ]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_stdev"] = statistics.stdev(values) if len(values) >= 2 else 0.0
        cache_hit_rates = [
            value
            for case_id in grouped_case_ids[group]
            if (value := read_cache_hit_rate(run_dir / f"{case_id}.log")) is not None
        ]
        row["cache_hit_rate_mean"] = statistics.fmean(cache_hit_rates) if cache_hit_rates else ""
        row["cache_hit_rate_stdev"] = (
            statistics.stdev(cache_hit_rates)
            if len(cache_hit_rates) >= 2
            else 0.0
            if cache_hit_rates
            else ""
        )
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
            else 0.0
            if mixed_chunk_sizes
            else ""
        )
        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["case", "repetitions"]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _workload_name(case: str) -> str:
    parts = case.split("__", 2)
    return parts[-1]


def compare_summaries(
    baseline_csv: Path, candidate_csv: Path, output_csv: Path
) -> list[dict[str, Any]]:
    """Compare matching workloads; positive percentages always mean improvement."""

    def read_rows(path: Path) -> dict[str, dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as handle:
            return {_workload_name(row["case"]): row for row in csv.DictReader(handle)}

    baseline_rows = read_rows(baseline_csv)
    candidate_rows = read_rows(candidate_csv)
    rows: list[dict[str, Any]] = []
    for workload in sorted(baseline_rows.keys() & candidate_rows.keys()):
        baseline = baseline_rows[workload]
        candidate = candidate_rows[workload]
        row: dict[str, Any] = {
            "workload": workload,
            "baseline_case": baseline["case"],
            "candidate_case": candidate["case"],
        }
        for metric, higher_is_better in COMPARISON_METRICS.items():
            if not baseline.get(metric) or not candidate.get(metric):
                continue
            baseline_value = float(baseline[metric])
            candidate_value = float(candidate[metric])
            row[f"baseline_{metric}"] = baseline_value
            row[f"candidate_{metric}"] = candidate_value
            # One-token output has no inter-token latency. A relative change
            # against zero is undefined, not a 0% improvement or a fatal error.
            if baseline_value == 0:
                row[f"{metric}_improvement_pct"] = ""
                continue
            improvement = (
                (candidate_value - baseline_value) / baseline_value
                if higher_is_better
                else (baseline_value - candidate_value) / baseline_value
            )
            row[f"{metric}_improvement_pct"] = improvement * 100
        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        list(dict.fromkeys(key for row in rows for key in row))
        if rows
        else ["workload", "baseline_case", "candidate_case"]
    )
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def check_run_equivalence(baseline_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    """Check deterministic output fields for matching workload repetitions."""

    def records(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
        output = {}
        for path in run_dir.glob("*.jsonl"):
            parts = path.stem.rsplit("__", 2)
            if len(parts) != 3:
                continue
            output[(parts[-2], parts[-1])] = read_last_json(path)
        return output

    baseline = records(baseline_dir)
    candidate = records(candidate_dir)
    matched = sorted(baseline.keys() & candidate.keys())
    fields = ("generated_texts", "output_lens", "errors")
    mismatches = [
        {"workload": key[0], "repetition": key[1], "field": field}
        for key in matched
        for field in fields
        if field not in baseline[key]
        or field not in candidate[key]
        or baseline[key][field] != candidate[key][field]
    ]
    mismatches.extend(
        {"workload": key[0], "repetition": key[1], "field": "missing_record"}
        for key in sorted(baseline.keys() ^ candidate.keys())
    )
    return {
        "matched_repetitions": len(matched),
        "equivalent": bool(matched) and not mismatches,
        "mismatches": mismatches,
    }


def audit_paired_run(root: Path, *, minimum_repetitions: int = 3) -> dict[str, Any]:
    """Reject incomplete, mixed-source or failed-request paired experiments."""
    failures = []
    source_ids = set()
    successful = 0
    workloads: dict[str, set[tuple[str, int]]] = {}
    for profile in ("prefill_off", "prefill_on"):
        directory = root / profile
        launches = json.loads((directory / "manifest.json").read_text())["launches"]
        repetitions = [launch["repetition"] for launch in launches]
        if len(repetitions) < minimum_repetitions or len(set(repetitions)) != len(repetitions):
            failures.append(f"{profile}: insufficient or duplicate repetitions")
        expected_files = set()
        workloads[profile] = set()
        for launch in launches:
            manifest = launch["manifest"]
            fingerprint = manifest.get("source_files_sha256")
            if not fingerprint or not fingerprint.get("runtime_candidate"):
                failures.append(f"{profile}: missing executed-source fingerprints")
            source_ids.add(json.dumps(fingerprint, sort_keys=True))
            for case in manifest["cases"]:
                name = case["case_id"].rsplit("__r", 1)[0] + f"__r{launch['repetition']}"
                expected_files.add(name + ".jsonl")
                workload = manifest["config"]["workloads"][case["workload"]]
                workloads[profile].add((case["workload"], launch["repetition"]))
                path = directory / (name + ".jsonl")
                if not path.is_file():
                    failures.append(f"{profile}: missing {path.name}")
                    continue
                record = read_last_json(path)
                expected = workload["num_prompts"] * workload.get("turns_per_conversation", 1)
                errors = record.get("errors")
                if record.get("completed") != expected:
                    failures.append(f"{name}: incomplete requests")
                if not isinstance(errors, list) or len(errors) != expected or any(errors):
                    failures.append(f"{name}: missing or nonempty request errors")
                lengths = record.get("output_lens")
                texts = record.get("generated_texts")
                if not isinstance(lengths, list) or len(lengths) != expected:
                    failures.append(f"{name}: missing output lengths")
                if not isinstance(texts, list) or len(texts) != expected:
                    failures.append(f"{name}: missing generated output records")
                successful += int(record.get("completed", 0))
                args = workload.get("args", [])
                fixed_output = None
                if workload["dataset"] == "agentic-trace" and "--sharegpt-output-len" in args:
                    fixed_output = int(args[args.index("--sharegpt-output-len") + 1])
                if workload["dataset"] == "generated-shared-prefix" and "--gsp-output-len" in args:
                    fixed_output = int(args[args.index("--gsp-output-len") + 1])
                if workload["dataset"] == "random-ids":
                    if "--tokenize-prompt" not in args:
                        failures.append(f"{name}: nominal text lengths are not native IDs")
                    ratio = args[args.index("--random-range-ratio") + 1]
                    if float(ratio) == 1:
                        length = int(args[args.index("--random-input-len") + 1])
                        if record.get("input_lens") != [length] * expected:
                            failures.append(f"{name}: fixed token lengths differ")
                        if "--random-output-len" in args:
                            fixed_output = int(args[args.index("--random-output-len") + 1])
                if fixed_output is not None and lengths != [fixed_output] * expected:
                    failures.append(f"{name}: fixed output token lengths differ")
        if expected_files != {path.name for path in directory.glob("*.jsonl")}:
            failures.append(f"{profile}: unexpected or missing output files")
    if len(source_ids) != 1:
        failures.append("Executed source bytes differ across OFF/ON launches")
    if workloads["prefill_off"] != workloads["prefill_on"]:
        failures.append("OFF/ON workload repetitions differ")
    equivalence = check_run_equivalence(root / "prefill_off", root / "prefill_on")
    if any(row["field"] == "output_lens" for row in equivalence["mismatches"]):
        failures.append("Paired output lengths differ; timing is not comparable")
    return {
        "passed": not failures,
        "failures": failures,
        "completed_requests": successful,
        "matched_repetitions": len(workloads["prefill_off"] & workloads["prefill_on"]),
        "output_equivalence": equivalence,
    }
