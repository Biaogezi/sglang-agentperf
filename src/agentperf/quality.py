from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SGLang returned HTTP {error.code}: {body}") from error
    result = json.loads(body)
    if not isinstance(result, dict):
        raise TypeError(f"Expected one response object, got {type(result).__name__}")
    return result


def extract_logprobs(response: dict[str, Any]) -> list[float]:
    """Extract finite prompt-token logprobs from SGLang's native response."""
    try:
        entries = response["meta_info"]["input_token_logprobs"]
    except (KeyError, TypeError) as error:
        raise ValueError("Response does not contain meta_info.input_token_logprobs") from error
    if not isinstance(entries, list):
        raise TypeError("input_token_logprobs must be a list")

    values: list[float] = []
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or not entry:
            continue
        value = entry[0]
        if value is None:
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Non-finite input logprob: {numeric}")
        values.append(numeric)
    if not values:
        raise ValueError("No scored prompt tokens were returned")
    return values


def _load_corpus(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            raise TypeError(f"Invalid corpus record at line {line_number}")
        text = record["text"].strip()
        if not text:
            raise ValueError(f"Empty text at line {line_number}")
        records.append({"id": str(record.get("id", line_number)), "text": text})
    if not records:
        raise ValueError("Corpus is empty")
    return records


def score_corpus(
    *,
    endpoint: str,
    corpus_path: Path,
    output_path: Path,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    records = _load_corpus(corpus_path)
    documents: list[dict[str, Any]] = []
    total_nll = 0.0
    total_tokens = 0

    for record in records:
        response = _post_json(
            endpoint.rstrip("/") + "/generate",
            {
                "text": record["text"],
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                "return_logprob": True,
                "return_input_logprob": True,
                "logprob_start_len": 0,
            },
            timeout_s,
        )
        logprobs = extract_logprobs(response)
        nll = -sum(logprobs)
        documents.append(
            {
                "id": record["id"],
                "scored_tokens": len(logprobs),
                "mean_nll": nll / len(logprobs),
            }
        )
        total_nll += nll
        total_tokens += len(logprobs)

    mean_nll = total_nll / total_tokens
    corpus_bytes = corpus_path.read_bytes()
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "corpus": str(corpus_path),
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "documents": documents,
        "document_count": len(documents),
        "scored_tokens": total_tokens,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def compare_quality(
    baseline_path: Path,
    candidate_path: Path,
    *,
    max_nll_increase: float,
) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if baseline["corpus_sha256"] != candidate["corpus_sha256"]:
        raise ValueError("Quality results use different corpora")
    if baseline["scored_tokens"] != candidate["scored_tokens"]:
        raise ValueError("Quality results scored different token counts")

    nll_delta = float(candidate["mean_nll"]) - float(baseline["mean_nll"])
    ppl_delta_pct = (
        (float(candidate["perplexity"]) / float(baseline["perplexity"])) - 1.0
    ) * 100.0
    return {
        "baseline_mean_nll": baseline["mean_nll"],
        "candidate_mean_nll": candidate["mean_nll"],
        "mean_nll_delta": nll_delta,
        "perplexity_delta_pct": ppl_delta_pct,
        "max_nll_increase": max_nll_increase,
        "passed": nll_delta <= max_nll_increase,
    }
