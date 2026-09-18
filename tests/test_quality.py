import json
from pathlib import Path

import pytest

from agentperf.quality import _load_corpus, compare_quality, extract_logprobs, score_corpus


def test_extract_logprobs_ignores_unscored_first_token() -> None:
    response = {
        "meta_info": {"input_token_logprobs": [[None, 1, "a"], [-0.5, 2, "b"], [-1.0, 3, "c"]]}
    }
    assert extract_logprobs(response) == [-0.5, -1.0]


def test_extract_logprobs_rejects_non_finite_value() -> None:
    with pytest.raises(ValueError, match="Non-finite"):
        extract_logprobs({"meta_info": {"input_token_logprobs": [[float("nan"), 1]]}})


def test_compare_quality_enforces_matching_corpus_and_threshold(tmp_path: Path) -> None:
    baseline = {
        "corpus_sha256": "same",
        "scored_tokens": 100,
        "mean_nll": 2.0,
        "perplexity": 7.389056,
    }
    candidate = {
        "corpus_sha256": "same",
        "scored_tokens": 100,
        "mean_nll": 2.015,
        "perplexity": 7.500711,
    }
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    result = compare_quality(baseline_path, candidate_path, max_nll_increase=0.02)
    assert result["passed"] is True
    assert result["mean_nll_delta"] == pytest.approx(0.015)


@pytest.mark.parametrize("ids", [[True, 2], [-1, 2], [1], [1, 2.5], "12"])
def test_corpus_rejects_invalid_ids(tmp_path: Path, ids) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps({"input_ids": ids}), encoding="utf-8")
    with pytest.raises(ValueError):
        _load_corpus(path)


def test_score_token_corpus_preserves_ids(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"id":"window","input_ids":[3,4,5]}', encoding="utf-8")
    payloads = []

    def post(url, payload, timeout):
        payloads.append(payload)
        return {"meta_info": {"input_token_logprobs": [[None, 3], [-1.0, 4], [-2.0, 5]]}}

    monkeypatch.setattr("agentperf.quality._post_json", post)
    result = score_corpus(
        endpoint="http://localhost", corpus_path=path, output_path=tmp_path / "result.json"
    )
    assert payloads[0]["input_ids"] == [3, 4, 5]
    assert "text" not in payloads[0]
    assert result["scored_tokens"] == 2
    assert result["mean_nll"] == 1.5
