import json
from pathlib import Path

import pytest

from agentperf.quality import compare_quality, extract_logprobs


def test_extract_logprobs_ignores_unscored_first_token() -> None:
    response = {
        "meta_info": {
            "input_token_logprobs": [[None, 1, "a"], [-0.5, 2, "b"], [-1.0, 3, "c"]]
        }
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
