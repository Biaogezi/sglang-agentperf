# Quantization and numerics quality gate

Performance alone is insufficient for a quantization or kernel change.

## Required checks

1. Fixed-prompt smoke set: no empty responses, NaNs, malformed UTF-8, or repeated-token collapse.
2. Deterministic greedy comparison: record exact-match rate against FP16 for short prompts.
3. Fixed-corpus prompt NLL smoke gate: run `agentperf score-corpus` against the versioned
   multilingual corpus and reject a mean-NLL increase above the predeclared threshold.
4. Perplexity or standard task subset: run the same dataset revision and few-shot settings.
5. Tool calling: JSON parse success, required-key accuracy and argument-value accuracy.
6. Long context: verify retrieval from at least 8K context and inspect degradation by position.

The prompt-NLL corpus is a fast numerical regression test, not a substitute for a standard
downstream evaluation. The primary downstream acceptance rule is less than one percentage point
loss on the task-level metric chosen before optimization. Exact token equality is diagnostic, not
a universal acceptance requirement.

## Planned commands

```bash
agentperf run-quality --config configs/experiment_matrix.json --model qwen3_8b_fp16 \
  --corpus configs/quality_corpus.jsonl --output-root quality/cloud
agentperf compare-quality --baseline quality/fp16/quality.json --candidate quality/awq/quality.json \
  --max-nll-increase 0.02
```

For application-level checks use the OpenAI-compatible SGLang endpoint and pin the evaluator and
dataset revisions in the run manifest before accepting results.
