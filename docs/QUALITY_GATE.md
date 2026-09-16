# Quantization and numerics quality gate

Performance alone is insufficient for a quantization or kernel change.

## Required checks

1. Fixed-prompt smoke set: no empty responses, NaNs, malformed UTF-8, or repeated-token collapse.
2. Deterministic greedy comparison: record exact-match rate against FP16 for short prompts.
3. Perplexity or standard task subset: run the same dataset revision and few-shot settings.
4. Tool calling: JSON parse success, required-key accuracy and argument-value accuracy.
5. Long context: verify retrieval from at least 8K context and inspect degradation by position.

The primary acceptance rule is less than one percentage point loss on the task-level metric chosen
before optimization. Exact token equality is diagnostic, not a universal acceptance requirement.

## Planned commands

Use the OpenAI-compatible SGLang endpoint for application-level checks and pin the evaluator commit.
The evaluator and dataset revisions will be selected on the GPU host only after model availability
is confirmed; they must be recorded in the run manifest before results are accepted.

