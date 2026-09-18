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

## Smoke commands

```bash
agentperf run-quality --config configs/experiment_matrix.json --model qwen3_8b_fp16 \
  --corpus configs/quality_corpus.jsonl --output-root quality/cloud
agentperf compare-quality --baseline quality/fp16/quality.json --candidate quality/awq/quality.json \
  --max-nll-increase 0.02
```

For application-level checks use the OpenAI-compatible SGLang endpoint and pin the evaluator and
dataset revisions in the run manifest before accepting results.

## Expanded numerical regression

The acceptance runner uses 64 deterministic windows of 128 native input IDs from the pinned
WikiText-2 test set, scoring 8,128 next-token log probabilities. Windows are independent; the
result is not directly comparable to published sliding-window/full-document WikiText perplexity.
For a source change, compare OFF and ON of the **same checkpoint**. For quantization selection,
compare each format to FP16 with both custom switches OFF and the identical corpus IDs.

The 40 generated task regressions include strict JSON (16), exact-format arithmetic (16), and
long key retrieval (8). Report each category separately. Greedy output agreement and task
correctness are different: two variants producing the same wrong-format answer are not two
correct answers. The early W8A8 run passed JSON and retrieval, but scored 0/16 on strict arithmetic
format even though its answers contained the expected numbers. This weakness must remain visible.

The corpus SHA, dataset revision/hash, tokenizer path, runtime source fingerprints and task-set
hash are retained. No task result here establishes general instruction-following or tool-use
quality. JSON extraction is a constrained regression, not a full tool-calling benchmark.

## Expanded checkpoint comparison (2026-09-18)

With both custom switches OFF, the identical 8,128-token corpus gives FP16 mean NLL
**2.94421470** and AWQ-Marlin **3.02027352**: delta **+0.07605882**, or +7.90% in
exponentiated mean NLL. AWQ **fails** the unchanged +0.02 numerical gate. Its historical
335-token smoke passed; the larger test supersedes that acceptance, not the recorded timings.
AWQ remains a quality/performance trade-off reference, not the release's quality-approved path.

FP16 scores 0/16 strict arithmetic-format, 16/16 JSON and 8/8 retrieval; AWQ scores 5/16,
16/16 and 8/8 respectively. These tiny task samples do not negate the numerical failure.
The W8A8 source candidate is evaluated separately against both FP16 and its same-checkpoint OFF
control. Published data: `evidence/a10/fp16_quality_expanded/` and
`evidence/a10/awq_quality_expanded/`, including original synthetic-task answers.
