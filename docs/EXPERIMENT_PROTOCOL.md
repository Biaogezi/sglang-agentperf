# Experiment protocol

## Required sequence

1. Record the GPU, driver, CUDA, PyTorch, Triton, SGLang commit, model revision and launch command.
2. Run the smoke suite and inspect generated text/errors before performance measurement.
3. Warm the server, then execute at least three repetitions per point.
4. Keep model, workload, randomization, request rate and server flags fixed between baseline and candidate.
5. Capture an online serving result first; use one-batch and kernel microbenchmarks only to explain it.
6. Profile prefill and decode separately. Do not infer decode behavior from a prefill-heavy trace.
7. Check GPU clocks, thermals, throttling and competing processes before accepting a regression.
8. Run the quality gate for every quantization or numerics change.

## Baseline ladder

For each model/precision pair, collect:

1. Default SGLang configuration.
2. CUDA Graph disabled.
3. Alternate compatible attention backend.
4. Candidate optimization on top of the best fair baseline.

This prevents attributing an existing command-line tuning opportunity to a code change.

## Workload rules

- `num_prompts >= 5 * max_concurrency` for steady-state serving runs.
- Random length ratio is fixed to 1 for shape-controlled comparisons.
- For `random-ids`, pass `--tokenize-prompt` to send native IDs. Without it, the benchmark
  decodes IDs to text and the server re-tokenizes; reported nominal lengths can differ from the
  actual GEMM row count. Check server-side rows, not just the CLI's dataset name.
- Shared-prefix tests report both cold-cache and warm-cache behavior.
- Agent traces retain request order and timestamps when replayed.
- Do not mix synthetic, ShareGPT and agent-trace results in one speedup number.

## Reporting rules

- Show absolute baseline and candidate values alongside speedup.
- Include mean and standard deviation across repetitions.
- Label throughput-bound and SLO-bound experiments separately.
- Report failed requests and output-token counts.
- A result with different output lengths is invalid unless normalized and explained.
- Preserve raw JSONL, logs, traces, environment manifests and Git diffs.
