# Synthetic multi-turn agent replay

This workload exercises real SGLang multi-turn chat serving, but its repository context and
tool observations are original synthetic fixtures. It is **not** SWE-bench, a downloaded agent
trace, or an agent task-success evaluation. No actual tools are executed.

`scripts/prepare_agent_trace.py` deterministically generates 24 conversations, each with a
repository listing followed by three tool-observation messages. The existing upstream
`AgenticTraceDataset` and `sglang-oai-chat` benchmark client send the four turns in order and
insert the server's actual assistant reply into the next turn's history. We integrate those
upstream facilities rather than claiming to have written their replay engine or prefix cache.

Protocol:

- 24 conversations × 4 turns = 96 requests per repetition; 4 concurrent conversations.
- Fixed 32 generated tokens per turn using the upstream ignore-EOS benchmark behavior.
- Explicit `stream_options.include_usage=true` requests server completion-token usage; the chat
  benchmark updates its output length from that usage rather than only the requested limit.
- Thinking disabled through `chat_template_kwargs`, temperature zero.
- Flush prefix cache before each measured run; retain cache reuse **within** a conversation.
- Record dataset SHA-256, source fingerprints, output token counts, output texts and errors.
- Alternating server restarts and three repetitions for each accepted OFF/ON result.
- Compare output histories as well as timing: if responses diverge, later prompts differ too.
  Such measurements cannot be called identical-input replay, even at temperature zero.

Commands (inside the pinned GPU container):

```bash
python scripts/prepare_agent_trace.py
python scripts/run_paired_prefill.py --suite agent --candidate gemm --repetitions 3
python scripts/audit_paired_run.py results/paired/RUN_DIRECTORY
```

Native fixed-token prefill micro-workloads and natural-text multi-turn replay answer different
questions; their absolute throughput must not be compared as though they were the same inputs.
Status/results are recorded in the final experiment report once measurements finish.

## Upstream metric limitation

At the pinned release, `AgenticTraceDataset` treats the first turn's `prompt_tokens` as
informational, and `benchmark.serving.calculate_metrics` receives `input_requests=None` for
multi-turn requests. The chat request client does not update `output.prompt_len` from server
usage. Therefore the emitted input tok/s and prompt-denominator cache-hit rate are **not valid
multi-turn metrics** (our fixtures leave that informational length zero). Do not report them.
Use completed requests, output tok/s, TTFT, TPOT and E2E for this replay; preserve the raw fields
so the limitation is auditable. This project has not patched that upstream benchmark limitation.
