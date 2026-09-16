# Upstream contribution strategy

The project uses two Git histories:

- this repository: experiment definitions, automation and evidence;
- `upstream/sglang`: runtime code based on the pinned release and rebased onto current main before PR.

## Patch sequence

1. A small benchmark/observability patch that is useful independently and easy to review.
2. The runtime optimization with unit tests and targeted benchmarks.
3. Documentation or tuning guidance only after performance evidence is stable.

Each commit must build and test independently. Avoid mixing formatting, refactoring, new metrics and
performance behavior in one patch. The resume may say “submitted upstream PR” only after a public PR
exists, and “contributed to SGLang” only when the contribution status is stated precisely.

## Branches

```bash
git -C upstream/sglang switch -c agentperf/baseline-v0.5.19
git -C upstream/sglang switch -c agentperf/optimization-name
```

Before submission, fetch `sgl-project/sglang` main, rebase, rerun correctness tests and repeat the
performance comparison against the rebased parent commit.

