# GPU runbook

## 1. Host checks

The host must provide an NVIDIA driver, Docker with the NVIDIA runtime, a mounted `/data` volume,
and at least 80 GiB free space. The project uses the host driver but pins all user-space CUDA and
SGLang dependencies in `CONTAINER.lock`.

For mainland-China hosts, `CONTAINER.lock` also records a mirror reference. The content digest is
identical to the official Docker Hub image; the bootstrap tries the official registry first and
falls back to the mirror without relaxing content verification.

Large model downloads must be verified before launch. For sources with pinned file metadata:

```bash
agentperf verify-model --model qwen3_8b_w8a8 --model-dir /data/models/Qwen3-8B-W8A8
```

```bash
nvidia-smi
docker info
df -h /data
```

If the 200 GiB data disk is not mounted, stop before formatting anything and identify the exact
unused block device with `lsblk -f`. Disk initialization is intentionally not automated because a
wrong target would be destructive.

## 2. Bootstrap

```bash
bash scripts/bootstrap_remote.sh
cp configs/host.env.example .env
```

Download only the two required checkpoints first:

```bash
source .venv/bin/activate
hf download Qwen/Qwen3-8B --local-dir /data/models/Qwen3-8B
hf download Qwen/Qwen3-8B-AWQ --local-dir /data/models/Qwen3-8B-AWQ
```

Model IDs and trust status are recorded in `configs/model_sources.json`. The GPTQ checkpoint is a
third-party comparison and must not be treated as an official Qwen baseline.

## 3. Smoke gate

```bash
bash scripts/run_gpu_smoke.sh
```

Do not start the matrix until all eight requests complete, generated outputs are non-empty, the
server log contains no fallback/OOM warnings, and three JSONL records exist.

## 4. Baseline matrix

Run from the pinned container shell:

```bash
bash scripts/container_shell.sh bash
python -m agentperf.cli run --config configs/experiment_matrix.json \
  --model qwen3_8b_fp16 --profile baseline --suite core
python -m agentperf.cli run --config configs/experiment_matrix.json \
  --model qwen3_8b_fp16 --profile eager --suite core
python -m agentperf.cli run --config configs/experiment_matrix.json \
  --model qwen3_8b_awq --profile baseline --suite core
```

Summarize each generated run directory separately:

```bash
python -m agentperf.cli summarize --run-dir results/RUN_DIRECTORY \
  --output results/RUN_DIRECTORY/summary.csv
```

## 5. Profiling gate

Start with SGLang's PyTorch profile around a representative online workload. Use Nsight only after
the online metrics reproduce, because trace collection perturbs timing. Capture prefill and decode
separately and retain raw traces outside Git under `profiles/`.

Before implementing a runtime change, add the top three time consumers and the chosen hypothesis
to `docs/DECISION_LOG.md`.

## 6. Candidate validation

Create an optimization branch in `upstream/sglang`, never modify the detached baseline commit.
Re-run the same commands, model revisions and workload inputs. A candidate is rejected unless it
passes correctness, quality, memory and performance gates in `docs/EXPERIMENT_PROTOCOL.md`.

## 7. Reproduce the custom-kernel experiment

First verify the source patch series on CPU; it constructs a temporary Git index and checks
the exact patched tree without changing the current checkout:

```bash
python scripts/check_patch_series.py
bash scripts/container_shell.sh python scripts/run_gpu_validation.py
```

The second command runs the integer GEMM, native method dispatch, bounded JIT-variant and native
norm-fusion tests sequentially. Every command, return code, source fingerprint and log is retained
under `results/gpu-validation/`. Do not run GPU microbenchmarks concurrently with serving tests.

Then collect unprofiled serving data. Each command uses alternating OFF/ON, ON/OFF, OFF/ON
server restarts; the default control has both custom switches disabled:

```bash
bash scripts/container_shell.sh python scripts/run_paired_prefill.py --suite short --candidate gemm
bash scripts/container_shell.sh python scripts/run_paired_prefill.py --suite short --candidate fusion
bash scripts/container_shell.sh python scripts/run_paired_prefill.py --suite short --candidate combined
python scripts/audit_paired_run.py results/paired/RUN_DIRECTORY
```

These are 96/128/160 **native-input-token**, 1-output-token, concurrency-one latency-isolation
experiments, not a representative chatbot throughput score. Output=1 has no TPOT; its relative
TPOT change is undefined. The 160-token point is the GEMM fallback control, but norm fusion still
applies there. Keep `--suite core` as a separate regression rather than averaging it with `short`.

Collect execution proof separately (profiling perturbs timing):

```bash
bash scripts/container_shell.sh python scripts/run_paired_prefill.py \
  --suite proof --candidate combined --repetitions 1
```

The ON trace must contain `_int8_prefill` and `_rmsnorm_quant_int8`; the OFF trace must not.
The quality runner is `scripts/run_acceptance_quality.py --candidate combined`. Its fixed-window
corpus is generated by `scripts/prepare_wikitext_quality.py` from the dataset revision/hash in the
script, using the pinned model tokenizer. This is independent-window regression NLL, not a
standard full-document WikiText perplexity or a broad language-model capability evaluation.

All switches remain opt-in. Do not enable a candidate on unmeasured hardware merely because it
passes CPU CI; CUDA numerical tests and serving measurements require the actual GPU.
