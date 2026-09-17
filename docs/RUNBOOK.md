# GPU runbook

## 1. Host checks

The host must provide an NVIDIA driver, Docker with the NVIDIA runtime, a mounted `/data` volume,
and at least 80 GiB free space. The project uses the host driver but pins all user-space CUDA and
SGLang dependencies in `CONTAINER.lock`.

For mainland-China hosts, `CONTAINER.lock` also records a mirror reference. The content digest is
identical to the official Docker Hub image; the bootstrap tries the official registry first and
falls back to the mirror without relaxing content verification.

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
