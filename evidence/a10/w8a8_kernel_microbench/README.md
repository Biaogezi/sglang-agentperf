# W8A8 kernel microbenchmarks

These JSON files were produced on the NVIDIA A10 with the same digest-pinned container and
SGLang worktree used by the serving runs.

- `w8a8_norm_quant_a10.json`: seven-round medians and every raw round for the unfused versus fused
  RMSNorm + per-token INT8 quantization operator pair.
- `w8a8_gemm_a10.json`: current CUTLASS W8A8 GEMM versus `torch._int_mm` plus PyTorch/custom
  Triton epilogues on exact Qwen3-8B projection shapes.

Reproduce with:

```bash
bash scripts/container_shell.sh python scripts/bench_w8a8_norm_quant.py \
  --rounds 7 --output results/micro/w8a8_norm_quant_a10.json
bash scripts/container_shell.sh python scripts/bench_w8a8_gemm.py \
  --output results/micro/w8a8_gemm_a10.json
```

SHA-256:

- `w8a8_norm_quant_a10.json`: `4f85becfb8574549deab77764595598cea8393acdcfe5150cd9bb586751b9c15`
- `w8a8_gemm_a10.json`: `a0857a94f1bf50439d6d7960ef055b3f759079b43e6eb9f6aa115cfbe9ef8b78`

Kernel-only percentages are never reported as serving speedups.

