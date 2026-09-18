"""Experimental INT8 GEMM with independent integer split-K partials.

Weight layout is the runtime's [K,N] column-major view. No float atomics:
partials are summed as INT32 before a single scale/cast epilogue.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(
    A,
    B,
    P,
    SA,
    SB,
    BIAS,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    ki = split * BK + tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.int32)
    for step in range(tl.cdiv(K, BK * SPLITS)):
        kk = ki + step * BK * SPLITS
        a = tl.load(
            A + mi[:, None] * K + kk[None, :], (mi[:, None] < M) & (kk[None, :] < K), other=0
        )
        b = tl.load(
            B + kk[:, None] + ni[None, :] * K, (kk[:, None] < K) & (ni[None, :] < N), other=0
        )
        acc += tl.dot(a, b)
    if SPLITS == 1:
        sa = tl.load(SA + mi, mi < M, other=0)
        sb = tl.load(SB + ni, ni < N, other=0)
        values = acc.to(tl.float32) * (sa[:, None] * sb[None, :])
        if HAS_BIAS:
            values += tl.load(BIAS + ni, ni < N, other=0).to(tl.float32)[None, :]
        tl.store(OUT + mi[:, None] * N + ni[None, :], values, (mi[:, None] < M) & (ni[None, :] < N))
    else:
        tl.store(
            P + split * M * N + mi[:, None] * N + ni[None, :],
            acc,
            (mi[:, None] < M) & (ni[None, :] < N),
        )


@triton.jit
def _reduce(
    P,
    SA,
    SB,
    BIAS,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < M * N
    acc = tl.full((BLOCK,), 0, tl.int32)
    for split in range(SPLITS):
        acc += tl.load(P + split * M * N + idx, mask, other=0)
    sa = tl.load(SA + idx // N, mask, other=0)
    sb = tl.load(SB + idx % N, mask, other=0)
    value = acc.to(tl.float32) * (sa * sb)
    if HAS_BIAS:
        value += tl.load(BIAS + idx % N, mask, other=0).to(tl.float32)
    tl.store(OUT + idx, value, mask)


def splitk_mm(
    a,
    b,
    sa,
    sb,
    *,
    bm=16,
    bn=64,
    bk=128,
    splits=4,
    warps=4,
    stages=3,
    dtype=torch.float16,
    bias=None,
):
    m, k = a.shape
    n = b.shape[1]
    assert a.is_contiguous() and b.stride() == (1, k)
    partial = (
        torch.empty((splits, m, n), device=a.device, dtype=torch.int32) if splits > 1 else None
    )
    out = torch.empty((m, n), device=a.device, dtype=dtype)
    _gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn), splits)](
        a,
        b,
        partial,
        sa,
        sb,
        bias,
        out,
        m,
        n,
        k,
        bm,
        bn,
        bk,
        splits,
        bias is not None,
        num_warps=warps,
        num_stages=stages,
        enable_fp_fusion=False,
    )
    if splits > 1:
        _reduce[(triton.cdiv(m * n, 256),)](
            partial, sa, sb, bias, out, m, n, splits, bias is not None, 256, enable_fp_fusion=False
        )
    return out
