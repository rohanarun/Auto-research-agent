# okc: problem=muse_moe_w4a16
"""Routed experts: one Triton routing kernel, two grouped NVFP4 GEMMs, one Triton post kernel.

v1 (submission_v1_correct_slow.py) passed public case 0 at 3.7e-3 relative L2 but ran about 6.5x
slower than the baseline, mostly from ~40 small torch launches. v2 moves the sort/padding into one
Triton kernel and the post-expert norm, route weighting and sum over the 8 routes into one Triton
kernel. E2M1 decode uses arithmetic instead of exp2. Not yet measured on the GPU."""

from typing import TypedDict

import torch
import triton
import triton.language as tl


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


BM = 16
BN = 64
BKB = 64  # packed bytes per K step: 128 elements, 8 scale blocks
POST_NORM_EPS = 1e-8


@triton.jit
def _e2m1(c):
    # c: int32 nibble. Bit 3 is the sign; bits 0-2 index {0, .5, 1, 1.5, 2, 3, 4, 6}.
    m = c & 7
    ex = m >> 1
    mant = (m & 1).to(tl.float32)
    pw = tl.where(ex == 1, 0.5, tl.where(ex == 2, 1.0, 2.0))
    mag = tl.where(ex == 0, mant * 0.5, (2.0 + mant) * pw)
    return tl.where(c >= 8, -mag, mag)


@triton.jit
def _mac(acc, xe, xo, wbase, sbase, rows, kb, K: tl.constexpr):
    # acc += x @ W^T over one K tile. W is packed E2M1 [N, K/2] with E4M3 block scales [N, K/16];
    # byte j holds elements 2j (low nibble) and 2j+1 (high nibble).
    wb = tl.load(wbase + rows[:, None] * (K // 2) + kb[None, :]).to(tl.int32)
    sc = tl.load(sbase + rows[:, None] * (K // 16) + (kb // 8)[None, :]).to(tl.float32)
    lo = (_e2m1(wb & 15) * sc).to(tl.bfloat16)
    hi = (_e2m1(wb >> 4) * sc).to(tl.bfloat16)
    return acc + tl.dot(xe, tl.trans(lo)) + tl.dot(xo, tl.trans(hi))


@triton.jit
def _route_kernel(ids_ptr, tok_ptr, dst_ptr, bexp_ptr, ls,
                  P: tl.constexpr, PP: tl.constexpr, PPAD: tl.constexpr, NL: tl.constexpr,
                  BM: tl.constexpr, TOP_K: tl.constexpr, NBV: tl.constexpr, NB: tl.constexpr):
    # One program. Pair p = token * TOP_K + slot. Local pairs get a row in an expert-sorted buffer
    # where each expert's run starts on a BM boundary; tok_ptr is pre-filled with -1 (padding).
    offs = tl.arange(0, PP)
    valid = offs < P
    ids = tl.load(ids_ptr + offs, mask=valid, other=-1)
    loc = valid & (ids >= ls) & (ids < ls + NL)
    le = tl.where(loc, ids - ls, NL)
    ex = tl.arange(0, NL)
    eq = (le[:, None] == ex[None, :]).to(tl.int32)
    cs = tl.cumsum(eq, axis=0)
    cnt = tl.sum(eq, axis=0)
    pc = (cnt + BM - 1) // BM * BM
    pend = tl.cumsum(pc, axis=0)
    pstart = pend - pc
    rank = tl.sum(eq * (cs - 1), axis=1)
    base = tl.sum(eq * pstart[None, :], axis=1)
    dst = tl.where(loc, base + rank, PPAD)
    tl.store(dst_ptr + offs, dst, mask=valid)
    tl.store(tok_ptr + dst, offs // TOP_K, mask=loc)
    bidx = tl.arange(0, NB)
    bexp = tl.sum((pend[None, :] <= bidx[:, None] * BM).to(tl.int32), axis=1)
    tl.store(bexp_ptr + bidx, bexp, mask=bidx < NBV)


@triton.jit
def _up_kernel(x_ptr, w_ptr, s_ptr, gs_ptr, tok_ptr, bexp_ptr, inter_ptr,
               H: tl.constexpr, I: tl.constexpr, NL: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BKB: tl.constexpr):
    b = tl.program_id(0)
    nt = tl.program_id(1)
    e = tl.load(bexp_ptr + b)
    if e < NL:
        offs_m = tl.arange(0, BM)
        offs_n = nt * BN + tl.arange(0, BN)
        offs_j = tl.arange(0, BKB)
        tok = tl.load(tok_ptr + b * BM + offs_m)
        live = tok >= 0
        xrow = x_ptr + tl.where(live, tok, 0)[:, None] * H
        wbase = w_ptr + e * (2 * I * (H // 2))
        sbase = s_ptr + e * (2 * I * (H // 16))
        acc_g = tl.zeros((BM, BN), dtype=tl.float32)
        acc_u = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, H // 2, BKB):
            kb = k0 + offs_j
            xe = tl.load(xrow + 2 * kb[None, :], mask=live[:, None], other=0.0)
            xo = tl.load(xrow + 2 * kb[None, :] + 1, mask=live[:, None], other=0.0)
            acc_g = _mac(acc_g, xe, xo, wbase, sbase, offs_n, kb, H)
            acc_u = _mac(acc_u, xe, xo, wbase, sbase, offs_n + I, kb, H)
        gs = tl.load(gs_ptr + e)
        g = acc_g * gs
        u = acc_u * gs
        inter = g / (1.0 + tl.exp(-g)) * u
        tl.store(inter_ptr + (b * BM + offs_m)[:, None] * I + offs_n[None, :], inter.to(tl.bfloat16))


@triton.jit
def _down_kernel(inter_ptr, w_ptr, s_ptr, gs_ptr, bexp_ptr, y_ptr,
                 H: tl.constexpr, I: tl.constexpr, NL: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BKB: tl.constexpr):
    b = tl.program_id(0)
    nt = tl.program_id(1)
    e = tl.load(bexp_ptr + b)
    if e < NL:
        offs_m = tl.arange(0, BM)
        offs_n = nt * BN + tl.arange(0, BN)
        offs_j = tl.arange(0, BKB)
        rows = b * BM + offs_m
        wbase = w_ptr + e * (H * (I // 2))
        sbase = s_ptr + e * (H * (I // 16))
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, I // 2, BKB):
            kb = k0 + offs_j
            ie = tl.load(inter_ptr + rows[:, None] * I + 2 * kb[None, :])
            io = tl.load(inter_ptr + rows[:, None] * I + 2 * kb[None, :] + 1)
            acc = _mac(acc, ie, io, wbase, sbase, offs_n, kb, I)
        gs = tl.load(gs_ptr + e)
        tl.store(y_ptr + rows[:, None] * H + offs_n[None, :], (acc * gs).to(tl.bfloat16))


@triton.jit
def _post_kernel(y_ptr, dst_ptr, tw_ptr, w_ptr, out_ptr, H: tl.constexpr, TOP_K: tl.constexpr,
                 PPAD: tl.constexpr, EPS: tl.constexpr):
    # One program per token: post-expert norm of each local route, weighted and summed.
    t = tl.program_id(0)
    offs = tl.arange(0, H)
    wp1 = 1.0 + tl.load(w_ptr + offs).to(tl.float32)
    acc = tl.zeros((H,), dtype=tl.float32)
    for s in tl.static_range(TOP_K):
        d = tl.load(dst_ptr + t * TOP_K + s)
        tw = tl.load(tw_ptr + t * TOP_K + s)
        yv = tl.load(y_ptr + d * H + offs).to(tl.float32)
        z = (yv * wp1).to(tl.bfloat16).to(tl.float32)
        z = z * tl.rsqrt(tl.sum(z * z, axis=0) / H + EPS)
        acc += tl.where(d < PPAD, z * tw, 0.0)
    tl.store(out_ptr + t * H + offs, acc.to(tl.bfloat16))


def prepare(weights: Weights) -> Weights:
    """Called once, untimed, before any case. Repack the weights as you like."""
    return {k: v.contiguous() for k, v in weights.items()}


def kernel_fn(data: Inputs, weights: Weights) -> torch.Tensor:
    """This GPU's share of the MoE output, bfloat16 [M, H]: for each token, the sum over its
    experts that are local of weight * post_expert_norm(expert(x)). Rows whose experts are
    all elsewhere are zero."""
    cfg = data["config"]
    x = data["hidden_states"]
    M, H = x.shape
    I = cfg["d_expert"]
    NL = cfg["local_count"]
    ls = cfg["local_start"]
    top_k = cfg["top_k"]
    dev = x.device
    P = M * top_k
    PP = triton.next_power_of_2(P)
    ppad = ((P + NL * (BM - 1) + BM - 1) // BM) * BM
    nb = ppad // BM
    NB = triton.next_power_of_2(nb)

    ids = data["topk_ids"].reshape(-1)
    tw = data["topk_weights"].reshape(-1)
    tok_row = torch.full((ppad,), -1, dtype=torch.int32, device=dev)
    dst = torch.empty((P,), dtype=torch.int32, device=dev)
    bexp = torch.empty((nb,), dtype=torch.int32, device=dev)
    _route_kernel[(1,)](ids, tok_row, dst, bexp, ls, P=P, PP=PP, PPAD=ppad, NL=NL, BM=BM,
                        TOP_K=top_k, NBV=nb, NB=NB, num_warps=8)

    inter = torch.empty((ppad, I), dtype=torch.bfloat16, device=dev)
    _up_kernel[(nb, I // BN)](x, weights["w13"], weights["w13_sf"], weights["w13_gs"], tok_row, bexp, inter,
                              H=H, I=I, NL=NL, BM=BM, BN=BN, BKB=BKB, num_warps=4, num_stages=3)
    y = torch.empty((ppad + 1, H), dtype=torch.bfloat16, device=dev)
    _down_kernel[(nb, H // BN)](inter, weights["w2"], weights["w2_sf"], weights["w2_gs"], bexp, y,
                                H=H, I=I, NL=NL, BM=BM, BN=BN, BKB=BKB, num_warps=4, num_stages=3)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    _post_kernel[(M,)](y, dst, tw, weights["post_norm_w"], out, H=H, TOP_K=top_k, PPAD=ppad,
                       EPS=POST_NORM_EPS, num_warps=4)
    return out
