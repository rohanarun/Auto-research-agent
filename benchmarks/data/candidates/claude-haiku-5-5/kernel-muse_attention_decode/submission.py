# okc: problem=muse_attention_decode
"""Attention step with the QK-norm, RoPE, FP8 cache write and query cast fused into one Triton
kernel. SGLang's path launches those as separate kernels (FusedQKNorm, store_cache, the query
cast); here they are one launch. FA3 and the output rms_norm_gate are SGLang's own kernels.

Queries are cast to FP8 straight from FP32 (no coarser than the baseline's bf16 -> FP8 cast)."""

from typing import TypedDict

import torch
import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases, 64 in some tests
    window: int           # 2048 on sliding-window layers; 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value


class Inputs(TypedDict):
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64]
    positions: torch.Tensor    # int64 [B*L]
    cache_loc: torch.Tensor    # int64 [B*L]
    cache_seqlens: torch.Tensor  # int32 [B]
    cu_seqlens_q: torch.Tensor   # int32 [B+1]
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    k_scale: torch.Tensor     # float32 []
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages]
    seq_lens: torch.Tensor    # int32 [B]
    config: Config


RMS_EPS = 1e-5
FP8 = torch.float8_e4m3fn
_INV_FREQ = {}


@triton.jit
def _qk_cache_kernel(qkv_ptr, pos_ptr, loc_ptr, invf_ptr, q_ptr, k_ptr, v_ptr, kscale_ptr, vscale_ptr,
                     stride_row, HQ: tl.constexpr, HKV: tl.constexpr, HALF: tl.constexpr,
                     ROPE: tl.constexpr, EPS: tl.constexpr):
    # One program per token. Each head's 64 dims are split into interleaved pairs (2i, 2i+1),
    # so RoPE and the pair-wise loads use [heads, HALF] tiles.
    t = tl.program_id(0)
    row = qkv_ptr + t * stride_row
    j = tl.arange(0, HALF)[None, :]
    hq = tl.arange(0, HQ)[:, None]
    hk = tl.arange(0, HKV)[:, None]
    if ROPE:
        pos = tl.load(pos_ptr + t).to(tl.float32)
        ang = pos * tl.load(invf_ptr + j)
        cos = tl.cos(ang)
        sin = tl.sin(ang)

    # Queries: scaleless RMSNorm per head (64 dims), RoPE on sliding layers, cast to FP8.
    qe = tl.load(row + hq * 64 + 2 * j).to(tl.float32)
    qo = tl.load(row + hq * 64 + 2 * j + 1).to(tl.float32)
    qr = tl.rsqrt(tl.sum(qe * qe + qo * qo, axis=1)[:, None] / 64.0 + EPS)
    qe = qe * qr
    qo = qo * qr
    if ROPE:
        qe_rot = qe * cos - qo * sin
        qo = qe * sin + qo * cos
        qe = qe_rot
    qdst = q_ptr + t * HQ * 64 + hq * 64 + 2 * j
    tl.store(qdst, qe.to(tl.float8e4nv))
    tl.store(qdst + 1, qo.to(tl.float8e4nv))

    # Keys: same norm and RoPE, then written to the FP8 cache row (divided by the cache scale).
    kbase = row + HQ * 64
    ke = tl.load(kbase + hk * 64 + 2 * j).to(tl.float32)
    ko = tl.load(kbase + hk * 64 + 2 * j + 1).to(tl.float32)
    kr = tl.rsqrt(tl.sum(ke * ke + ko * ko, axis=1)[:, None] / 64.0 + EPS)
    ke = ke * kr
    ko = ko * kr
    if ROPE:
        ke_rot = ke * cos - ko * sin
        ko = ke * sin + ko * cos
        ke = ke_rot
    slot = tl.load(loc_ptr + t)
    ks = tl.load(kscale_ptr).to(tl.float32)
    kdst = k_ptr + slot * (HKV * 64) + hk * 64 + 2 * j
    tl.store(kdst, (ke / ks).to(tl.float8e4nv))
    tl.store(kdst + 1, (ko / ks).to(tl.float8e4nv))

    # Values: no norm, divided by the value scale, written to the same cache row.
    vbase = row + (HQ + HKV) * 64
    ve = tl.load(vbase + hk * 64 + 2 * j).to(tl.float32)
    vo = tl.load(vbase + hk * 64 + 2 * j + 1).to(tl.float32)
    vs = tl.load(vscale_ptr).to(tl.float32)
    vdst = v_ptr + slot * (HKV * 64) + hk * 64 + 2 * j
    tl.store(vdst, (ve / vs).to(tl.float8e4nv))
    tl.store(vdst + 1, (vo / vs).to(tl.float8e4nv))


def _inv_freq(theta, d, device):
    """Constant RoPE inverse frequencies, built as SGLang's RotaryEmbedding does (fp32)."""
    key = (theta, d, device)
    if key not in _INV_FREQ:
        _INV_FREQ[key] = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
    return _INV_FREQ[key]


def kernel_fn(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device
    theta = cfg["rope_theta"]
    k_cache, v_cache = data["k_cache"], data["v_cache"]

    q = torch.empty((tokens, hq, d), dtype=FP8, device=device)
    invf = _inv_freq(theta, d, device) if theta else qkv
    _qk_cache_kernel[(tokens,)](
        qkv, data["positions"], data["cache_loc"], invf, q, k_cache, v_cache,
        data["k_scale"], data["v_scale"], qkv.stride(0),
        HQ=hq, HKV=hkv, HALF=d // 2, ROPE=bool(theta), EPS=RMS_EPS,
    )

    scaled = cfg["kv_scale"] != 1.0
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    window = cfg["window"]
    out = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        # The 5.473 query factor is folded into the softmax scale, as SGLang does.
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=RMS_EPS,
                      has_norm=True)
    return o.view(batch, length, hq * d)
