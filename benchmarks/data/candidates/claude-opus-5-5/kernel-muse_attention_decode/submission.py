# okc: problem=muse_attention_decode
"""SGLang's FA3 attention step with the prologue fused: one Triton kernel does the per-head
RMSNorm of q and k, interleaved RoPE on sliding layers, the FP8 E4M3 cast of q, and the FP8
casts and cache stores of the new keys and values. That replaces FusedQKNorm, the k/v casts,
store_cache and q.to(fp8) (five launches) with one. FA3 and rms_norm_gate are as in SGLang."""

from typing import TypedDict

import torch
import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate


class Config(TypedDict):
    batch: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float


class Inputs(TypedDict):
    qkv: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    config: Config


RMS_EPS = 1e-5
FP8 = torch.float8_e4m3fn
_ROPE = {}
ROPE_POSITIONS = 1 << 21


def _cos_sin(theta, d, device):
    """[2^21, d] float32 cos | sin, as SGLang's RotaryEmbedding builds it, cached per theta."""
    key = (theta, d, str(device))
    cache = _ROPE.get(key)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = _ROPE[key] = torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()
    return cache


@triton.jit
def _prologue(qkv_ptr, qkv_stride, pos_ptr, loc_ptr, cs_ptr, ks_ptr, vs_ptr, q_out, k_cache, v_cache,
              HQ: tl.constexpr, HKV: tl.constexpr, HQP: tl.constexpr, HKVP: tl.constexpr, D: tl.constexpr,
              ROPE: tl.constexpr, SCALED: tl.constexpr, EPS: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    HALF: tl.constexpr = D // 2
    i = tl.arange(0, HALF)
    row = qkv_ptr + t * qkv_stride
    loc = tl.load(loc_ptr + t).to(tl.int64)
    if ROPE:
        p = tl.load(pos_ptr + t).to(tl.int64)
        c = tl.load(cs_ptr + p * D + i)[None, :]
        s = tl.load(cs_ptr + p * D + HALF + i)[None, :]

    # Queries: RMSNorm (the 5.473 factor is folded into the softmax scale), RoPE, FP8.
    hq = tl.arange(0, HQP)
    mq = (hq < HQ)[:, None]
    off = hq[:, None] * D + 2 * i[None, :]
    qe = tl.load(row + off, mask=mq, other=0.).to(tl.float32)
    qo = tl.load(row + off + 1, mask=mq, other=0.).to(tl.float32)
    r = tl.rsqrt((tl.sum(qe * qe, 1) + tl.sum(qo * qo, 1)) / D + EPS)[:, None]
    qe = qe * r
    qo = qo * r
    if ROPE:
        qe2 = qe * c - qo * s
        qo = qe * s + qo * c
        qe = qe2
    qbase = q_out + t * (HQ * D)
    tl.store(qbase + off, qe.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=mq)
    tl.store(qbase + off + 1, qo.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=mq)

    # New keys: RMSNorm, RoPE, divided by the cache scale, FP8, stored at the token's row.
    hk = tl.arange(0, HKVP)
    mk = (hk < HKV)[:, None]
    koff = HQ * D + hk[:, None] * D + 2 * i[None, :]
    ke = tl.load(row + koff, mask=mk, other=0.).to(tl.float32)
    ko = tl.load(row + koff + 1, mask=mk, other=0.).to(tl.float32)
    r = tl.rsqrt((tl.sum(ke * ke, 1) + tl.sum(ko * ko, 1)) / D + EPS)[:, None]
    ke = ke * r
    ko = ko * r
    if ROPE:
        ke2 = ke * c - ko * s
        ko = ke * s + ko * c
        ke = ke2
    if SCALED:
        ks = tl.load(ks_ptr)
        ke = ke / ks
        ko = ko / ks
    cbase = loc * (HKV * D)
    coff = hk[:, None] * D + 2 * i[None, :]
    tl.store(k_cache + cbase + coff, ke.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=mk)
    tl.store(k_cache + cbase + coff + 1, ko.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=mk)

    # New values: divided by the cache scale, FP8, stored.
    j = tl.arange(0, D)
    voff = hk[:, None] * D + j[None, :]
    v = tl.load(row + (HQ + HKV) * D + voff, mask=mk, other=0.).to(tl.float32)
    if SCALED:
        v = v / tl.load(vs_ptr)
    tl.store(v_cache + cbase + voff, v.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=mk)


def kernel_fn(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device

    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    rope = bool(cfg["rope_theta"])
    cs = _cos_sin(cfg["rope_theta"], d, device) if rope else qkv
    q8 = torch.empty((tokens, hq, d), dtype=FP8, device=device)
    _prologue[(tokens,)](
        qkv, qkv.stride(0), data["positions"], data["cache_loc"], cs, data["k_scale"], data["v_scale"],
        q8.view(torch.uint8), k_cache.view(torch.uint8), v_cache.view(torch.uint8),
        HQ=hq, HKV=hkv, HQP=triton.next_power_of_2(hq), HKVP=triton.next_power_of_2(hkv), D=d,
        ROPE=rope, SCALED=scaled, EPS=RMS_EPS, num_warps=2)

    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q8,
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
