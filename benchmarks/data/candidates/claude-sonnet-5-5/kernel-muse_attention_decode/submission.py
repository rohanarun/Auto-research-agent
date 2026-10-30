# okc: problem=muse_attention_decode
"""SGLang's Muse attention with the pre-attention kernels fused: one Triton launch does the per-head
q/k RMSNorm (+ interleaved RoPE on sliding layers), the FP8 cast + store of k and v into the cache,
and the FP8 cast of q (replacing FusedQKNorm, store_cache and q.to(FP8)). FA3 and rms_norm_gate are
unchanged."""

from typing import TypedDict

import torch
import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate

try:
    from triton.language.extra import libdevice
    _cos, _sin = libdevice.cos, libdevice.sin
except Exception:  # pragma: no cover
    _cos, _sin = tl.cos, tl.sin


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
_INV_FREQ = {}


def _inv_freq(theta, d, device):
    key = (theta, d, str(device))
    f = _INV_FREQ.get(key)
    if f is None:
        f = _INV_FREQ[key] = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
    return f


@triton.jit
def _to_fp8_bytes(x):
    # round through bf16 first, as the baseline's bf16 activations are, then to E4M3
    return x.to(tl.bfloat16).to(tl.float32).to(tl.float8e4nv).to(tl.uint8, bitcast=True)


@triton.jit
def _prep_kernel(qkv_ptr, pos_ptr, loc_ptr, invf_ptr, ks_ptr, vs_ptr, q_ptr, kc_ptr, vc_ptr, stride_qkv,
                 HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, ROPE: tl.constexpr,
                 SCALED: tl.constexpr, EPS: tl.constexpr):
    t = tl.program_id(0)
    H2: tl.constexpr = D // 2
    i = tl.arange(0, H2)
    d = tl.arange(0, D)
    base = qkv_ptr + t.to(tl.int64) * stride_qkv
    if ROPE:
        ang = tl.load(pos_ptr + t).to(tl.float32) * tl.load(invf_ptr + i)
        c = _cos(ang)[None, :]
        s = _sin(ang)[None, :]

    # queries
    hq = tl.arange(0, HQ)[:, None]
    offs = hq * D + 2 * i[None, :]
    e = tl.load(base + offs).to(tl.float32)
    o = tl.load(base + offs + 1).to(tl.float32)
    r = tl.rsqrt((tl.sum(e * e + o * o, axis=1) / D) + EPS)[:, None]
    e = e * r
    o = o * r
    if ROPE:
        e, o = e * c - o * s, e * s + o * c
    qo = q_ptr + t.to(tl.int64) * (HQ * D) + offs
    tl.store(qo, _to_fp8_bytes(e))
    tl.store(qo + 1, _to_fp8_bytes(o))

    # keys
    row = tl.load(loc_ptr + t).to(tl.int64) * (HKV * D)
    hk = tl.arange(0, HKV)[:, None]
    koffs = hk * D + 2 * i[None, :]
    e = tl.load(base + HQ * D + koffs).to(tl.float32)
    o = tl.load(base + HQ * D + koffs + 1).to(tl.float32)
    r = tl.rsqrt((tl.sum(e * e + o * o, axis=1) / D) + EPS)[:, None]
    e = e * r
    o = o * r
    if ROPE:
        e, o = e * c - o * s, e * s + o * c
    if SCALED:
        ksc = tl.load(ks_ptr)
        e = e.to(tl.bfloat16).to(tl.float32) / ksc
        o = o.to(tl.bfloat16).to(tl.float32) / ksc
    tl.store(kc_ptr + row + koffs, _to_fp8_bytes(e))
    tl.store(kc_ptr + row + koffs + 1, _to_fp8_bytes(o))

    # values
    voffs = hk * D + d[None, :]
    v = tl.load(base + (HQ + HKV) * D + voffs).to(tl.float32)
    if SCALED:
        v = v / tl.load(vs_ptr)
    tl.store(vc_ptr + row + voffs, _to_fp8_bytes(v))


def kernel_fn(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device
    rope = bool(cfg["rope_theta"])
    scaled = cfg["kv_scale"] != 1.0
    k_cache, v_cache = data["k_cache"], data["v_cache"]

    q8 = torch.empty((tokens, hq, d), dtype=torch.float8_e4m3fn, device=device)
    invf = _inv_freq(cfg["rope_theta"], d, device) if rope else qkv
    _prep_kernel[(tokens,)](
        qkv, data["positions"], data["cache_loc"], invf, data["k_scale"], data["v_scale"],
        q8.view(torch.uint8), k_cache.view(torch.uint8), v_cache.view(torch.uint8), qkv.stride(0),
        HQ=hq, HKV=hkv, D=d, ROPE=rope, SCALED=scaled, EPS=RMS_EPS, num_warps=2)

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
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    out = out.reshape(tokens, hq * d)
    if out.dtype != torch.bfloat16:
        out = out.to(torch.bfloat16)
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(out, gate, None, head_dim=d, eps=RMS_EPS, has_norm=True)
    return o.view(batch, length, hq * d)
