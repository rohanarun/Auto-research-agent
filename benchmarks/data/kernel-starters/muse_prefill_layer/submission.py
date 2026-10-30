# okc: problem=muse_prefill_layer
"""Starter: one decoder layer in plain PyTorch, correct but slow. Each request attends over
every position its page table can hold, with the rest masked out, and every local expert
runs on every token, kept by the token's routing weight (zero where it wasn't routed
there). The NVFP4 expert weights are expanded to BF16 on every call.

Implement kernel_fn, and prepare to repack the weights (called once, untimed)."""

from typing import TypedDict

import torch

from nvfp4 import dequantize


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it); 200-16384 on muse_prefill_layer
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32: this GPU computes experts local_start .. local_start + 31


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream (the previous layer's boundary wrote it)
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T], each new token's position (seq_lens[b] + i)
    cache_loc: torch.Tensor   # int64 [T], each new token's flat cache row (page * page_size + offset)
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule (get_scheduler_metadata) on
                                             # full-attention decode steps, else None
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale: 1 in the scored cases, 1/64 in some tests
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    local_start: torch.Tensor # int32 [1], this GPU's first expert (a multiple of 32; its weights are w13[0], ...)
    config: Config


HIDDEN = 8192
D_EXPERT = 4096
LOCAL_EXPERTS = 32
UNPOOL_SLICE = 1024
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8


def prepare(weights):
    """Called once, untimed, before any case."""
    return weights


def rms(x, eps=RMS_EPS):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def gain(w):
    """Muse's norm weights are zero-centered: the scale is 1 + w."""
    return 1.0 + w.float()


def rope(x, pos, theta):
    """Interleaved: pairs (2i, 2i+1). x [T, heads, d], pos [T]."""
    d = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device).float() / d))
    ang = pos.float()[:, None, None] * inv_freq
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even * ang.cos() - odd * ang.sin(), even * ang.sin() + odd * ang.cos()], -1).flatten(-2)


def residual_coefficients(gate):
    w = gate.float() / 0.3
    beta = torch.sigmoid(w)
    return torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3)), beta


def post_expert_norm(y, w):
    """Scale by (1 + w), round to BF16, then RMSNorm."""
    return rms((y.float() * gain(w)).to(torch.bfloat16), POST_NORM_EPS)


def round_queries(q, factor):
    """The precision queries attend in: BF16 here (FA3 with an FP8 cache uses FP8)."""
    return (q * factor).to(torch.bfloat16)


def unpool(m, w):
    """Up to 8 tokens, this GPU's slice of the unpooler, then the all-gather as the
    identity (zeros in the other columns); above that, the whole unpooler."""
    if len(m) > 8:
        return (m @ w.T).float()
    return gather_unpooled(m @ w[:UNPOOL_SLICE].T)


def gather_unpooled(u):
    """The all-gather, as the identity: this GPU's slice in its columns, zeros elsewhere."""
    return torch.nn.functional.pad(u.float(), (0, HIDDEN - u.shape[1]))


def attention(data, x, weights):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    qkv = x @ weights["qkv_gate"].T
    q, k, v, g = qkv.split([hq * d, hkv * d, hkv * d, hq * d], dim=-1)
    pos = data["positions"]
    q = rms(q.view(-1, hq, d))
    k = rms(k.view(-1, hkv, d))
    if cfg["rope_theta"]:
        q, k = rope(q, pos, cfg["rope_theta"]), rope(k, pos, cfg["rope_theta"])
    q = round_queries(q, cfg["qk_scale_factor"])

    table = data["page_table"].long()
    slots = data["cache_loc"]
    k_cache = data["k_cache"].view(-1, hkv, d)
    v_cache = data["v_cache"].view(-1, hkv, d)
    k_scale, v_scale = data["k_scale"], data["v_scale"]
    k_cache[slots] = (k / k_scale).to(torch.float8_e4m3fn)
    v_cache[slots] = (v.view(-1, hkv, d).float() / v_scale).to(torch.float8_e4m3fn)

    # Every position a request's page table can hold; unused pages (-1) read page 0 and
    # are masked out with everything else past the token's own position.
    t = torch.arange(table.shape[1] * page, device=x.device)
    pos = pos.view(batch, length)
    q = q.view(batch, length, hkv, hq // hkv, d).float()
    out = torch.empty((batch, length, hq, d), device=x.device)
    for b in range(batch):
        rows = table[b].clamp(min=0)[t // page] * page + t % page
        keys = k_cache[rows].float() * k_scale  # [S, Hkv, d]
        vals = v_cache[rows].float() * v_scale
        for c in range(0, length, 512):  # queries in chunks, so long prefill blocks fit in memory
            j = slice(c, min(c + 512, length))
            visible = t[None, :] <= pos[b, j][:, None]  # [L, S]: causal, including the block's own tokens
            if cfg["window"]:
                visible &= t[None, :] > pos[b, j][:, None] - cfg["window"]
            att = torch.einsum("lkgd,skd->lkgs", q[b, j], keys) * cfg["sm_scale"]
            att = att.masked_fill(~visible[:, None, None, :], float("-inf")).softmax(-1)
            out[b, j] = torch.einsum("lkgs,skd->lkgd", att, vals).reshape(-1, hq, d)
    return (rms(out).reshape(-1, hq * d) * torch.sigmoid(g.float())).to(torch.bfloat16)


def route(h, weights, top_k):
    """Sigmoid scores; the bias only chooses; the chosen scores are renormalized."""
    scores = torch.sigmoid((h.float() @ weights["router"].T) * 0.1767766952966369)
    ids = (scores + weights["router_bias"]).topk(top_k, dim=-1).indices
    picked = scores.gather(1, ids)
    return ids, picked / (picked.sum(-1, keepdim=True) + 1e-15)


def experts(x, ids, topk_weights, weights):
    """Every local expert on every token, each token's output kept by its routing weight
    for that expert (zero unless it is among the token's top 8)."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in range(LOCAL_EXPERTS):
        g = (topk_weights * (ids == e)).sum(-1)
        w13 = dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]).to(torch.bfloat16)
        h = x @ w13.T
        inter = torch.nn.functional.silu(h[:, :D_EXPERT]) * h[:, D_EXPERT:]
        w2 = dequantize(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]).to(torch.bfloat16)
        out += g[:, None] * post_expert_norm(inter @ w2.T, weights["post_expert_norm"])
    return out.to(torch.bfloat16)


def kernel_fn(data: Inputs, weights: Weights):
    """Returns (next layer's attention input bfloat16 [T, 8192], residual stream float32
    [T, 8192], experts per token int32 [T, 8])."""
    cfg = data["config"]
    a = attention(data, data["attn_input"], weights) @ weights["o"].T
    alpha, beta = residual_coefficients(weights["attn_residual_gate"])
    stream = alpha * data["stream"] + beta * rms(a, POST_NORM_EPS)
    h = (rms(stream) * gain(weights["pre_ffn_norm"])).to(torch.bfloat16)
    ids, topk_weights = route(h, weights, cfg["top_k"])
    p = (rms(h @ weights["pooler"].T) * gain(weights["pre_expert_norm"])).to(torch.bfloat16)
    local = (ids - data["local_start"]) % 256  # this GPU's experts as 0-31
    f = unpool(experts(p, local, topk_weights, weights), weights["unpooler"]) * gain(weights["post_ffn_norm"])
    alpha, beta = residual_coefficients(weights["ffn_residual_gate"])
    stream = alpha * stream + beta * rms(f, POST_NORM_EPS)
    hidden = (rms(stream) * gain(weights["next_pre_attn_norm"])).to(torch.bfloat16)
    return hidden, stream, ids.to(torch.int32)
