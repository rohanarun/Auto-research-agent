"""One Muse Spark Open decoder layer on one GPU of an 8xH200 deployment (TP8/EP8), without
the cross-GPU communication.

Input: the FP32 residual stream and the BF16 attention input (the stream after the previous
layer's fused norm). Output: the next layer's attention input and residual stream, the experts
each token routes to, and the new KV cache entries.

The all-reduces after the output projection and the experts are the identity on this GPU's
partial sums. Up to 8 tokens, the unpooler computes only this GPU's 1/8 of its output columns
(rank 0: the first 1024) and the rest stay zero; above 8 tokens its output is complete.

The KV cache is FP8 E4M3 in pages of config["page_size"] tokens. RoPE is the interleaved
(GPT-J) form, applied after the QK-norm.

Semantics follow transformers' MuseSparkTextDecoderLayer and SGLang's MuseSpark1xDecoderLayer
(commit a55ab7d5). FP8 queries follow FA3's FP8 path: the normalized query is cast to E4M3
with the 5.473 factor folded into the softmax scale.
"""

import functools
import math

import torch

from checks import compare, query_precision_bound, relative_l2
from nvfp4 import BLOCK, dequantize, random_weights

# The model's config, per GPU under TP8/EP8.
HIDDEN = 8192
HEAD_DIM = 64
HQ, HKV = 16, 2  # query and key/value heads on this GPU
Q_SIZE, KV_SIZE = HQ * HEAD_DIM, HKV * HEAD_DIM
QKV_SIZE = 2 * Q_SIZE + 2 * KV_SIZE  # q, k, v, then the output gate
SLIDING_WINDOW = 2048
ROPE_THETA = 500_000.0
QK_SCALE_FACTOR = 5.4730064863875
SM_SCALE = HEAD_DIM ** -0.5
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
PAGE = 64  # default cache page size in tokens; scored cases use 1
QUERY_CHUNK = 2048  # queries attended at a time in the reference (long prefill blocks)
FUSED_MIN_QUERIES = 256  # blocks this long use the fused attention path
FP8 = torch.float8_e4m3fn
K_SCALE = V_SCALE = 1.0  # default per-tensor FP8 cache scales; kernels must apply k_scale and v_scale
D_EXPERT = 4096  # the MoE block's width, and each expert's intermediate size
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32  # this GPU holds 32 experts: local_start .. local_start + 31
# local_start is an int32 tensor input, set per input to the 32-expert range its routing sends the
# most pairs to (case_fields).
ROUTER_MULTIPLIER = 0.1767766952966369
ROUTE_SCALE = 1.0
ROUTE_EPS = 1e-15
RESIDUAL_TEMPERATURE = 0.3
RESIDUAL_GATE_EPS = 1e-3
# Up to SLICED_MAX_TOKENS tokens, this GPU computes only its UNPOOL_SLICE columns of the unpooler output.
UNPOOL_SLICE = HIDDEN // 8
SLICED_MAX_TOKENS = 8
WEIGHTS_SEED = 5353

# Output checks: the residual update and the next attention input must each be within a relative
# L2 bound of the reference. The bound is FP8_QUERY_FACTOR times the reference's own error with
# FP8 E4M3 queries, clamped to [FP8_QUERY_FLOOR, REL_L2_TOL].
REL_L2_TOL = 0.15
FP8_QUERY_FACTOR = 2.5
FP8_QUERY_FLOOR = 0.03
CACHE_TOL = 8e-2  # relative error allowed on the new cache entries
# Routing checks: each chosen expert's biased score must be within ROUTE_TOL of the reference's
# 8th best. Each of the reference's local experts that beats its 9th best by more than KEEP_TOL
# must be kept. Routes to local experts, and distinct local experts touched, may fall short of
# the reference's by at most LOCAL_SLACK and DISTINCT_SLACK.
ROUTE_TOL = 5e-2
LOCAL_SLACK = (0.08, 4)  # (fraction of the reference's count, absolute)
DISTINCT_SLACK = (0.15, 2)
KEEP_TOL = (0.03, 0.04)  # (up to KEEP_TOKENS tokens, more)
KEEP_TOKENS = 256

# Input distribution: the residual stream has a fixed mean direction and low-rank shared
# structure (from STREAM_SEED), and the router has a component along the mean, so a few experts
# take most tokens.
STREAM_SEED = 5354
STREAM_MEAN = 0.7
STREAM_SHARED, STREAM_RANK = 0.9, 4
ROUTER_STD, ROUTER_ALIGN, ROUTER_BIAS_STD = 0.045, 0.07, 0.055
DRAFT_SHARE = 0.6  # draft rows of a verify block keep this fraction of the shared structure
MAX_REPEATS = 20


def generate_weights(seed: int, device: str = "cuda", router_seed: int | None = None):
    """The layer's weights from `seed`. With router_seed, the router and its bias come from that
    seed instead."""
    gen = torch.Generator(device=device).manual_seed(seed)

    def dense(n, k, dtype=torch.bfloat16):
        return (torch.randn((n, k), generator=gen, device=device) / math.sqrt(k)).to(dtype)

    def vec(n, std):
        return (std * torch.randn((n,), generator=gen, device=device)).to(torch.bfloat16)

    w13, w13_sf, gs13 = random_weights(2 * D_EXPERT, D_EXPERT, generator=gen, device=device, leading=(LOCAL_EXPERTS,))
    w2, w2_sf, gs2 = random_weights(D_EXPERT, D_EXPERT, generator=gen, device=device, leading=(LOCAL_EXPERTS,))
    # Per-expert output scales vary.
    gs13 = gs13 * torch.exp(torch.randn((LOCAL_EXPERTS,), generator=gen, device=device))
    gs2 = gs2 * torch.exp(torch.randn((LOCAL_EXPERTS,), generator=gen, device=device))
    o, attn_gate, pre_ffn = dense(HIDDEN, Q_SIZE), vec(HIDDEN, 0.3), vec(HIDDEN, 0.1)
    weights = {
        "qkv_gate": dense(QKV_SIZE, HIDDEN),
        "o": o,
        "attn_residual_gate": attn_gate,
        "pre_ffn_norm": pre_ffn,
        **_router(gen, device, pre_ffn),
        "pooler": dense(D_EXPERT, HIDDEN),
        "pre_expert_norm": vec(D_EXPERT, 0.1),
        "w13": w13,
        "w13_sf": w13_sf,
        "w13_gs": gs13,
        "w2": w2,
        "w2_sf": w2_sf,
        "w2_gs": gs2,
        "post_expert_norm": vec(D_EXPERT, 0.1),
        "unpooler": dense(HIDDEN, D_EXPERT),
        "post_ffn_norm": vec(HIDDEN, 0.1),
        "ffn_residual_gate": vec(HIDDEN, 0.3),
        "next_pre_attn_norm": vec(HIDDEN, 0.1),
    }
    if router_seed is not None:
        weights.update(_router(torch.Generator(device=device).manual_seed(router_seed), device, pre_ffn))
    return weights


@functools.cache
def _stream_basis(device: str):
    """The residual stream's mean direction [8192] and shared directions [STREAM_RANK, 8192]."""
    gen = torch.Generator().manual_seed(STREAM_SEED)
    mean = torch.randn((HIDDEN,), generator=gen)
    shared = torch.randn((STREAM_RANK, HIDDEN), generator=gen)
    return mean.to(device), shared.to(device)


def _router(gen, device, pre_ffn):
    """FP32 router rows, each with a component along the stream's mean direction, and the router
    bias. pre_ffn is unused."""
    mean, _ = _stream_basis(str(device))
    w = ROUTER_STD * torch.randn((N_EXPERTS, HIDDEN), generator=gen, device=device)
    align = ROUTER_ALIGN * torch.randn((N_EXPERTS, 1), generator=gen, device=device)
    rows = w + align * mean[None, :] / math.sqrt(HIDDEN)
    bias = ROUTER_BIAS_STD * torch.randn((N_EXPERTS,), generator=gen, device=device)
    return {"router": rows.contiguous(), "router_bias": bias.contiguous()}


# Request lengths by spread: "fixed" (all ctx), "agentic" (uniform in ctx +/- AGENTIC_SPREAD;
# the scored batched cases), "mixed" (log-normal around ctx, up to 4x it).
AGENTIC_SPREAD = 0.075


def _longest(ctx: int, spread: str) -> int:
    """The longest length _lengths can draw."""
    if spread == "fixed":
        return ctx
    if spread == "agentic":
        return math.ceil(ctx * (1 + AGENTIC_SPREAD))
    return 4 * ctx


def _lengths(gen, batch: int, ctx: int, spread: str, device) -> torch.Tensor:
    if spread == "fixed":
        return torch.full((batch,), ctx, dtype=torch.int64, device=device)
    if spread == "agentic":
        u = torch.rand((batch,), generator=gen, device=device)
        return (ctx * (1 + AGENTIC_SPREAD * (2 * u - 1))).long().clamp(1, _longest(ctx, spread))
    # Log-normal around ctx: many short requests, a long tail.
    z = torch.randn((batch,), generator=gen, device=device)
    return (ctx * torch.exp(z - 0.5)).clamp(16, 4 * ctx).long()


def _shortest_share(spread: str) -> float:
    """A lower bound on mean length / ctx, for the speed-of-light bound."""
    return {"fixed": 1.0, "agentic": 1 - AGENTIC_SPREAD}.get(spread, 0.5)


def _quant(x, scale):
    return (x.float() / scale).to(FP8)


def _dequant(x, scale):
    return x.float() * scale


def _rms(x, eps=RMS_EPS):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


CACHE_CHUNK = 1 << 20  # cache rows generated at a time, so large pools stay within memory


# dist="soft": each KV head's keys share a direction u: k = sqrt(c2) u + sqrt(1 - c2) z (unit
# RMS), with c2 drawn per head uniformly from SOFT_C2[layer]. Attention spreads over many keys.
SOFT_C2 = {"sliding": (0.3, 0.75), "full": (0.5, 0.62)}


def _fill_cache(gen, shape, device, scale=K_SCALE, seed=0, dist="normal", layer="full"):
    """FP8 key and value caches of `shape`: N(0, 1) entries, or soft keys for dist="soft"."""
    hkv = shape[-2]
    k_cache = torch.empty(shape, dtype=FP8, device=device)
    v_cache = torch.empty(shape, dtype=FP8, device=device)
    mix = None
    if dist == "soft":
        # A separate generator: the other inputs match the same seed's dist="normal" case.
        soft = torch.Generator(device=device).manual_seed(seed + 0x50F7)
        u = torch.randn((hkv, HEAD_DIM), generator=soft, device=device)
        u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True))
        lo, hi = SOFT_C2[layer]
        c2 = lo + (hi - lo) * torch.rand((hkv, 1), generator=soft, device=device)
        mix = (c2.sqrt() * u, (1 - c2).sqrt())
    for dst, keys in ((k_cache, True), (v_cache, False)):
        flat = dst.view(-1, hkv, HEAD_DIM)
        for at in range(0, flat.shape[0], CACHE_CHUNK):
            n = min(CACHE_CHUNK, flat.shape[0] - at)
            x = torch.randn((n, hkv, HEAD_DIM), generator=gen, device=device)
            if keys and mix is not None:
                x = mix[0] + mix[1] * x
            flat[at:at + n] = _quant(x, scale)
    return k_cache, v_cache


def _fa3_scheduler_metadata(cache_seqlens, cu_seqlens_q, hq, hkv, page, q_len, layer):
    """FA3's tile schedule for a decode step on a full-attention layer
    (sgl_kernel.flash_attn.get_scheduler_metadata), which SGLang builds once per step from the
    batch's lengths and shares across those layers. None on sliding-window layers and verify
    blocks, where SGLang leaves it to each call."""
    if layer != "full" or q_len != 1:
        return None
    try:
        from sgl_kernel.flash_attn import get_scheduler_metadata
    except ImportError:
        return None
    return get_scheduler_metadata(
        batch_size=cache_seqlens.shape[0], max_seqlen_q=1, max_seqlen_k=int(cache_seqlens.max()),
        num_heads=hq, num_heads_k=hkv, headdim=HEAD_DIM, cache_seqlens=cache_seqlens, qkv_dtype=FP8,
        cu_seqlens_q=cu_seqlens_q, page_size=page, causal=True, num_splits=0)


def generate_input(batch: int, ctx: int, layer: str, seed: int, q_len: int = 1, spread: str = "fixed",
                   dist: str = "normal", page: int = PAGE, kv_scale: float = K_SCALE, local_start: int = 0,
                   device: str = "cuda", **_):
    """layer: "sliding" (2,048-token window, RoPE) or "full" (every cached token, no RoPE).
    q_len: 1 for a decode step, 8 for a speculative-decoding verify block, 200-16384 for a
    prefill chunk; causal inside the block.
    dist="outliers": 1% of residual-stream channels are 10x larger.
    dist="soft": soft cached keys (see SOFT_C2)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    tokens = batch * q_len
    seq_lens = _lengths(gen, batch, ctx, spread, device)
    pages_per = (seq_lens + q_len + page - 1) // page  # room for the new tokens
    # Sized for the case, not the lengths drawn, so every repetition has the same shapes
    # (the judge replays one captured CUDA graph on fresh inputs).
    longest = _longest(ctx, spread)
    max_pages = (longest + q_len + page - 1) // page
    total = batch * max_pages
    perm = torch.randperm(total, generator=gen, device=device).to(torch.int32)  # scattered pages
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=device)
    offsets = torch.cumsum(pages_per, 0) - pages_per
    for b in range(batch):
        n, o = int(pages_per[b]), int(offsets[b])
        page_table[b, :n] = perm[o:o + n]
    positions = (seq_lens[:, None] + torch.arange(q_len, device=device)).reshape(-1)
    pos = positions.view(batch, q_len)
    cache_loc = (torch.gather(page_table.long(), 1, pos // page) * page + pos % page).reshape(-1)
    shape = (total, page, HKV, HEAD_DIM)
    # Cached keys are post-norm (unit RMS per head); values are projections, scale ~1.
    k_cache, v_cache = _fill_cache(gen, shape, device, kv_scale, seed, dist, layer)

    # Residual stream: a fixed mean direction, low-rank shared structure, and noise.
    mean, shared = _stream_basis(str(device))
    coef = torch.randn((tokens, STREAM_RANK), generator=gen, device=device)
    # In a verify block, rows after the first (draft tokens) keep DRAFT_SHARE of the shared structure.
    scale = torch.ones((batch, q_len, 1), device=device)
    if 1 < q_len <= 8:
        scale[:, 1:] = DRAFT_SHARE
    scale = scale.reshape(tokens, 1)
    stream = (scale * (STREAM_MEAN * mean + math.sqrt(STREAM_SHARED / STREAM_RANK) * coef @ shared)
              + (1 - STREAM_SHARED * scale * scale).sqrt()
              * torch.randn((tokens, HIDDEN), generator=gen, device=device))
    if dist == "outliers":
        idx = torch.randperm(HIDDEN, generator=gen, device=device)[: HIDDEN // 100]
        stream[:, idx] *= 10
    cache_seqlens = (seq_lens + q_len).to(torch.int32)
    cu_seqlens_q = torch.arange(0, tokens + 1, q_len, dtype=torch.int32, device=device)
    return {
        "stream": stream,
        # The stream, RMS-normalized, in BF16.
        "attn_input": _rms(stream).to(torch.bfloat16),
        # Step metadata: each new token's position and flat cache row, per-request cache
        # lengths and query offsets, and on full-attention decode steps FA3's tile schedule.
        "positions": positions,
        "cache_loc": cache_loc,
        "cache_seqlens": cache_seqlens,
        "cu_seqlens_q": cu_seqlens_q,
        "scheduler_metadata": _fa3_scheduler_metadata(cache_seqlens, cu_seqlens_q, HQ, HKV, page, q_len, layer),
        "k_cache": k_cache,
        "v_cache": v_cache,
        "k_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),
        "v_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),
        "page_table": page_table,
        "seq_lens": seq_lens.to(torch.int32),
        "local_start": torch.tensor([local_start], dtype=torch.int32, device=device),
        "config": {
            "batch": batch,
            "q_len": q_len,
            "tokens": tokens,
            "hidden": HIDDEN,
            "num_q_heads": HQ,
            "num_kv_heads": HKV,
            "head_dim": HEAD_DIM,
            "page_size": page,
            "window": SLIDING_WINDOW if layer == "sliding" else 0,
            "rope_theta": ROPE_THETA if layer == "sliding" else 0.0,
            "qk_scale_factor": QK_SCALE_FACTOR,
            "sm_scale": SM_SCALE,
            "kv_scale": kv_scale,
            "n_experts": N_EXPERTS,
            "top_k": TOP_K,
            "local_experts": LOCAL_EXPERTS,
        },
    }


def _rope(x, pos, theta):
    """Interleaved (GPT-J) RoPE: pairs (2i, 2i+1) rotate by pos * theta^(-2i/64).
    x [T, heads, 64], pos [T]."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, device=x.device).float() / HEAD_DIM))
    ang = pos.float()[:, None] * inv_freq  # [T, 32]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)


def residual_coefficients(gate):
    """MuseSpark1xNormPreservingResidualConnection: new = alpha * stream + beta * branch."""
    w = gate.float() / RESIDUAL_TEMPERATURE
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=RESIDUAL_GATE_EPS))
    return alpha, beta


def new_slots(data) -> torch.Tensor:
    """[B, L] flat cache rows where the block's new tokens go."""
    cfg = data["config"]
    pos = data["seq_lens"].long()[:, None] + torch.arange(cfg["q_len"], device=data["seq_lens"].device)
    size = cfg["page_size"]
    page = torch.gather(data["page_table"].long(), 1, pos // size)
    return page * size + pos % size


def _attention(data, x, weights, query_dtype=torch.bfloat16):
    """Attention input [T, 8192] -> gated attention output [T, 1024] and the new cache
    entries (FP8 [B, L, Hkv, 64])."""
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    qkv = x @ weights["qkv_gate"].float().T
    q, k, v, gate = qkv.split([Q_SIZE, KV_SIZE, KV_SIZE, Q_SIZE], dim=-1)
    pos = (data["seq_lens"].long()[:, None] + torch.arange(length, device=x.device)).reshape(-1)
    q = _rms(q.view(-1, HQ, HEAD_DIM))
    k = _rms(k.view(-1, HKV, HEAD_DIM))
    if cfg["rope_theta"]:
        q, k = _rope(q, pos, cfg["rope_theta"]), _rope(k, pos, cfg["rope_theta"])
    # The reference rounds queries to BF16; kernels may use BF16 or FP8 queries.
    if query_dtype == torch.bfloat16:
        q = (q * cfg["qk_scale_factor"]).to(torch.bfloat16).float()
    else:  # FP8 on the normalized query, qk_scale_factor applied after
        q = q.to(query_dtype).float() * cfg["qk_scale_factor"]
    q = q.view(batch, length, HQ, HEAD_DIM)
    k_new = _quant(k, data["k_scale"]).view(batch, length, HKV, HEAD_DIM)
    v_new = _quant(v.view(-1, HKV, HEAD_DIM), data["v_scale"]).view(batch, length, HKV, HEAD_DIM)
    k_flat = data["k_cache"].view(-1, HKV, HEAD_DIM)
    v_flat = data["v_cache"].view(-1, HKV, HEAD_DIM)
    group, window = HQ // HKV, cfg["window"]
    out = torch.empty((batch, length, HQ, HEAD_DIM), device=x.device)
    steps = torch.arange(length, device=x.device)
    for b in range(batch):
        seq = int(data["seq_lens"][b])
        first = max(0, seq - window + 1) if window else 0  # the oldest key any block token sees
        past = torch.arange(first, seq, device=x.device)
        size = cfg["page_size"]
        rows = data["page_table"][b].long()[past // size] * size + past % size
        keys = torch.cat([_dequant(k_flat[rows], data["k_scale"]), _dequant(k_new[b], data["k_scale"])])
        vals = torch.cat([_dequant(v_flat[rows], data["v_scale"]), _dequant(v_new[b], data["v_scale"])])
        key_pos = torch.cat([past, seq + steps])
        fused = length >= FUSED_MIN_QUERIES
        if fused:  # each query head reads its KV head's keys and values: [HQ, S, 64]
            keys_h = keys.repeat_interleave(group, dim=1).transpose(0, 1)
            vals_h = vals.repeat_interleave(group, dim=1).transpose(0, 1)
        # Queries in chunks, so long prefill blocks fit in memory.
        for c in range(0, length, QUERY_CHUNK):
            j = slice(c, min(c + QUERY_CHUNK, length))
            query_pos = seq + steps[j]
            visible = key_pos[None, :] <= query_pos[:, None]
            if window:
                visible &= key_pos[None, :] > query_pos[:, None] - window
            if fused:
                out[b, j] = _softmax_attention(q[b, j].transpose(0, 1), keys_h, vals_h, visible,
                                               cfg["sm_scale"]).transpose(0, 1)
                continue
            n = query_pos.numel()
            att = torch.einsum("lkgd,tkd->lkgt", q[b, j].reshape(n, HKV, group, HEAD_DIM), keys) * cfg["sm_scale"]
            att = att.masked_fill(~visible[:, None, None, :], float("-inf")).softmax(-1)
            out[b, j] = torch.einsum("lkgt,tkd->lkgd", att, vals).reshape(n, HQ, HEAD_DIM)
    out = _rms(out).reshape(-1, Q_SIZE) * torch.sigmoid(gate)
    return out, k_new, v_new


def _softmax_attention(q, k, v, visible, scale):
    """softmax(q k^T * scale, masked to `visible`) v in FP32: q [H, n, 64], k and v [H, S, 64],
    visible [n, S] -> [H, n, 64]. Same result as the einsum path."""
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=visible[None], scale=scale)


def route(h, weights):
    """Router scores for the FFN input h [T, 8192]: (unbiased sigmoid scores, the biased
    scores that choose the top 8), both [T, 256] FP32."""
    logits = h.double() @ weights["router"].double().T
    scores = torch.sigmoid(logits * ROUTER_MULTIPLIER).float()
    return scores, scores + weights["router_bias"].float()


def post_expert_norm(y, w):
    """Scale by (1 + w) first, then RMSNorm (eps 1e-8)."""
    y = y.float() * (1.0 + w.float())
    return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)


def _experts(x, ids, topk_weights, weights):
    """The local experts' weighted, post-normed outputs, summed per token: [T, 4096]."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in torch.unique(ids).tolist():
        if not 0 <= e < LOCAL_EXPERTS:
            continue  # another GPU's expert: its contribution arrives in the all-reduce
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]).T
        inter = torch.nn.functional.silu(h[:, :D_EXPERT]) * h[:, D_EXPERT:]
        y = inter @ dequantize(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]).T
        out.index_add_(0, tok, post_expert_norm(y, weights["post_expert_norm"]) * topk_weights[tok, slot, None])
    return out


def ref_front(data, weights, query_dtype=torch.bfloat16):
    """Everything up to the router: the attention, its residual boundary and the router's
    scores (the routing itself is checked against these)."""
    o, k_new, v_new = _attention(data, data["attn_input"].float(), weights, query_dtype)
    a = o @ weights["o"].float().T  # the all-reduce: identity
    alpha1, beta1 = residual_coefficients(weights["attn_residual_gate"])
    stream1 = alpha1 * data["stream"].float() + beta1 * _rms(a, POST_NORM_EPS)
    h = _rms(stream1) * (1.0 + weights["pre_ffn_norm"].float())
    scores, biased = route(h, weights)
    return {"stream1": stream1, "h": h, "scores": scores, "biased": biased, "k_new": k_new, "v_new": v_new,
            "local_start": int(data["local_start"])}


def ref_back(front, weights, ids):
    """The MoE block and the FFN boundary, with the given experts per token."""
    picked = front["scores"].gather(1, ids.long())
    topk_weights = picked / (picked.sum(-1, keepdim=True) + ROUTE_EPS) * ROUTE_SCALE
    p = front["h"] @ weights["pooler"].float().T
    p = _rms(p) * (1.0 + weights["pre_expert_norm"].float())
    local = (ids.long() - front["local_start"]) % N_EXPERTS  # this GPU's experts are 0-31 here
    moe = _experts(p, local, topk_weights, weights)  # the all-reduce: identity
    if len(moe) <= SLICED_MAX_TOKENS:
        f = torch.zeros_like(front["stream1"])
        f[:, :UNPOOL_SLICE] = moe @ weights["unpooler"][:UNPOOL_SLICE].float().T  # the all-gather: identity
    else:
        f = moe @ weights["unpooler"].float().T
    f = f * (1.0 + weights["post_ffn_norm"].float())
    alpha2, beta2 = residual_coefficients(weights["ffn_residual_gate"])
    stream2 = alpha2 * front["stream1"] + beta2 * _rms(f, POST_NORM_EPS)
    hidden = _rms(stream2) * (1.0 + weights["next_pre_attn_norm"].float())
    return hidden.to(torch.bfloat16), stream2


def case_fields(case: dict, weights) -> dict:
    """Run by the judge for each input: local_start, the 32-expert range this input's routing
    sends the most pairs to."""
    data = generate_input(**{**case, "local_start": 0})
    ids = ref_front(data, weights)["biased"].topk(TOP_K, dim=-1).indices
    pairs = torch.bincount((ids // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    return {"local_start": LOCAL_EXPERTS * int(pairs.argmax())}


# The judge picks each run's router seed so the timed GPU gets its usual amount of expert work.
LOAD_TOL = 0.015  # relative tolerance on the target load
LOAD_ATTEMPTS = 5000  # router seeds tried
SAMPLE_TOKENS, ROUTE_SAMPLES = 256, 8  # cases this small average the load over this many inputs


def _busiest_load(h, router) -> tuple[int, int]:
    """(routes, distinct experts) on the range of LOCAL_EXPERTS the routing sends the most to."""
    ids = route(h, router)[1].topk(TOP_K, dim=-1).indices
    pairs = torch.bincount((ids // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    return _local_load(ids, LOCAL_EXPERTS * int(pairs.argmax()))


def _busiest_routes(h, router) -> int:
    return _busiest_load(h, router)[0]


def _mean_busiest_load(h, router, inputs: int):
    """_busiest_load of each of `inputs` equal-length inputs stacked in h, averaged: a [2]
    tensor (routes, distinct experts)."""
    ids = route(h, router)[1].topk(TOP_K, dim=-1).indices.view(inputs, -1)
    ranges = N_EXPERTS // LOCAL_EXPERTS
    pairs = torch.zeros((inputs, ranges), device=h.device)
    pairs.scatter_add_(1, ids // LOCAL_EXPERTS, torch.ones_like(ids, dtype=torch.float))
    busiest = pairs.argmax(1)
    used = torch.zeros((inputs, N_EXPERTS), device=h.device).scatter_(1, ids, 1.0)
    used = used.view(inputs, ranges, LOCAL_EXPERTS)
    distinct = used[torch.arange(inputs, device=h.device), busiest].sum(1)
    return torch.stack([pairs.max(1).values.mean(), distinct.mean()])


def weights_fields(weights, cases: list[dict], seed: int) -> dict:
    """Run by the judge once per timed run: a router_seed whose routing gives each load (the
    timed cases sharing a `load`) its target MoE load, within LOAD_TOL for every load (else the
    draw of LOAD_ATTEMPTS whose worst load is closest). {} without targets."""
    timed = [c for c in cases if c.get("router_load")]
    if not timed:
        return {}
    keys = [k for k in ("router_load", "router_distinct") if all(c.get(k) for c in timed)]
    cols = [("router_load", "router_distinct").index(k) for k in keys]
    loads: dict[str, list[int]] = {}
    for i, c in enumerate(timed):
        loads.setdefault(str(c.get("load", f"case {i}")), []).append(i)
    # Each load's value is its cases' weighted mean: a [loads, cases] averaging matrix.
    w = torch.tensor([float(c.get("weight", 1)) for c in timed], dtype=torch.float64)
    avg = torch.zeros((len(loads), len(timed)), dtype=torch.float64)
    for g, members in enumerate(loads.values()):
        avg[g, members] = w[members] / w[members].sum()
    targets = avg @ torch.tensor([[float(c[k]) for k in keys] for c in timed], dtype=torch.float64)
    hs = []
    with torch.no_grad():
        for c in timed:
            n = ROUTE_SAMPLES if c.get("batch", 1) * c.get("q_len", 1) <= SAMPLE_TOKENS else 1
            seeds = [c["seed"]] + [(c["seed"] + j * 0x2545F4914F6CDD1D) % (2**63 - 1) for j in range(1, n)]
            hs.append((torch.cat([ref_front(generate_input(**{**c, "seed": s, "local_start": 0}), weights)["h"]
                                  for s in seeds]), n))
        device = hs[0][0].device
        avg, targets = avg.to(device), targets.to(device)
        best = (math.inf, 0)
        for i in range(1, LOAD_ATTEMPTS + 1):
            router_seed = (seed + i * 0x9E3779B97F4A7C15) % (2**63 - 1)
            router = _router(torch.Generator(device=device).manual_seed(router_seed), device, weights["pre_ffn_norm"])
            per_case = torch.stack([_mean_busiest_load(h, router, n) for h, n in hs])[:, cols].double()
            err = float(((avg @ per_case) / targets - 1).abs().max())
            best = min(best, (err, router_seed))
            if best[0] <= LOAD_TOL:
                break
    return {"router_seed": best[1]}


def ref_kernel(data, weights, topk_ids=None):
    """Returns (next layer's attention input bf16 [T, 8192], residual stream fp32 [T, 8192],
    experts int32 [T, 8], new cache keys and values FP8 [B, L, Hkv, 64]). With topk_ids,
    the MoE uses those experts instead of the reference's own choice."""
    front = ref_front(data, weights)
    ids = front["biased"].topk(TOP_K, dim=-1).indices.to(torch.int32) if topk_ids is None else topk_ids
    hidden, stream = ref_back(front, weights, ids)
    return hidden, stream, ids, front["k_new"], front["v_new"]


def _shapes(cfg):
    t, b, length = cfg["tokens"], cfg["batch"], cfg["q_len"]
    return [((t, HIDDEN), torch.bfloat16), ((t, HIDDEN), torch.float32), ((t, TOP_K), torch.int32),
            ((b, length, HKV, HEAD_DIM), FP8), ((b, length, HKV, HEAD_DIM), FP8)]


def collect(data, output) -> torch.Tensor:
    """Runs in the submission's process after a step: its three outputs, then the cache
    entries the step wrote, as raw bytes (see unpack)."""
    cfg = data["config"]
    want = _shapes(cfg)[:3]
    if type(output) not in (tuple, list) or len(output) != 3 or any(
            type(t) is not torch.Tensor or t.dtype != dtype or tuple(t.shape) != shape
            for t, (shape, dtype) in zip(output, want)):
        raise ValueError("expected (hidden bfloat16 [T, 8192], stream float32 [T, 8192], topk_ids int32 [T, 8])")
    slots = new_slots(data)
    k = data["k_cache"].view(-1, HKV, HEAD_DIM)[slots]
    v = data["v_cache"].view(-1, HKV, HEAD_DIM)[slots]
    return torch.cat([p.contiguous().view(torch.uint8).reshape(-1) for p in (*output, k, v)])


def unpack(raw: torch.Tensor, cfg: dict):
    shapes = _shapes(cfg)
    sizes = [math.prod(shape) * dtype.itemsize for shape, dtype in shapes]
    if raw.dtype != torch.uint8 or raw.numel() != sum(sizes):
        return None
    out, at = [], 0
    for (shape, dtype), n in zip(shapes, sizes):
        out.append(raw[at:at + n].view(dtype).view(shape))
        at += n
    return out


def _local_load(ids, local_start=0):
    """Routes to this GPU's experts (local_start .. local_start + 31), and how many distinct
    ones they touch."""
    local = ids[(ids - local_start) % N_EXPERTS < LOCAL_EXPERTS]
    return int(local.numel()), int(torch.unique(local).numel())


def check_routing(ids, biased, local_start=0):
    """None if ids [T, 8] are an acceptable top-8 under the reference's biased scores."""
    if ids.min() < 0 or ids.max() >= N_EXPERTS:
        return "expert id out of range"
    ranked = ids.long().sort(-1).values
    if (ranked[:, 1:] == ranked[:, :-1]).any():
        return "a token routed to the same expert twice"
    best = biased.topk(TOP_K, dim=-1)
    gap = best.values[:, -1:] - biased.gather(1, ids.long())
    if gap.max() > ROUTE_TOL:
        t = int(gap.max(-1).values.argmax())
        return (f"token {t} routed to an expert whose biased score is {float(gap[t].max()):.2e} below its 8th best"
                f" (> {ROUTE_TOL:g})")
    ninth = biased.topk(TOP_K + 1, dim=-1).values[:, -1:]
    keep_tol = KEEP_TOL[0] if len(ids) <= KEEP_TOKENS else KEEP_TOL[1]
    kept = (best.indices[:, :, None] == ids.long()[:, None, :]).any(-1)
    local = (best.indices - local_start) % N_EXPERTS < LOCAL_EXPERTS
    lost = local & ~kept & (best.values - ninth > keep_tol)
    if lost.any():
        t, slot = (int(i) for i in lost.nonzero()[0])
        return (f"token {t} dropped expert {int(best.indices[t, slot])}, on this GPU, whose biased score is"
                f" {float(best.values[t, slot] - ninth[t]):.2e} above its 9th best (> {keep_tol:g})")
    routes, distinct = _local_load(ids.long(), local_start)
    ref_routes, ref_distinct = _local_load(best.indices, local_start)
    if routes < ref_routes * (1 - LOCAL_SLACK[0]) - LOCAL_SLACK[1]:
        return f"{routes} routes to this GPU's experts where the reference has {ref_routes}"
    if distinct < ref_distinct * (1 - DISTINCT_SLACK[0]) - DISTINCT_SLACK[1]:
        return f"{distinct} of this GPU's experts touched where the reference touches {ref_distinct}"
    return None


def check_implementation(data, raw, weights):
    parts = unpack(raw, data["config"])
    if parts is None:
        return False, "malformed output"
    hidden, stream, ids, k, v = parts
    front = ref_front(data, weights)
    error = check_routing(ids, front["biased"], front["local_start"])
    if error:
        return False, f"routing: {error}"
    ref_hidden, ref_stream = ref_back(front, weights, ids)
    # The residual stream mostly carries its input through; compare what the layer added.
    alpha1, _ = residual_coefficients(weights["attn_residual_gate"])
    alpha2, _ = residual_coefficients(weights["ffn_residual_gate"])
    carried = alpha1 * alpha2 * data["stream"].float()
    # The bounds scale with how far E4M3 queries move the reference (same experts).
    e4m3_hidden, e4m3_stream = ref_back(ref_front(data, weights, torch.float8_e4m3fn), weights, ids)
    bound = lambda err: query_precision_bound(err, FP8_QUERY_FACTOR, FP8_QUERY_FLOOR, REL_L2_TOL)  # noqa: E731
    tol = bound(relative_l2(e4m3_stream - carried, ref_stream - carried))
    hidden_tol = bound(relative_l2(e4m3_hidden, ref_hidden))
    ok, msg = compare(stream - carried, ref_stream - carried, tol)
    msg = f"{msg} [bound {tol:.3f}]"
    if not ok:
        return False, f"residual update: {msg}"
    ok, hidden_msg = compare(hidden, ref_hidden, hidden_tol)
    hidden_msg = f"{hidden_msg} [bound {hidden_tol:.3f}]"
    if not ok:
        return False, f"next attention input: {hidden_msg}"
    for name, got, want, scale in (("key", k, front["k_new"], data["k_scale"]),
                                   ("value", v, front["v_new"], data["v_scale"])):
        ok, kv_msg = compare(_dequant(got, scale), _dequant(want, scale), CACHE_TOL)
        if not ok:
            return False, f"new {name} cache entries: {kv_msg}"
    return True, f"update {msg}; attention input {hidden_msg}"


# A case's router_load and router_distinct (task.yml) are the mean routes and distinct experts on
# this GPU's range under the run's router. Each repetition's routing varies around them, so the
# bound counts this share of each.
ROUTING_SHARE = 0.75


def speed_of_light_ns(batch: int, ctx: int, layer: str, hw: dict, q_len: int = 1, spread: str = "fixed",
                      router_load: float = 0.0, router_distinct: float = 0.0, **_) -> float:
    tokens = batch * q_len
    unpooled = UNPOOL_SLICE if tokens <= SLICED_MAX_TOKENS else HIDDEN
    dense = QKV_SIZE * HIDDEN + HIDDEN * Q_SIZE + D_EXPERT * HIDDEN + unpooled * D_EXPERT
    # Lower bounds on the local experts touched and the routes landing on this GPU. It is the
    # busiest, so it has at least the mean routes per GPU (one per token).
    touched = max(1 if tokens < 16 else 2 if tokens < 128 else 6, ROUTING_SHARE * router_distinct)
    local_routes = max(tokens, ROUTING_SHARE * router_load)
    per_expert = 3 * D_EXPERT * D_EXPERT
    attended = min(ctx, SLIDING_WINDOW) if layer == "sliding" else ctx
    kv_tokens = batch * attended * _shortest_share(spread)  # uneven lengths: stay a lower bound
    bytes_moved = (2 * dense + 4 * N_EXPERTS * HIDDEN + touched * per_expert * (0.5 + 1 / BLOCK)
                   + kv_tokens * KV_SIZE * 2 + tokens * HIDDEN * (4 + 2) * 2)
    flops = (2.0 * tokens * (dense + N_EXPERTS * HIDDEN) + 2.0 * local_routes * per_expert
             + 4.0 * kv_tokens * q_len * Q_SIZE)
    return 1e9 * max(bytes_moved / hw["hbm_bytes_per_s"], flops / hw["bf16_dense_flops_per_s"])
