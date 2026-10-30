# okc: problem=muse_moe_w4a16
"""Starter: the MoE experts in plain PyTorch, correct but slow. Every local expert runs on
every token, kept by the token's routing weight (zero where it wasn't routed there), and
the NVFP4 weights are expanded to BF16 on every call.

Implement kernel_fn, and prepare to repack the weights (called once, untimed)."""

from typing import TypedDict

import torch

from nvfp4 import dequantize


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


def prepare(weights: Weights) -> Weights:
    """Called once, untimed, before any case. Repack the weights as you like."""
    return weights


def post_expert_norm(y, w, eps=1e-8):
    """Scale by (1 + w), round to BF16, then RMSNorm."""
    y = (y.float() * (1.0 + w.float())).to(torch.bfloat16).float()
    return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)


def kernel_fn(data: Inputs, weights: Weights) -> torch.Tensor:
    """This GPU's share of the MoE output, bfloat16 [M, H]: for each token, the sum over its
    experts that are local of weight * post_expert_norm(expert(x)). Rows whose experts are
    all elsewhere are zero."""
    cfg = data["config"]
    x = data["hidden_states"]
    d_expert = cfg["d_expert"]
    ids = data["topk_ids"].long()
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in range(cfg["local_start"], cfg["local_start"] + cfg["local_count"]):
        # Each token's routing weight for expert e: zero unless e is among its top 8.
        g = (data["topk_weights"] * (ids == e)).sum(-1)
        w13 = dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]).to(torch.bfloat16)
        h = x @ w13.T
        inter = torch.nn.functional.silu(h[:, :d_expert]) * h[:, d_expert:]
        w2 = dequantize(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]).to(torch.bfloat16)
        y = post_expert_norm(inter @ w2.T, weights["post_norm_w"])
        out += g[:, None] * y
    return out.to(torch.bfloat16)
