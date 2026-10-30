"""The routed experts of Muse Spark Open's MoE layer on one GPU, NVFP4 weights, BF16 activations.

Each token routes to 8 of 256 experts. "ep8": this GPU holds experts 0-31 and computes only
the (token, expert) pairs routed to them. "full": it holds all 256. Weights are generated
once (generate_weights) and passed to prepare(), untimed.

Per local pair: h = x @ W13^T; inter = silu(h[:, :D_EXPERT]) * h[:, D_EXPERT:];
y = post_expert_norm(inter @ W2^T); out[token] += topk_weight * y. Tokens with no local
pair get zeros.
"""

import torch

from checks import compare
from nvfp4 import BLOCK, dequantize, outlier_channels, random_weights

# Input width (the MoE block's 4096-wide space) and expert intermediate width.
D_HIDDEN = 4096
D_EXPERT = 4096
N_EXPERTS = 256
ROUTED_PER_TOKEN = 8
LOCAL = {"ep8": (0, 32), "full": (0, N_EXPERTS)}  # (first local expert, count)
# Each expert's output gets an RMSNorm (scale by 1 + w first, then normalize) before weighting.
POST_NORM_EPS = 1e-8
REL_L2_TOL = 1.5e-2
WEIGHTS_SEED = 5252
# Scale of the per-expert popularity (Student-t, 2 dof) added to uniform router scores.
POPULARITY = 0.03
# Timed repetitions per shape; each is checked.
MAX_REPEATS = 100


def generate_weights(seed: int, device: str = "cuda"):
    """NVFP4 per expert, W = e2m1 * block scale * global scale (nvfp4.dequantize):
    w13 [256, 2*D_EXPERT, D_HIDDEN/2] uint8, two E2M1 codes per byte (low nibble first), rows
    [:D_EXPERT] gate and [D_EXPERT:] up; w13_sf [256, 2*D_EXPERT, D_HIDDEN/16] float8_e4m3fn,
    one scale per 16 values; w13_gs [256] float32. w2, w2_sf, w2_gs likewise for
    [D_HIDDEN, D_EXPERT]. post_norm_w [D_HIDDEN] bf16."""
    gen = torch.Generator(device=device).manual_seed(seed)
    w13, w13_sf, gs13 = random_weights(2 * D_EXPERT, D_HIDDEN, generator=gen, device=device, leading=(N_EXPERTS,))
    w2, w2_sf, gs2 = random_weights(D_HIDDEN, D_EXPERT, generator=gen, device=device, leading=(N_EXPERTS,))
    return {
        "w13": w13,
        "w13_sf": w13_sf,
        "w13_gs": gs13.expand(N_EXPERTS).contiguous(),
        "w2": w2,
        "w2_sf": w2_sf,
        "w2_gs": gs2.expand(N_EXPERTS).contiguous(),
        "post_norm_w": (0.1 * torch.randn((D_HIDDEN,), generator=gen, device=device)).to(torch.bfloat16),
    }


def generate_input(bs: int, sharding: str, seed: int, dist: str = "normal", rank: str = "busiest",
                   device: str = "cuda", **_):
    """hidden_states [bs, D_HIDDEN] bf16; topk_ids [bs, 8] int32 (global expert ids);
    topk_weights [bs, 8] fp32, summing to 1 per token.

    dist="normal": top-8 of uniform scores plus a heavy-tailed per-expert popularity.
    dist="skewed": a few hot experts take most tokens, plus outlier activation channels.
    rank="busiest" (ep8): expert ids are rotated by multiples of 32 so experts 0-31 have the
    most pairs of any GPU."""
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((bs, D_HIDDEN), generator=gen, device=device).to(torch.bfloat16)
    scores = torch.rand((bs, N_EXPERTS), generator=gen, device=device)
    if dist == "normal":  # per-expert popularity, heavy-tailed
        z = torch.randn((N_EXPERTS,), generator=gen, device=device)
        u = torch.rand((N_EXPERTS,), generator=gen, device=device).clamp(min=1e-12)
        scores = scores + POPULARITY * z / (-u.log()).sqrt()  # Student-t, 2 degrees of freedom
    if dist == "skewed":
        heat = torch.rand((N_EXPERTS,), generator=gen, device=device).pow(6) * 4
        scores = scores + heat
        x = outlier_channels(x, generator=gen)
    routed = scores.topk(ROUTED_PER_TOKEN, dim=-1)
    start, count = LOCAL[sharding]
    ids = routed.indices
    if rank == "busiest" and count < N_EXPERTS:
        pairs = torch.bincount((ids // count).flatten(), minlength=N_EXPERTS // count)
        ids = (ids - count * pairs.argmax()) % N_EXPERTS  # busiest GPU's experts become 0-31
    return {
        "hidden_states": x,
        "topk_ids": ids.to(torch.int32),
        "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float(),
        "config": {"bs": bs, "d_hidden": D_HIDDEN, "d_expert": D_EXPERT, "n_experts": N_EXPERTS,
                   "local_start": start, "local_count": count, "top_k": ROUTED_PER_TOKEN, "sharding": sharding},
    }


def post_expert_norm(y, w):
    y = y.float() * (1.0 + w.float())
    return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)


def ref_kernel(data, weights):
    cfg = data["config"]
    x = data["hidden_states"].float()
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    ids = data["topk_ids"].long()
    lo, hi = cfg["local_start"], cfg["local_start"] + cfg["local_count"]
    for e in torch.unique(ids).tolist():
        if not lo <= e < hi:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]).T
        inter = torch.nn.functional.silu(h[:, :D_EXPERT]) * h[:, D_EXPERT:]
        y = inter @ dequantize(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]).T
        out.index_add_(0, tok, post_expert_norm(y, weights["post_norm_w"]) * data["topk_weights"][tok, slot].unsqueeze(1))
    return out.to(torch.bfloat16)


def check_implementation(data, output, weights):
    """bf16 [bs, D_HIDDEN]; relative L2 error within REL_L2_TOL over the whole output and
    within 5x REL_L2_TOL per row (checks.compare)."""
    ref = ref_kernel(data, weights)
    if not (ref != 0).any():  # never with rank="busiest"
        return False, "this case's inputs route no token to a local expert, so there is nothing to check"
    return compare(output, ref, REL_L2_TOL)


# A case's router_load and router_distinct (task.yml) are the mean routes and distinct experts
# that generate_input gives the local experts. Each repetition's routing varies around them, so
# the bound counts this share of each.
ROUTING_SHARE = 0.75


def speed_of_light_ns(bs: int, sharding: str, hw: dict, router_load: float = 0.0, router_distinct: float = 0.0,
                      **_) -> float:
    _, count = LOCAL[sharding]
    # Only local experts that receive at least one token need their weights read. This GPU is
    # the busiest, so it has at least the mean routes per GPU, on at least one expert.
    touched = max(1.0, ROUTING_SHARE * router_distinct)
    routes = max(bs * ROUTED_PER_TOKEN * count / N_EXPERTS, ROUTING_SHARE * router_load)
    per_expert = 3 * D_EXPERT * D_HIDDEN
    weight_bytes = touched * per_expert * (0.5 + 1 / BLOCK)
    act_bytes = bs * D_HIDDEN * 2 * 2
    flops = 2.0 * routes * per_expert
    return 1e9 * max((weight_bytes + act_bytes) / hw["hbm_bytes_per_s"], flops / hw["bf16_dense_flops_per_s"])
