"""The dense projections of Muse Spark Open, at per-GPU shapes for 8xH200 (TP8).

The attention projections, pooler and unpooler have BF16 weights; the router has FP32
weights. Weights are generated once (generate_weights) and passed to prepare(); each
case supplies only activations.
"""

import math

import torch

from checks import compare

# Per-GPU weights: [N, K].
LAYERS = {
    "qkv_gate": (2304, 8192),  # q 16x64 + k 2x64 + v 2x64 + gate 16x64, fused
    "o": (8192, 1024),         # 16x64 -> hidden, row-parallel
    "pooler": (4096, 8192),    # hidden -> MoE width, replicated on every rank
    "unpooler": (8192, 4096),  # MoE width -> hidden, replicated (see out_features)
    "router": (256, 8192),     # hidden -> 256 expert logits, FP32 weights and output
}
# At up to 8 tokens the unpooler computes only this GPU's slice: the first 1024 rows of W.
UNPOOL_SLICE = 1024
SLICED_MAX_TOKENS = 8
FP32_LAYERS = {"router"}
WEIGHTS_SEED = 5151
REL_L2_TOL = 1e-2
# Router tolerance. Splitting the FP32 weights into BF16 hi + lo passes; BF16 weights alone fail.
ROUTER_TOL = 1e-4
# Probe rows: x has a single nonzero entry, a power of two, so the output row is a scaled
# weight column. Each probe row must be within PROBE_TOL relative error, so weights must be
# used at their given precision (BF16, or FP32 / BF16 hi + lo for the router).
PROBE_EVERY = 8  # about one row in 8; at least one when M >= 8
PROBE_TOL = 1e-3


def out_features(layer: str, m: int) -> int:
    """N: the output width one GPU computes for this layer at M tokens."""
    if layer == "unpooler" and m <= SLICED_MAX_TOKENS:
        return UNPOOL_SLICE
    return LAYERS[layer][0]


def generate_weights(seed: int, device: str = "cuda"):
    """W [N, K] per layer, normal with std 1/sqrt(K): bfloat16, float32 for the router."""
    gen = torch.Generator(device=device).manual_seed(seed)
    return {name: (torch.randn((n, k), generator=gen, device=device) / math.sqrt(k)).to(
                torch.float32 if name in FP32_LAYERS else torch.bfloat16)
            for name, (n, k) in LAYERS.items()}


def generate_input(m: int, layer: str, seed: int, dist: str = "normal", device: str = "cuda", **_):
    """x: [M, K] bfloat16, standard normal, with probe rows. dist="outliers": 1% of channels scaled 30x."""
    gen = torch.Generator(device=device).manual_seed(seed)
    k = LAYERS[layer][1]
    n = out_features(layer, m)
    x = torch.randn((m, k), generator=gen, device=device)
    if dist == "outliers":
        idx = torch.randperm(k, generator=gen, device=device)[: max(1, k // 100)]
        x[:, idx] *= 30
    probes = torch.rand((m,), generator=gen, device=device) < 1 / PROBE_EVERY
    if m >= PROBE_EVERY:  # at least one, at a seeded row
        probes[torch.randint(m, (1,), generator=gen, device=device)] = True
    cols = torch.randint(k, (m,), generator=gen, device=device)
    signs = torch.randint(2, (m,), generator=gen, device=device) * 2 - 1
    powers = torch.randint(3, 8, (m,), generator=gen, device=device)  # |entry| 8 to 128
    spike = torch.zeros_like(x).scatter_(1, cols[:, None], (signs * 2.0 ** powers).float()[:, None])
    x = torch.where(probes[:, None], spike, x)
    return {"x": x.to(torch.bfloat16), "config": {"m": m, "n": n, "k": k, "layer": layer}}


def ref_kernel(data, weights):
    """out = x @ W[:N]^T: bfloat16 for the BF16 projections, float32 for the router."""
    layer, n = data["config"]["layer"], data["config"]["n"]
    out = data["x"].double() @ weights[layer][:n].double().T
    return out.to(torch.float32 if layer in FP32_LAYERS else torch.bfloat16)


def probe_rows(x: torch.Tensor) -> torch.Tensor:
    """Indices of rows with exactly one nonzero entry."""
    return ((x != 0).sum(-1) == 1).nonzero().flatten()


def check_implementation(data, output, weights):
    """Relative L2 error within tol (REL_L2_TOL, or ROUTER_TOL for the router) over the whole
    output and within 5x tol per row (checks.compare); each probe row within PROBE_TOL."""
    tol = ROUTER_TOL if data["config"]["layer"] in FP32_LAYERS else REL_L2_TOL
    ref = ref_kernel(data, weights)
    ok, msg = compare(output, ref, tol)
    rows = probe_rows(data["x"])
    if not ok or not len(rows):
        return ok, msg
    diff = (output[rows].double() - ref[rows].double()).norm(dim=-1) / ref[rows].double().norm(dim=-1).clamp(min=1e-30)
    worst = int(diff.argmax())
    if float(diff[worst]) > PROBE_TOL:
        return False, (f"{msg}, but probe row {int(rows[worst])} (one nonzero input, so a weight column, exact"
                       f" with the weights as given) is off by {float(diff[worst]):.2e} > {PROBE_TOL:g}: are the weights"
                       " used at their own precision?")
    return True, f"{msg}; {len(rows)} probe rows exact to {float(diff.max()):.1e}"


def speed_of_light_ns(m: int, layer: str, hw: dict, **_) -> float:
    n, k = out_features(layer, m), LAYERS[layer][1]
    w_bytes = 4 if layer in FP32_LAYERS else 2
    bytes_moved = n * k * w_bytes + m * k * 2 + m * n * w_bytes
    flops = 2.0 * m * n * k
    return 1e9 * max(bytes_moved / hw["hbm_bytes_per_s"], flops / hw["bf16_dense_flops_per_s"])
