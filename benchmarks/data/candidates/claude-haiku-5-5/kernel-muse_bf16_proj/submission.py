# okc: problem=muse_bf16_proj
"""Dense projections in plain PyTorch, with a one-GEMM router above 8 tokens.

bf16 layers: x @ W[:n].T, as in the starter.
router, m <= 8: FP32 F.linear, as SGLang does.
router, m > 8: BF16 hi + lo split stacked along N into one [2N, K] weight. One GEMM with
FP32 output, then the hi and lo halves are added. The baseline runs two GEMMs and an add.
"""

from typing import TypedDict

import torch
import torch.nn.functional as F


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8


def prepare(weights: Weights) -> Weights:
    """Called once, untimed, before any case. Stacks the router's BF16 hi and lo halves."""
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_lo"] = torch.cat([hi, lo], dim=0).contiguous()  # [512, 8192]
    return state


def kernel_fn(data: Inputs, weights: Weights) -> torch.Tensor:
    x, config = data["x"], data["config"]
    layer = config["layer"]
    if layer == "router":
        if x.shape[0] <= ROUTER_FP32_MAX_TOKENS:
            return F.linear(x.float(), weights["router"])
        acc = torch.mm(x, weights["router_hi_lo"].t(), out_dtype=torch.float32)
        half = acc.shape[1] // 2
        return acc[:, :half] + acc[:, half:]
    return x @ weights[layer][: config["n"]].t()
