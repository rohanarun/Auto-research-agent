"""SGLang's dense projections for Muse Spark Open on H200, at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f:

  qkv_gate, o, pooler    bf16_gemm_dispatch (cuBLAS on H200)
  unpooler               bf16_gemm_dispatch: up to 8 tokens on this GPU's 1/8 slice of the
                         weight, above that on the whole weight
  router                 FP32 weight: F.linear in FP32 up to 8 tokens, above that two BF16
                         GEMMs on a hi/lo split of the weight with FP32 accumulation
                         (MuseSpark1xRouter.forward_split2)

Differs from SGLang: the sliced unpooler's all-gather is not run.
"""

from typing import TypedDict

import torch
import torch.nn.functional as F
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl



class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]; up to 8 tokens, only rows 0-1023 (this GPU's slice)
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [M, K]
    config: Config


# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8


def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    return state


def kernel_fn(data, state):
    x, layer = data["x"], data["config"]["layer"]
    if layer != "router":
        # The unpooler's slice is its first N rows (all of them above 8 tokens).
        return _bf16_gemm_dispatch_impl(x, state[layer][: data["config"]["n"]], None)
    if x.shape[0] <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc
