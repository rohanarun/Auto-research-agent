# okc: problem=muse_bf16_proj
"""Router at m<=32: single-kernel hi/lo split-K Triton GEMM with FP32 atomic accumulation;
everything else falls back to the SGLang path."""

from typing import TypedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    pooler: torch.Tensor
    unpooler: torch.Tensor
    router: torch.Tensor


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


ROUTER_FP32_MAX_TOKENS = 8
SMALL_M = 32


@triton.jit
def _router_kernel(x_ptr, hi_ptr, lo_ptr, o_ptr, M, N, K_SPLIT, stride_xm, stride_wn, stride_om,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, NUM_STAGES: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_k * K_SPLIT + tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :]
    h_ptrs = hi_ptr + rn[:, None] * stride_wn + rk[None, :]
    l_ptrs = lo_ptr + rn[:, None] * stride_wn + rk[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in tl.range(0, K_SPLIT // BLOCK_K, num_stages=NUM_STAGES):
        x = tl.load(x_ptrs, mask=rm[:, None] < M, other=0.0)
        hi = tl.load(h_ptrs)
        lo = tl.load(l_ptrs)
        acc = tl.dot(x, tl.trans(hi), acc)
        acc = tl.dot(x, tl.trans(lo), acc)
        x_ptrs += BLOCK_K
        h_ptrs += BLOCK_K
        l_ptrs += BLOCK_K
    o_ptrs = o_ptr + rm[:, None] * stride_om + rn[None, :]
    tl.atomic_add(o_ptrs, acc, mask=rm[:, None] < M)


def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    return state


def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    layer, m, n = cfg["layer"], cfg["m"], cfg["n"]
    if layer == "router":
        if m <= SMALL_M:
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            bm = 16
            bn, bk, split = 16, 256, 8
            k = x.shape[1]
            _router_kernel[(n // bn, split, triton.cdiv(m, bm))](
                x, state["router_hi"], state["router_lo"], out, m, n, k // split,
                x.stride(0), k, out.stride(0),
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, NUM_STAGES=4, num_warps=4)
            return out
        if m <= ROUTER_FP32_MAX_TOKENS:
            return F.linear(x.float(), state["router"])
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
