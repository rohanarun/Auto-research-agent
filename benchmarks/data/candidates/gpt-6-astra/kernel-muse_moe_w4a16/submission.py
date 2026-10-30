# okc: problem=muse_moe_w4a16
"""NVFP4/BF16 Marlin MoE with GPU-local route compaction.

Weights use SGLang's lossless Marlin repacking. Every timed call rebuilds
alignment from the current router IDs; GEMMs retain FP32 reduction and BF16
activations. The route compaction and block-size changes are GPU-unverified.
"""

from typing import TypedDict

import types

import torch
import triton as tr
import triton.language as tl
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin


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


N_EXPERTS = 256
TOP_K = 8
POST_NORM_EPS = 1e-8


def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        # One global scale per expert for gate and up, as in SGLang's Marlin path.
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def prepare(weights):
    device = weights["w13"].device
    # One repacked bank: this GPU's 32 experts under EP8.
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    torch.cuda.synchronize()
    return {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        # MuseSpark1xFp4Experts binds the effective (1 + w) weight of the post-expert norm.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }


def kernel_fn(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    # Compact current routes on the GPU; no input-derived state survives a call.
    topk_weights = data["topk_weights"].contiguous()
    block_size_m = 8 if num_tokens <= 32 else (16 if num_tokens <= 128 else 32)
    route_count = num_tokens * TOP_K
    count = bank["count"]
    local_ids = torch.empty_like(data["topk_ids"], dtype=torch.int32)
    routes = torch.empty((count, route_count), device=x.device, dtype=torch.int32)
    counts = torch.empty((count,), device=x.device, dtype=torch.int32)
    capacity = route_count + count * (block_size_m - 1)
    sorted_ids = torch.empty((capacity,), device=x.device, dtype=torch.int32)
    block_experts = torch.empty((tr.cdiv(capacity, block_size_m),), device=x.device, dtype=torch.int32)
    n_padded = torch.empty((1,), device=x.device, dtype=torch.int32)
    _collect_routes[(count,)](
        data["topk_ids"], local_ids, routes, counts,
        route_count, cfg["local_start"], count, tr.next_power_of_2(route_count),
    )
    _align_routes[(count,)](
        routes, counts, sorted_ids, block_experts, n_padded,
        route_count, count, block_size_m,
        tr.next_power_of_2(route_count), tr.next_power_of_2(count),
    )

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


@tr.jit
def _collect_routes(Ids, Local, Routes, Counts,
                    R: tl.constexpr, START: tl.constexpr, E: tl.constexpr,
                    P: tl.constexpr):
    expert = tl.program_id(0)
    r = tl.arange(0, P)
    ids = tl.load(Ids + r, r < R, other=-1)
    local = (ids >= START) & (ids < START + E)
    if expert == 0:
        tl.store(Local + r, tl.where(local, ids - START, -1), r < R)
    match = (r < R) & (ids == START + expert)
    position = tl.cumsum(match.to(tl.int32), 0) - 1
    tl.store(Routes + expert * R + position, r, match)
    tl.store(Counts + expert, tl.sum(match.to(tl.int32), 0))


@tr.jit
def _align_routes(Routes, Counts, Sorted, Experts, Total,
                  R: tl.constexpr, E: tl.constexpr, BM: tl.constexpr,
                  P: tl.constexpr, EP: tl.constexpr):
    expert = tl.program_id(0)
    es = tl.arange(0, EP)
    counts = tl.load(Counts + es, es < E, other=0)
    padded = tl.cdiv(counts, BM) * BM
    offset = tl.sum(tl.where(es < expert, padded, 0), 0)
    count = tl.sum(tl.where(es == expert, counts, 0), 0)
    size = tl.cdiv(count, BM) * BM
    if expert == 0:
        tl.store(Total, tl.sum(padded, 0))
    # P is a power of two and at least BM for all task inputs. Thus rounding
    # a count <= R up to BM never exceeds P.
    r = tl.arange(0, P)
    route = tl.load(Routes + expert * R + r, r < count, other=R)
    tl.store(Sorted + offset + r, route, r < size)
    b = tl.arange(0, P // BM)
    tl.store(Experts + offset // BM + b, expert, b < size // BM)
